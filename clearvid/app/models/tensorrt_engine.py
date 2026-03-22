"""TensorRT / torch.compile() inference accelerator for Real-ESRGAN models.

Provides two tiers of acceleration:
- **torch.compile**: Uses PyTorch 2.x inductor backend. Always available with
  PyTorch ≥2.0. Typical speedup: 1.3-2× on first-run compilation, then cached.
- **TensorRT**: Exports model to ONNX → builds TensorRT engine. Requires the
  ``tensorrt`` package. Typical speedup: 2-4× over eager PyTorch.

The accelerator wraps the model's forward pass transparently — the rest of the
RealESRGANer tiling/padding logic is untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class InferenceAccelerator(str, Enum):
    NONE = "none"
    COMPILE = "compile"
    TENSORRT = "tensorrt"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def accelerate_model(
    model: object,
    accelerator: InferenceAccelerator,
    *,
    fp16: bool = True,
    tile_size: int = 512,
    batch_size: int = 1,
    cache_dir: Path | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> object:
    """Wrap *model* with the requested accelerator.  Returns the (possibly
    wrapped) model — always usable as a drop-in replacement for the original.

    If acceleration is unavailable or fails, logs a warning and returns the
    original model unchanged.
    """
    if accelerator == InferenceAccelerator.NONE:
        return model

    if accelerator == InferenceAccelerator.COMPILE:
        return _apply_torch_compile(model)

    if accelerator == InferenceAccelerator.TENSORRT:
        return _apply_tensorrt(
            model, fp16=fp16, tile_size=tile_size, batch_size=batch_size,
            cache_dir=cache_dir, progress_callback=progress_callback,
        )

    return model


# ---------------------------------------------------------------------------
# Tier 1: torch.compile()
# ---------------------------------------------------------------------------

def _apply_torch_compile(model: object) -> object:
    """Use ``torch.compile()`` with the inductor backend."""
    try:
        import torch  # noqa: F811

        if not hasattr(torch, "compile"):
            logger.warning("torch.compile 不可用 (需要 PyTorch ≥ 2.0)，回退到标准推理")
            return model

        # inductor backend requires triton for CUDA kernel compilation
        try:
            import triton  # noqa: F401
        except ImportError:
            logger.warning("triton 未安装，torch.compile (inductor) 不可用，回退到标准推理。安装: pip install triton")
            return model

        compiled = torch.compile(model, mode="max-autotune", backend="inductor")
        logger.info("已启用 torch.compile (inductor) 加速")
        return compiled
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile 初始化失败，回退到标准推理: %s", exc)
        return model


# ---------------------------------------------------------------------------
# Tier 2: ONNX → TensorRT (subprocess-isolated build)
# ---------------------------------------------------------------------------

def _apply_tensorrt(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    batch_size: int = 1,
    cache_dir: Path | None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> object:
    """Export to ONNX, then build/load a TensorRT engine."""
    try:
        import torch  # noqa: F811
        import tensorrt  # noqa: F401 — ensure tensorrt is importable
    except ImportError:
        logger.warning("tensorrt 包未安装，回退到标准推理。安装: pip install tensorrt")
        return model

    if cache_dir is None:
        from clearvid.app.bootstrap.paths import TRT_CACHE_DIR
        cache_dir = TRT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine_path = _get_or_build_engine(
            model, fp16=fp16, tile_size=tile_size, batch_size=batch_size,
            cache_dir=cache_dir, progress_callback=progress_callback,
        )
        wrapper = _TensorRTModelWrapper(engine_path, fp16=fp16)
        logger.info("已启用 TensorRT 加速 (engine: %s)", engine_path.name)
        return wrapper
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorRT 引擎构建失败，回退到标准推理: %s", exc, exc_info=True)
        return model


def _engine_cache_key(model: object, fp16: bool, tile_size: int, batch_size: int) -> str:
    """Deterministic cache key from model state + build params."""
    key_data = f"tile{tile_size}_fp16{fp16}_batch{batch_size}"
    state_bytes = str(sum(p.numel() for p in model.parameters())).encode()
    return hashlib.sha256(state_bytes + key_data.encode()).hexdigest()[:16]


def _get_or_build_engine(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    batch_size: int = 1,
    cache_dir: Path,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    """Return path to a cached TRT engine, building one if necessary."""
    import torch

    digest = _engine_cache_key(model, fp16, tile_size, batch_size)
    engine_path = cache_dir / f"realesrgan_{digest}.engine"

    if engine_path.exists() and engine_path.stat().st_size > 0:
        logger.info("加载缓存的 TensorRT 引擎: %s", engine_path.name)
        return engine_path

    # Step 1: export ONNX (quick, in-process)
    onnx_path = cache_dir / f"realesrgan_{digest}.onnx"
    if not onnx_path.exists():
        logger.info("正在导出 ONNX 模型 (tile=%d, batch=%d) ...", tile_size, batch_size)
        dummy = torch.randn(batch_size, 3, tile_size, tile_size).to(
            next(model.parameters()).device,
        )
        if fp16:
            dummy = dummy.half()
            model = model.half()
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 2: "h", 3: "w"},
                "output": {0: "batch", 2: "h4", 3: "w4"},
            },
        )
        logger.info("ONNX 导出完成: %s", onnx_path.name)

    # Step 2: build TensorRT engine in ISOLATED SUBPROCESS
    # This prevents GPU-intensive TRT builder from freezing the main app.
    # Dynamic timeout based on model complexity (RRDB ~16.7M params needs much more time)
    param_count = sum(p.numel() for p in model.parameters())
    if param_count > 10_000_000:      # RRDB-class models
        build_timeout = 1800            # 30 min
    elif param_count > 1_000_000:
        build_timeout = 600             # 10 min
    else:                               # SRVGGNetCompact etc.
        build_timeout = 300             # 5 min
    logger.info(
        "模型参数量: %.2fM → TRT 构建超时: %ds",
        param_count / 1e6, build_timeout,
    )
    if progress_callback is not None:
        progress_callback(
            11,
            f"首次构建 TensorRT 引擎 ({param_count/1e6:.1f}M 参数, "
            f"最长 {build_timeout//60} 分钟, 后续秒级加载)...",
        )
    logger.info("正在构建 TensorRT 引擎 (首次构建, 子进程隔离) ...")
    _build_trt_engine_subprocess(
        onnx_path, engine_path,
        fp16=fp16, tile_size=tile_size, batch_size=batch_size,
        timeout=build_timeout,
        progress_callback=progress_callback,
    )

    if not engine_path.exists() or engine_path.stat().st_size == 0:
        raise RuntimeError("TRT 引擎文件未生成")

    logger.info("TensorRT 引擎构建完成: %s", engine_path.name)
    return engine_path


# -- Self-contained build script executed in subprocess ---------------------
_TRT_BUILD_SCRIPT = r'''
import json, os, sys
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

args = json.loads(sys.argv[1])

import tensorrt as trt

trt_logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(trt_logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, trt_logger)

with open(args["onnx_path"], "rb") as f:
    ok = parser.parse(f.read())
if not ok:
    errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
    print(f"ONNX_PARSE_FAILED: {errors}", file=sys.stderr)
    sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

# Optimization level 2 (default=3): much faster build, ~5% less optimized kernels
if hasattr(config, "builder_optimization_level"):
    config.builder_optimization_level = 2

if args["fp16"]:
    config.set_flag(trt.BuilderFlag.FP16)

tile = args["tile_size"]
batch = args["batch_size"]
min_hw = max(64, tile // 4)
max_hw = min(tile + 256, 768)
max_batch = min(batch, 8)

profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, min_hw, min_hw),
    opt=(batch, 3, tile, tile),
    max=(max_batch, 3, max_hw, max_hw),
)
config.add_optimization_profile(profile)

print(f"Building TRT engine: opt=({batch},3,{tile},{tile}) max=({max_batch},3,{max_hw},{max_hw})", flush=True)
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    print("ENGINE_BUILD_RETURNED_NONE", file=sys.stderr)
    sys.exit(1)

with open(args["engine_path"], "wb") as f:
    f.write(engine_bytes)

print("OK", flush=True)
'''


def _build_trt_engine_subprocess(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    tile_size: int,
    batch_size: int,
    timeout: int = 600,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    """Build TRT engine in an isolated subprocess with timeout.

    - Subprocess gets its own CUDA context → main app stays responsive
    - On timeout the process is killed → GPU memory released immediately
    - Lower priority on Windows to reduce desktop compositor stutter
    """
    import time as _time

    args_json = json.dumps({
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "fp16": fp16,
        "tile_size": tile_size,
        "batch_size": batch_size,
    })

    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = (
            subprocess.BELOW_NORMAL_PRIORITY_CLASS
            | subprocess.CREATE_NO_WINDOW
        )

    proc = subprocess.Popen(
        [sys.executable, "-c", _TRT_BUILD_SCRIPT, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )

    # Poll subprocess with periodic progress updates
    t_start = _time.monotonic()
    poll_interval = 10  # seconds
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=poll_interval)
                # Process finished
                break
            except subprocess.TimeoutExpired:
                elapsed = _time.monotonic() - t_start
                if elapsed > timeout:
                    raise  # will be caught by outer handler
                if progress_callback is not None:
                    progress_callback(
                        11,
                        f"TensorRT 引擎构建中... ({elapsed:.0f}s / 最长{timeout}s)",
                    )
                continue

        if proc.returncode != 0:
            err_msg = stderr.decode(errors="replace").strip()
            if engine_path.exists():
                try:
                    engine_path.unlink()
                except OSError:
                    pass
            raise RuntimeError(f"TRT 引擎构建失败 (exit={proc.returncode}): {err_msg}")
        logger.info("TRT 子进程输出: %s", stdout.decode(errors="replace").strip())
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        # Clean up partial engine file
        if engine_path.exists():
            try:
                engine_path.unlink()
            except OSError:
                pass
        raise TimeoutError(
            f"TRT 引擎构建超时 ({timeout}s)。"
            "建议减小 batch_size 或使用标准 PyTorch 推理。"
        )


class _TensorRTModelWrapper:
    """Drop-in replacement for a PyTorch ``nn.Module`` that runs inference
    through a TensorRT engine.  Supports the ``__call__(tensor) → tensor``
    interface used by ``RealESRGANer``.
    """

    def __init__(self, engine_path: Path, *, fp16: bool = True) -> None:
        import tensorrt as trt

        self._fp16 = fp16
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

    def __call__(self, x):  # noqa: ANN001, ANN204
        """Run TensorRT inference — compatible with ``model(tensor)`` call."""
        import torch

        b, c, h, w = x.shape
        self._context.set_input_shape("input", (b, c, h, w))

        output_shape = (b, c, h * 4, w * 4)
        output = torch.empty(output_shape, dtype=x.dtype, device=x.device)

        self._context.set_tensor_address("input", x.data_ptr())
        self._context.set_tensor_address("output", output.data_ptr())
        self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        return output

    def parameters(self):
        """Compatibility: return empty iterator (TRT engine has no PyTorch params)."""
        return iter([])

    def half(self):
        """No-op — TRT engine precision is fixed at build time."""
        return self

    def eval(self):
        """No-op — TRT engine is always in eval mode."""
        return self

    def to(self, *args, **kwargs):
        """No-op — TRT engine memory is managed internally."""
        return self


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def detect_best_accelerator() -> InferenceAccelerator:
    """Auto-detect the best available accelerator."""
    try:
        import tensorrt  # noqa: F401
        return InferenceAccelerator.TENSORRT
    except ImportError:
        pass

    try:
        import torch
        if hasattr(torch, "compile"):
            import triton  # noqa: F401
            return InferenceAccelerator.COMPILE
    except ImportError:
        pass

    return InferenceAccelerator.NONE


def describe_accelerator(accel: InferenceAccelerator) -> str:
    """Human-readable description for GUI/logs."""
    return {
        InferenceAccelerator.NONE: "标准 PyTorch",
        InferenceAccelerator.COMPILE: "torch.compile (inductor)",
        InferenceAccelerator.TENSORRT: "TensorRT",
    }.get(accel, str(accel))
