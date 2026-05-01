"""TensorRT / torch.compile() inference accelerator for Real-ESRGAN models.

Provides two tiers of acceleration:
- **torch.compile**: Uses PyTorch 2.x inductor backend. Always available with
  PyTorch >=2.0. Typical speedup: 1.3-2x on first-run compilation, then cached.
- **TensorRT**: Exports model to ONNX -> builds TensorRT engine. Requires the
  ``tensorrt`` package. Typical speedup: 2-4x over eager PyTorch.

Engine building is a **one-time deployment step** that can take 10-30 minutes.
Once cached, subsequent loads are near-instant.  The build can be done:
- Via the GUI "Deploy TensorRT Engine" button (recommended)
- Via CLI ``clearvid warmup``
- Automatically (opt-in, off by default during export)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import subprocess
import sys
import time as _time_module
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

TRT_MAX_BATCH = 16
TRT_PREFERRED_BATCHES = (16, 8, 4, 2, 1)
TRT_PREFERRED_TILE_SIZES = (1024, 768, 512, 256, 128)


class InferenceAccelerator(str, Enum):
    NONE = "none"
    COMPILE = "compile"
    TENSORRT = "tensorrt"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def check_engine_ready(
    model: object,
    *,
    fp16: bool = True,
    tile_size: int = 512,
    batch_size: int = 1,
    cache_dir: Path | None = None,
    weight_path: Path | None = None,
) -> tuple[bool, str]:
    """Check whether a cached TRT engine exists for *model*.

    Returns ``(ready, message)``.  *ready* is True when the ``.engine`` file
    is cached on disk and non-empty.  *message* is a human-readable status
    string suitable for UI display.
    """
    if cache_dir is None:
        from clearvid.app.bootstrap.paths import TRT_CACHE_DIR
        cache_dir = TRT_CACHE_DIR
    batch_size = _normalize_trt_batch(batch_size)

    digest = _engine_cache_key(model, fp16, tile_size, batch_size, weight_path)
    engine_path = cache_dir / f"realesrgan_{digest}.engine"
    failed_path = cache_dir / f"realesrgan_{digest}.failed"

    if engine_path.exists() and engine_path.stat().st_size > 0:
        return True, "TensorRT 引擎已就绪"
    prev = _read_failed_marker(failed_path)
    if prev is not None:
        age_h = (_time_module.time() - prev["timestamp"]) / 3600
        return False, f"上次部署失败 ({age_h:.1f} 小时前)"
    return False, "TensorRT 引擎尚未部署"


def find_compatible_engine(
    model: object,
    *,
    fp16: bool = True,
    cache_dir: Path | None = None,
    weight_path: Path | None = None,
) -> tuple[int, int, str] | None:
    """Scan *cache_dir* for any deployed engine matching the same weight + fp16.

    Searches across common tile sizes and batch sizes, preferring higher batch
    engines first since they give
    better GPU utilisation.  Returns ``(tile_size, batch_size, engine_path_str)``
    for the first match found, or ``None`` if no compatible engine exists.
    """
    if cache_dir is None:
        from clearvid.app.bootstrap.paths import TRT_CACHE_DIR
        cache_dir = TRT_CACHE_DIR

    # Prefer larger batch (better GPU utilisation), then larger tiles (fewer calls)
    for candidate_batch in TRT_PREFERRED_BATCHES:
        for candidate_tile in TRT_PREFERRED_TILE_SIZES:
            digest = _engine_cache_key(model, fp16, candidate_tile, candidate_batch, weight_path)
            engine_path = cache_dir / f"realesrgan_{digest}.engine"
            if engine_path.exists() and engine_path.stat().st_size > 0:
                return candidate_tile, candidate_batch, str(engine_path)
    return None


def accelerate_model(
    model: object,
    accelerator: InferenceAccelerator,
    *,
    fp16: bool = True,
    tile_size: int = 512,
    batch_size: int = 1,
    cache_dir: Path | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    weight_path: Path | None = None,
    trt_build_timeout: int | None = None,
    build_if_missing: bool = True,
    low_load: bool = True,
) -> object:
    """Wrap *model* with the requested accelerator.

    Parameters
    ----------
    weight_path:
        Path to the model weight file (.pth). Used to compute a cache key.
    trt_build_timeout:
        Override the auto-detected TRT engine build timeout (seconds).
    build_if_missing:
        When ``True`` (default), a missing TRT engine triggers an automatic
        build.  Set to ``False`` during exports to prevent blocking the user
        with a long first-time build.
    low_load:
        When ``True`` (default), the TRT build uses reduced GPU workspace,
        lower optimization level, and idle CPU priority to avoid freezing
        the system.
    """
    if accelerator == InferenceAccelerator.NONE:
        return model

    if accelerator == InferenceAccelerator.COMPILE:
        return _apply_torch_compile(model)

    if accelerator == InferenceAccelerator.TENSORRT:
        return _apply_tensorrt(
            model, fp16=fp16, tile_size=tile_size, batch_size=batch_size,
            cache_dir=cache_dir, progress_callback=progress_callback,
            weight_path=weight_path, build_timeout=trt_build_timeout,
            build_if_missing=build_if_missing, low_load=low_load,
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
            logger.warning("torch.compile 不可用 (需要 PyTorch >= 2.0)，回退到标准推理")
            return model

        try:
            import triton  # noqa: F401
        except ImportError:
            logger.warning("triton 未安装，torch.compile (inductor) 不可用，回退到标准推理。安装: pip install triton")
            return model

        compiled = torch.compile(model, mode="reduce-overhead", backend="inductor")
        logger.info("已启用 torch.compile (inductor) 加速")
        return compiled
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile 初始化失败，回退到标准推理: %s", exc)
        return model


# ---------------------------------------------------------------------------
# Tier 2: ONNX -> TensorRT (subprocess-isolated build)
# ---------------------------------------------------------------------------

def _apply_tensorrt(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    batch_size: int = 1,
    cache_dir: Path | None,
    progress_callback: Callable[[int, str], None] | None = None,
    weight_path: Path | None = None,
    build_timeout: int | None = None,
    build_if_missing: bool = True,
    low_load: bool = True,
) -> object:
    """Export to ONNX, then build/load a TensorRT engine.

    When *build_if_missing* is False and no cached engine exists, raises
    ``RuntimeError`` immediately instead of starting a long build.
    """
    try:
        import torch  # noqa: F811
        import tensorrt  # noqa: F401
    except ImportError:
        logger.warning("tensorrt 包未安装，回退到标准推理。安装: pip install tensorrt")
        if progress_callback is not None:
            progress_callback(11, "TensorRT 未安装，使用标准 PyTorch 推理")
        return model

    # Pre-check for ONNX — needed by torch.onnx.export()
    try:
        import onnx  # noqa: F401
    except Exception as _e:
        detail = str(_e)
        logger.error("onnx import 失败: %s (Python: %s)", detail, sys.executable)
        msg = (
            f"onnx 包导入失败: {detail}。"
            "请确保 onnx 已安装在当前 Python 环境中:\n"
            f"  \"{sys.executable}\" -m pip install onnx>=1.17,<2.0"
        )
        if progress_callback is not None:
            progress_callback(11, msg)
        raise RuntimeError(msg) from None

    if cache_dir is None:
        from clearvid.app.bootstrap.paths import TRT_CACHE_DIR
        cache_dir = TRT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size = _normalize_trt_batch(batch_size)

    try:
        engine_path = _get_or_build_engine(
            model, fp16=fp16, tile_size=tile_size, batch_size=batch_size,
            cache_dir=cache_dir, progress_callback=progress_callback,
            weight_path=weight_path, build_timeout=build_timeout,
            build_if_missing=build_if_missing, low_load=low_load,
        )
        wrapper = _TensorRTModelWrapper(engine_path, fp16=fp16)
        logger.info("已启用 TensorRT 加速 (engine: %s)", engine_path.name)
        return wrapper
    except RuntimeError:
        # build_if_missing=False: engine not ready, re-raise to caller
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("TensorRT 引擎构建失败，回退到标准推理: %s", exc, exc_info=True)
        if progress_callback is not None:
            progress_callback(
                11,
                "TensorRT 构建失败，尝试 torch.compile 降级加速...",
            )
        compiled = _apply_torch_compile(model)
        if compiled is not model:
            logger.info("TensorRT 失败，已降级到 torch.compile 加速")
            if progress_callback is not None:
                progress_callback(
                    11,
                    f"推理加速就绪: torch.compile (TensorRT 构建失败: {exc})",
                )
            return compiled
        if progress_callback is not None:
            progress_callback(
                11,
                "推理加速不可用，使用标准 PyTorch (TensorRT/torch.compile 均失败)",
            )
        return model


# ---------------------------------------------------------------------------
# GPU detection helpers
# ---------------------------------------------------------------------------

def _get_gpu_sm_version() -> int:
    """Return the CUDA compute capability SM version * 10 (e.g. 89 for SM 8.9).
    Returns 0 when CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return cap[0] * 10 + cap[1]
    except Exception:
        pass
    return 0


def _resolve_trt_timeout(
    param_count: int, sm_version: int, user_override: int | None, low_load: bool,
) -> int:
    """Choose a sensible TRT engine build timeout.

    In *low_load* mode, timeouts are doubled to allow the slower, low-resource
    build to complete without being killed prematurely.
    """
    if user_override is not None:
        return max(60, min(user_override, 7200))

    # Base timeout by model complexity
    if param_count > 10_000_000:      # RRDB-class (~16.7M)
        base = 1800                     # 30 min
    elif param_count > 1_000_000:
        base = 600                      # 10 min
    else:                               # SRVGGNetCompact etc.
        base = 300                      # 5 min

    # Low-load mode: allow longer build time
    if low_load:
        base = int(base * 2.0)

    # Scale by GPU generation
    if sm_version >= 100:               # Blackwell / RTX 50 series
        base = int(base * 0.5)
    elif sm_version >= 90:
        base = int(base * 0.55)
    elif sm_version >= 89:              # Ada Lovelace / RTX 40 series
        base = int(base * 0.65)
    elif sm_version >= 80:              # Ampere / RTX 30 series
        base = int(base * 0.8)

    return max(180, base)               # never shorter than 3 minutes


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _engine_cache_key(
    model: object,
    fp16: bool,
    tile_size: int,
    batch_size: int,
    weight_path: Path | None = None,
) -> str:
    """Deterministic cache key from model parameters + weight identity."""
    key_data = f"tile{tile_size}_fp16{fp16}_batch{batch_size}"

    if weight_path is not None and weight_path.exists():
        with open(weight_path, "rb") as fh:
            weight_prefix = fh.read(8192)
        weight_hash = hashlib.sha256(weight_prefix).hexdigest()[:8]
    else:
        weight_hash = str(sum(p.numel() for p in model.parameters()))

    return hashlib.sha256(
        f"{weight_hash}|{key_data}".encode()
    ).hexdigest()[:16]


def _normalize_trt_batch(batch_size: int) -> int:
    return max(1, min(int(batch_size), TRT_MAX_BATCH))


# ---------------------------------------------------------------------------
# Failed-marker helpers
# ---------------------------------------------------------------------------

_FAILED_MARKER_TTL = 86400  # 24 hours


def _read_failed_marker(marker_path: Path) -> dict | None:
    if not marker_path.exists():
        return None
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
        age = _time_module.time() - data.get("timestamp", 0)
        if age > _FAILED_MARKER_TTL:
            marker_path.unlink(missing_ok=True)
            return None
        return data
    except Exception:
        marker_path.unlink(missing_ok=True)
        return None


def _write_failed_marker(marker_path: Path, reason: str) -> None:
    marker_path.write_text(
        json.dumps({
            "timestamp": _time_module.time(),
            "reason": reason[:500],
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("已记录 TRT 构建失败标记: %s", marker_path.name)


def _clear_failed_marker(marker_path: Path) -> None:
    marker_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Engine build orchestration
# ---------------------------------------------------------------------------

def _get_or_build_engine(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    batch_size: int = 1,
    cache_dir: Path,
    progress_callback: Callable[[int, str], None] | None = None,
    weight_path: Path | None = None,
    build_timeout: int | None = None,
    build_if_missing: bool = True,
    low_load: bool = True,
) -> Path:
    """Return path to a cached TRT engine.

    When *build_if_missing* is False and no cached engine exists, raises
    ``RuntimeError`` immediately so the caller can prompt the user to deploy.
    """
    import torch
    batch_size = _normalize_trt_batch(batch_size)

    digest = _engine_cache_key(model, fp16, tile_size, batch_size, weight_path)
    engine_path = cache_dir / f"realesrgan_{digest}.engine"
    failed_path = cache_dir / f"realesrgan_{digest}.failed"

    # --- Cache hit -----------------------------------------------------------
    if engine_path.exists() and engine_path.stat().st_size > 0:
        _clear_failed_marker(failed_path)
        logger.info("加载缓存的 TensorRT 引擎: %s", engine_path.name)
        return engine_path

    # --- Build policy check --------------------------------------------------
    if not build_if_missing:
        prev = _read_failed_marker(failed_path)
        hint = ""
        if prev is not None:
            age_h = (_time_module.time() - prev["timestamp"]) / 3600
            hint = f"。上次部署在 {age_h:.1f} 小时前失败"
        raise RuntimeError(
            f"TensorRT 引擎尚未部署，请先点击'部署 TensorRT 引擎'按钮完成首次构建{hint}"
        )

    # --- Previous failure check (auto-build path only) ----------------------
    # When the user explicitly clicks deploy (build_if_missing=True), clear
    # any stale failure marker and retry.  Only block on the failed marker
    # for background auto-builds (build_if_missing=False, handled above).
    prev_failure = _read_failed_marker(failed_path)
    if prev_failure is not None:
        age_h = (_time_module.time() - prev_failure["timestamp"]) / 3600
        logger.info(
            "发现历史失败标记 (%.1f 小时前)，本次显式部署将清除并重试: %s",
            age_h, failed_path.name,
        )
        _clear_failed_marker(failed_path)

    # --- Step 1: export ONNX (in-process) ------------------------------------
    onnx_path = cache_dir / f"realesrgan_{digest}.onnx"
    if not onnx_path.exists():
        logger.info("正在导出 ONNX 模型 (tile=%d, batch=%d) ...", tile_size, batch_size)
        export_model = copy.deepcopy(model)
        dummy = torch.randn(batch_size, 3, tile_size, tile_size).to(
            next(model.parameters()).device,
        )
        if fp16:
            dummy = dummy.half()
            export_model = export_model.half()
        torch.onnx.export(
            export_model,
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
        # Free GPU memory before the heavy build
        del export_model, dummy
        torch.cuda.empty_cache()

    # --- Step 2: resolve build timeout ---------------------------------------
    param_count = sum(p.numel() for p in model.parameters())
    sm_version = _get_gpu_sm_version()
    timeout = _resolve_trt_timeout(param_count, sm_version, build_timeout, low_load)

    load_label = "低负载" if low_load else "标准"
    logger.info(
        "模型参数量: %.2fM | GPU SM: %d | 模式: %s | TRT 构建超时: %ds",
        param_count / 1e6, sm_version, load_label, timeout,
    )
    if progress_callback is not None:
        progress_callback(
            0,
            f"部署 TensorRT 引擎 ({param_count/1e6:.1f}M 参数, "
            f"{load_label}模式, 最长 {timeout//60} 分钟)...",
        )

    # --- Step 3: build engine in subprocess ----------------------------------
    logger.info("正在构建 TensorRT 引擎 (%s, 子进程隔离) ...", load_label)
    try:
        _build_trt_engine_subprocess(
            onnx_path, engine_path,
            fp16=fp16, tile_size=tile_size, batch_size=batch_size,
            timeout=timeout,
            progress_callback=progress_callback,
            low_load=low_load,
        )
    except Exception as exc:
        _write_failed_marker(failed_path, str(exc))
        raise

    if not engine_path.exists() or engine_path.stat().st_size == 0:
        _write_failed_marker(failed_path, "引擎文件为空")
        raise RuntimeError("TRT 引擎文件未生成")

    # --- Success -------------------------------------------------------------
    _clear_failed_marker(failed_path)
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

# --- Low-load build settings ---
# Reduce GPU workspace and skip heavy kernel auto-tuning to keep the system
# responsive during the build.  The resulting engine is ~5-10% slower at
# inference but the build completes without freezing the desktop.
low_load = args.get("low_load", True)
if low_load:
    # 256 MB workspace (was 1 GB) -- enough for SR models, less GPU pressure
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    # Level 0: skip most kernel auto-tuning. Builds 3-5x faster with minimal
    # inference speed penalty for the simple feed-forward SR architectures.
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 0
    # Limit auxiliary streams to reduce GPU context switching overhead
    if hasattr(config, "max_aux_streams"):
        config.max_aux_streams = 1
else:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 2

if args["fp16"]:
    config.set_flag(trt.BuilderFlag.FP16)

tile = args["tile_size"]
batch = args["batch_size"]

min_hw = max(32, tile // 4)
max_hw = tile + 32
max_batch = min(batch, 16)

profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, min_hw, min_hw),
    opt=(batch, 3, tile, tile),
    max=(max_batch, 3, max_hw, max_hw),
)
config.add_optimization_profile(profile)

mode_label = "low-load" if low_load else "standard"
print(
    f"Building TRT engine ({mode_label}): opt=({batch},3,{tile},{tile}) "
    f"range=[{min_hw}..{max_hw}] max_batch={max_batch}",
    flush=True,
)
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
    low_load: bool = True,
) -> None:
    """Build TRT engine in an isolated subprocess.

    In *low_load* mode (default), the subprocess runs at idle priority and
    uses minimal GPU resources to avoid freezing the desktop.
    """
    import time as _time

    args_json = json.dumps({
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "fp16": fp16,
        "tile_size": tile_size,
        "batch_size": batch_size,
        "low_load": low_load,
    })

    creation_flags = 0
    if sys.platform == "win32":
        # IDLE priority: only runs when the system has nothing else to do
        creation_flags = (
            subprocess.IDLE_PRIORITY_CLASS
            | subprocess.CREATE_NO_WINDOW
        )

    proc = subprocess.Popen(
        [sys.executable, "-c", _TRT_BUILD_SCRIPT, args_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )

    t_start = _time.monotonic()
    poll_interval = 15  # seconds — longer interval in low-load mode
    try:
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                elapsed = _time.monotonic() - t_start
                if elapsed > timeout:
                    raise
                if progress_callback is not None:
                    progress_callback(
                        int(min(elapsed / max(timeout, 1), 1.0) * 100),
                        f"TensorRT 引擎部署中... ({elapsed:.0f}s / 最长{timeout}s)",
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
    through a TensorRT engine."""

    def __init__(self, engine_path: Path, *, fp16: bool = True) -> None:
        import tensorrt as trt

        self._fp16 = fp16
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self.max_batch = 1
        self.opt_shape: tuple[int, ...] | None = None
        self.max_shape: tuple[int, ...] | None = None
        try:
            profile_shape = _read_engine_profile_shapes(self._engine, "input", 0)
            self.opt_shape = tuple(profile_shape[1])
            self.max_shape = tuple(profile_shape[2])
            self.max_batch = int(self.max_shape[0])
        except Exception:
            pass

        # Warm up: one dummy inference to trigger lazy CUDA init
        import torch
        try:
            opt_shape = self.opt_shape
            if opt_shape is None:
                opt_shape = tuple(_read_engine_profile_shapes(self._engine, "input", 0)[1])
            dummy = torch.randn(*opt_shape, device="cuda", dtype=torch.float16 if fp16 else torch.float32)
            self.__call__(dummy)
            del dummy
        except Exception:
            pass

    def __call__(self, x):  # noqa: ANN001, ANN204
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
        return iter([])

    def half(self):
        return self

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _read_engine_profile_shapes(engine: Any, tensor_name: str, profile_index: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if hasattr(engine, "get_tensor_profile_shape"):
        min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(tensor_name, profile_index)
    else:
        min_shape, opt_shape, max_shape = engine.get_profile_shape(tensor_name, profile_index)
    return _shape_tuple(min_shape), _shape_tuple(opt_shape), _shape_tuple(max_shape)


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
