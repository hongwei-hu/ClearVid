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
import logging
from enum import Enum
from pathlib import Path

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
    cache_dir: Path | None = None,
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
        return _apply_tensorrt(model, fp16=fp16, tile_size=tile_size, cache_dir=cache_dir)

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

        compiled = torch.compile(model, mode="max-autotune", backend="inductor")
        logger.info("已启用 torch.compile (inductor) 加速")
        return compiled
    except Exception as exc:  # noqa: BLE001
        logger.warning("torch.compile 初始化失败，回退到标准推理: %s", exc)
        return model


# ---------------------------------------------------------------------------
# Tier 2: ONNX → TensorRT
# ---------------------------------------------------------------------------

def _apply_tensorrt(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    cache_dir: Path | None,
) -> object:
    """Export to ONNX, then build/load a TensorRT engine."""
    try:
        import torch
        import tensorrt  # noqa: F401 — ensure tensorrt is importable
    except ImportError:
        logger.warning("tensorrt 包未安装，回退到标准推理。安装: pip install tensorrt")
        return model

    if cache_dir is None:
        cache_dir = Path.cwd() / "weights" / "trt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine_path = _get_or_build_engine(model, fp16=fp16, tile_size=tile_size, cache_dir=cache_dir)
        wrapper = _TensorRTModelWrapper(engine_path, fp16=fp16)
        logger.info("已启用 TensorRT 加速 (engine: %s)", engine_path.name)
        return wrapper
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorRT 引擎构建失败，回退到标准推理: %s", exc)
        return model


def _get_or_build_engine(
    model: object,
    *,
    fp16: bool,
    tile_size: int,
    cache_dir: Path,
) -> Path:
    """Return path to a cached TRT engine, building one if necessary."""
    import torch

    # Deterministic cache key from model state + build params
    key_data = f"tile{tile_size}_fp16{fp16}"
    state_bytes = str(sum(p.numel() for p in model.parameters())).encode()
    digest = hashlib.sha256(state_bytes + key_data.encode()).hexdigest()[:16]
    engine_path = cache_dir / f"realesrgan_{digest}.engine"

    if engine_path.exists():
        logger.info("加载缓存的 TensorRT 引擎: %s", engine_path.name)
        return engine_path

    # Step 1: export ONNX
    onnx_path = cache_dir / f"realesrgan_{digest}.onnx"
    if not onnx_path.exists():
        logger.info("正在导出 ONNX 模型 (tile=%d) ...", tile_size)
        dummy = torch.randn(1, 3, tile_size, tile_size).to(next(model.parameters()).device)
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
            dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h4", 3: "w4"}},
        )
        logger.info("ONNX 导出完成: %s", onnx_path.name)

    # Step 2: build TensorRT engine
    logger.info("正在构建 TensorRT 引擎 (首次可能需要几分钟) ...")
    _build_trt_engine(onnx_path, engine_path, fp16=fp16, tile_size=tile_size)
    logger.info("TensorRT 引擎构建完成: %s", engine_path.name)
    return engine_path


def _build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    fp16: bool,
    tile_size: int,
) -> None:
    """Build a TensorRT engine from an ONNX model."""
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"ONNX 解析失败:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    min_hw = max(64, tile_size // 4)
    max_hw = tile_size * 2
    profile.set_shape(
        "input",
        min=(1, 3, min_hw, min_hw),
        opt=(1, 3, tile_size, tile_size),
        max=(1, 3, max_hw, max_hw),
    )
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT 引擎序列化失败")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)


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
        torch.cuda.current_stream().synchronize()

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
