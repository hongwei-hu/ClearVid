from __future__ import annotations

import importlib
import logging
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

import cv2
import numpy as np

from clearvid.app.models import stream_codec
from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.models.gfpgan_runner import GFPGANRestorer
from clearvid.app.models.perf_diagnostics import (
    CpuUsageTracker,
    GpuSampler,
    format_gpu_snapshot,
    format_gpu_summary,
    format_queue_info,
)
from clearvid.app.models.tensorrt_engine import (
    InferenceAccelerator,
    TRT_MAX_BATCH,
    accelerate_model,
    check_engine_ready,
    describe_accelerator,
    detect_best_accelerator,
)
from clearvid.app.bootstrap.paths import (
    REALESRGAN_WEIGHTS_DIR,
    TRT_CACHE_DIR,
    WEIGHTS_DIR,
)
from clearvid.app.postprocess.enhance import apply_sharpening
from clearvid.app.postprocess.temporal_stabilizer import TemporalStabilizer
from clearvid.app.schemas.models import (
    EnhancementConfig,
    FaceRestoreModel,
    InferenceAccelerator as InferenceAcceleratorEnum,
    QualityMode,
    TargetProfile,
    UpscaleModel,
    VideoMetadata,
)
from clearvid.app.export_control import ExportCancelled, ExportControl
from clearvid.app.utils.subprocess_utils import run_command

DEFAULT_MODEL_SCALE = 4


class FrameSkipper:
    """Skip super-resolution inference for near-static frames.

    When consecutive raw frames differ by less than *threshold* (mean absolute
    pixel difference), the previous enhanced result is reused instead of
    running the full model forward pass.  This is most effective for
    talking-head videos, tutorials, and screen recordings.

    The threshold is expressed as a mean pixel difference (0-255 range).
    Recommended values: 2.0 (conservative) to 5.0 (aggressive).
    Set to 0 to disable.
    """

    def __init__(self, threshold: float = 0.0):
        self._threshold = max(0.0, threshold)
        self._prev_raw: np.ndarray | None = None
        self._prev_enhanced: np.ndarray | None = None
        self.skip_count = 0

    @property
    def active(self) -> bool:
        return self._threshold > 0

    def should_skip(self, raw: np.ndarray) -> bool:
        if not self.active or self._prev_raw is None:
            return False
        diff = np.abs(raw.astype(np.float32) - self._prev_raw.astype(np.float32))
        return float(diff.mean()) < self._threshold

    def record(self, raw: np.ndarray, enhanced: np.ndarray) -> None:
        self._prev_raw = raw.copy()
        self._prev_enhanced = enhanced.copy()

    def get_cached(self) -> np.ndarray:
        """Return a copy of the cached enhanced frame."""
        assert self._prev_enhanced is not None
        return self._prev_enhanced.copy()


# ---------------------------------------------------------------------------
# Model registry — each entry describes architecture, weights, and download URL
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, dict] = {
    "general_v3": {
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "arch": "srvgg",
        "scale": 4,
    },
    "x4plus": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "arch": "rrdb",
        "scale": 4,
    },
}

# Backward-compatible aliases
DEFAULT_MODEL_FILENAME = _MODEL_REGISTRY["general_v3"]["filename"]
DEFAULT_MODEL_URL = _MODEL_REGISTRY["general_v3"]["url"]


def find_realesrgan_weights(weights_path: Path | None) -> list[Path]:
    if not weights_path or not weights_path.exists():
        return []
    if weights_path.is_file() and weights_path.suffix.lower() == ".pth":
        return [weights_path]
    return sorted(path for path in weights_path.glob("*.pth") if path.is_file())


def inspect_realesrgan_runtime(weights_path: Path | None = None) -> tuple[bool, str, str | None, bool, bool]:
    torch_version: str | None = None
    cuda_available = False
    gpu_compatible = False

    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch is not installed: {exc}", torch_version, cuda_available, gpu_compatible

    torch_version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return False, "CUDA is not available in the current PyTorch runtime.", torch_version, False, False

    try:
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        arch_list = set(torch.cuda.get_arch_list())
        gpu_compatible = required_arch in arch_list
    except Exception:  # noqa: BLE001
        gpu_compatible = False

    if not gpu_compatible:
        return (
            False,
            "Current PyTorch build does not support this GPU architecture. Install a build that supports RTX 5090.",
            torch_version,
            True,
            False,
        )

    try:
        _load_runtime_components()
    except ImportError as exc:
        return False, f"Real-ESRGAN is not installed: {exc}", torch_version, True, True
    except Exception as exc:  # noqa: BLE001
        return False, f"Real-ESRGAN 运行时初始化失败: {exc}", torch_version, True, True

    discovered_weights = find_realesrgan_weights(weights_path)
    if weights_path and not discovered_weights:
        return (
            True,
            f"Real-ESRGAN 运行时可用，缺少默认权重；首次运行时将自动下载到: {weights_path}",
            torch_version,
            True,
            True,
        )

    if discovered_weights:
        return True, f"Real-ESRGAN ready: {discovered_weights[0].name}", torch_version, True, True
    return True, "Real-ESRGAN runtime is available", torch_version, True, True


def validate_realesrgan_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    available, message, _, _, _ = inspect_realesrgan_runtime(weights_path)
    return available, message


def _apply_quality_mode_overrides(config: EnhancementConfig) -> EnhancementConfig:
    """Apply pipeline behaviour overrides based on quality_mode.

    FAST mode disables heavy postprocessing for maximum speed.
    BALANCED mode keeps postprocessing but uses lighter defaults.
    QUALITY mode uses all settings as-is.
    """
    if config.quality_mode == QualityMode.FAST:
        config = config.model_copy(update={
            "face_restore_enabled": False,
            "temporal_stabilize_enabled": False,
            "sharpen_enabled": False,
        })
    return config


def _auto_batch_size(
    config: EnhancementConfig, width: int, height: int, model_arch: str = "rrdb",
) -> int:
    """Choose batch_size based on VRAM, resolution, and model architecture.

    RRDB is very VRAM-hungry (23 residual blocks) — keep batches small.
    SRVGG is lightweight — moderate batches to balance latency vs throughput.
    """
    # 0 = GUI "auto"; 4 = legacy schema default / CLI default — both trigger VRAM detection.
    # Any other explicit value is respected as-is.
    if config.batch_size not in (0, 4):
        return config.batch_size
    try:
        import torch
        if not torch.cuda.is_available():
            return config.batch_size
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return config.batch_size
    megapixels = (width * height) / 1_000_000
    if model_arch == "srvgg":
        return _auto_srvgg_batch_size(vram_mb, megapixels)
    return _auto_rrdb_batch_size(vram_mb, megapixels)


def _auto_srvgg_batch_size(vram_mb: int, megapixels: float) -> int:
    # SRVGG is lightweight; high-VRAM GPUs benefit from fatter TRT tile batches.
    if vram_mb >= 24_000 and megapixels <= 0.6:
        return 16
    if vram_mb >= 16_000 and megapixels <= 1.0:
        return 8
    if vram_mb >= 8_000:
        return 4
    return 2


def _auto_rrdb_batch_size(vram_mb: int, megapixels: float) -> int:
    # RRDB: conservative — intermediate activations are VRAM-hungry
    if vram_mb >= 24_000 and megapixels <= 0.5:
        return 4
    if vram_mb >= 16_000 and megapixels <= 0.5:
        return 2
    return 1


def _resolve_trt_batch(
    config_batch: int,
    accel: InferenceAccelerator,
    config_accel: InferenceAcceleratorEnum,
) -> int:
    """Return the effective batch_size for TRT engine lookup.

    TRT forces tiling → batching is disabled in the pipeline.  Using a
    batch_size that differs from the deployed engine's profile causes a
    cache miss and a misleading "not deployed" error.

    For explicit TensorRT mode, always pin to 1:
    - The GUI deploy button always builds with batch=1 (spin at "自动" → 0 or 1 = 1).
    - Auto-detected batch (e.g. 8 on high-VRAM machines) must not be treated
      as a "user-explicit" override — it causes a hash mismatch with the
      deployed engine.
    """
    # AUTO or COMPILE: keep auto-detected batch
    if accel != InferenceAccelerator.TENSORRT:
        return config_batch
    if config_accel.value == "auto":
        return config_batch
    # Explicitly selected TensorRT: use a deployed high-throughput profile when
    # available.  TensorRT batching here means batched tiles, not batched whole
    # frames; cap to the engine builder's supported maximum.
    return max(1, min(config_batch, TRT_MAX_BATCH))


def _auto_trt_tile_size(config_tile_size: int, width: int, height: int) -> int:
    """Choose a TRT tile profile that spends more VRAM to reduce tile calls."""
    if config_tile_size > 0:
        return config_tile_size
    try:
        import torch

        if not torch.cuda.is_available():
            return 512
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 512
    long_edge = max(width, height)
    if vram_mb >= 24_000 and long_edge <= 1080:
        return 1024
    if vram_mb >= 12_000 and long_edge <= 1080:
        return 768
    return 512


def run_realesrgan_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    progress_callback: Callable[[int, str], None] | None = None,
    control: ExportControl | None = None,
    preview_callback: Callable[[str], None] | None = None,
) -> None:
    config = _apply_quality_mode_overrides(config)
    t_start = time.perf_counter()

    # Resolve accelerator FIRST so model selection can consider it
    accel = _resolve_accelerator(config.inference_accelerator)
    accel_label = describe_accelerator(accel)

    _emit_progress(progress_callback, 6, "正在准备 Real-ESRGAN 权重")
    weights_dir = REALESRGAN_WEIGHTS_DIR
    model_key = resolve_upscale_model(config.upscale_model, config.quality_mode, accel)
    model_arch = _MODEL_REGISTRY[model_key]["arch"]
    config = config.model_copy(update={
        "batch_size": _auto_batch_size(config, metadata.width, metadata.height, model_arch),
    })
    model_path = ensure_realesrgan_weights(weights_dir, model_key)
    model_label = _MODEL_REGISTRY[model_key]["filename"]
    _emit_progress(progress_callback, 10, f"正在初始化 Real-ESRGAN ({model_label})")
    upsampler = _build_upsampler(config, model_path, model_key, metadata.width, metadata.height)
    tile_size = getattr(upsampler, "tile_size", getattr(upsampler, "tile", "?"))
    if accel != InferenceAccelerator.NONE:
        _emit_progress(progress_callback, 11, f"正在应用推理加速: {accel_label}")
        t_accel = time.perf_counter()
        trt_tile = _auto_trt_tile_size(config.tile_size, metadata.width, metadata.height)

        # When TensorRT is explicitly selected (not AUTO), require a pre-built
        # engine.  TRT engines are built for a specific (tile, batch) profile.
        # Since TRT forces tiling which disables batching, batch_size=1 is the
        # only value that makes sense — override auto-detected batch here.
        trt_batch = _resolve_trt_batch(config.batch_size, accel, config.inference_accelerator)

        if accel == InferenceAccelerator.TENSORRT and config.inference_accelerator.value != "auto":
            ready, _msg = check_engine_ready(
                upsampler.model,
                fp16=config.fp16_enabled,
                tile_size=trt_tile,
                batch_size=trt_batch,
                cache_dir=TRT_CACHE_DIR,
                weight_path=model_path,
            )
            if not ready:
                # Try to find a compatible engine with a different tile/batch size
                from clearvid.app.models.tensorrt_engine import find_compatible_engine
                compat = find_compatible_engine(
                    upsampler.model,
                    fp16=config.fp16_enabled,
                    cache_dir=TRT_CACHE_DIR,
                    weight_path=model_path,
                )
                if compat is not None:
                    found_tile, found_batch, _ = compat
                    logger.info(
                        "TRT 引擎 tile=%d batch=%d 未找到，自动切换到已部署的 tile=%d batch=%d 引擎",
                        trt_tile, trt_batch, found_tile, found_batch,
                    )
                    trt_tile = found_tile
                    trt_batch = found_batch
                    config = config.model_copy(update={"tile_size": found_tile, "batch_size": found_batch})
                else:
                    raise RuntimeError(
                        f"TensorRT 引擎尚未部署 (tile={trt_tile}, batch={trt_batch})。"
                        "请先使用 GUI 中的'部署 TensorRT 引擎'按钮 "
                        "或运行 `clearvid warmup` 命令完成首次构建。\n"
                        "或切换到'自动检测'模式以自动选择可用加速方案。"
                    )
            build_if_missing = True
            # TRT forces tiling → batching is disabled in the pipeline.  Pin
            # config.batch_size to match the deployed engine profile so logs
            # and diagnostics are accurate.
            if trt_batch != config.batch_size:
                config = config.model_copy(update={"batch_size": trt_batch})
        else:
            build_if_missing = False

        upsampler.model = accelerate_model(
            upsampler.model,
            accel,
            fp16=config.fp16_enabled,
            tile_size=trt_tile,
            batch_size=trt_batch,
            cache_dir=TRT_CACHE_DIR,
            progress_callback=progress_callback,
            weight_path=model_path,
            trt_build_timeout=getattr(config, 'trt_build_timeout', None),
            build_if_missing=build_if_missing,
            low_load=True,
        )
        accel_actual = "TensorRT" if hasattr(upsampler.model, '_engine') else "PyTorch (回退)"
        # When TRT is active, force tiling so input shapes stay within the
        # TRT engine profile.  Whole-frame mode (tile=0) would send shapes
        # like 854×480 that exceed the profile's max_hw.
        if hasattr(upsampler.model, '_engine') and getattr(upsampler, 'tile', 0) == 0:
            upsampler.tile = trt_tile
            upsampler.tile_size = trt_tile
            logger.info("TRT 激活: 强制 tiling=%d (整帧模式不兼容 TRT profile)", trt_tile)

        # Performance advisory: TRT batch=1 with large free VRAM is GPU-underutilised.
        # Tile inference is sequential — each TRT call covers one tile, leaving GPU
        # idle between kernel launches.  batch=4+ in the engine profile would allow
        # multi-tile fusing for ~3-4× better throughput.
        if hasattr(upsampler.model, '_engine') and trt_batch == 1:
            try:
                import torch
                if torch.cuda.is_available():
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    if free_bytes > 8 * 1024 ** 3:
                        logger.warning(
                            "TRT batch=1 性能提示: GPU 剩余显存 %.0fGB，当前引擎每次仅处理 1 个 tile，"
                            "GPU 利用率偏低。建议点击『部署 TRT 引擎』按钮重新部署 (batch=4) 以获得 3-4× 速度提升；"
                            "或在『加速方式』下拉菜单切换为『自动检测』改用 PyTorch 批量模式，无需重新部署。",
                            free_bytes / 1024 ** 3,
                        )
            except Exception:  # noqa: BLE001
                pass

        _emit_progress(
            progress_callback, 11,
            f"推理加速就绪: {accel_actual} ({time.perf_counter() - t_accel:.1f}s)",
        )

    _emit_progress(progress_callback, 12, "正在初始化人脸修复")
    codeformer_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)
    _emit_progress(progress_callback, 14, "正在初始化时序稳定器")
    stabilizer = _build_temporal_stabilizer(config)

    # Log pipeline configuration summary
    t_init = time.perf_counter() - t_start
    actual_tile = getattr(upsampler, 'tile', tile_size)
    batching = "整帧" if actual_tile == 0 else f"tile={actual_tile}"
    face_label = "关闭" if codeformer_restorer is None else type(codeformer_restorer).__name__
    _emit_progress(
        progress_callback, 15,
        f"初始化完成 ({t_init:.1f}s) | 模型={model_label} 加速={accel_label} "
        f"batch={config.batch_size} {batching} fp16={'是' if config.fp16_enabled else '否'} "
        f"人脸={face_label} {'async' if config.async_pipeline else 'sync'}",
    )

    # Emit structured task config block for diagnostics
    fps_src = metadata.fps or 0.0
    dur_s = metadata.duration_seconds or 0.0
    total_frames_est = int(dur_s * fps_src) if dur_s > 0 and fps_src > 0 else 0
    dur_str = f"{int(dur_s // 60)}m{int(dur_s % 60)}s" if dur_s > 0 else "未知"
    stabilizer_label = "开启" if config.temporal_stabilize_enabled else "关闭"
    sharpen_label = f"{config.sharpen_strength:.1f}" if config.sharpen_enabled else "关闭"
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _gpu_name = _torch.cuda.get_device_properties(0).name
            _vram_total_gb = _torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            gpu_line = f"{_gpu_name} ({_vram_total_gb:.0f}GB VRAM)"
        else:
            gpu_line = "未检测到 CUDA GPU"
    except Exception:  # noqa: BLE001
        gpu_line = "未知"
    _emit_progress(
        progress_callback, 15,
        f"[任务配置] "
        f"输入: {metadata.width}×{metadata.height} {fps_src:.3f}fps ~{total_frames_est or '?'}帧({dur_str}) | "
        f"输出: {output_width}×{output_height} 缩放×{output_width // max(metadata.width, 1)} | "
        f"模型: {model_label} | 加速: {accel_label} batch={config.batch_size} {batching} fp16={'是' if config.fp16_enabled else '否'} | "
        f"后处理: 人脸={face_label} 稳定={stabilizer_label} 锐化={sharpen_label} | "
        f"GPU: {gpu_line}",
    )

    temp_root = Path(tempfile.mkdtemp(prefix="clearvid-realesrgan-"))
    temp_video_path = temp_root / "enhanced_video.mp4"
    preview_path = temp_root / "preview.mp4"

    # Build a mux trigger that creates a playable preview with audio
    _preview_mux_lock = threading.Lock()
    _preview_mux_thread: list[threading.Thread | None] = [None]

    def _on_preview_mux_needed(frames_done: int) -> None:
        """Called from monitor loop when enough new frames warrant a preview update."""
        if preview_callback is None:
            return
        fps = metadata.fps or 30.0
        duration_sec = frames_done / fps
        if duration_sec < 10:  # not enough content yet
            return
        with _preview_mux_lock:
            if _preview_mux_thread[0] is not None and _preview_mux_thread[0].is_alive():
                return  # previous mux still running

            def _do_mux() -> None:
                ok = _mux_preview(config, temp_video_path, preview_path, duration_sec)
                if ok and preview_callback is not None:
                    preview_callback(str(preview_path))

            t = threading.Thread(target=_do_mux, daemon=True, name="preview-mux")
            _preview_mux_thread[0] = t
            t.start()

    try:
        outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
        t_process = time.perf_counter()
        _stream_process_video(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            stabilizer=stabilizer,
            temp_video_path=temp_video_path,
            progress_callback=progress_callback,
            control=control,
            preview_mux_trigger=_on_preview_mux_needed,
        )
        t_process_elapsed = time.perf_counter() - t_process
        total_frames = _estimate_total_frames(metadata, config.preview_seconds) or 0
        avg_fps = total_frames / t_process_elapsed if t_process_elapsed > 0 and total_frames else 0
        _emit_progress(
            progress_callback, 95,
            f"帧处理完成: {total_frames}帧 {t_process_elapsed:.1f}s (平均 {avg_fps:.2f} fps)",
        )
        # Wait for any in-flight preview mux before final mux
        with _preview_mux_lock:
            if _preview_mux_thread[0] is not None:
                _preview_mux_thread[0].join(timeout=10)
        t_mux = time.perf_counter()
        _emit_progress(progress_callback, 96, "正在封装音频与元数据")
        _mux_output(config, temp_video_path)
        _emit_progress(
            progress_callback, 100,
            f"导出完成 | 总耗时 {time.perf_counter() - t_start:.1f}s "
            f"(初始化 {t_init:.1f}s + 帧处理 {t_process_elapsed:.1f}s + 封装 {time.perf_counter() - t_mux:.1f}s)",
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def enhance_single_frame(
    frame: np.ndarray,
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Enhance a single BGR frame using the full pipeline (upscale + face + sharpen).

    This is used by the preview feature to show a Before/After comparison
    without launching the full streaming pipeline.
    """
    weights_dir = REALESRGAN_WEIGHTS_DIR
    model_key = resolve_upscale_model(config.upscale_model, config.quality_mode)
    model_path = ensure_realesrgan_weights(weights_dir, model_key)
    upsampler = _build_upsampler(config, model_path, model_key, metadata.width, metadata.height)

    outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
    enhanced, _ = upsampler.enhance(frame, outscale=outscale)

    face_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)
    if face_restorer is not None:
        enhanced = face_restorer.restore_faces(enhanced)

    enhanced = _resize_for_target(enhanced, output_width, output_height, config.target_profile)

    if config.sharpen_enabled and config.sharpen_strength > 0:
        enhanced = apply_sharpening(enhanced, config.sharpen_strength)

    return enhanced


def extract_frame(video_path: Path, timestamp_sec: float = 0.0, width: int = 0, height: int = 0) -> np.ndarray:
    return stream_codec.extract_frame(
        video_path,
        timestamp_sec=timestamp_sec,
        width=width,
        height=height,
        run_factory=subprocess.run,
    )


def ensure_realesrgan_weights(weights_path: Path, model_key: str = "general_v3") -> Path:
    """Ensure weights for *model_key* exist, downloading if needed."""
    weights_path.mkdir(parents=True, exist_ok=True)
    entry = _MODEL_REGISTRY[model_key]
    target = weights_path / entry["filename"]
    if target.exists():
        return target

    # Fallback: check for any .pth already present (backward compat)
    discovered = find_realesrgan_weights(weights_path)
    for p in discovered:
        if p.name == entry["filename"]:
            return p

    _, _, _, load_file_from_url = _load_runtime_components()
    try:
        downloaded_path = load_file_from_url(
            url=entry["url"],
            model_dir=str(weights_path),
            progress=True,
            file_name=entry["filename"],
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"自动下载 Real-ESRGAN 权重失败 ({entry['filename']}): {exc}") from exc

    model_path = Path(downloaded_path)
    if not model_path.exists():
        raise RuntimeError(f"Real-ESRGAN 权重下载后未找到: {model_path}")
    return model_path


def _load_runtime_components() -> tuple[type, type, type, object]:
    realesrgan_utils = importlib.import_module("realesrgan.utils")
    srvgg_arch = importlib.import_module("realesrgan.archs.srvgg_arch")
    rrdb_arch = importlib.import_module("basicsr.archs.rrdbnet_arch")
    download_util = importlib.import_module("basicsr.utils.download_util")
    return (
        realesrgan_utils.RealESRGANer,
        srvgg_arch.SRVGGNetCompact,
        rrdb_arch.RRDBNet,
        download_util.load_file_from_url,
    )


def resolve_upscale_model(
    upscale_model: UpscaleModel,
    quality_mode: QualityMode,
    accel: InferenceAccelerator | None = None,
) -> str:
    """Resolve AUTO to a concrete model key based on quality mode.

    When TensorRT is available, ``general_v3`` is preferred even in QUALITY mode
    because TRT-accelerated general_v3 is faster than pure-PyTorch x4plus while
    delivering comparable subjective quality.  x4plus is reserved for QUALITY
    mode when no hardware acceleration is available (pure PyTorch).
    """
    if upscale_model != UpscaleModel.AUTO:
        return upscale_model.value
    if quality_mode == QualityMode.QUALITY:
        if accel is not None and accel != InferenceAccelerator.NONE:
            # Hardware acceleration available: general_v3 is fast enough
            return "general_v3"
        # No acceleration: use x4plus for best quality (slower but worth it)
        return "x4plus"
    return "general_v3"


def _build_upsampler(
    config: EnhancementConfig, model_path: Path, model_key: str,
    input_width: int, input_height: int,
) -> object:
    real_esrganer_cls, srvgg_cls, rrdb_cls, _ = _load_runtime_components()
    entry = _MODEL_REGISTRY[model_key]

    if entry["arch"] == "rrdb":
        model = rrdb_cls(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=entry["scale"],
        )
    else:
        model = srvgg_cls(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=entry["scale"],
            act_type="prelu",
        )

    tile = _resolve_tile_size(config.tile_size, input_width, input_height, config.fp16_enabled)
    return real_esrganer_cls(
        scale=entry["scale"],
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=config.tile_pad,
        pre_pad=0,
        half=config.fp16_enabled,
    )


def _build_codeformer_restorer(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> CodeFormerRestorer | GFPGANRestorer | None:
    if not config.face_restore_enabled:
        return None

    upscale_factor = _resolve_outscale(metadata, output_width, output_height, config.target_profile)

    if config.face_restore_model == FaceRestoreModel.GFPGAN:
        return GFPGANRestorer(
            upscale_factor=upscale_factor,
            weights_root=WEIGHTS_DIR,
        )

    return CodeFormerRestorer(
        fidelity_weight=config.face_restore_strength,
        upscale_factor=upscale_factor,
        weights_root=WEIGHTS_DIR,
        use_poisson_blend=config.face_poisson_blend,
    )


def _build_temporal_stabilizer(config: EnhancementConfig) -> TemporalStabilizer | None:
    if not config.temporal_stabilize_enabled:
        return None
    return TemporalStabilizer(strength=config.temporal_stabilize_strength)


def _resolve_accelerator(requested: InferenceAcceleratorEnum) -> InferenceAccelerator:
    """Map config enum to runtime accelerator, resolving AUTO."""
    if requested == InferenceAcceleratorEnum.AUTO:
        return detect_best_accelerator()
    return InferenceAccelerator(requested.value)


def _stream_process_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | None,
    stabilizer: TemporalStabilizer | None,
    temp_video_path: Path,
    progress_callback: Callable[[int, str], None] | None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
) -> None:
    decode_command = _build_decode_command(config, metadata)
    encode_command = _build_encode_command(config, metadata, output_width, output_height, temp_video_path)
    decoder, encoder = _start_stream_processes(decode_command, encode_command)

    try:
        _process_stream_frames(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            stabilizer=stabilizer,
            decoder=decoder,
            encoder=encoder,
            progress_callback=progress_callback,
            control=control,
            preview_mux_trigger=preview_mux_trigger,
        )
        _finalize_stream_processes(decoder, encoder)
    finally:
        _cleanup_stream_processes(decoder, encoder)


def _start_stream_processes(
    decode_command: list[str],
    encode_command: list[str],
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    return stream_codec.start_stream_processes(
        decode_command,
        encode_command,
        popen_factory=subprocess.Popen,
    )


def _mux_preview(
    config: EnhancementConfig,
    temp_video_path: Path,
    preview_path: Path,
    duration_sec: float,
) -> bool:
    return stream_codec.mux_preview(
        config,
        temp_video_path,
        preview_path,
        duration_sec,
        run_factory=subprocess.run,
    )


def _mux_output(config: EnhancementConfig, temp_video_path: Path) -> None:
    stream_codec.mux_output(config, temp_video_path, command_runner=run_command)


def _resolve_outscale(
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> float:
    if target_profile == TargetProfile.SOURCE:
        return 1.0
    if target_profile == TargetProfile.SCALE2X:
        return 2.0
    if target_profile == TargetProfile.SCALE4X:
        return 4.0
    return min(output_width / metadata.width, output_height / metadata.height)


def _resize_for_target(
    frame: np.ndarray,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> np.ndarray:
    if target_profile in {TargetProfile.FHD, TargetProfile.UHD4K}:
        return _fit_and_pad_frame(frame, output_width, output_height)
    if frame.shape[1] == output_width and frame.shape[0] == output_height:
        return frame
    return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)


def _fit_and_pad_frame(frame: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(output_width / width, output_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)

    if resized_width == output_width and resized_height == output_height:
        return resized

    canvas = np.zeros((output_height, output_width, 3), dtype=frame.dtype)
    offset_x = (output_width - resized_width) // 2
    offset_y = (output_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


def _read_exact_bytes(stream: object, byte_count: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < byte_count:
        chunk = stream.read(byte_count - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)

    if not buffer:
        return None
    if len(buffer) != byte_count:
        raise RuntimeError("视频帧流被意外截断。")
    return bytes(buffer)


def _estimate_total_frames(metadata: VideoMetadata, preview_seconds: int | None) -> int | None:
    duration_seconds = preview_seconds or metadata.duration_seconds
    if duration_seconds <= 0 or metadata.fps <= 0:
        return None
    return max(1, int(round(duration_seconds * metadata.fps)))


def _map_frame_progress(processed_frames: int, total_frames: int | None) -> int:
    if not total_frames:
        return 18
    ratio = min(max(processed_frames / total_frames, 0.0), 1.0)
    return 18 + int(76 * ratio)


def _build_decode_command(config: EnhancementConfig, metadata: VideoMetadata) -> list[str]:
    return stream_codec.build_decode_command(config, metadata)


def _build_encode_command(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    temp_video_path: Path,
) -> list[str]:
    return stream_codec.build_encode_command(config, metadata, output_width, output_height, temp_video_path)


_FRAME_QUEUE_DEPTH = 64


def _resolve_tile_size(config_tile_size: int, width: int, height: int, fp16: bool) -> int:
    """Return effective tile size. 0 = auto based on VRAM."""
    if config_tile_size > 0:
        return config_tile_size
    try:
        import torch

        if not torch.cuda.is_available():
            return 512
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 512
    megapixels = (width * height) / 1_000_000
    estimated_mb = int(megapixels * (2000 if fp16 else 4000))
    if estimated_mb < vram_mb * 0.5:
        return 0
    return 512


def _start_decode_thread(
    decoder_stdout: object,
    frame_size: int,
    abort: threading.Event,
) -> tuple[queue.Queue, list[BaseException], threading.Thread]:
    # Limit raw-frame queue to ~256 MB so 4K processing doesn't exhaust RAM.
    # 1080p (6 MB/frame) → ~42 frames capped at 48.
    # 4K   (24 MB/frame) → ~10 frames (well within limit).
    _depth = max(4, min(48, 256 * 1024 * 1024 // max(frame_size, 1)))
    raw_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_depth)
    errors: list[BaseException] = []

    def _loop() -> None:
        try:
            while not abort.is_set():
                raw = _read_exact_bytes(decoder_stdout, frame_size)
                if raw is None:
                    break
                while not abort.is_set():
                    try:
                        raw_queue.put(raw, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            abort.set()
        finally:
            # Timeout-aware sentinel: keep trying until downstream picks it up
            # or the pipeline is aborted entirely.
            while not abort.is_set():
                try:
                    raw_queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return raw_queue, errors, thread


def _collect_batch(raw_queue: queue.Queue, batch_size: int) -> list[bytes]:
    """Collect up to *batch_size* frames.

    The first frame blocks indefinitely (waits for decode). Subsequent
    frames use a short timeout so the GPU gets a reasonably full batch
    without waiting an entire decode cycle when frames trickle in slowly.
    """
    batch: list[bytes] = []
    # First frame: block until available
    raw = raw_queue.get()
    if raw is None:
        raw_queue.put(None)
        return batch
    batch.append(raw)
    # Remaining frames: timeout = ~1 video frame at 30 fps (33 ms) so the GPU
    # collects a reasonably full batch without stalling on slow-decode sources.
    # The original 5 ms timeout caused near-constant batch=1 when decode was
    # even slightly bottlenecked (e.g. nlmeans on CPU), starving the GPU.
    for _ in range(batch_size - 1):
        try:
            raw = raw_queue.get(timeout=0.033)
        except queue.Empty:
            break
        if raw is None:
            raw_queue.put(None)
            break
        batch.append(raw)
    return batch


def _enhance_frames_batch(
    frames: list[np.ndarray],
    upsampler: object,
    outscale: float,
) -> list[np.ndarray]:
    """Batch-enhance multiple frames bypassing per-frame overhead."""
    import torch

    # Vectorized CPU preprocessing: BGR→RGB, HWC→CHW, normalize, stack
    tensors = []
    for frame in frames:
        img = frame[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)  # BGR→RGB + normalize
        tensors.append(torch.from_numpy(np.ascontiguousarray(img).transpose(2, 0, 1)))

    batch = torch.stack(tensors)
    if upsampler.half:
        batch = batch.half()
    batch = batch.to(upsampler.device, non_blocking=True)

    with torch.inference_mode():
        output = upsampler.model(batch)

    # Batch GPU→CPU transfer (single DMA instead of per-frame)
    output_cpu = output.float().cpu().clamp_(0, 1)

    results: list[np.ndarray] = []
    factor = outscale / upsampler.scale
    need_resize = abs(factor - 1.0) > 1e-6
    for i in range(output_cpu.shape[0]):
        out = output_cpu[i].numpy()
        out = np.ascontiguousarray(out[[2, 1, 0], :, :].transpose(1, 2, 0))  # RGB CHW→BGR HWC
        if need_resize:
            h, w = out.shape[:2]
            out = cv2.resize(
                out, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_LANCZOS4,
            )
        results.append((out * 255.0).round().astype(np.uint8))
    return results


def _is_trt_upsampler(upsampler: object) -> bool:
    return hasattr(getattr(upsampler, "model", None), "_engine")


_TrtTileInfo = tuple[int, int, int, int, int, int, int, int]
_TrtPendingTile = tuple[object, _TrtTileInfo]


def _pad_trt_tile_to_shape(tile: object, target_h: int, target_w: int) -> object:
    tile_h = int(tile.shape[2])
    tile_w = int(tile.shape[3])
    if tile_h == target_h and tile_w == target_w:
        return tile
    padded = tile.new_zeros((tile.shape[0], tile.shape[1], target_h, target_w))
    padded[:, :, :tile_h, :tile_w] = tile
    return padded


def _flush_trt_tile_batch(
    pending: list[_TrtPendingTile],
    upsampler: object,
    output: object,
    tile_stats: dict[str, float] | None = None,
) -> None:
    if not pending:
        return
    import torch

    tiles = torch.cat([item[0] for item in pending], dim=0)
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = upsampler.model(tiles)
    if tile_stats is not None:
        tile_items = float(tiles.shape[0])
        tile_stats["tile_batches"] = tile_stats.get("tile_batches", 0.0) + 1.0
        tile_stats["tiles"] = tile_stats.get("tiles", 0.0) + tile_items
        tile_stats["tile_infer_ms"] = tile_stats.get("tile_infer_ms", 0.0) + (time.perf_counter() - t0) * 1000
        tile_stats["tile_batch_max"] = max(tile_stats.get("tile_batch_max", 0.0), tile_items)
    output_offset = 0
    for tile, info in pending:
        (
            output_start_x, output_end_x, output_start_y, output_end_y,
            output_start_x_tile, output_end_x_tile,
            output_start_y_tile, output_end_y_tile,
        ) = info
        tile_batch = int(tile.shape[0])
        output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = outputs[
            output_offset:output_offset + tile_batch,
            :,
            output_start_y_tile:output_end_y_tile,
            output_start_x_tile:output_end_x_tile,
        ]
        output_offset += tile_batch


def _trt_output_to_frames(
    output_tensor: object,
    upsampler: object,
    outscale: float,
    input_width: int,
    input_height: int,
    tile_stats: dict[str, float] | None = None,
) -> list[np.ndarray]:
    t_post = time.perf_counter()
    upsampler.output = output_tensor
    result_tensor = upsampler.post_process().float().clamp_(0, 1)
    if tile_stats is not None:
        tile_stats["trt_post_ms"] = tile_stats.get("trt_post_ms", 0.0) + (time.perf_counter() - t_post) * 1000

    if outscale is not None and outscale != float(upsampler.scale):
        import torch.nn.functional as F

        t_resize = time.perf_counter()
        result_tensor = F.interpolate(
            result_tensor,
            size=(max(1, int(input_height * outscale)), max(1, int(input_width * outscale))),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0, 1)
        if tile_stats is not None:
            tile_stats["trt_resize_ms"] = tile_stats.get("trt_resize_ms", 0.0) + (time.perf_counter() - t_resize) * 1000

    import torch

    t_pack = time.perf_counter()
    packed_tensor = (
        (result_tensor[:, [2, 1, 0], :, :] * 255.0)
        .round()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    if tile_stats is not None:
        tile_stats["trt_pack_ms"] = tile_stats.get("trt_pack_ms", 0.0) + (time.perf_counter() - t_pack) * 1000

    t_cpu = time.perf_counter()
    output_frames = packed_tensor.cpu().numpy()
    if tile_stats is not None:
        tile_stats["trt_cpu_ms"] = tile_stats.get("trt_cpu_ms", 0.0) + (time.perf_counter() - t_cpu) * 1000
    return [np.ascontiguousarray(output_frames[i]) for i in range(output_frames.shape[0])]


def _trt_output_to_frame(
    output_tensor: object,
    upsampler: object,
    outscale: float,
    input_width: int,
    input_height: int,
    tile_stats: dict[str, float] | None = None,
) -> np.ndarray:
    return _trt_output_to_frames(
        output_tensor, upsampler, outscale, input_width, input_height, tile_stats,
    )[0]


def _enhance_frames_trt_tiled(
    frames: list[np.ndarray],
    upsampler: object,
    outscale: float,
    tile_stats: dict[str, float] | None = None,
) -> list[np.ndarray]:
    """Enhance one or more frames by batching RealESRGAN tiles through TensorRT.

    The upstream RealESRGANer processes tiles one at a time.  With TensorRT this
    underutilizes high-end GPUs, so this path groups same-size padded tiles from
    one or more frames into batches up to the engine profile's max batch.
    """
    if not frames:
        return []
    h_input, w_input = frames[0].shape[:2]
    preprocessed = []
    for frame in frames:
        if frame.shape[:2] != (h_input, w_input):
            raise ValueError("TRT frame batch requires equal frame sizes")
        img = cv2.cvtColor(frame.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_BGR2RGB)
        upsampler.pre_process(img)
        preprocessed.append(upsampler.img)
    import torch

    upsampler.img = torch.cat(preprocessed, dim=0)

    batch, channel, height, width = upsampler.img.shape
    output = upsampler.img.new_zeros((batch, channel, height * upsampler.scale, width * upsampler.scale))
    tile_size = int(getattr(upsampler, "tile_size", 0) or 0)
    tile_pad = int(getattr(upsampler, "tile_pad", 0) or 0)
    max_batch = max(1, int(getattr(upsampler.model, "max_batch", 1)))
    if tile_stats is not None:
        tile_stats["engine_max_batch"] = float(max_batch)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    target_tile_h = min(height, tile_size + tile_pad * 2)
    target_tile_w = min(width, tile_size + tile_pad * 2)

    pending: list[_TrtPendingTile] = []
    pending_hw: tuple[int, int] | None = None
    pending_items = 0

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            ofs_x = tile_x * tile_size
            ofs_y = tile_y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = upsampler.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
            input_tile = _pad_trt_tile_to_shape(input_tile, target_tile_h, target_tile_w)
            tile_hw = (int(input_tile.shape[2]), int(input_tile.shape[3]))
            tile_items = int(input_tile.shape[0])
            if pending_hw is not None and (
                tile_hw != pending_hw or pending_items + tile_items > max_batch
            ):
                _flush_trt_tile_batch(pending, upsampler, output, tile_stats=tile_stats)
                pending = []
                pending_hw = None
                pending_items = 0
            pending_hw = tile_hw

            output_start_x = input_start_x * upsampler.scale
            output_end_x = input_end_x * upsampler.scale
            output_start_y = input_start_y * upsampler.scale
            output_end_y = input_end_y * upsampler.scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * upsampler.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * upsampler.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * upsampler.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * upsampler.scale
            pending.append((
                input_tile,
                (
                    output_start_x, output_end_x, output_start_y, output_end_y,
                    output_start_x_tile, output_end_x_tile,
                    output_start_y_tile, output_end_y_tile,
                ),
            ))
            pending_items += tile_items
    _flush_trt_tile_batch(pending, upsampler, output, tile_stats=tile_stats)

    return _trt_output_to_frames(output, upsampler, outscale, w_input, h_input, tile_stats)


def _enhance_frame_trt_tiled(
    frame: np.ndarray,
    upsampler: object,
    outscale: float,
    tile_stats: dict[str, float] | None = None,
) -> np.ndarray:
    return _enhance_frames_trt_tiled([frame], upsampler, outscale, tile_stats=tile_stats)[0]


def _fetch_enhanced_frames(
    raw_queue: queue.Queue,
    use_batching: bool,
    batch_size: int,
    height: int,
    width: int,
    upsampler: object,
    outscale: float,
    skipper: FrameSkipper | None = None,
    trt_tile_stats: dict[str, float] | None = None,
) -> list[np.ndarray] | None:
    """Fetch and enhance the next frame(s). Returns None at end of stream."""
    if use_batching:
        batch_raw = _collect_batch(raw_queue, batch_size)
        if not batch_raw:
            return None
        images = [np.frombuffer(r, dtype=np.uint8).reshape((height, width, 3)) for r in batch_raw]
        return _enhance_frames_batch(images, upsampler, outscale)

    if (
        not (skipper is not None and skipper.active)
        and _is_trt_upsampler(upsampler)
        and getattr(upsampler, "tile_size", 0) > 0
        and batch_size > 1
    ):
        trt_frame_batch = min(batch_size, max(1, int(getattr(upsampler.model, "max_batch", 1))))
        batch_raw = _collect_batch(raw_queue, trt_frame_batch)
        if not batch_raw:
            return None
        images = [np.frombuffer(r, dtype=np.uint8).reshape((height, width, 3)) for r in batch_raw]
        return _enhance_frames_trt_tiled(images, upsampler, outscale, tile_stats=trt_tile_stats)

    raw_frame = raw_queue.get()
    if raw_frame is None:
        return None
    image = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

    if skipper is not None and skipper.should_skip(image):
        skipper.skip_count += 1
        return [skipper.get_cached()]

    if _is_trt_upsampler(upsampler) and getattr(upsampler, "tile_size", 0) > 0:
        enhanced = _enhance_frame_trt_tiled(image, upsampler, outscale, tile_stats=trt_tile_stats)
    else:
        enhanced, _ = upsampler.enhance(image, outscale=outscale)
    if skipper is not None:
        skipper.record(image, enhanced)
    return [enhanced]


def _prepare_encoder_frame(frame: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim != 3:
        raise ValueError(f"编码帧维度异常: {frame.shape}")
    elif frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] > 3:
        frame = frame[:, :, :3]

    if frame.shape[1] != output_width or frame.shape[0] != output_height:
        raise ValueError(
            f"编码帧尺寸不匹配: got {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {output_width}x{output_height}"
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_encoder_frame(
    encoder_stdin: object,
    frame: np.ndarray,
    output_width: int,
    output_height: int,
) -> None:
    prepared = _prepare_encoder_frame(frame, output_width, output_height)
    encoder_stdin.write(memoryview(prepared).cast("B"))


def _write_enhanced_frames(
    enhanced_list: list[np.ndarray],
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    encoder_stdin: object,
    sharpen_strength: float = 0.0,
) -> int:
    """Post-process and write frames to encoder. Returns count written.

    Used by the synchronous pipeline path.  The async pipeline uses
    ``_write_finalized_frames`` instead (GPU work done in Stage 2).
    """
    count = 0
    for enhanced in enhanced_list:
        if codeformer_restorer is not None:
            enhanced = codeformer_restorer.restore_faces(enhanced)
        if stabilizer is not None:
            enhanced = stabilizer.stabilize(enhanced)
        if sharpen_strength > 0:
            enhanced = apply_sharpening(enhanced, sharpen_strength)
        finalized = _resize_for_target(enhanced, output_width, output_height, target_profile)
        _write_encoder_frame(encoder_stdin, finalized, output_width, output_height)
        count += 1
    return count


def _write_finalized_frames(
    finalized_list: list[np.ndarray],
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    encoder_stdin: object,
) -> int:
    """Resize and write pre-processed frames to encoder (I/O only).

    Used by the async pipeline Stage 3 — all GPU-heavy work (face restore,
    temporal stabilize, sharpen) has already been done in Stage 2.
    """
    count = 0
    for frame in finalized_list:
        resized = _resize_for_target(frame, output_width, output_height, target_profile)
        _write_encoder_frame(encoder_stdin, resized, output_width, output_height)
        count += 1
    return count


def _process_stream_frames(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    decoder: object,
    encoder: object,
    progress_callback: Callable[[int, str], None] | None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
) -> None:
    frame_size = metadata.width * metadata.height * 3
    total_frames = _estimate_total_frames(metadata, config.preview_seconds)

    assert decoder.stdout is not None
    assert encoder.stdin is not None
    skipper = FrameSkipper(getattr(config, 'skip_frame_threshold', 0.0))
    _emit_progress(
        progress_callback, 18,
        f"正在流式处理视频帧 (batch={config.batch_size}, "
        f"{'async' if config.async_pipeline else 'sync'}"
        f"{', 智能跳帧=%d' % int(skipper._threshold) if skipper.active else ''})",
    )

    abort = threading.Event()
    raw_queue, decode_errors, reader = _start_decode_thread(decoder.stdout, frame_size, abort)

    if config.async_pipeline:
        _process_frames_async(
            config, metadata, output_width, output_height, outscale,
            upsampler, codeformer_restorer, stabilizer,
            raw_queue, encoder, total_frames, progress_callback,
            abort=abort, control=control,
            preview_mux_trigger=preview_mux_trigger,
            skipper=skipper,
        )
    else:
        _process_frames_sync(
            config, metadata, output_width, output_height, outscale,
            upsampler, codeformer_restorer, stabilizer,
            raw_queue, encoder, total_frames, progress_callback,
            preview_mux_trigger=preview_mux_trigger,
            skipper=skipper,
        )

    if skipper.skip_count > 0:
        logger.info("智能跳帧: 跳过 %d 帧 (%.1f%%)",
                     skipper.skip_count,
                     skipper.skip_count / max(total_frames or 1, 1) * 100)

    reader.join()
    if decode_errors:
        raise decode_errors[0]


def _process_frames_sync(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    raw_queue: queue.Queue,
    encoder: object,
    total_frames: int | None,
    progress_callback: Callable[[int, str], None] | None,
    preview_mux_trigger: Callable[[int], None] | None = None,
    skipper: FrameSkipper | None = None,
) -> None:
    """Original synchronous processing: inference and encode on main thread."""
    use_batching = getattr(upsampler, "tile_size", 0) == 0 and config.batch_size > 1
    processed_frames = 0
    last_reported_progress = -1
    fps = metadata.fps or 30.0
    _preview_interval_frames = int(fps * 60)  # ~60 seconds of video
    _last_preview_at = 0
    _t_sync_start = time.perf_counter()

    while True:
        enhanced_list = _fetch_enhanced_frames(
            raw_queue, use_batching, config.batch_size, metadata.height, metadata.width, upsampler, outscale,
            skipper=skipper,
        )
        if enhanced_list is None:
            break
        processed_frames += _write_enhanced_frames(
            enhanced_list, codeformer_restorer, stabilizer, output_width, output_height, config.target_profile, encoder.stdin,
            sharpen_strength=config.sharpen_strength if config.sharpen_enabled else 0.0,
        )
        last_reported_progress = _report_stream_progress(
            processed_frames, total_frames, last_reported_progress, progress_callback,
            start_time=_t_sync_start,
        )
        if preview_mux_trigger and processed_frames - _last_preview_at >= _preview_interval_frames:
            _last_preview_at = processed_frames
            preview_mux_trigger(processed_frames)

    encoder.stdin.close()


_ENHANCED_QUEUE_DEPTH = 16


def _process_frames_async(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    raw_queue: queue.Queue,
    encoder: object,
    total_frames: int | None,
    progress_callback: Callable[[int, str], None] | None,
    *,
    abort: threading.Event | None = None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
    skipper: FrameSkipper | None = None,
) -> None:
    """Async pipeline with abort-safe queue operations.

    All queue.put() calls use a timeout loop that checks ``abort``.
    If any stage fails it sets ``abort`` so all other stages unblock
    and drain promptly instead of deadlocking.
    """
    if abort is None:
        abort = threading.Event()

    def _safe_put(q: queue.Queue, item: object) -> bool:
        """Put item into queue, respecting abort. Returns False if aborted."""
        while not abort.is_set():
            try:
                q.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _safe_get(q: queue.Queue) -> tuple[bool, object]:
        """Get item from queue, respecting abort. Returns (ok, item)."""
        while not abort.is_set():
            try:
                return True, q.get(timeout=1.0)
            except queue.Empty:
                continue
        return False, None

    sharpen_strength = config.sharpen_strength if config.sharpen_enabled else 0.0
    postprocess_needed = (
        codeformer_restorer is not None
        or stabilizer is not None
        or sharpen_strength > 0
    )

    if postprocess_needed:
        # Limit enhanced-frame queue to ~128 MB.  Each queue slot is a *batch*
        # of frames; with batch=4 and 4K output each slot is ~96 MB, so minimum
        # depth of 4 still gives the async stages enough room to overlap.
        _enhanced_item_bytes = output_width * output_height * 3 * max(1, config.batch_size)
        _eq_depth = max(4, min(16, 128 * 1024 * 1024 // max(_enhanced_item_bytes, 1)))
        enhanced_queue: queue.Queue[list[np.ndarray] | None] = queue.Queue(
            maxsize=_eq_depth,
        )
        finalized_queue: queue.Queue[list[np.ndarray] | None] = queue.Queue(
            maxsize=_eq_depth,
        )
    else:
        _enhanced_item_bytes = output_width * output_height * 3 * max(1, config.batch_size)
        _eq_depth = max(4, min(16, 128 * 1024 * 1024 // max(_enhanced_item_bytes, 1)))
        finalized_queue = queue.Queue(maxsize=_eq_depth)
        enhanced_queue = finalized_queue

    enhance_errors: list[BaseException] = []
    write_errors: list[BaseException] = []
    frames_written = [0]

    # -- Diagnostics counters (shared across threads) -----------------------
    _diag_enhance_batches = [0]
    _diag_enhance_frames = [0]
    _diag_enhance_infer_ms = [0.0]
    _diag_enhance_wait_ms = [0.0]
    _diag_write_frames = [0]
    _diag_write_ms = [0.0]
    _diag_batch_sizes: list[int] = []       # capped at 100 entries — see append below
    _diag_face_frames = [0]                 # frames where faces were detected
    _diag_face_total = [0]                  # total faces restored
    _diag_postprocess_ms = [0.0]
    _diag_face_ms = [0.0]
    _diag_stabilize_ms = [0.0]
    _diag_sharpen_ms = [0.0]
    _diag_trt_tile_stats: dict[str, float] = {}

    # -- Stage 2: super-resolution thread -----------------------------------
    use_batching = getattr(upsampler, "tile_size", 0) == 0 and config.batch_size > 1

    def _enhance_loop() -> None:
        try:
            while not abort.is_set():
                if control is not None:
                    control.check()
                t_wait = time.perf_counter()
                enhanced_list = _fetch_enhanced_frames(
                    raw_queue, use_batching, config.batch_size,
                    metadata.height, metadata.width, upsampler, outscale,
                    skipper=skipper,
                    trt_tile_stats=_diag_trt_tile_stats,
                )
                t_done = time.perf_counter()
                if enhanced_list is None:
                    break
                n = len(enhanced_list)
                _diag_enhance_infer_ms[0] += (t_done - t_wait) * 1000
                _diag_enhance_batches[0] += 1
                _diag_enhance_frames[0] += n
                if len(_diag_batch_sizes) >= 100:
                    _diag_batch_sizes.pop(0)
                _diag_batch_sizes.append(n)
                if not _safe_put(enhanced_queue, enhanced_list):
                    break
        except ExportCancelled:
            abort.set()
        except Exception as exc:  # noqa: BLE001
            enhance_errors.append(exc)
            abort.set()
        finally:
            while not abort.is_set():
                try:
                    enhanced_queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue

    enhance_thread = threading.Thread(
        target=_enhance_loop, daemon=True, name="sr-enhance",
    )

    # -- Stage 4: write thread -----------------------------------------------
    def _write_loop() -> None:
        try:
            while not abort.is_set():
                ok, item = _safe_get(finalized_queue)
                if not ok or item is None:
                    break
                t_w = time.perf_counter()
                n = _write_finalized_frames(
                    item, output_width, output_height,
                    config.target_profile, encoder.stdin,
                )
                _diag_write_ms[0] += (time.perf_counter() - t_w) * 1000
                _diag_write_frames[0] += n
                frames_written[0] += n
        except Exception as exc:  # noqa: BLE001
            write_errors.append(exc)
            abort.set()
        finally:
            try:
                encoder.stdin.close()
            except Exception:  # noqa: BLE001
                pass

    write_thread = threading.Thread(
        target=_write_loop, daemon=True, name="frame-writer",
    )

    # -- Stage 3 (thread): face restore + stabilize + sharpen ----------------
    stage3_errors: list[BaseException] = []
    postprocess_thread: threading.Thread | None = None

    if postprocess_needed:
        def _postprocess_loop() -> None:
            _stab_pool: ThreadPoolExecutor | None = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="stabilizer")
                if stabilizer is not None else None
            )
            _stab_future: Future[np.ndarray] | None = None

            def _stabilize_and_sharpen(frame: np.ndarray) -> np.ndarray:
                t_stage = time.perf_counter()
                result = frame
                if stabilizer is not None:
                    t_stab = time.perf_counter()
                    result = stabilizer.stabilize(result)
                    _diag_stabilize_ms[0] += (time.perf_counter() - t_stab) * 1000
                if sharpen_strength > 0:
                    t_sharp = time.perf_counter()
                    result = apply_sharpening(result, sharpen_strength)
                    _diag_sharpen_ms[0] += (time.perf_counter() - t_sharp) * 1000
                _diag_postprocess_ms[0] += (time.perf_counter() - t_stage) * 1000
                return result

            try:
                while not abort.is_set():
                    ok, enhanced_list = _safe_get(enhanced_queue)
                    if not ok or enhanced_list is None:
                        break

                    finalized_list: list[np.ndarray] = []
                    for enhanced in enhanced_list:
                        if abort.is_set():
                            break
                        if control is not None:
                            control.check()
                        if codeformer_restorer is not None:
                            t_face = time.perf_counter()
                            enhanced = codeformer_restorer.restore_faces(enhanced)
                            face_ms = (time.perf_counter() - t_face) * 1000
                            _diag_face_ms[0] += face_ms
                            _diag_postprocess_ms[0] += face_ms

                        if _stab_future is not None:
                            finalized_list.append(_stab_future.result())
                            _stab_future = None

                        if _stab_pool is not None:
                            _stab_future = _stab_pool.submit(
                                _stabilize_and_sharpen, enhanced,
                            )
                        else:
                            if sharpen_strength > 0:
                                t_pp = time.perf_counter()
                                t_sharp = time.perf_counter()
                                enhanced = apply_sharpening(enhanced, sharpen_strength)
                                _diag_sharpen_ms[0] += (time.perf_counter() - t_sharp) * 1000
                                _diag_postprocess_ms[0] += (time.perf_counter() - t_pp) * 1000
                            finalized_list.append(enhanced)

                    if finalized_list and not abort.is_set():
                        _safe_put(finalized_queue, finalized_list)

                if _stab_future is not None and not abort.is_set():
                    _safe_put(finalized_queue, [_stab_future.result()])
            except ExportCancelled:
                abort.set()
            except Exception as exc:  # noqa: BLE001
                stage3_errors.append(exc)
                abort.set()
            finally:
                while not abort.is_set():
                    try:
                        finalized_queue.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                if _stab_pool is not None:
                    _stab_pool.shutdown(wait=False)

        postprocess_thread = threading.Thread(
            target=_postprocess_loop, daemon=True, name="postprocess",
        )

    # -- Start background stages ---------------------------------------------
    enhance_thread.start()
    if postprocess_thread is not None:
        postprocess_thread.start()
    write_thread.start()

    # -- Performance diagnostics samplers ------------------------------------
    _gpu_sampler = GpuSampler(interval=5.0)
    _cpu_tracker = CpuUsageTracker()

    # -- Main thread: progress reporting + periodic preview mux ---------------
    monitor_thread = postprocess_thread if postprocess_thread is not None else enhance_thread
    last_reported_progress = -1
    fps = metadata.fps or 30.0
    _preview_interval_frames = int(fps * 60)  # ~60 seconds of video
    _last_preview_at = 0
    _t_async_start = time.perf_counter()
    _last_diag_time = _t_async_start
    _DIAG_INTERVAL = 10.0  # log diagnostics every 10 seconds
    while monitor_thread.is_alive():
        monitor_thread.join(timeout=0.5)
        current_written = frames_written[0]
        now = time.perf_counter()
        last_reported_progress = _report_stream_progress(
            current_written, total_frames, last_reported_progress,
            progress_callback, start_time=_t_async_start,
        )
        if preview_mux_trigger and current_written - _last_preview_at >= _preview_interval_frames:
            _last_preview_at = current_written
            preview_mux_trigger(current_written)
        # Periodic diagnostics
        if now - _last_diag_time >= _DIAG_INTERVAL and progress_callback:
            _last_diag_time = now
            cpu_pct = _cpu_tracker.sample_percent(now)
            rq = raw_queue.qsize()
            eq = enhanced_queue.qsize() if enhanced_queue is not finalized_queue else -1
            fq = finalized_queue.qsize()
            avg_batch = (
                sum(_diag_batch_sizes[-20:]) / len(_diag_batch_sizes[-20:])
                if _diag_batch_sizes else 0
            )
            elapsed_s = now - _t_async_start
            enh_fps = _diag_enhance_frames[0] / elapsed_s if elapsed_s > 0 else 0
            wr_fps = _diag_write_frames[0] / elapsed_s if elapsed_s > 0 else 0
            pp_avg = (
                _diag_postprocess_ms[0] / _diag_write_frames[0]
                if _diag_write_frames[0] > 0 else 0
            )
            avg_infer_ms = (
                _diag_enhance_infer_ms[0] / _diag_enhance_batches[0]
                if _diag_enhance_batches[0] > 0 else 0
            )
            trt_tile_batches = _diag_trt_tile_stats.get("tile_batches", 0.0)
            trt_tiles = _diag_trt_tile_stats.get("tiles", 0.0)
            trt_avg_tile_batch = trt_tiles / trt_tile_batches if trt_tile_batches > 0 else 0.0
            queue_info = format_queue_info(rq, eq, fq, _FRAME_QUEUE_DEPTH, _ENHANCED_QUEUE_DEPTH)
            pp_info = f" 后处理={pp_avg:.0f}ms/帧" if pp_avg > 0 else ""
            infer_info = f" 推理={avg_infer_ms:.0f}ms/批" if avg_infer_ms > 0 else ""
            trt_info = f" TRT-tile-batch={trt_avg_tile_batch:.1f}" if trt_tile_batches > 0 else ""
            gpu_info = format_gpu_snapshot(_gpu_sampler.snapshot())
            _emit_progress(
                progress_callback,
                last_reported_progress,
                f"[诊断] {queue_info} | 增强 {enh_fps:.1f}fps 写入 {wr_fps:.1f}fps"
                f" | 平均帧batch={avg_batch:.1f}{trt_info}{infer_info}{pp_info} CPU={cpu_pct:.0f}%{gpu_info}",
            )

    _report_stream_progress(
        frames_written[0], total_frames, last_reported_progress,
        progress_callback, start_time=_t_async_start,
    )

    # Stop GPU sampler and collect final summary
    _gpu_sampler.stop()
    _gpu_summary = _gpu_sampler.summary()

    # ---- Comprehensive performance report ----------------------------------
    total_elapsed = time.perf_counter() - _t_async_start
    if progress_callback and total_elapsed > 0:
        enh_batches = _diag_enhance_batches[0]
        enh_frames = _diag_enhance_frames[0]
        wr_frames = _diag_write_frames[0]
        frame_denominator = max(wr_frames, 1)
        avg_batch = enh_frames / enh_batches if enh_batches else 0
        avg_infer_ms = _diag_enhance_infer_ms[0] / enh_batches if enh_batches else 0
        avg_write_ms = _diag_write_ms[0] / wr_frames if wr_frames else 0
        avg_pp_ms = _diag_postprocess_ms[0] / frame_denominator
        avg_face_ms = _diag_face_ms[0] / frame_denominator
        avg_stabilize_ms = _diag_stabilize_ms[0] / frame_denominator
        avg_sharpen_ms = _diag_sharpen_ms[0] / frame_denominator
        total_fps = enh_frames / total_elapsed
        cpu_total_pct = _cpu_tracker.total_percent(total_elapsed)
        skip_frames = skipper.skip_count if skipper is not None else 0
        skip_pct = skip_frames * 100.0 / max(enh_frames + skip_frames, 1)
        trt_tile_batches = _diag_trt_tile_stats.get("tile_batches", 0.0)
        trt_tiles = _diag_trt_tile_stats.get("tiles", 0.0)
        trt_tile_ms = _diag_trt_tile_stats.get("tile_infer_ms", 0.0)
        trt_tile_max_batch = _diag_trt_tile_stats.get("tile_batch_max", 0.0)
        trt_engine_max_batch = _diag_trt_tile_stats.get("engine_max_batch", 0.0)
        trt_post_ms = _diag_trt_tile_stats.get("trt_post_ms", 0.0)
        trt_resize_ms = _diag_trt_tile_stats.get("trt_resize_ms", 0.0)
        trt_pack_ms = _diag_trt_tile_stats.get("trt_pack_ms", 0.0)
        trt_cpu_ms = _diag_trt_tile_stats.get("trt_cpu_ms", 0.0)
        trt_avg_tile_batch = trt_tiles / trt_tile_batches if trt_tile_batches > 0 else 0.0
        trt_tile_ms_per_batch = trt_tile_ms / trt_tile_batches if trt_tile_batches > 0 else 0.0
        trt_frames = max(enh_frames, 1)

        # Stage throughput breakdown (ms/frame)
        infer_per_frame = avg_infer_ms / max(avg_batch, 1)
        trt_kernel_ms_per_frame = trt_tile_ms / trt_frames
        trt_post_ms_per_frame = trt_post_ms / trt_frames
        trt_resize_ms_per_frame = trt_resize_ms / trt_frames
        trt_pack_ms_per_frame = trt_pack_ms / trt_frames
        trt_cpu_ms_per_frame = trt_cpu_ms / trt_frames
        trt_other_ms_per_frame = max(
            0.0,
            infer_per_frame
            - trt_kernel_ms_per_frame
            - trt_post_ms_per_frame
            - trt_resize_ms_per_frame
            - trt_pack_ms_per_frame
            - trt_cpu_ms_per_frame,
        )

        gpu_line = format_gpu_summary(_gpu_summary)

        pp_line = (
            f"  后处理:    平均 {avg_pp_ms:.1f}ms/帧  "
            f"face={avg_face_ms:.1f}ms  stabilize={avg_stabilize_ms:.1f}ms  "
            f"sharpen={avg_sharpen_ms:.1f}ms"
        ) if postprocess_needed else "  后处理: 未启用"
        trt_line = (
            f"  TRT tiles: {trt_tiles:.0f}个 / {trt_tile_batches:.0f}批  "
            f"平均 {trt_avg_tile_batch:.2f} tiles/批  峰值 {trt_tile_max_batch:.0f}  "
            f"engine_max_batch={trt_engine_max_batch:.0f}  TRT调用 {trt_tile_ms_per_batch:.1f}ms/批"
        ) if trt_tile_batches > 0 else "  TRT tiles: 未使用 TensorRT tile 批处理"
        trt_post_line = (
            f"  TRT后处理: post={trt_post_ms_per_frame:.1f}ms/帧  "
            f"resize={trt_resize_ms_per_frame:.1f}ms/帧  "
            f"pack={trt_pack_ms_per_frame:.1f}ms/帧  "
            f"GPU→CPU={trt_cpu_ms_per_frame:.1f}ms/帧  "
            f"其他={trt_other_ms_per_frame:.1f}ms/帧"
        ) if trt_tile_batches > 0 else ""

        report_lines = [
            "═" * 46,
            "              [性能报告]",
            "═" * 46,
            f"  输出帧数:  {enh_frames}帧  耗时 {total_elapsed:.1f}s",
            f"  总体吞吐:  {total_fps:.2f} fps",
            f"  推理批次:  {enh_batches}批  平均 {avg_batch:.2f}帧/批",
            f"  推理延迟:  {avg_infer_ms:.1f}ms/批  ({infer_per_frame:.1f}ms/帧)",
            trt_line,
            trt_post_line,
            f"  写入延迟:  {avg_write_ms:.2f}ms/帧",
            pp_line,
            f"  CPU占用:   进程平均 {cpu_total_pct:.0f}%  (按 {_cpu_tracker.cpu_count} 核归一化)",
            f"  跳帧统计:  跳过 {skip_frames}帧 ({skip_pct:.1f}%)",
            "  " + "-" * 42,
            gpu_line,
            "═" * 46,
        ]
        for line in report_lines:
            _emit_progress(progress_callback, last_reported_progress, line)

    # -- Join background threads (with timeout to prevent hang) ---------------
    for t in (enhance_thread, postprocess_thread, write_thread):
        if t is not None:
            t.join(timeout=10.0)

    if enhance_errors:
        raise enhance_errors[0]
    if stage3_errors:
        raise stage3_errors[0]
    if write_errors:
        raise write_errors[0]


def _report_stream_progress(
    processed_frames: int,
    total_frames: int | None,
    last_reported_progress: int,
    progress_callback: Callable[[int, str], None] | None,
    start_time: float = 0.0,
) -> int:
    progress = _map_frame_progress(processed_frames, total_frames)
    if progress > last_reported_progress:
        elapsed = time.perf_counter() - start_time if start_time > 0 else 0
        fps_str = f" ({processed_frames / elapsed:.2f} fps)" if elapsed > 1 else ""
        _emit_progress(
            progress_callback, progress,
            f"正在增强视频帧 {processed_frames}/{total_frames or '?'}{fps_str}",
        )
        return progress
    return last_reported_progress


def _finalize_stream_processes(decoder: object, encoder: object) -> None:
    stream_codec.finalize_stream_processes(decoder, encoder)


def _cleanup_stream_processes(decoder: object, encoder: object) -> None:
    stream_codec.cleanup_stream_processes(decoder, encoder)


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)
