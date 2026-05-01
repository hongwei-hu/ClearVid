from __future__ import annotations

import importlib
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from clearvid.app.bootstrap.paths import REALESRGAN_WEIGHTS_DIR, TRT_CACHE_DIR, WEIGHTS_DIR
from clearvid.app.export_control import ExportControl
from clearvid.app.models import stream_codec
from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.models.gfpgan_runner import GFPGANRestorer
from clearvid.app.models.tensorrt_engine import (
    InferenceAccelerator,
    TRT_MAX_BATCH,
    check_engine_ready,
    describe_accelerator,
    detect_best_accelerator,
)
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
from clearvid.app.utils.subprocess_utils import run_command

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SCALE = 4


class FrameSkipper:
    """Skip super-resolution inference for near-static frames."""

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
        assert self._prev_enhanced is not None
        return self._prev_enhanced.copy()


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
    except Exception:
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
    except Exception as exc:
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
    if config.batch_size not in (0, 4):
        return config.batch_size
    try:
        import torch

        if not torch.cuda.is_available():
            return config.batch_size
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        return config.batch_size
    megapixels = (width * height) / 1_000_000
    if model_arch == "srvgg":
        return _auto_srvgg_batch_size(vram_mb, megapixels)
    return _auto_rrdb_batch_size(vram_mb, megapixels)


def _auto_srvgg_batch_size(vram_mb: int, megapixels: float) -> int:
    if vram_mb >= 24_000 and megapixels <= 0.6:
        return 16
    if vram_mb >= 16_000 and megapixels <= 1.0:
        return 8
    if vram_mb >= 8_000:
        return 4
    return 2


def _auto_rrdb_batch_size(vram_mb: int, megapixels: float) -> int:
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
    if accel != InferenceAccelerator.TENSORRT:
        return config_batch
    if config_accel.value == "auto":
        return config_batch
    return max(1, min(config_batch, TRT_MAX_BATCH))


def _auto_trt_tile_size(config_tile_size: int, width: int, height: int) -> int:
    if config_tile_size > 0:
        return config_tile_size
    try:
        import torch

        if not torch.cuda.is_available():
            return 512
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        return 512
    long_edge = max(width, height)
    if vram_mb >= 24_000 and long_edge <= 1080:
        return 1024
    if vram_mb >= 12_000 and long_edge <= 1080:
        return 768
    return 512


def ensure_realesrgan_weights(weights_path: Path, model_key: str = "general_v3") -> Path:
    weights_path.mkdir(parents=True, exist_ok=True)
    entry = _MODEL_REGISTRY[model_key]
    target = weights_path / entry["filename"]
    if target.exists():
        return target

    discovered = find_realesrgan_weights(weights_path)
    for path in discovered:
        if path.name == entry["filename"]:
            return path

    _, _, _, load_file_from_url = _load_runtime_components()
    try:
        downloaded_path = load_file_from_url(
            url=entry["url"],
            model_dir=str(weights_path),
            progress=True,
            file_name=entry["filename"],
        )
    except Exception as exc:
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
    if upscale_model != UpscaleModel.AUTO:
        return upscale_model.value
    if quality_mode == QualityMode.QUALITY:
        if accel is not None and accel != InferenceAccelerator.NONE:
            return "general_v3"
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
    if requested == InferenceAcceleratorEnum.AUTO:
        return detect_best_accelerator()
    return InferenceAccelerator(requested.value)


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
    if frame.shape[1] == output_width and frame.shape[0] == output_height:
        return frame
    if target_profile in {TargetProfile.FHD, TargetProfile.UHD4K}:
        return _fit_and_pad_frame(frame, output_width, output_height)
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


def _resolve_tile_size(config_tile_size: int, width: int, height: int, fp16: bool) -> int:
    if config_tile_size > 0:
        return config_tile_size
    try:
        import torch

        if not torch.cuda.is_available():
            return 512
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:
        return 512
    megapixels = (width * height) / 1_000_000
    estimated_mb = int(megapixels * (2000 if fp16 else 4000))
    if estimated_mb < vram_mb * 0.5:
        return 0
    return 512


__all__ = [
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_MODEL_SCALE",
    "DEFAULT_MODEL_URL",
    "FrameSkipper",
    "_MODEL_REGISTRY",
    "_apply_quality_mode_overrides",
    "_auto_batch_size",
    "_auto_rrdb_batch_size",
    "_auto_srvgg_batch_size",
    "_auto_trt_tile_size",
    "_build_codeformer_restorer",
    "_build_decode_command",
    "_build_encode_command",
    "_build_temporal_stabilizer",
    "_build_upsampler",
    "_estimate_total_frames",
    "_fit_and_pad_frame",
    "_load_runtime_components",
    "_map_frame_progress",
    "_mux_output",
    "_mux_preview",
    "_resolve_accelerator",
    "_resolve_outscale",
    "_resolve_tile_size",
    "_resolve_trt_batch",
    "_resize_for_target",
    "ensure_realesrgan_weights",
    "find_realesrgan_weights",
    "inspect_realesrgan_runtime",
    "resolve_upscale_model",
    "validate_realesrgan_environment",
]