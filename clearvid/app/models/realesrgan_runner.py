from __future__ import annotations

import importlib
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.schemas.models import EnhancementConfig, TargetProfile, VideoMetadata
from clearvid.app.utils.subprocess_utils import run_command

DEFAULT_MODEL_FILENAME = "realesr-general-x4v3.pth"
DEFAULT_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
DEFAULT_MODEL_SCALE = 4


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


def run_realesrgan_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> None:
    weights_dir = Path.cwd() / "weights" / "realesrgan"
    model_path = ensure_realesrgan_weights(weights_dir)
    upsampler = _build_upsampler(config, model_path)
    codeformer_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)

    temp_root = Path(tempfile.mkdtemp(prefix="clearvid-realesrgan-"))
    input_frames_dir = temp_root / "input_frames"
    output_frames_dir = temp_root / "output_frames"
    input_frames_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        _extract_frames(config, input_frames_dir)
        frame_paths = sorted(input_frames_dir.glob("*.png"))
        if not frame_paths:
            raise RuntimeError("Real-ESRGAN 未提取到任何视频帧。")

        outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
        for frame_path in frame_paths:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取提取帧: {frame_path}")

            enhanced, _ = upsampler.enhance(image, outscale=outscale)
            if codeformer_restorer is not None:
                enhanced = codeformer_restorer.restore_faces(enhanced)
            finalized = _resize_for_target(enhanced, output_width, output_height, config.target_profile)
            output_frame_path = output_frames_dir / frame_path.name
            if not cv2.imwrite(str(output_frame_path), finalized):
                raise RuntimeError(f"无法写出增强帧: {output_frame_path}")

        temp_video_path = temp_root / "enhanced_video.mp4"
        _encode_video(config, metadata, output_frames_dir, temp_video_path)
        _mux_output(config, temp_video_path)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def ensure_realesrgan_weights(weights_path: Path) -> Path:
    weights_path.mkdir(parents=True, exist_ok=True)
    discovered = find_realesrgan_weights(weights_path)
    if discovered:
        return discovered[0]

    _, _, load_file_from_url = _load_runtime_components()
    try:
        downloaded_path = load_file_from_url(
            url=DEFAULT_MODEL_URL,
            model_dir=str(weights_path),
            progress=True,
            file_name=DEFAULT_MODEL_FILENAME,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"自动下载 Real-ESRGAN 权重失败: {exc}") from exc

    model_path = Path(downloaded_path)
    if not model_path.exists():
        raise RuntimeError(f"Real-ESRGAN 权重下载后未找到: {model_path}")
    return model_path


def _load_runtime_components() -> tuple[type, type, object]:
    realesrgan_utils = importlib.import_module("realesrgan.utils")
    srvgg_arch = importlib.import_module("realesrgan.archs.srvgg_arch")
    download_util = importlib.import_module("basicsr.utils.download_util")
    return realesrgan_utils.RealESRGANer, srvgg_arch.SRVGGNetCompact, download_util.load_file_from_url


def _build_upsampler(config: EnhancementConfig, model_path: Path) -> object:
    real_esrganer_cls, srvgg_cls, _ = _load_runtime_components()
    model = srvgg_cls(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=DEFAULT_MODEL_SCALE,
        act_type="prelu",
    )
    return real_esrganer_cls(
        scale=DEFAULT_MODEL_SCALE,
        model_path=str(model_path),
        model=model,
        tile=config.tile_size,
        tile_pad=config.tile_pad,
        pre_pad=0,
        half=config.fp16_enabled,
    )


def _build_codeformer_restorer(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> CodeFormerRestorer | None:
    if not config.face_restore_enabled:
        return None

    upscale_factor = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
    return CodeFormerRestorer(
        fidelity_weight=config.face_restore_strength,
        upscale_factor=upscale_factor,
        weights_root=Path.cwd() / "weights",
    )


def _extract_frames(config: EnhancementConfig, output_dir: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-i",
        str(config.input_path),
    ]
    if config.preview_seconds:
        command.extend(["-t", str(config.preview_seconds)])
    command.extend([
        "-vsync",
        "0",
        str(output_dir / "%08d.png"),
    ])
    run_command(command)


def _encode_video(config: EnhancementConfig, metadata: VideoMetadata, frames_dir: Path, temp_video_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-framerate",
        f"{metadata.fps:.6f}",
        "-i",
        str(frames_dir / "%08d.png"),
        "-c:v",
        config.encoder,
        "-preset",
        config.encoder_preset,
    ]

    if config.video_bitrate:
        command.extend(["-b:v", config.video_bitrate])
    else:
        command.extend(["-cq", "18"])

    command.extend([
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(temp_video_path),
    ])
    run_command(command)


def _mux_output(config: EnhancementConfig, temp_video_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(temp_video_path),
        "-i",
        str(config.input_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
    ]

    if config.preserve_audio:
        command.extend(["-map", "1:a?", "-c:a", "copy"])
    else:
        command.append("-an")

    if config.preserve_subtitles:
        command.extend(["-map", "1:s?", "-c:s", "copy"])
    else:
        command.append("-sn")

    if config.preserve_metadata:
        command.extend(["-map_metadata", "1", "-map_chapters", "1"])
    else:
        command.extend(["-map_metadata", "-1", "-map_chapters", "-1"])

    command.extend(["-shortest", str(config.output_path)])
    run_command(command)


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
