from __future__ import annotations

from pathlib import Path

from clearvid.app.io.probe import collect_environment_info
from clearvid.app.models.realesrgan_runner import resolve_upscale_model, validate_realesrgan_environment
from clearvid.app.preprocess.filters import build_preprocess_filters
from clearvid.app.schemas.models import BackendType, EnhancementConfig, ExecutionPlan, InferenceAccelerator, TargetProfile, VideoMetadata


def build_execution_plan(config: EnhancementConfig, metadata: VideoMetadata) -> ExecutionPlan:
    output_width, output_height = resolve_output_size(metadata.width, metadata.height, config.target_profile)
    resolved_backend = resolve_backend(config.backend)
    notes = [
        f"Input: {metadata.width}x{metadata.height} @ {metadata.fps:.3f} fps",
        f"Output: {output_width}x{output_height}",
        f"Backend: {resolved_backend.value}",
    ]

    if resolved_backend != config.backend:
        notes.append(f"Resolved backend: {resolved_backend.value}")

    if resolved_backend == BackendType.BASELINE:
        command = build_baseline_command(config, output_width, output_height)
        notes.append("Using FFmpeg baseline backend.")
        return ExecutionPlan(
            command=command,
            output_width=output_width,
            output_height=output_height,
            backend=resolved_backend,
            notes=notes,
        )

    available, message = validate_realesrgan_environment(Path("weights") / "realesrgan")
    if not available:
        raise RuntimeError(message)

    notes.append("Using Real-ESRGAN backend.")
    resolved_model = resolve_upscale_model(config.upscale_model, config.quality_mode)
    notes.append(f"Upscale model: {resolved_model}")
    if config.temporal_stabilize_enabled:
        notes.append(f"Temporal stabilization: ON (strength={config.temporal_stabilize_strength:.2f})")
    preprocess_filters = build_preprocess_filters(config, metadata)
    if preprocess_filters:
        notes.append(f"Preprocess filters: {', '.join(preprocess_filters)}")
    else:
        notes.append("Preprocess filters: none")
    accel_label = config.inference_accelerator.value
    notes.append(f"Inference accelerator: {accel_label}")
    notes.append(f"Async pipeline: {'ON' if config.async_pipeline else 'OFF'}")
    return ExecutionPlan(
        output_width=output_width,
        output_height=output_height,
        backend=resolved_backend,
        notes=notes,
    )


def resolve_backend(requested_backend: BackendType) -> BackendType:
    if requested_backend != BackendType.AUTO:
        return requested_backend

    environment = collect_environment_info()
    if environment.realesrgan_available:
        return BackendType.REALESRGAN
    return BackendType.BASELINE


def build_baseline_command(config: EnhancementConfig, output_width: int, output_height: int) -> list[str]:
    video_filters = [_build_scale_filter(config.target_profile, output_width, output_height)]

    if config.denoise_strength > 0:
        video_filters.append("hqdn3d=1.5:1.5:6:6")

    if config.sharpen_strength > 0:
        video_filters.append("unsharp=5:5:0.5:5:5:0.0")

    video_filters.append("format=yuv420p")
    filter_graph = ",".join(video_filters)

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
        "-map",
        "0",
        "-vf",
        filter_graph,
        "-c:v",
        config.encoder,
        "-preset",
        config.encoder_preset,
    ])

    if config.video_bitrate:
        command.extend(["-b:v", config.video_bitrate])
    else:
        command.extend(["-cq", "18"])

    command.extend(["-pix_fmt", "yuv420p"])

    if config.preserve_audio:
        command.extend(["-c:a", "copy"])
    else:
        command.extend(["-an"])

    if config.preserve_subtitles:
        command.extend(["-c:s", "copy"])
    else:
        command.extend(["-sn"])

    if config.preserve_metadata:
        command.extend(["-map_metadata", "0"])
    else:
        command.extend(["-map_metadata", "-1"])

    command.append(str(config.output_path))
    return command


def resolve_output_size(width: int, height: int, target_profile: TargetProfile) -> tuple[int, int]:
    if target_profile == TargetProfile.SOURCE:
        return _make_even(width), _make_even(height)
    if target_profile == TargetProfile.SCALE2X:
        return _make_even(width * 2), _make_even(height * 2)
    if target_profile == TargetProfile.SCALE4X:
        return _make_even(width * 4), _make_even(height * 4)
    if target_profile == TargetProfile.FHD:
        return 1920, 1080
    if target_profile == TargetProfile.UHD4K:
        return 3840, 2160
    return _make_even(width), _make_even(height)


def _build_scale_filter(target_profile: TargetProfile, output_width: int, output_height: int) -> str:
    if target_profile in {TargetProfile.FHD, TargetProfile.UHD4K}:
        return (
            f"scale={output_width}:{output_height}:"
            f"force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={output_width}:{output_height}:(ow-iw)/2:(oh-ih)/2:black"
        )
    return f"scale={output_width}:{output_height}:flags=lanczos"


def _fit_to_height(width: int, height: int, target_height: int) -> tuple[int, int]:
    scale = target_height / height
    scaled_width = int(round(width * scale))
    return _make_even(scaled_width), _make_even(target_height)


def _make_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1
