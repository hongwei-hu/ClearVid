from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from clearvid.app.bootstrap.paths import (
    REALESRGAN_WEIGHTS_DIR,
    ffmpeg_path as _ffmpeg_path,
    ffprobe_path as _ffprobe_path,
)
from clearvid.app.models.realesrgan_runner import inspect_realesrgan_runtime
from clearvid.app.schemas.models import EnvironmentInfo, StreamInfo, VideoMetadata


def _which(binary_name: str) -> str | None:
    return shutil.which(binary_name)


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
        encoding="utf-8", errors="replace",
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip() or completed.stderr.strip()


def probe_video(input_path: Path) -> VideoMetadata:
    fp = _ffprobe_path() or _which("ffprobe")
    if not fp:
        raise RuntimeError("ffprobe is not available on PATH")

    command = [
        fp,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(input_path),
    ]
    payload = json.loads(_run_text(command) or "{}")
    streams = payload.get("streams", [])
    format_info = payload.get("format", {})

    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise RuntimeError(f"No video stream found in {input_path}")

    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    subtitle_streams = [stream for stream in streams if stream.get("codec_type") == "subtitle"]

    fps_value = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = _parse_fps(fps_value)

    normalized_streams = [
        StreamInfo(
            index=stream.get("index", 0),
            codec_type=stream.get("codec_type", "unknown"),
            codec_name=stream.get("codec_name"),
            width=stream.get("width"),
            height=stream.get("height"),
            channels=stream.get("channels"),
            sample_rate=stream.get("sample_rate"),
            language=(stream.get("tags") or {}).get("language"),
        )
        for stream in streams
    ]

    return VideoMetadata(
        input_path=input_path,
        container=format_info.get("format_name"),
        duration_seconds=float(format_info.get("duration", 0.0)),
        bit_rate=_to_int(format_info.get("bit_rate")),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        video_codec=video_stream.get("codec_name", "unknown"),
        audio_codec=audio_streams[0].get("codec_name") if audio_streams else None,
        audio_streams=len(audio_streams),
        subtitle_streams=len(subtitle_streams),
        is_interlaced=_detect_interlaced(video_stream),
        streams=normalized_streams,
        color_primaries=_normalise_color_tag(video_stream.get("color_primaries")),
        color_space=_normalise_color_tag(video_stream.get("color_space")),
    )


def collect_environment_info() -> EnvironmentInfo:
    ffmpeg_bin = _ffmpeg_path() or _which("ffmpeg")
    ffprobe_bin = _ffprobe_path() or _which("ffprobe")
    ffmpeg_available = ffmpeg_bin is not None
    ffprobe_available = ffprobe_bin is not None
    nvidia_smi_available = _which("nvidia-smi") is not None

    ffmpeg_version = None
    ffmpeg_hwaccels: list[str] = []
    ffmpeg_encoders: list[str] = []
    if ffmpeg_available:
        version_text = _run_text([ffmpeg_bin, "-hide_banner", "-version"])
        ffmpeg_version = version_text.splitlines()[0] if version_text else None
        hwaccel_text = _run_text([ffmpeg_bin, "-hide_banner", "-hwaccels"])
        ffmpeg_hwaccels = _parse_hwaccels(hwaccel_text)
        encoder_text = _run_text([ffmpeg_bin, "-hide_banner", "-encoders"])
        ffmpeg_encoders = _parse_nvenc_encoders(encoder_text)

    gpu_name = None
    gpu_driver_version = None
    gpu_memory_mb = None
    if nvidia_smi_available:
        query_text = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        first_line = query_text.splitlines()[0] if query_text else ""
        if first_line:
            parts = [part.strip() for part in first_line.split(",")]
            if len(parts) >= 3:
                gpu_name, gpu_driver_version, memory_text = parts[:3]
                gpu_memory_mb = _to_int(memory_text)

    realtime_weights_path = REALESRGAN_WEIGHTS_DIR
    realesrgan_available, realesrgan_message, torch_version, torch_cuda_available, torch_gpu_compatible = (
        inspect_realesrgan_runtime(realtime_weights_path)
    )

    return EnvironmentInfo(
        ffmpeg_available=ffmpeg_available,
        ffprobe_available=ffprobe_available,
        nvidia_smi_available=nvidia_smi_available,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_hwaccels=ffmpeg_hwaccels,
        ffmpeg_encoders=ffmpeg_encoders,
        gpu_name=gpu_name,
        gpu_driver_version=gpu_driver_version,
        gpu_memory_mb=gpu_memory_mb,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_gpu_compatible=torch_gpu_compatible,
        preferred_backend=EnvironmentInfo.model_fields["preferred_backend"].annotation.REALESRGAN if realesrgan_available else EnvironmentInfo.model_fields["preferred_backend"].annotation.BASELINE,
        realesrgan_available=realesrgan_available,
        realesrgan_message=realesrgan_message,
    )


def _parse_fps(raw_fps: str) -> float:
    numerator_text, _, denominator_text = raw_fps.partition("/")
    numerator = float(numerator_text or 0.0)
    denominator = float(denominator_text or 1.0)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse_hwaccels(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.lower().startswith("hardware acceleration")]


def _parse_nvenc_encoders(text: str) -> list[str]:
    matches = re.findall(r"\b(?:av1|h264|hevc)_nvenc\b", text)
    return sorted(set(matches))


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _detect_interlaced(video_stream: dict) -> bool:
    """Detect interlaced content from ffprobe stream metadata."""
    field_order = (video_stream.get("field_order") or "").lower()
    if field_order in {"tt", "bb", "tb", "bt"}:
        return True
    if field_order == "progressive":
        return False
    # Fallback: check codec-level flags
    coded_height = _to_int(video_stream.get("coded_height")) or 0
    height = _to_int(video_stream.get("height")) or 0
    if coded_height > 0 and height > 0 and coded_height != height:
        return True
    return False


def _normalise_color_tag(value: object) -> str | None:
    """Return the colour tag string if known, or None for unspecified/unknown.

    ffprobe reports 'unknown' for streams where the codec bitstream does not
    carry colour metadata (ITU-T H.264 colour_primaries = 2).  We treat this
    the same as a missing field so callers can safely skip colour-space
    conversion when the value is None.
    """
    if not value:
        return None
    s = str(value).lower().strip()
    if s in {"unknown", "unspecified", "reserved", ""}:
        return None
    return s
