"""Smart parameter recommendation engine.

Analyzes input video metadata and hardware environment to suggest optimal
processing parameters. Used by both GUI ("一键最佳" button) and CLI plan command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clearvid.app.schemas.models import EnvironmentInfo, VideoMetadata


@dataclass
class Recommendation:
    """Recommended parameter values with human-readable explanations."""

    target_profile: str
    quality_mode: str
    upscale_model: str
    face_restore_enabled: bool
    face_restore_model: str
    temporal_stabilize_enabled: bool
    sharpen_enabled: bool
    sharpen_strength: float
    encoder: str
    encoder_crf: int | None
    output_pixel_format: str
    inference_accelerator: str
    async_pipeline: bool
    tile_size: int
    notes: list[str]


def recommend(metadata: VideoMetadata, environment: EnvironmentInfo) -> Recommendation:
    """Generate processing recommendations based on video and hardware analysis."""
    notes: list[str] = []
    vram = environment.gpu_memory_mb or 0
    w, h = metadata.width, metadata.height
    pixels = w * h
    duration = metadata.duration_seconds
    bitrate_kbps = (metadata.bit_rate or 0) // 1000

    # --- Target profile ---
    if pixels <= 921_600:  # ≤ ~720p (1280x720)
        target_profile = "fhd"
        notes.append(f"输入 {w}x{h} 较低，推荐提升至 1080p")
    elif pixels <= 2_073_600:  # ≤ 1080p
        target_profile = "uhd4k"
        notes.append(f"输入 {w}x{h}，推荐提升至 4K")
    else:
        target_profile = "source"
        notes.append(f"输入已是 {w}x{h} 高分辨率，建议保持原始分辨率")

    # --- Quality mode ---
    if vram >= 12_000:
        quality_mode = "quality"
    elif vram >= 6_000:
        quality_mode = "balanced"
    else:
        quality_mode = "fast"
        notes.append("显存有限，推荐快速模式以避免 OOM")

    # --- Upscale model ---
    # When hardware acceleration is available, general_v3 under TensorRT is
    # faster than pure-PyTorch x4plus with comparable quality.  Reserve x4plus
    # for quality mode when no acceleration is available.
    gpu_capable = (
        environment.torch_gpu_compatible
        and environment.nvidia_smi_available
        and vram >= 6_000
    )
    if quality_mode == "quality" and vram >= 8_000 and not gpu_capable:
        upscale_model = "x4plus"
        notes.append("无硬件加速 + 高质量模式，使用 RRDB x4plus 模型以获得最佳细节")
    else:
        upscale_model = "general_v3"
        if quality_mode == "quality" and gpu_capable:
            notes.append("检测到 GPU 加速可用，使用 general_v3 (TRT 加速后性能更优)")

    # --- Tile size based on VRAM ---
    if vram >= 16_000:
        tile_size = 0  # no tiling needed
        notes.append("大显存，禁用分块以获得最佳质量")
    elif vram >= 8_000:
        tile_size = 512
    elif vram >= 4_000:
        tile_size = 256
    else:
        tile_size = 128
        notes.append("低显存，使用 128 分块尺寸")

    # --- Face restoration ---
    # FAST mode auto-disables in pipeline; here we set the GUI default.
    face_restore_enabled = quality_mode != "fast"
    face_restore_model = "codeformer"

    # --- Temporal stabilization ---
    # FAST mode auto-disables in pipeline; here we set the GUI default.
    temporal_stabilize_enabled = duration > 1.0 and quality_mode != "fast"

    # --- Sharpening ---
    sharpen_enabled = True
    sharpen_strength = 0.12

    # --- Encoder selection ---
    encoders = environment.ffmpeg_encoders
    if "av1_nvenc" in encoders and vram >= 8_000:
        encoder = "av1_nvenc"
        notes.append("检测到 AV1 硬件编码支持，使用 AV1 以获得更高压缩效率")
    elif "hevc_nvenc" in encoders:
        encoder = "hevc_nvenc"
    else:
        encoder = "libx264"
        notes.append("未检测到硬件编码器，回退到 CPU 编码 (libx264)")

    # --- CRF ---
    encoder_crf: int | None = None  # let encoder default

    # --- Pixel format ---
    output_pixel_format = "yuv420p"

    # --- Inference accelerator ---
    if environment.torch_gpu_compatible:
        inference_accelerator = "auto"
    else:
        inference_accelerator = "none"

    # --- Async pipeline ---
    async_pipeline = True

    # --- High bitrate warning ---
    if bitrate_kbps > 15_000 and pixels >= 2_073_600:
        notes.append(
            f"输入码率已较高 ({bitrate_kbps} kbps)，超分提升可能有限，"
            "可考虑保持原始分辨率或仅做人脸修复"
        )

    return Recommendation(
        target_profile=target_profile,
        quality_mode=quality_mode,
        upscale_model=upscale_model,
        face_restore_enabled=face_restore_enabled,
        face_restore_model=face_restore_model,
        temporal_stabilize_enabled=temporal_stabilize_enabled,
        sharpen_enabled=sharpen_enabled,
        sharpen_strength=sharpen_strength,
        encoder=encoder,
        encoder_crf=encoder_crf,
        output_pixel_format=output_pixel_format,
        inference_accelerator=inference_accelerator,
        async_pipeline=async_pipeline,
        tile_size=tile_size,
        notes=notes,
    )
