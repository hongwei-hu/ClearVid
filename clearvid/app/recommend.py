"""Smart parameter recommendation engine.

Analyzes input video metadata and hardware environment to suggest optimal
processing parameters. Used by both GUI ("一键最佳" button) and CLI plan command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from clearvid.app.recommendation_rules import (
    add_high_bitrate_note,
    choose_encoder,
    choose_quality_mode,
    choose_target_profile,
    choose_tile_size,
    choose_upscale_model,
    has_gpu_acceleration,
    should_denoise,
)

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
    preprocess_denoise: bool
    notes: list[str]


def recommend(metadata: VideoMetadata, environment: EnvironmentInfo) -> Recommendation:
    """Generate processing recommendations based on video and hardware analysis."""
    notes: list[str] = []
    vram = environment.gpu_memory_mb or 0
    duration = metadata.duration_seconds
    bitrate_kbps = (metadata.bit_rate or 0) // 1000

    target_profile = choose_target_profile(metadata, notes)
    quality_mode = choose_quality_mode(vram, notes)
    gpu_capable = has_gpu_acceleration(environment, vram)
    upscale_model = choose_upscale_model(quality_mode, vram, gpu_capable, notes)
    tile_size = choose_tile_size(vram, notes)

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

    encoder = choose_encoder(environment, vram, notes)

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

    preprocess_denoise, bpp_val = should_denoise(metadata)
    if preprocess_denoise:
        notes.append(f"低码率源 (bpp={bpp_val:.3f})，自动开启降噪预处理")

    add_high_bitrate_note(metadata, bitrate_kbps, notes)

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
        preprocess_denoise=preprocess_denoise,
        notes=notes,
    )
