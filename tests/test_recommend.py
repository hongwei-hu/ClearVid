"""Tests for clearvid.app.recommend."""
from __future__ import annotations

from pathlib import Path

from clearvid.app.recommend import Recommendation, recommend
from clearvid.app.schemas.models import EnvironmentInfo, VideoMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    bit_rate: int = 3_000_000,
    duration: float = 10.0,
) -> VideoMetadata:
    return VideoMetadata(
        input_path=Path("in.mp4"),
        width=width,
        height=height,
        fps=fps,
        bit_rate=bit_rate,
        duration_seconds=duration,
        video_codec="h264",
    )


def _make_env(
    vram_mb: int = 8_000,
    torch_gpu: bool = True,
    nvidia_smi: bool = True,
    encoders: list[str] | None = None,
    ffmpeg_available: bool = True,
) -> EnvironmentInfo:
    return EnvironmentInfo(
        ffmpeg_available=ffmpeg_available,
        ffprobe_available=True,
        nvidia_smi_available=nvidia_smi,
        gpu_memory_mb=vram_mb,
        torch_cuda_available=torch_gpu,
        torch_gpu_compatible=torch_gpu,
        ffmpeg_encoders=encoders or ["hevc_nvenc"],
    )


# ---------------------------------------------------------------------------
# Target profile recommendations
# ---------------------------------------------------------------------------

def test_recommend_low_res_fhd() -> None:
    """<=720p input → recommend FHD."""
    rec = recommend(_make_meta(width=854, height=480), _make_env())
    assert rec.target_profile == "fhd"


def test_recommend_mid_res_uhd4k() -> None:
    """1080p input → recommend 4K."""
    rec = recommend(_make_meta(width=1920, height=1080), _make_env())
    assert rec.target_profile == "uhd4k"


def test_recommend_high_res_source() -> None:
    """4K input → keep source."""
    rec = recommend(_make_meta(width=3840, height=2160), _make_env())
    assert rec.target_profile == "source"


# ---------------------------------------------------------------------------
# Quality mode recommendations
# ---------------------------------------------------------------------------

def test_recommend_quality_mode_high_vram() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=16_000))
    assert rec.quality_mode == "quality"


def test_recommend_quality_mode_mid_vram() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=8_000))
    assert rec.quality_mode == "balanced"


def test_recommend_quality_mode_low_vram() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=2_000))
    assert rec.quality_mode == "fast"


def test_recommend_quality_mode_balanced_vram() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=6_000))
    assert rec.quality_mode == "balanced"


# ---------------------------------------------------------------------------
# Tile size recommendations
# ---------------------------------------------------------------------------

def test_recommend_tile_size_no_tiling_large_vram() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=20_000))
    assert rec.tile_size == 0


def test_recommend_tile_size_512_for_midrange() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=10_000))
    assert rec.tile_size == 512


def test_recommend_tile_size_256_for_limited() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=4_000))
    assert rec.tile_size == 256


def test_recommend_tile_size_128_for_low() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=2_000))
    assert rec.tile_size == 128


# ---------------------------------------------------------------------------
# Face restoration
# ---------------------------------------------------------------------------

def test_face_restore_enabled_for_quality_mode() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=16_000))
    assert rec.face_restore_enabled is True


def test_face_restore_disabled_for_fast_mode() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=1_000))
    assert rec.face_restore_enabled is False


def test_face_restore_model_is_codeformer() -> None:
    rec = recommend(_make_meta(), _make_env(vram_mb=16_000))
    assert rec.face_restore_model == "codeformer"


# ---------------------------------------------------------------------------
# Encoder selection
# ---------------------------------------------------------------------------

def test_encoder_av1_when_available_high_vram() -> None:
    env = _make_env(vram_mb=12_000, encoders=["av1_nvenc", "hevc_nvenc"])
    rec = recommend(_make_meta(), env)
    assert rec.encoder == "av1_nvenc"


def test_encoder_hevc_nvenc_fallback() -> None:
    env = _make_env(vram_mb=8_000, encoders=["hevc_nvenc"])
    rec = recommend(_make_meta(), env)
    assert rec.encoder == "hevc_nvenc"


def test_encoder_libx264_cpu_fallback() -> None:
    env = EnvironmentInfo(
        ffmpeg_available=True,
        ffprobe_available=True,
        nvidia_smi_available=False,
        gpu_memory_mb=0,
        ffmpeg_encoders=[],  # empty: no NVENC
    )
    rec = recommend(_make_meta(), env)
    assert rec.encoder == "libx264"


# ---------------------------------------------------------------------------
# Temporal stabilization
# ---------------------------------------------------------------------------

def test_temporal_stabilize_disabled_short_video() -> None:
    """Very short clip (≤1s) → disable temporal stabilization."""
    rec = recommend(_make_meta(duration=0.5), _make_env(vram_mb=16_000))
    assert rec.temporal_stabilize_enabled is False


def test_temporal_stabilize_enabled_long_video() -> None:
    rec = recommend(_make_meta(duration=30.0), _make_env(vram_mb=16_000))
    assert rec.temporal_stabilize_enabled is True


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_recommend_returns_recommendation_instance() -> None:
    rec = recommend(_make_meta(), _make_env())
    assert isinstance(rec, Recommendation)


def test_recommend_notes_is_list_of_strings() -> None:
    rec = recommend(_make_meta(), _make_env())
    assert isinstance(rec.notes, list)
    for note in rec.notes:
        assert isinstance(note, str)


# ---------------------------------------------------------------------------
# Preprocess denoise
# ---------------------------------------------------------------------------

def test_preprocess_denoise_enabled_for_low_bpp() -> None:
    """Very low bitrate → enable preprocess denoise."""
    meta = _make_meta(width=1280, height=720, fps=30.0, bit_rate=10_000)
    rec = recommend(meta, _make_env())
    assert rec.preprocess_denoise is True


def test_preprocess_denoise_disabled_for_high_bpp() -> None:
    meta = _make_meta(width=1280, height=720, fps=30.0, bit_rate=100_000_000)
    rec = recommend(meta, _make_env())
    assert rec.preprocess_denoise is False
