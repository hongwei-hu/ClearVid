"""Tests for clearvid.app.preprocess.filters."""
from __future__ import annotations

from pathlib import Path

from clearvid.app.preprocess.filters import (
    _bits_per_pixel,
    _colorspace_filter,
    _deblock_filter,
    _deinterlace_filter,
    _denoise_filter,
    _estimate_denoise_strength,
    build_preprocess_filters,
)
from clearvid.app.schemas.models import EnhancementConfig, VideoMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(**kwargs) -> VideoMetadata:
    defaults = dict(
        input_path=Path("in.mp4"),
        duration_seconds=10.0,
        width=1280,
        height=720,
        fps=30.0,
        video_codec="h264",
        bit_rate=5_000_000,
    )
    defaults.update(kwargs)
    return VideoMetadata(**defaults)


def _make_cfg(**kwargs) -> EnhancementConfig:
    defaults = dict(input_path=Path("in.mp4"), output_path=Path("out.mp4"))
    defaults.update(kwargs)
    return EnhancementConfig(**defaults)


# ---------------------------------------------------------------------------
# _bits_per_pixel
# ---------------------------------------------------------------------------

def test_bits_per_pixel_normal() -> None:
    meta = _make_meta(bit_rate=3_686_400, width=1280, height=720, fps=30.0)
    bpp = _bits_per_pixel(meta)
    assert bpp is not None
    assert abs(bpp - 3_686_400 / (1280 * 720 * 30)) < 1e-9


def test_bits_per_pixel_no_bitrate() -> None:
    meta = _make_meta(bit_rate=None)
    assert _bits_per_pixel(meta) is None


def test_bits_per_pixel_zero_fps() -> None:
    meta = _make_meta(fps=0.0)
    assert _bits_per_pixel(meta) is None


# ---------------------------------------------------------------------------
# _estimate_denoise_strength
# ---------------------------------------------------------------------------

def test_estimate_denoise_strength_very_low_bpp() -> None:
    # bpp < 0.03 → s=8
    meta = _make_meta(bit_rate=100_000, width=1280, height=720, fps=30.0)
    assert _estimate_denoise_strength(meta) == 8.0


def test_estimate_denoise_strength_high_bpp() -> None:
    # bpp > 0.20 → s=3
    meta = _make_meta(bit_rate=100_000_000, width=1280, height=720, fps=30.0)
    assert _estimate_denoise_strength(meta) == 3.0


def test_estimate_denoise_strength_no_bitrate() -> None:
    meta = _make_meta(bit_rate=None)
    assert _estimate_denoise_strength(meta) == 4.0  # safe default


# ---------------------------------------------------------------------------
# _deinterlace_filter
# ---------------------------------------------------------------------------

def test_deinterlace_filter_off() -> None:
    cfg = _make_cfg(preprocess_deinterlace="off")
    meta = _make_meta(is_interlaced=True)
    assert _deinterlace_filter(cfg, meta) == []


def test_deinterlace_filter_auto_progressive() -> None:
    cfg = _make_cfg(preprocess_deinterlace="auto")
    meta = _make_meta(is_interlaced=False)
    assert _deinterlace_filter(cfg, meta) == []


def test_deinterlace_filter_auto_interlaced() -> None:
    cfg = _make_cfg(preprocess_deinterlace="auto")
    meta = _make_meta(is_interlaced=True)
    result = _deinterlace_filter(cfg, meta)
    assert len(result) == 1
    assert "bwdif" in result[0]


# ---------------------------------------------------------------------------
# _denoise_filter
# ---------------------------------------------------------------------------

def test_denoise_filter_disabled() -> None:
    cfg = _make_cfg(preprocess_denoise=False)
    meta = _make_meta()
    assert _denoise_filter(cfg, meta) == []


def test_denoise_filter_enabled() -> None:
    cfg = _make_cfg(preprocess_denoise=True)
    meta = _make_meta(bit_rate=500_000)
    result = _denoise_filter(cfg, meta)
    assert len(result) == 1
    assert "nlmeans" in result[0]
    assert "s=" in result[0]


# ---------------------------------------------------------------------------
# _deblock_filter
# ---------------------------------------------------------------------------

def test_deblock_filter_disabled() -> None:
    cfg = _make_cfg(preprocess_deblock=False)
    meta = _make_meta(video_codec="h264", bit_rate=500_000)
    assert _deblock_filter(cfg, meta) == []


def test_deblock_filter_not_block_codec() -> None:
    cfg = _make_cfg(preprocess_deblock=True)
    meta = _make_meta(video_codec="vp9", bit_rate=500_000)
    assert _deblock_filter(cfg, meta) == []


def test_deblock_filter_high_bitrate_skipped() -> None:
    """bpp > 0.15 → skip deblock."""
    cfg = _make_cfg(preprocess_deblock=True)
    meta = _make_meta(video_codec="h264", bit_rate=500_000_000)
    assert _deblock_filter(cfg, meta) == []


def test_deblock_filter_low_bitrate_h264() -> None:
    cfg = _make_cfg(preprocess_deblock=True)
    # Very low bpp → strong deblock
    meta = _make_meta(video_codec="h264", bit_rate=100_000, width=1280, height=720, fps=30.0)
    result = _deblock_filter(cfg, meta)
    assert len(result) == 1
    assert "deblock" in result[0]
    assert "alpha=1.0" in result[0]


def test_deblock_filter_medium_bitrate_h264() -> None:
    cfg = _make_cfg(preprocess_deblock=True)
    # Medium bpp (0.05 < bpp < 0.10)
    meta = _make_meta(video_codec="h264", bit_rate=2_000_000, width=1280, height=720, fps=30.0)
    result = _deblock_filter(cfg, meta)
    assert len(result) == 1
    assert "deblock" in result[0]
    assert "alpha=0.6" in result[0]


def test_deblock_filter_mild_bitrate_h264() -> None:
    cfg = _make_cfg(preprocess_deblock=True)
    # bpp > 0.10 but < 0.15 → weak
    meta = _make_meta(video_codec="h264", bit_rate=4_000_000, width=1280, height=720, fps=30.0)
    result = _deblock_filter(cfg, meta)
    assert len(result) == 1
    assert "alpha=0.3" in result[0]


def test_deblock_filter_mpeg2() -> None:
    cfg = _make_cfg(preprocess_deblock=True)
    meta = _make_meta(video_codec="mpeg2video", bit_rate=200_000, width=720, height=480, fps=25.0)
    result = _deblock_filter(cfg, meta)
    assert len(result) == 1
    assert "deblock" in result[0]


# ---------------------------------------------------------------------------
# _colorspace_filter
# ---------------------------------------------------------------------------

def test_colorspace_filter_disabled() -> None:
    cfg = _make_cfg(preprocess_colorspace_normalize=False)
    meta = _make_meta()
    assert _colorspace_filter(cfg, meta) == []


def test_colorspace_filter_unknown_primaries() -> None:
    """None primaries → skip to avoid FFmpeg error."""
    cfg = _make_cfg(preprocess_colorspace_normalize=True)
    meta = _make_meta(color_primaries=None)
    assert _colorspace_filter(cfg, meta) == []


def test_colorspace_filter_already_bt709() -> None:
    cfg = _make_cfg(preprocess_colorspace_normalize=True)
    meta = _make_meta(color_primaries="bt709")
    assert _colorspace_filter(cfg, meta) == []


def test_colorspace_filter_non_bt709() -> None:
    cfg = _make_cfg(preprocess_colorspace_normalize=True)
    meta = _make_meta(color_primaries="smpte170m")
    result = _colorspace_filter(cfg, meta)
    assert len(result) == 1
    assert "colorspace" in result[0]
    assert "bt709" in result[0]


# ---------------------------------------------------------------------------
# build_preprocess_filters — integration
# ---------------------------------------------------------------------------

def test_build_preprocess_filters_no_filters() -> None:
    cfg = _make_cfg(
        preprocess_denoise=False,
        preprocess_deblock=False,
        preprocess_deinterlace="off",
        preprocess_colorspace_normalize=False,
    )
    meta = _make_meta(is_interlaced=False)
    result = build_preprocess_filters(cfg, meta)
    assert result == []


def test_build_preprocess_filters_returns_list() -> None:
    cfg = _make_cfg()
    meta = _make_meta()
    result = build_preprocess_filters(cfg, meta)
    assert isinstance(result, list)
    for f in result:
        assert isinstance(f, str)


def test_build_preprocess_filters_interlaced_deblock() -> None:
    cfg = _make_cfg(
        preprocess_denoise=False,
        preprocess_deblock=True,
        preprocess_deinterlace="auto",
        preprocess_colorspace_normalize=False,
    )
    # interlaced + low-bitrate h264 → deinterlace + deblock
    meta = _make_meta(is_interlaced=True, video_codec="h264", bit_rate=100_000, width=720, height=480, fps=25.0)
    result = build_preprocess_filters(cfg, meta)
    texts = " ".join(result)
    assert "bwdif" in texts
    assert "deblock" in texts
