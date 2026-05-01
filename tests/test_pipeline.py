"""Tests for clearvid.app.pipeline."""
from __future__ import annotations

from pathlib import Path

from clearvid.app.pipeline import (
    _build_scale_filter,
    _fit_to_height,
    _make_even,
    build_baseline_command,
    resolve_output_size,
)
from clearvid.app.schemas.models import (
    EnhancementConfig,
    TargetProfile,
)


# ---------------------------------------------------------------------------
# _make_even
# ---------------------------------------------------------------------------

def test_make_even_already_even() -> None:
    assert _make_even(100) == 100


def test_make_even_odd() -> None:
    assert _make_even(101) == 102


def test_make_even_zero() -> None:
    assert _make_even(0) == 0


def test_make_even_one() -> None:
    assert _make_even(1) == 2


# ---------------------------------------------------------------------------
# resolve_output_size
# ---------------------------------------------------------------------------

def test_resolve_output_size_fhd() -> None:
    width, height = resolve_output_size(854, 480, TargetProfile.FHD)
    assert (width, height) == (1920, 1080)


def test_resolve_output_size_uhd4k() -> None:
    assert resolve_output_size(854, 480, TargetProfile.UHD4K) == (3840, 2160)


def test_resolve_output_size_scale2x() -> None:
    w, h = resolve_output_size(640, 360, TargetProfile.SCALE2X)
    assert w == 1280
    assert h == 720


def test_resolve_output_size_scale4x() -> None:
    width, height = resolve_output_size(854, 480, TargetProfile.SCALE4X)
    assert (width, height) == (3416, 1920)


def test_resolve_output_size_source_even() -> None:
    w, h = resolve_output_size(1920, 1080, TargetProfile.SOURCE)
    assert (w, h) == (1920, 1080)


def test_resolve_output_size_source_odd_dimensions() -> None:
    """Odd source dimensions must be rounded up to even numbers."""
    w, h = resolve_output_size(1279, 719, TargetProfile.SOURCE)
    assert w % 2 == 0
    assert h % 2 == 0


def test_resolve_output_size_scale2x_result_is_even() -> None:
    w, h = resolve_output_size(641, 361, TargetProfile.SCALE2X)
    assert w % 2 == 0
    assert h % 2 == 0


# ---------------------------------------------------------------------------
# _fit_to_height
# ---------------------------------------------------------------------------

def test_fit_to_height_simple() -> None:
    w, h = _fit_to_height(1280, 720, 1080)
    assert h == 1080
    assert w % 2 == 0


def test_fit_to_height_same_size() -> None:
    w, h = _fit_to_height(1920, 1080, 1080)
    assert h == 1080
    assert w == 1920


# ---------------------------------------------------------------------------
# _build_scale_filter
# ---------------------------------------------------------------------------

def test_build_scale_filter_fhd_contains_pad() -> None:
    f = _build_scale_filter(TargetProfile.FHD, 1920, 1080)
    assert "pad=" in f
    assert "1920" in f
    assert "1080" in f


def test_build_scale_filter_uhd4k_contains_pad() -> None:
    f = _build_scale_filter(TargetProfile.UHD4K, 3840, 2160)
    assert "pad=" in f


def test_build_scale_filter_scale4x_no_pad() -> None:
    f = _build_scale_filter(TargetProfile.SCALE4X, 3840, 2160)
    assert "pad=" not in f
    assert "lanczos" in f


def test_build_scale_filter_source_no_pad() -> None:
    f = _build_scale_filter(TargetProfile.SOURCE, 1280, 720)
    assert "pad=" not in f


# ---------------------------------------------------------------------------
# build_baseline_command
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> EnhancementConfig:
    defaults = dict(input_path=Path("in.mp4"), output_path=Path("out.mp4"))
    defaults.update(kwargs)
    return EnhancementConfig(**defaults)


def test_build_baseline_command_includes_input_output() -> None:
    cfg = _make_config()
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "in.mp4" in " ".join(cmd)
    assert "out.mp4" in cmd[-1]


def test_build_baseline_command_preserve_audio() -> None:
    cfg = _make_config(preserve_audio=True)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-c:a" in cmd
    audio_idx = cmd.index("-c:a")
    assert cmd[audio_idx + 1] == "copy"


def test_build_baseline_command_no_audio() -> None:
    cfg = _make_config(preserve_audio=False)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-an" in cmd


def test_build_baseline_command_preserve_subtitles() -> None:
    cfg = _make_config(preserve_subtitles=True)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-c:s" in cmd
    sub_idx = cmd.index("-c:s")
    assert cmd[sub_idx + 1] == "copy"


def test_build_baseline_command_no_subtitles() -> None:
    cfg = _make_config(preserve_subtitles=False)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-sn" in cmd


def test_build_baseline_command_preserve_metadata() -> None:
    cfg = _make_config(preserve_metadata=True)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-map_metadata" in cmd
    idx = cmd.index("-map_metadata")
    assert cmd[idx + 1] == "0"


def test_build_baseline_command_no_metadata() -> None:
    cfg = _make_config(preserve_metadata=False)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-map_metadata" in cmd
    idx = cmd.index("-map_metadata")
    assert cmd[idx + 1] == "-1"


def test_build_baseline_command_preview_seconds() -> None:
    cfg = _make_config(preview_seconds=10)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-t" in cmd
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1] == "10"


def test_build_baseline_command_video_bitrate() -> None:
    cfg = _make_config(video_bitrate="8M")
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-b:v" in cmd
    idx = cmd.index("-b:v")
    assert cmd[idx + 1] == "8M"


def test_build_baseline_command_no_bitrate_uses_cq() -> None:
    cfg = _make_config(video_bitrate=None)
    cmd = build_baseline_command(cfg, 1920, 1080)
    assert "-cq" in cmd


def test_build_baseline_command_denoise_filter() -> None:
    cfg = _make_config(denoise_strength=0.5)
    cmd = build_baseline_command(cfg, 1920, 1080)
    vf_idx = cmd.index("-vf")
    vf_str = cmd[vf_idx + 1]
    assert "hqdn3d" in vf_str


def test_build_baseline_command_no_denoise_at_zero() -> None:
    cfg = _make_config(denoise_strength=0.0)
    cmd = build_baseline_command(cfg, 1920, 1080)
    vf_idx = cmd.index("-vf")
    vf_str = cmd[vf_idx + 1]
    assert "hqdn3d" not in vf_str


def test_build_baseline_command_sharpen_filter() -> None:
    cfg = _make_config(sharpen_strength=0.5)
    cmd = build_baseline_command(cfg, 1920, 1080)
    vf_idx = cmd.index("-vf")
    vf_str = cmd[vf_idx + 1]
    assert "unsharp" in vf_str


def test_build_baseline_command_no_sharpen_at_zero() -> None:
    cfg = _make_config(sharpen_strength=0.0)
    cmd = build_baseline_command(cfg, 1920, 1080)
    vf_idx = cmd.index("-vf")
    vf_str = cmd[vf_idx + 1]
    assert "unsharp" not in vf_str
