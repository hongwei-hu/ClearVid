"""Tests for clearvid.app.bootstrap.paths."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


from clearvid.app.bootstrap.paths import (
    APP_ROOT,
    CODEFORMER_WEIGHTS_DIR,
    FACELIB_WEIGHTS_DIR,
    GFPGAN_WEIGHTS_DIR,
    OUTPUTS_DIR,
    REALESRGAN_WEIGHTS_DIR,
    TRT_CACHE_DIR,
    WEIGHTS_DIR,
    ffmpeg_path,
    ffprobe_path,
    installed_lib_version,
)


# ---------------------------------------------------------------------------
# APP_ROOT
# ---------------------------------------------------------------------------

def test_app_root_is_absolute() -> None:
    assert APP_ROOT.is_absolute()


def test_app_root_contains_pyproject() -> None:
    """APP_ROOT should resolve to the project root which has pyproject.toml."""
    assert (APP_ROOT / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# Derived directory paths
# ---------------------------------------------------------------------------

def test_weights_dir_is_under_app_root() -> None:
    assert WEIGHTS_DIR.parent == APP_ROOT


def test_outputs_dir_is_under_app_root() -> None:
    assert OUTPUTS_DIR.parent == APP_ROOT


def test_trt_cache_dir_is_under_weights() -> None:
    assert TRT_CACHE_DIR.parent == WEIGHTS_DIR


def test_realesrgan_weights_dir_is_under_weights() -> None:
    assert REALESRGAN_WEIGHTS_DIR.parent == WEIGHTS_DIR


def test_codeformer_weights_dir_is_under_weights() -> None:
    assert CODEFORMER_WEIGHTS_DIR.parent == WEIGHTS_DIR


def test_gfpgan_weights_dir_is_under_weights() -> None:
    assert GFPGAN_WEIGHTS_DIR.parent == WEIGHTS_DIR


def test_facelib_weights_dir_is_under_weights() -> None:
    assert FACELIB_WEIGHTS_DIR.parent == WEIGHTS_DIR


# ---------------------------------------------------------------------------
# ffmpeg_path / ffprobe_path
# ---------------------------------------------------------------------------

def test_ffmpeg_path_returns_str_or_none() -> None:
    result = ffmpeg_path()
    assert result is None or isinstance(result, str)


def test_ffprobe_path_returns_str_or_none() -> None:
    result = ffprobe_path()
    assert result is None or isinstance(result, str)


def test_ffmpeg_path_env_override(tmp_path: Path) -> None:
    """CLEARVID_ROOT override must be respected at runtime via _find_binary."""
    from clearvid.app.bootstrap import paths as paths_module
    import sys

    fake_ffmpeg = tmp_path / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    fake_ffmpeg.write_bytes(b"")

    original_root = paths_module.APP_ROOT
    try:
        paths_module.APP_ROOT = tmp_path
        result = paths_module._find_binary("ffmpeg")
        assert result == str(fake_ffmpeg)
    finally:
        paths_module.APP_ROOT = original_root


# ---------------------------------------------------------------------------
# installed_lib_version
# ---------------------------------------------------------------------------

def test_installed_lib_version_returns_none_when_missing() -> None:
    from clearvid.app.bootstrap import paths as paths_module
    original = paths_module.VERSION_FILE
    try:
        paths_module.VERSION_FILE = Path("/nonexistent/path/.clearvid_version")
        result = installed_lib_version()
        assert result is None
    finally:
        paths_module.VERSION_FILE = original


def test_installed_lib_version_reads_file(tmp_path: Path) -> None:
    from clearvid.app.bootstrap import paths as paths_module
    version_file = tmp_path / ".clearvid_version"
    version_file.write_text("1.2.3\n", encoding="utf-8")
    original = paths_module.VERSION_FILE
    try:
        paths_module.VERSION_FILE = version_file
        result = installed_lib_version()
        assert result == "1.2.3"
    finally:
        paths_module.VERSION_FILE = original
