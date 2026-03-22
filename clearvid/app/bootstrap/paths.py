"""Centralized path resolution for ClearVid.

All runtime paths (weights, ffmpeg, outputs, configs) are resolved relative to
the *application root* — not ``Path.cwd()``.  This makes ClearVid work correctly
whether run from source, from a venv, or from a portable / PyInstaller build.

Other modules should import ``APP_ROOT``, ``WEIGHTS_DIR``, ``ffmpeg_path()`` etc.
from this module instead of building paths by hand.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Application root — the directory that contains ``clearvid/``, ``weights/``,
# ``ffmpeg.exe``, etc.
#
# Resolution order:
#   1. CLEARVID_ROOT environment variable (explicit override)
#   2. PyInstaller frozen bundle (sys._MEIPASS → one-dir layout parent)
#   3. Walk up from this file until we find pyproject.toml or weights/
# ---------------------------------------------------------------------------

def _find_app_root() -> Path:
    # 1. Env var override
    env = os.environ.get("CLEARVID_ROOT")
    if env:
        return Path(env).resolve()

    # 2. PyInstaller one-dir bundle
    if getattr(sys, "frozen", False):
        # In one-dir mode _MEIPASS == exe directory
        return Path(sys.executable).resolve().parent

    # 3. Walk up from this file: bootstrap/ → app/ → clearvid/ → <root>
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "pyproject.toml").exists() or (candidate / "weights").exists():
        return candidate

    # Fallback to cwd
    return Path.cwd().resolve()


APP_ROOT: Path = _find_app_root()

# ---------------------------------------------------------------------------
# Derived directories
# ---------------------------------------------------------------------------

WEIGHTS_DIR: Path = APP_ROOT / "weights"
OUTPUTS_DIR: Path = APP_ROOT / "outputs"
LIB_DIR: Path = APP_ROOT / "lib"
TRT_CACHE_DIR: Path = WEIGHTS_DIR / "trt_cache"

# Per-model weight directories
REALESRGAN_WEIGHTS_DIR: Path = WEIGHTS_DIR / "realesrgan"
CODEFORMER_WEIGHTS_DIR: Path = WEIGHTS_DIR / "codeformer"
GFPGAN_WEIGHTS_DIR: Path = WEIGHTS_DIR / "gfpgan"
FACELIB_WEIGHTS_DIR: Path = WEIGHTS_DIR / "facelib"

# ---------------------------------------------------------------------------
# FFmpeg / FFprobe binary resolution
# ---------------------------------------------------------------------------

def _find_binary(name: str) -> str | None:
    """Find *name* (.exe on Windows).  Check APP_ROOT first, then PATH."""
    suffix = ".exe" if sys.platform == "win32" else ""
    local = APP_ROOT / f"{name}{suffix}"
    if local.is_file():
        return str(local)
    return shutil.which(name)


def ffmpeg_path() -> str | None:
    """Return the absolute path to ``ffmpeg``, or *None*."""
    return _find_binary("ffmpeg")


def ffprobe_path() -> str | None:
    """Return the absolute path to ``ffprobe``, or *None*."""
    return _find_binary("ffprobe")


# ---------------------------------------------------------------------------
# Version stamp for installed dependencies
# ---------------------------------------------------------------------------

VERSION_FILE: Path = LIB_DIR / ".clearvid_version"


def installed_lib_version() -> str | None:
    """Return the version string from the lib stamp file, or *None*."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text(encoding="utf-8").strip()
    return None


def write_lib_version(version: str) -> None:
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(version, encoding="utf-8")
