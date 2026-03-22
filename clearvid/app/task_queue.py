from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}


def discover_video_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
