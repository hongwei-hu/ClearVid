from __future__ import annotations

from pathlib import Path


def validate_realesrgan_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    try:
        import realesrgan  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return False, f"Real-ESRGAN dependencies are not installed: {exc}"

    if weights_path and not weights_path.exists():
        return False, f"Real-ESRGAN weights not found: {weights_path}"

    return True, "Real-ESRGAN environment looks available"
