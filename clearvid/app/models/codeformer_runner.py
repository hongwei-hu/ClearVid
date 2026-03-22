from __future__ import annotations

from pathlib import Path


def validate_codeformer_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return False, f"CodeFormer dependencies are not installed: {exc}"

    if weights_path and not weights_path.exists():
        return False, f"CodeFormer weights not found: {weights_path}"

    return True, "CodeFormer environment looks available"
