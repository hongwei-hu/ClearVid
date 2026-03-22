"""GFPGAN face restoration runner — drop-in alternative to CodeFormer.

Uses the ``gfpgan`` package (already in project dependencies) with the same
face-detection + alignment pipeline from ``facexlib``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

GFPGAN_MODEL_FILENAME = "GFPGANv1.4.pth"
GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


class GFPGANRestorer:
    """GFPGAN-based face restorer sharing the facexlib detection pipeline."""

    def __init__(
        self,
        upscale_factor: float,
        weights_root: Path,
    ) -> None:
        gfpgan_module = importlib.import_module("gfpgan")
        weight_path = ensure_gfpgan_weights(weights_root / "gfpgan")

        self._restorer = gfpgan_module.GFPGANer(
            model_path=str(weight_path),
            upscale=upscale_factor,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cuda",
        )
        # Override facexlib model root so detection weights are found
        self._restorer.face_helper.model_rootpath = str(weights_root / "facelib")
        logger.info("GFPGAN restorer initialized (upscale=%.1f)", upscale_factor)

    def restore_faces(self, frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Detect, align, restore and paste-back faces in *frame*.

        Returns the frame with all detected faces restored.  If no faces are
        found, returns the original frame unchanged.
        """
        _, _, restored = self._restorer.enhance(
            frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )
        return restored if restored is not None else frame


def validate_gfpgan_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    try:
        importlib.import_module("gfpgan")
    except ImportError as exc:
        return False, f"GFPGAN package not installed: {exc}"

    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch not installed: {exc}"

    if not torch.cuda.is_available():
        return False, "GFPGAN requires CUDA."

    try:
        ensure_gfpgan_weights(weights_path or Path.cwd() / "weights" / "gfpgan")
    except Exception as exc:  # noqa: BLE001
        return False, f"GFPGAN weights not available: {exc}"

    return True, "GFPGAN environment available"


def ensure_gfpgan_weights(weights_dir: Path) -> Path:
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_file = weights_dir / GFPGAN_MODEL_FILENAME
    if weight_file.exists():
        return weight_file

    download_util = importlib.import_module("basicsr.utils.download_util")
    downloaded = download_util.load_file_from_url(
        url=GFPGAN_MODEL_URL,
        model_dir=str(weights_dir),
        progress=True,
        file_name=GFPGAN_MODEL_FILENAME,
    )
    resolved = Path(downloaded)
    if not resolved.exists():
        raise RuntimeError(f"GFPGAN weight download failed: {resolved}")
    return resolved
