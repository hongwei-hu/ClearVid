"""Post-processing enhancements applied after super-resolution and before encoding.

Provides:
- **Adaptive sharpening**: Unsharp Mask with configurable strength, applied only
  when ``sharpen_strength > 0``.
- **Color correction**: Per-channel histogram matching that aligns the enhanced
  frame's color distribution to the source frame, correcting any model-induced
  color drift.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_sharpening(frame: np.ndarray, strength: float) -> np.ndarray:
    """Apply Unsharp Mask sharpening.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 image.
    strength : float
        Sharpening intensity in [0, 1].  0 = no-op, 0.12 = subtle,
        0.3 = moderate, 0.6+ = aggressive.
    """
    if strength <= 0:
        return frame

    # Gaussian kernel radius scales with strength; sigma kept moderate to avoid
    # amplifying noise.  amount maps strength to the unsharp mask weight.
    sigma = 1.0 + strength * 2.0        # 1.0 – 3.0
    amount = 0.3 + strength * 1.2       # 0.3 – 1.5
    ksize = 0  # auto from sigma

    blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def apply_color_correction(
    enhanced: np.ndarray,
    source: np.ndarray,
) -> np.ndarray:
    """Match the color histogram of *enhanced* to *source* per channel.

    Uses cumulative-distribution-function (CDF) matching in YCrCb space so that
    luminance and chrominance are independently corrected.  This prevents the
    super-resolution model from shifting overall color or brightness.

    Both inputs must be BGR uint8 arrays.  *source* is typically the original
    (pre-upscale) frame resized to *enhanced*'s resolution.
    """
    if source.shape[:2] != enhanced.shape[:2]:
        source = cv2.resize(source, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA)

    src_ycrcb = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    enh_ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)

    matched = np.empty_like(enh_ycrcb)
    for ch in range(3):
        matched[:, :, ch] = _match_histogram_channel(
            enh_ycrcb[:, :, ch], src_ycrcb[:, :, ch]
        )

    return cv2.cvtColor(matched, cv2.COLOR_YCrCb2BGR)


def _match_histogram_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map *source* channel's histogram to match *reference* using CDF lookup."""
    src_hist = cv2.calcHist([source], [0], None, [256], [0, 256]).ravel()
    ref_hist = cv2.calcHist([reference], [0], None, [256], [0, 256]).ravel()

    src_cdf = src_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    # Normalize to [0, 255]
    src_cdf = (src_cdf / src_cdf[-1] * 255).astype(np.uint8) if src_cdf[-1] > 0 else np.arange(256, dtype=np.uint8)
    ref_cdf = (ref_cdf / ref_cdf[-1] * 255).astype(np.uint8) if ref_cdf[-1] > 0 else np.arange(256, dtype=np.uint8)

    # Build lookup: for each src CDF value, find the closest ref CDF value
    lookup = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
            ref_idx += 1
        lookup[src_val] = ref_idx

    return lookup[source]
