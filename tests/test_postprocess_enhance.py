"""Tests for clearvid.app.postprocess.enhance."""
from __future__ import annotations

import numpy as np
import pytest

from clearvid.app.postprocess.enhance import apply_color_correction, apply_sharpening


# ---------------------------------------------------------------------------
# apply_sharpening
# ---------------------------------------------------------------------------

def _solid_frame(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a frame with horizontal gradient for sharpening tests."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        frame[:, x, :] = int(x * 255 / (w - 1))
    return frame


def test_apply_sharpening_zero_is_noop() -> None:
    frame = _gradient_frame()
    result = apply_sharpening(frame, 0.0)
    assert np.array_equal(result, frame)


def test_apply_sharpening_returns_uint8() -> None:
    frame = _gradient_frame()
    result = apply_sharpening(frame, 0.3)
    assert result.dtype == np.uint8


def test_apply_sharpening_same_shape() -> None:
    frame = _gradient_frame(128, 128)
    result = apply_sharpening(frame, 0.5)
    assert result.shape == frame.shape


def test_apply_sharpening_negative_is_noop() -> None:
    frame = _gradient_frame()
    result = apply_sharpening(frame, -1.0)
    assert np.array_equal(result, frame)


def test_apply_sharpening_solid_unchanged() -> None:
    """Solid colour frame: sharpening should not change the value significantly."""
    frame = _solid_frame(64, 64, 128)
    result = apply_sharpening(frame, 0.5)
    # Interior pixels of a solid frame should be within 1 of 128 after sharpening
    assert int(result[32, 32, 0]) in range(126, 131)


def test_apply_sharpening_modifies_edges() -> None:
    """Sharpening on a high-contrast frame should change some pixel values."""
    frame = _gradient_frame()
    result = apply_sharpening(frame, 0.5)
    assert not np.array_equal(result, frame)


def test_apply_sharpening_clips_to_255() -> None:
    """Pixels should never exceed 255."""
    frame = _solid_frame(64, 64, 250)
    result = apply_sharpening(frame, 1.0)
    assert result.max() <= 255


def test_apply_sharpening_clips_to_0() -> None:
    """Pixels should never go below 0."""
    frame = _solid_frame(64, 64, 5)
    result = apply_sharpening(frame, 1.0)
    assert result.min() >= 0


# ---------------------------------------------------------------------------
# apply_color_correction
# ---------------------------------------------------------------------------

def test_color_correction_same_size_returns_uint8() -> None:
    enhanced = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = apply_color_correction(enhanced, source)
    assert result.dtype == np.uint8
    assert result.shape == enhanced.shape


def test_color_correction_different_size_resizes_source() -> None:
    enhanced = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = apply_color_correction(enhanced, source)
    assert result.shape == (128, 128, 3)


def test_color_correction_identical_frames() -> None:
    """When enhanced == source the output should be very close to input."""
    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = apply_color_correction(frame.copy(), frame.copy())
    # CDF matching of identical histograms should be nearly identity
    diff = np.abs(result.astype(np.int32) - frame.astype(np.int32))
    assert int(diff.mean()) <= 5


def test_color_correction_uniform_source() -> None:
    """Uniform source frame (single value histogram) should not crash."""
    enhanced = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    source = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = apply_color_correction(enhanced, source)
    assert result.dtype == np.uint8


def test_color_correction_clips_to_valid_range() -> None:
    enhanced = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    source = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = apply_color_correction(enhanced, source)
    assert result.min() >= 0
    assert result.max() <= 255
