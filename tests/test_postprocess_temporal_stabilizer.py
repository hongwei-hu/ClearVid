"""Tests for clearvid.app.postprocess.temporal_stabilizer."""
from __future__ import annotations

import numpy as np
import pytest

from clearvid.app.postprocess.temporal_stabilizer import TemporalStabilizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _noise(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Constructor / defaults
# ---------------------------------------------------------------------------

def test_default_strength_clipped() -> None:
    ts = TemporalStabilizer(strength=2.0)
    assert ts._strength <= 1.0


def test_negative_strength_clipped_to_zero() -> None:
    ts = TemporalStabilizer(strength=-0.5)
    assert ts._strength == 0.0


# ---------------------------------------------------------------------------
# First frame — should be returned unchanged
# ---------------------------------------------------------------------------

def test_first_frame_returned_unchanged() -> None:
    ts = TemporalStabilizer(strength=0.6)
    frame = _solid(64, 64, 100)
    result = ts.stabilize(frame)
    assert np.array_equal(result, frame)


def test_first_frame_result_is_uint8() -> None:
    ts = TemporalStabilizer()
    frame = _noise()
    result = ts.stabilize(frame)
    assert result.dtype == np.uint8


def test_first_frame_shape_preserved() -> None:
    ts = TemporalStabilizer()
    frame = _noise(128, 128)
    result = ts.stabilize(frame)
    assert result.shape == (128, 128, 3)


# ---------------------------------------------------------------------------
# Subsequent frames
# ---------------------------------------------------------------------------

def test_second_frame_has_same_shape() -> None:
    ts = TemporalStabilizer(strength=0.6)
    ts.stabilize(_solid(64, 64, 50))
    result = ts.stabilize(_solid(64, 64, 60))
    assert result.shape == (64, 64, 3)


def test_second_frame_is_uint8() -> None:
    ts = TemporalStabilizer(strength=0.6)
    ts.stabilize(_noise())
    result = ts.stabilize(_noise(seed=1))
    assert result.dtype == np.uint8


def test_identical_frames_second_close_to_first() -> None:
    """Two identical static frames: blending should yield the same value."""
    ts = TemporalStabilizer(strength=0.9)
    frame = _solid(64, 64, 128)
    ts.stabilize(frame)
    result = ts.stabilize(frame.copy())
    # Result should be very close to 128 (no motion, no scene change)
    assert abs(int(result.mean()) - 128) <= 2


def test_scene_change_resets_buffer() -> None:
    """Very high motion → scene_threshold exceeded → return current frame unchanged."""
    ts = TemporalStabilizer(strength=0.8, scene_threshold=0.1)
    ts.stabilize(_solid(64, 64, 0))
    # Completely different frame → large flow magnitude
    result = ts.stabilize(_solid(64, 64, 255))
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_clears_state() -> None:
    ts = TemporalStabilizer()
    ts.stabilize(_noise())
    assert ts._prev_frame is not None
    ts.reset()
    assert ts._prev_frame is None
    assert ts._prev_gray_small is None


def test_after_reset_next_frame_returned_unchanged() -> None:
    ts = TemporalStabilizer(strength=0.8)
    ts.stabilize(_noise())
    ts.stabilize(_noise(seed=1))
    ts.reset()
    frame = _solid(64, 64, 200)
    result = ts.stabilize(frame)
    assert np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# _blend helper (static method)
# ---------------------------------------------------------------------------

def test_blend_zero_weight_returns_current() -> None:
    current = _solid(64, 64, 100)
    warped = _solid(64, 64, 200)
    weight = np.zeros((64, 64), dtype=np.float32)
    result = TemporalStabilizer._blend(current, warped, weight)
    assert np.allclose(result.astype(np.float32), current.astype(np.float32), atol=1)


def test_blend_full_weight_returns_warped() -> None:
    current = _solid(64, 64, 100)
    warped = _solid(64, 64, 200)
    weight = np.ones((64, 64), dtype=np.float32)
    result = TemporalStabilizer._blend(current, warped, weight)
    assert np.allclose(result.astype(np.float32), warped.astype(np.float32), atol=1)


def test_blend_result_clipped() -> None:
    current = _solid(64, 64, 255)
    warped = _solid(64, 64, 255)
    weight = np.full((64, 64), 0.5, dtype=np.float32)
    result = TemporalStabilizer._blend(current, warped, weight)
    assert result.max() <= 255
    assert result.min() >= 0
