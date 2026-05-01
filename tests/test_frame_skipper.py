"""Tests for FrameSkipper in clearvid.app.models.realesrgan_runner."""
from __future__ import annotations

import numpy as np
import pytest

from clearvid.app.models.realesrgan_runner import FrameSkipper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(value: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Constructor / active property
# ---------------------------------------------------------------------------

def test_frame_skipper_disabled_by_default() -> None:
    fs = FrameSkipper(threshold=0.0)
    assert fs.active is False


def test_frame_skipper_enabled_with_threshold() -> None:
    fs = FrameSkipper(threshold=2.0)
    assert fs.active is True


def test_frame_skipper_negative_threshold_clamped_to_zero() -> None:
    fs = FrameSkipper(threshold=-5.0)
    assert fs._threshold == 0.0
    assert fs.active is False


def test_initial_skip_count_is_zero() -> None:
    fs = FrameSkipper(threshold=3.0)
    assert fs.skip_count == 0


# ---------------------------------------------------------------------------
# should_skip — inactive (threshold=0)
# ---------------------------------------------------------------------------

def test_should_skip_returns_false_when_inactive() -> None:
    fs = FrameSkipper(threshold=0.0)
    fs.record(_frame(100), _frame(200))
    assert fs.should_skip(_frame(100)) is False


def test_should_skip_returns_false_without_prior_record() -> None:
    fs = FrameSkipper(threshold=5.0)
    assert fs.should_skip(_frame(100)) is False


# ---------------------------------------------------------------------------
# should_skip — active
# ---------------------------------------------------------------------------

def test_should_skip_identical_frame() -> None:
    fs = FrameSkipper(threshold=5.0)
    raw = _frame(100)
    enhanced = _frame(200)
    fs.record(raw, enhanced)
    # Same frame → mean diff = 0 < threshold → skip
    assert fs.should_skip(raw.copy()) is True


def test_should_skip_very_different_frame() -> None:
    fs = FrameSkipper(threshold=5.0)
    raw = _frame(0)
    enhanced = _frame(200)
    fs.record(raw, enhanced)
    # Very different frame (mean diff = 200) → no skip
    assert fs.should_skip(_frame(200)) is False


def test_should_skip_borderline_below_threshold() -> None:
    """Mean diff just below threshold → skip."""
    fs = FrameSkipper(threshold=10.0)
    raw = _frame(100)
    fs.record(raw, _frame(200))
    # Add 5 to each pixel → mean diff = 5 < 10 → skip
    close_frame = _frame(105)
    assert fs.should_skip(close_frame) is True


def test_should_skip_borderline_above_threshold() -> None:
    """Mean diff just above threshold → no skip."""
    fs = FrameSkipper(threshold=5.0)
    raw = _frame(100)
    fs.record(raw, _frame(200))
    # Add 10 to each pixel → mean diff = 10 > 5 → no skip
    different_frame = _frame(110)
    assert fs.should_skip(different_frame) is False


# ---------------------------------------------------------------------------
# record / get_cached
# ---------------------------------------------------------------------------

def test_get_cached_after_record() -> None:
    fs = FrameSkipper(threshold=3.0)
    enhanced = _frame(255)
    fs.record(_frame(100), enhanced)
    cached = fs.get_cached()
    assert np.array_equal(cached, enhanced)


def test_get_cached_returns_copy() -> None:
    """Mutation of cached result should not affect internal state."""
    fs = FrameSkipper(threshold=3.0)
    enhanced = _frame(200)
    fs.record(_frame(100), enhanced)
    cached = fs.get_cached()
    cached[:] = 0
    assert not np.array_equal(fs.get_cached(), cached)


def test_record_updates_state() -> None:
    fs = FrameSkipper(threshold=3.0)
    fs.record(_frame(50), _frame(100))
    fs.record(_frame(200), _frame(255))
    cached = fs.get_cached()
    assert np.array_equal(cached, _frame(255))
