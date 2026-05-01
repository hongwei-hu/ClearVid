"""Tests for clearvid.app.export_control."""
from __future__ import annotations

import threading
import time

import pytest

from clearvid.app.export_control import ExportCancelled, ExportControl


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_not_cancelled() -> None:
    ctrl = ExportControl()
    assert ctrl.is_cancelled is False


def test_initial_not_paused() -> None:
    ctrl = ExportControl()
    assert ctrl.is_paused is False


def test_initial_check_does_not_raise() -> None:
    ctrl = ExportControl()
    ctrl.check()  # should not raise


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------

def test_cancel_sets_cancelled() -> None:
    ctrl = ExportControl()
    ctrl.cancel()
    assert ctrl.is_cancelled is True


def test_cancel_check_raises() -> None:
    ctrl = ExportControl()
    ctrl.cancel()
    with pytest.raises(ExportCancelled):
        ctrl.check()


def test_cancel_twice_still_raises() -> None:
    ctrl = ExportControl()
    ctrl.cancel()
    ctrl.cancel()
    with pytest.raises(ExportCancelled):
        ctrl.check()


def test_export_cancelled_is_exception() -> None:
    assert issubclass(ExportCancelled, Exception)


# ---------------------------------------------------------------------------
# pause() / resume()
# ---------------------------------------------------------------------------

def test_pause_sets_paused() -> None:
    ctrl = ExportControl()
    ctrl.pause()
    assert ctrl.is_paused is True


def test_resume_clears_paused() -> None:
    ctrl = ExportControl()
    ctrl.pause()
    ctrl.resume()
    assert ctrl.is_paused is False


def test_resume_after_pause_check_does_not_raise() -> None:
    ctrl = ExportControl()
    ctrl.pause()
    ctrl.resume()
    ctrl.check()  # should not raise


def test_cancel_while_paused_unblocks() -> None:
    """cancel() must unblock a thread waiting in check() due to pause."""
    ctrl = ExportControl()
    ctrl.pause()

    raised: list[bool] = []

    def worker() -> None:
        try:
            ctrl.check()
            raised.append(False)
        except ExportCancelled:
            raised.append(True)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.05)  # give worker time to block
    ctrl.cancel()
    t.join(timeout=2.0)
    assert not t.is_alive(), "Worker thread did not unblock within timeout"
    assert raised == [True]


def test_pause_resume_multiple_times() -> None:
    ctrl = ExportControl()
    for _ in range(5):
        ctrl.pause()
        assert ctrl.is_paused is True
        ctrl.resume()
        assert ctrl.is_paused is False
