"""Thread-safe export control for pause/resume/cancel.

A single ``ExportControl`` instance is shared between the GUI thread and
the worker thread.  The GUI calls ``pause()`` / ``resume()`` / ``cancel()``
and each stage of the pipeline calls ``check()`` every frame to respect
the signals.
"""

from __future__ import annotations

import threading


class ExportCancelled(Exception):
    """Raised inside the worker when the user cancels the export."""


class ExportControl:
    """Lightweight controller passed from GUI → Worker → Pipeline → Runner."""

    def __init__(self) -> None:
        self._cancelled = False
        self._paused = threading.Event()
        self._paused.set()  # starts in "not paused" state

    # -- GUI side ----------------------------------------------------------

    def pause(self) -> None:
        """Request the pipeline to pause at the next frame boundary."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume a paused pipeline."""
        self._paused.set()

    def cancel(self) -> None:
        """Request the pipeline to stop. Also resumes if paused."""
        self._cancelled = True
        self._paused.set()  # unblock so the cancel is noticed immediately

    # -- Worker / pipeline side --------------------------------------------

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_paused(self) -> bool:
        return not self._paused.is_set()

    def check(self) -> None:
        """Call once per frame. Blocks while paused; raises on cancel."""
        self._paused.wait()  # block if paused
        if self._cancelled:
            raise ExportCancelled("导出已被用户取消")
