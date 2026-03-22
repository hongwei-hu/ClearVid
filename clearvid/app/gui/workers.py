"""Background worker threads for video processing and preview generation."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import EnhancementConfig


class Worker(QThread):
    """Background thread for single-file video enhancement."""

    completed = Signal(str)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, config: EnhancementConfig) -> None:
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            result = Orchestrator().run_single(
                self._config, progress_callback=self._emit_progress
            )
            self.completed.emit(result.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _emit_progress(self, percent: int, message: str) -> None:
        self.progress.emit(percent, message)


class PreviewWorker(QThread):
    """Background thread for single-frame preview generation."""

    finished = Signal(object, object)  # (original_bgr, enhanced_bgr)
    failed = Signal(str)

    def __init__(self, config: EnhancementConfig, timestamp_sec: float) -> None:
        super().__init__()
        self._config = config
        self._timestamp_sec = timestamp_sec

    def run(self) -> None:
        try:
            original, enhanced, _ = Orchestrator().preview_frame(
                self._config, self._timestamp_sec
            )
            self.finished.emit(original, enhanced)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
