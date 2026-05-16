"""Export queue worker: runs multiple enhancement jobs sequentially with cancel support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from clearvid.app.export_control import ExportCancelled, ExportControl
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import EnhancementConfig


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportJob:
    """Single export job in the queue."""

    id: int
    config: EnhancementConfig
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    message: str = ""
    elapsed_sec: float = 0.0
    output_size_bytes: int = 0

    @property
    def display_name(self) -> str:
        return Path(self.config.input_path).name


class QueueWorker(QThread):
    """Process a list of ExportJobs sequentially. Supports cancel."""

    # Signals
    job_started = Signal(int)           # job id
    job_progress = Signal(int, int, str)  # job id, percent, message
    job_completed = Signal(int, str)    # job id, result_json
    job_failed = Signal(int, str)       # job id, error
    queue_finished = Signal()           # all done

    def __init__(self, jobs: list[ExportJob]) -> None:
        super().__init__()
        self._jobs = jobs
        self._cancel_requested = False
        self._current_job_id: int | None = None
        self._current_control: ExportControl | None = None

    def cancel(self) -> None:
        """Request cancellation. Current job is interrupted and remaining jobs are skipped."""
        self._cancel_requested = True
        if self._current_control is not None:
            self._current_control.cancel()

    @property
    def current_job_id(self) -> int | None:
        return self._current_job_id

    def run(self) -> None:
        orch = Orchestrator()
        for job in self._jobs:
            if self._cancel_requested:
                job.status = JobStatus.CANCELLED
                continue

            job.status = JobStatus.RUNNING
            self._current_job_id = job.id
            self._current_control = ExportControl()
            self.job_started.emit(job.id)

            start = time.monotonic()
            try:
                result = orch.run_single(
                    job.config,
                    progress_callback=lambda pct, msg, _id=job.id: self._on_progress(_id, pct, msg),
                    control=self._current_control,
                )
                job.elapsed_sec = time.monotonic() - start
                out_path = Path(job.config.output_path)
                job.output_size_bytes = out_path.stat().st_size if out_path.exists() else 0
                job.status = JobStatus.COMPLETED
                job.progress = 100
                self.job_completed.emit(job.id, result.model_dump_json(indent=2))
            except ExportCancelled as exc:
                job.elapsed_sec = time.monotonic() - start
                job.status = JobStatus.CANCELLED
                job.message = str(exc)
                self._cancel_requested = True
            except Exception as exc:  # noqa: BLE001
                job.elapsed_sec = time.monotonic() - start
                job.status = JobStatus.FAILED
                job.message = str(exc)
                self.job_failed.emit(job.id, str(exc))
            finally:
                self._current_control = None

        self._current_job_id = None
        self.queue_finished.emit()

    def _on_progress(self, job_id: int, pct: int, msg: str) -> None:
        self.job_progress.emit(job_id, pct, msg)
