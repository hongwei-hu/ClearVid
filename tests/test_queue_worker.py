from __future__ import annotations

import threading
import time
from pathlib import Path

from clearvid.app.export_control import ExportCancelled
from clearvid.app.gui.queue_worker import ExportJob, JobStatus, QueueWorker
from clearvid.app.schemas.models import EnhancementConfig


def _config(name: str) -> EnhancementConfig:
    return EnhancementConfig(
        input_path=Path(f"{name}.mp4"),
        output_path=Path(f"{name}_out.mp4"),
    )


def test_queue_cancel_interrupts_current_job(monkeypatch) -> None:
    control_ready = threading.Event()
    controls = []

    class _Orchestrator:
        def run_single(self, config, progress_callback=None, control=None):
            controls.append(control)
            control_ready.set()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if control is not None and control.is_cancelled:
                    raise ExportCancelled("cancelled")
                time.sleep(0.01)
            raise AssertionError("queue cancellation did not reach current job")

    monkeypatch.setattr("clearvid.app.gui.queue_worker.Orchestrator", _Orchestrator)

    jobs = [ExportJob(id=0, config=_config("a")), ExportJob(id=1, config=_config("b"))]
    worker = QueueWorker(jobs)
    thread = threading.Thread(target=worker.run)
    thread.start()

    assert control_ready.wait(timeout=1)
    worker.cancel()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert controls and controls[0] is not None
    assert jobs[0].status == JobStatus.CANCELLED
    assert jobs[1].status == JobStatus.CANCELLED
