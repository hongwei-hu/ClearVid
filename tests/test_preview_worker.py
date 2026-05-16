from __future__ import annotations

from pathlib import Path

from clearvid.app.bootstrap.weight_manager import WeightSpec
from clearvid.app.gui.workers import PreviewWorker, WeightDownloadWorker
from clearvid.app.schemas.models import EnhancementConfig


def test_preview_worker_cancel_before_run_emits_cancelled() -> None:
    worker = PreviewWorker(
        EnhancementConfig(input_path=Path("in.mp4"), output_path=Path("out.mp4")),
        timestamp_sec=0.0,
    )
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.cancel()
    worker.run()

    assert cancelled == [True]


def test_weight_download_worker_reports_completion(monkeypatch, tmp_path: Path) -> None:
    spec = WeightSpec(
        name="demo",
        filename="demo.pth",
        directory=tmp_path,
        url="https://example.invalid/demo.pth",
        size_mb=1,
    )

    def _download(spec, on_progress=None):
        if on_progress is not None:
            on_progress(50)
            on_progress(100)
        return True

    monkeypatch.setattr("clearvid.app.gui.workers.download_weight", _download)
    worker = WeightDownloadWorker([spec])
    completed: list[bool] = []
    progress: list[int] = []
    worker.completed.connect(lambda: completed.append(True))
    worker.progress.connect(lambda pct, _msg: progress.append(pct))

    worker.run()

    assert completed == [True]
    assert progress[-1] == 100
