from __future__ import annotations

import io
import subprocess

from clearvid.app.utils import subprocess_utils


class _FakeProgressProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO("out_time_ms=500000\n")
        self.returncode = 0

    def poll(self):
        return self.returncode if self.stdout.tell() == len(self.stdout.getvalue()) else None

    def wait(self):
        return self.returncode


def test_run_ffmpeg_with_progress_merges_stderr_to_stdout(monkeypatch) -> None:
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return _FakeProgressProcess()

    monkeypatch.setattr(subprocess_utils.subprocess, "Popen", fake_popen)

    subprocess_utils.run_ffmpeg_with_progress(["ffmpeg", "-i", "in", "out.mp4"])

    assert captured["stdout"] == subprocess.PIPE
    assert captured["stderr"] == subprocess.STDOUT
