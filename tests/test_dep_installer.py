"""Tests for dependency installer subprocess handling."""
from __future__ import annotations

from pathlib import Path

from clearvid.app.bootstrap import dep_installer
from clearvid.app.bootstrap.dep_installer import InstallPlan


class _FakeStdout:
    def __iter__(self):
        yield "first line\n"


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = _FakeStdout()
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.returncode is None else self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_install_all_terminates_process_when_output_callback_fails(monkeypatch, tmp_path: Path) -> None:
    fake_proc = _FakeProcess()
    monkeypatch.setattr(
        dep_installer,
        "build_install_steps",
        lambda plan: [("fake step", ["python", "-m", "pip", "install", "demo"])],
    )
    monkeypatch.setattr(dep_installer.subprocess, "Popen", lambda *args, **kwargs: fake_proc)

    def failing_output(_line: str) -> None:
        raise RuntimeError("callback failed")

    ok = dep_installer.run_install(InstallPlan(target_dir=tmp_path), on_output=failing_output)

    assert ok is False
    assert fake_proc.terminated is True
