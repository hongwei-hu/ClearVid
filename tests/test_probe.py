"""Tests for clearvid.app.io.probe."""
from __future__ import annotations

import subprocess

import pytest

from clearvid.app.io import probe
from clearvid.app.schemas.models import BackendType


def test_run_text_uses_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    assert probe._run_text(["ffprobe", "input.mp4"], timeout=12.5) == "ok"
    assert captured["timeout"] == pytest.approx(12.5)


def test_run_text_timeout_returns_empty(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(probe.subprocess, "run", fake_run)

    assert probe._run_text(["ffprobe", "bad.mp4"], timeout=0.01) == ""


def test_collect_environment_info_uses_backend_enum(monkeypatch) -> None:
    monkeypatch.setattr(probe, "_ffmpeg_path", lambda: None)
    monkeypatch.setattr(probe, "_ffprobe_path", lambda: None)
    monkeypatch.setattr(probe, "_which", lambda _: None)
    monkeypatch.setattr(
        probe,
        "inspect_realesrgan_runtime",
        lambda _: (True, "ok", "2.6.0", True, True),
    )

    env = probe.collect_environment_info()

    assert env.preferred_backend == BackendType.REALESRGAN
