"""Tests for Real-ESRGAN stream process startup."""
from __future__ import annotations

import pytest

from clearvid.app.models import realesrgan_runner


class _FakeDecoder:
    stdout = object()
    stderr = object()

    def __init__(self) -> None:
        self.killed = False
        self.waited = False

    def poll(self):
        return None

    def kill(self) -> None:
        self.killed = True

    def wait(self):
        self.waited = True


def test_start_stream_processes_cleans_decoder_when_encoder_start_fails(monkeypatch) -> None:
    decoder = _FakeDecoder()
    calls = {"count": 0}

    def fake_popen(command, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return decoder
        raise FileNotFoundError("ffmpeg missing")

    monkeypatch.setattr(realesrgan_runner.subprocess, "Popen", fake_popen)

    with pytest.raises(RuntimeError, match="无法启动"):
        realesrgan_runner._start_stream_processes(["ffmpeg", "-i", "in"], ["ffmpeg", "out"])

    assert decoder.killed is True
    assert decoder.waited is True