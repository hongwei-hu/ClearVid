"""Tests for Real-ESRGAN stream process startup."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from clearvid.app.models import realesrgan_runner
from clearvid.app.models.tensorrt_engine import InferenceAccelerator
from clearvid.app.schemas.models import InferenceAccelerator as ConfigAccelerator


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


def test_resolve_trt_batch_caps_to_engine_max_profile() -> None:
    assert realesrgan_runner._resolve_trt_batch(
        8,
        InferenceAccelerator.TENSORRT,
        ConfigAccelerator.TENSORRT,
    ) == 4


def test_prepare_encoder_frame_rejects_wrong_size() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="编码帧尺寸不匹配"):
        realesrgan_runner._prepare_encoder_frame(frame, output_width=16, output_height=8)


def test_prepare_encoder_frame_converts_gray_to_bgr24() -> None:
    frame = np.zeros((8, 16), dtype=np.uint8)

    prepared = realesrgan_runner._prepare_encoder_frame(frame, output_width=16, output_height=8)

    assert prepared.shape == (8, 16, 3)
    assert prepared.dtype == np.uint8
    assert prepared.flags.c_contiguous is True


class _FakeTrtModel:
    _engine = object()
    max_batch = 2

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(tensor.shape[0]))
        return torch.nn.functional.interpolate(tensor, scale_factor=2, mode="nearest")


class _FakeUpsampler:
    scale = 2
    tile_size = 4
    tile_pad = 0

    def __init__(self) -> None:
        self.model = _FakeTrtModel()

    def pre_process(self, img: np.ndarray) -> None:
        tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = tensor.unsqueeze(0)

    def post_process(self) -> torch.Tensor:
        return self.output


def test_enhance_frame_trt_tiled_batches_tiles_with_expected_shape() -> None:
    frame = np.full((8, 8, 3), 128, dtype=np.uint8)
    upsampler = _FakeUpsampler()

    enhanced = realesrgan_runner._enhance_frame_trt_tiled(frame, upsampler, outscale=2.0)

    assert enhanced.shape == (16, 16, 3)
    assert enhanced.dtype == np.uint8
    assert upsampler.model.batch_sizes == [2, 2]