"""Tests for Real-ESRGAN stream process startup."""
from __future__ import annotations

import pytest
import numpy as np
import torch

from clearvid.app.models import realesrgan_runner
from clearvid.app.models.perf_diagnostics import GpuSampler
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
    tile_stats: dict[str, float] = {}

    enhanced = realesrgan_runner._enhance_frame_trt_tiled(
        frame, upsampler, outscale=2.0, tile_stats=tile_stats,
    )

    assert enhanced.shape == (16, 16, 3)
    assert enhanced.dtype == np.uint8
    assert upsampler.model.batch_sizes == [2, 2]
    assert tile_stats["tiles"] == 4
    assert tile_stats["tile_batches"] == 2
    assert tile_stats["tile_batch_max"] == 2
    assert tile_stats["engine_max_batch"] == 2
    assert tile_stats["tile_infer_ms"] >= 0


def test_enhance_frame_trt_tiled_pads_edge_tiles_to_batch() -> None:
    frame = np.full((4, 6, 3), 128, dtype=np.uint8)
    upsampler = _FakeUpsampler()
    tile_stats: dict[str, float] = {}

    enhanced = realesrgan_runner._enhance_frame_trt_tiled(
        frame, upsampler, outscale=2.0, tile_stats=tile_stats,
    )

    assert enhanced.shape == (8, 12, 3)
    assert upsampler.model.batch_sizes == [2]
    assert tile_stats["tiles"] == 2
    assert tile_stats["tile_batches"] == 1
    assert tile_stats["tile_batch_max"] == 2
    assert tile_stats["engine_max_batch"] == 2


def test_enhance_frame_trt_tiled_resizes_on_tensor_before_cpu() -> None:
    frame = np.full((4, 6, 3), 128, dtype=np.uint8)
    upsampler = _FakeUpsampler()
    tile_stats: dict[str, float] = {}

    enhanced = realesrgan_runner._enhance_frame_trt_tiled(
        frame, upsampler, outscale=1.0, tile_stats=tile_stats,
    )

    assert enhanced.shape == (4, 6, 3)
    assert enhanced.dtype == np.uint8
    assert tile_stats["trt_resize_ms"] >= 0
    assert tile_stats["trt_cpu_ms"] >= 0
    assert tile_stats["trt_post_ms"] >= 0


def test_gpu_sampler_selects_torch_device_uuid(monkeypatch) -> None:
    monkeypatch.setattr(GpuSampler, "_detect_torch_device_uuid", staticmethod(lambda: "bbbb"))

    sampler = GpuSampler.__new__(GpuSampler)
    sampler._target_uuid = "bbbb"

    selected = sampler._select_gpu_line([
        "0, GPU A, GPU-aaaa, 00000000:01:00.0, 10, 1, 1000, 2000, 40, 1000, 2000, 50",
        "1, GPU B, GPU-bbbb, 00000000:02:00.0, 80, 5, 3000, 4000, 50, 1800, 2200, 180",
    ])

    assert selected.startswith("1, GPU B")