"""Tests for Real-ESRGAN stream process startup."""
from __future__ import annotations

import io
import queue

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
        32,
        InferenceAccelerator.TENSORRT,
        ConfigAccelerator.TENSORRT,
    ) == 16


def test_auto_trt_tile_size_uses_larger_profile_on_high_vram(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _idx: type("Props", (), {"total_memory": 32 * 1024 ** 3})(),
    )

    assert realesrgan_runner._auto_trt_tile_size(0, width=854, height=480) == 1024


def test_auto_trt_tile_size_respects_explicit_value() -> None:
    assert realesrgan_runner._auto_trt_tile_size(512, width=854, height=480) == 512


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


def test_write_encoder_frame_writes_raw_bgr_bytes() -> None:
    frame = np.arange(8 * 16 * 3, dtype=np.uint8).reshape((8, 16, 3))
    sink = io.BytesIO()

    realesrgan_runner._write_encoder_frame(sink, frame, output_width=16, output_height=8)

    assert sink.getvalue() == frame.tobytes()


class _CountingSink:
    def __init__(self) -> None:
        self.data = bytearray()
        self.write_count = 0

    def write(self, data) -> int:  # noqa: ANN001
        self.write_count += 1
        self.data.extend(data)
        return len(data)


def test_write_finalized_frames_writes_batch_in_one_pipe_call() -> None:
    frames = np.arange(2 * 8 * 16 * 3, dtype=np.uint8).reshape((2, 8, 16, 3))
    sink = _CountingSink()

    count = realesrgan_runner._write_finalized_frames(
        frames,
        output_width=16,
        output_height=8,
        target_profile=realesrgan_runner.TargetProfile.SOURCE,
        encoder_stdin=sink,
    )

    assert count == 2
    assert sink.write_count == 1
    assert bytes(sink.data) == frames.tobytes()


def test_write_finalized_frames_list_avoids_extra_batch_copy() -> None:
    frames = [
        np.arange(8 * 16 * 3, dtype=np.uint8).reshape((8, 16, 3)),
        np.full((8, 16, 3), 7, dtype=np.uint8),
    ]
    sink = _CountingSink()

    count = realesrgan_runner._write_finalized_frames(
        frames,
        output_width=16,
        output_height=8,
        target_profile=realesrgan_runner.TargetProfile.SOURCE,
        encoder_stdin=sink,
    )

    assert count == 2
    assert sink.write_count == 2
    assert bytes(sink.data) == b"".join(frame.tobytes() for frame in frames)


def test_pack_trt_output_tensor_matches_original_bgr_rounding() -> None:
    result = torch.tensor(
        [[
            [[0.0, 0.25], [0.5, 1.0]],
            [[1.0, 0.5], [0.25, 0.0]],
            [[0.1, 0.2], [0.3, 0.4]],
        ]],
        dtype=torch.float32,
    )
    expected = (
        (result[:, [2, 1, 0], :, :] * 255.0)
        .round()
        .clamp_(0, 255)
        .to(dtype=torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )

    packed = realesrgan_runner._pack_trt_output_tensor(result.clone())

    assert torch.equal(packed, expected)


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


def test_enhance_frames_trt_tiled_batches_tiles_across_frames() -> None:
    frames = [
        np.full((8, 8, 3), 64, dtype=np.uint8),
        np.full((8, 8, 3), 128, dtype=np.uint8),
    ]
    upsampler = _FakeUpsampler()
    upsampler.model.max_batch = 4
    tile_stats: dict[str, float] = {}

    enhanced = realesrgan_runner._enhance_frames_trt_tiled(
        frames, upsampler, outscale=2.0, tile_stats=tile_stats,
    )

    assert len(enhanced) == 2
    assert [frame.shape for frame in enhanced] == [(16, 16, 3), (16, 16, 3)]
    assert upsampler.model.batch_sizes == [4, 4]
    assert tile_stats["tiles"] == 8
    assert tile_stats["tile_batches"] == 2
    assert tile_stats["tile_batch_max"] == 4


def test_fetch_enhanced_frames_trt_batches_when_skipper_inactive() -> None:
    raw_queue: queue.Queue[bytes | None] = queue.Queue()
    for value in (64, 128, 192):
        raw_queue.put(np.full((4, 4, 3), value, dtype=np.uint8).tobytes())
    raw_queue.put(None)
    upsampler = _FakeUpsampler()
    tile_stats: dict[str, float] = {}

    enhanced = realesrgan_runner._fetch_enhanced_frames(
        raw_queue,
        use_batching=False,
        batch_size=4,
        height=4,
        width=4,
        upsampler=upsampler,
        outscale=2.0,
        trt_tile_stats=tile_stats,
    )

    assert enhanced is not None
    assert len(enhanced) == 2
    assert upsampler.model.batch_sizes == [2]
    assert tile_stats["tile_batch_max"] == 2


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
    assert tile_stats["trt_pack_ms"] >= 0
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