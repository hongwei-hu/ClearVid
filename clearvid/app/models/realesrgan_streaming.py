from __future__ import annotations

import math
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from clearvid.app.export_control import ExportCancelled, ExportControl
from clearvid.app.models import stream_codec
from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.models.gfpgan_runner import GFPGANRestorer
from clearvid.app.models.perf_diagnostics import (
    CpuUsageTracker,
    GpuSampler,
    format_gpu_snapshot,
    format_gpu_summary,
    format_queue_info,
)
from clearvid.app.models.realesrgan_support import (
    FrameSkipper,
    _estimate_total_frames,
    _map_frame_progress,
    _resize_for_target,
)
from clearvid.app.postprocess.enhance import apply_sharpening
from clearvid.app.postprocess.temporal_stabilizer import TemporalStabilizer
from clearvid.app.schemas.models import EnhancementConfig, TargetProfile, VideoMetadata

if TYPE_CHECKING:
    from collections.abc import Callable


_FRAME_QUEUE_DEPTH = 64
_ENHANCED_QUEUE_DEPTH = 16
_PACKED_LIST_WRITE_MIN_BATCH = 4
_WRITE_COALESCE_MAX_ITEMS = 4
_WRITE_COALESCE_MAX_FRAMES = 64
_TrtTileInfo = tuple[int, int, int, int, int, int, int, int]
_TrtPendingTile = tuple[object, _TrtTileInfo]
_FramePayload = list[np.ndarray] | np.ndarray


def _read_exact_bytes(stream: object, byte_count: int) -> bytes | None:
    buffer = bytearray()
    while len(buffer) < byte_count:
        chunk = stream.read(byte_count - len(buffer))
        if not chunk:
            break
        buffer.extend(chunk)

    if not buffer:
        return None
    if len(buffer) != byte_count:
        raise RuntimeError("视频帧流被意外截断。")
    return bytes(buffer)


def _start_decode_thread(
    decoder_stdout: object,
    frame_size: int,
    abort: threading.Event,
) -> tuple[queue.Queue, list[BaseException], threading.Thread]:
    _depth = max(4, min(48, 256 * 1024 * 1024 // max(frame_size, 1)))
    raw_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_depth)
    errors: list[BaseException] = []

    def _loop() -> None:
        try:
            while not abort.is_set():
                raw = _read_exact_bytes(decoder_stdout, frame_size)
                if raw is None:
                    break
                while not abort.is_set():
                    try:
                        raw_queue.put(raw, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:
            errors.append(exc)
            abort.set()
        finally:
            while not abort.is_set():
                try:
                    raw_queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return raw_queue, errors, thread


def _collect_batch(
    raw_queue: queue.Queue,
    batch_size: int,
    fetch_stats: dict[str, float] | None = None,
) -> list[bytes]:
    batch: list[bytes] = []
    t_wait = time.perf_counter()
    raw = raw_queue.get()
    if fetch_stats is not None:
        fetch_stats["fetch_wait_ms"] = fetch_stats.get("fetch_wait_ms", 0.0) + (time.perf_counter() - t_wait) * 1000
    if raw is None:
        raw_queue.put(None)
        return batch
    batch.append(raw)
    for _ in range(batch_size - 1):
        try:
            t_wait = time.perf_counter()
            raw = raw_queue.get(timeout=0.033)
            if fetch_stats is not None:
                fetch_stats["fetch_wait_ms"] = fetch_stats.get("fetch_wait_ms", 0.0) + (time.perf_counter() - t_wait) * 1000
        except queue.Empty:
            break
        if raw is None:
            raw_queue.put(None)
            break
        batch.append(raw)
    return batch


def _enhance_frames_batch(
    frames: list[np.ndarray],
    upsampler: object,
    outscale: float,
) -> list[np.ndarray]:
    import torch

    frame_count = len(frames)
    if frame_count == 0:
        return []
    batch_np = np.stack(frames, axis=0)
    batch_np = np.ascontiguousarray(batch_np[:, :, :, ::-1].transpose(0, 3, 1, 2), dtype=np.float32)
    batch_np *= (1.0 / 255.0)

    batch = torch.from_numpy(batch_np)
    if upsampler.half:
        batch = batch.half()
    batch = batch.to(upsampler.device, non_blocking=True)

    with torch.inference_mode():
        output = upsampler.model(batch)

    output_cpu = output.float().cpu().clamp_(0, 1)

    results: list[np.ndarray] = []
    factor = outscale / upsampler.scale
    need_resize = abs(factor - 1.0) > 1e-6
    for index in range(output_cpu.shape[0]):
        out = output_cpu[index].numpy()
        out = np.ascontiguousarray(out[[2, 1, 0], :, :].transpose(1, 2, 0))
        if need_resize:
            height, width = out.shape[:2]
            out = cv2.resize(
                out, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=cv2.INTER_LANCZOS4,
            )
        results.append((out * 255.0).round().astype(np.uint8))
    return results


def _is_trt_upsampler(upsampler: object) -> bool:
    return hasattr(getattr(upsampler, "model", None), "_engine")


@dataclass
class _PendingTrtFrameBatch:
    cpu_tensor: object
    gpu_tensor: object | None
    event: object | None
    tile_stats: dict[str, float] | None = None

    def resolve(self) -> np.ndarray:
        t_resolve = time.perf_counter()
        t_cpu = time.perf_counter()
        if self.event is not None:
            self.event.synchronize()
        if self.tile_stats is not None:
            self.tile_stats["trt_cpu_ms"] = self.tile_stats.get("trt_cpu_ms", 0.0) + (time.perf_counter() - t_cpu) * 1000
        output_frames = self.cpu_tensor.numpy()
        self.gpu_tensor = None
        if self.tile_stats is not None:
            self.tile_stats["trt_resolve_ms"] = self.tile_stats.get("trt_resolve_ms", 0.0) + (time.perf_counter() - t_resolve) * 1000
        return np.ascontiguousarray(output_frames)


def _resolve_frame_payload(payload: _FramePayload | _PendingTrtFrameBatch) -> _FramePayload:
    if isinstance(payload, _PendingTrtFrameBatch):
        return payload.resolve()
    return payload


def _frame_payload_count(payload: _FramePayload) -> int:
    return int(payload.shape[0]) if isinstance(payload, np.ndarray) and payload.ndim == 4 else len(payload)


def _pad_trt_tile_to_shape(tile: object, target_h: int, target_w: int) -> object:
    tile_h = int(tile.shape[2])
    tile_w = int(tile.shape[3])
    if tile_h == target_h and tile_w == target_w:
        return tile
    padded = tile.new_zeros((tile.shape[0], tile.shape[1], target_h, target_w))
    padded[:, :, :tile_h, :tile_w] = tile
    return padded


def _flush_trt_tile_batch(
    pending: list[_TrtPendingTile],
    upsampler: object,
    output: object,
    tile_stats: dict[str, float] | None = None,
) -> None:
    if not pending:
        return
    import torch

    tiles = _make_trt_tile_batch(pending)
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = upsampler.model(tiles)
    if tile_stats is not None:
        tile_items = float(tiles.shape[0])
        tile_stats["tile_batches"] = tile_stats.get("tile_batches", 0.0) + 1.0
        tile_stats["tiles"] = tile_stats.get("tiles", 0.0) + tile_items
        tile_stats["tile_infer_ms"] = tile_stats.get("tile_infer_ms", 0.0) + (time.perf_counter() - t0) * 1000
        tile_stats["tile_batch_max"] = max(tile_stats.get("tile_batch_max", 0.0), tile_items)
    t_stitch = time.perf_counter()
    output_offset = 0
    for tile, info in pending:
        (
            output_start_x, output_end_x, output_start_y, output_end_y,
            output_start_x_tile, output_end_x_tile,
            output_start_y_tile, output_end_y_tile,
        ) = info
        tile_batch = int(tile.shape[0])
        output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = outputs[
            output_offset:output_offset + tile_batch,
            :,
            output_start_y_tile:output_end_y_tile,
            output_start_x_tile:output_end_x_tile,
        ]
        output_offset += tile_batch
    if tile_stats is not None:
        tile_stats["trt_stitch_ms"] = tile_stats.get("trt_stitch_ms", 0.0) + (time.perf_counter() - t_stitch) * 1000


def _make_trt_tile_batch(pending: list[_TrtPendingTile]) -> object:
    import torch

    if len(pending) == 1:
        tile = pending[0][0]
        return tile if tile.is_contiguous() else tile.contiguous()
    return torch.cat([item[0] for item in pending], dim=0)


def _trt_output_to_frames(
    output_tensor: object,
    upsampler: object,
    outscale: float,
    input_width: int,
    input_height: int,
    tile_stats: dict[str, float] | None = None,
    *,
    async_cpu_transfer: bool = False,
) -> _FramePayload | _PendingTrtFrameBatch:
    t_post = time.perf_counter()
    upsampler.output = output_tensor
    result_tensor = upsampler.post_process().float().clamp_(0, 1)
    if tile_stats is not None:
        tile_stats["trt_post_ms"] = tile_stats.get("trt_post_ms", 0.0) + (time.perf_counter() - t_post) * 1000

    if outscale is not None and outscale != float(upsampler.scale):
        import torch.nn.functional as F

        t_resize = time.perf_counter()
        result_tensor = F.interpolate(
            result_tensor,
            size=(max(1, int(input_height * outscale)), max(1, int(input_width * outscale))),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp_(0, 1)
        if tile_stats is not None:
            tile_stats["trt_resize_ms"] = tile_stats.get("trt_resize_ms", 0.0) + (time.perf_counter() - t_resize) * 1000

    packed_tensor = _pack_trt_output_tensor(result_tensor, tile_stats)
    return _copy_trt_frames_to_cpu(packed_tensor, tile_stats, async_cpu_transfer=async_cpu_transfer)


def _pack_trt_output_tensor(result_tensor: object, tile_stats: dict[str, float] | None = None) -> object:
    import torch

    t_pack = time.perf_counter()
    scaled_tensor = result_tensor.mul(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8)
    batch, _, height, width = scaled_tensor.shape
    packed_tensor = torch.empty(
        (batch, height, width, 3),
        dtype=torch.uint8,
        device=scaled_tensor.device,
    )
    packed_tensor[..., 0].copy_(scaled_tensor[:, 2, :, :])
    packed_tensor[..., 1].copy_(scaled_tensor[:, 1, :, :])
    packed_tensor[..., 2].copy_(scaled_tensor[:, 0, :, :])
    if tile_stats is not None:
        tile_stats["trt_pack_ms"] = tile_stats.get("trt_pack_ms", 0.0) + (time.perf_counter() - t_pack) * 1000
    return packed_tensor


def _copy_trt_frames_to_cpu(
    packed_tensor: object,
    tile_stats: dict[str, float] | None = None,
    *,
    async_cpu_transfer: bool = False,
) -> np.ndarray | _PendingTrtFrameBatch:
    t_cpu = time.perf_counter()
    if async_cpu_transfer:
        pending = _try_async_trt_cpu_transfer(packed_tensor, tile_stats)
        if pending is not None:
            if tile_stats is not None:
                tile_stats["trt_cpu_schedule_ms"] = tile_stats.get("trt_cpu_schedule_ms", 0.0) + (time.perf_counter() - t_cpu) * 1000
            return pending

    output_frames = packed_tensor.cpu().numpy()
    if tile_stats is not None:
        tile_stats["trt_cpu_ms"] = tile_stats.get("trt_cpu_ms", 0.0) + (time.perf_counter() - t_cpu) * 1000
    return np.ascontiguousarray(output_frames)


def _try_async_trt_cpu_transfer(
    packed_tensor: object,
    tile_stats: dict[str, float] | None,
) -> _PendingTrtFrameBatch | None:
    try:
        import torch

        if not getattr(packed_tensor, "is_cuda", False) or not torch.cuda.is_available():
            return None
        copy_stream = torch.cuda.Stream(device=packed_tensor.device)
        cpu_tensor = torch.empty(
            tuple(packed_tensor.shape),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        copy_stream.wait_stream(torch.cuda.current_stream(packed_tensor.device))
        with torch.cuda.stream(copy_stream):
            cpu_tensor.copy_(packed_tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        return _PendingTrtFrameBatch(cpu_tensor=cpu_tensor, gpu_tensor=packed_tensor, event=event, tile_stats=tile_stats)
    except Exception:
        return None


def _trt_output_to_frame(
    output_tensor: object,
    upsampler: object,
    outscale: float,
    input_width: int,
    input_height: int,
    tile_stats: dict[str, float] | None = None,
) -> np.ndarray:
    frames = _resolve_frame_payload(_trt_output_to_frames(
        output_tensor, upsampler, outscale, input_width, input_height, tile_stats,
    ))
    return np.ascontiguousarray(frames[0])


def _enhance_frames_trt_tiled(
    frames: list[np.ndarray],
    upsampler: object,
    outscale: float,
    tile_stats: dict[str, float] | None = None,
    *,
    async_cpu_transfer: bool = False,
) -> _FramePayload | _PendingTrtFrameBatch:
    if not frames:
        return []
    frame_count = len(frames)
    h_input, w_input = frames[0].shape[:2]
    preprocess = upsampler.pre_process

    def _supports_fast_batch_preprocess() -> bool:
        if not hasattr(upsampler, "device") or not hasattr(upsampler, "half"):
            return False
        pre_pad = int(getattr(upsampler, "pre_pad", 0) or 0)
        if pre_pad != 0:
            return False
        mod_scale = None
        if int(getattr(upsampler, "scale", 0) or 0) == 2:
            mod_scale = 2
        elif int(getattr(upsampler, "scale", 0) or 0) == 1:
            mod_scale = 4
        if mod_scale is None:
            return True
        return (h_input % mod_scale == 0) and (w_input % mod_scale == 0)

    def _fast_batch_preprocess() -> object:
        import torch

        batch_np = np.stack(frames, axis=0)
        batch_np = np.ascontiguousarray(batch_np[:, :, :, ::-1].transpose(0, 3, 1, 2), dtype=np.float32)
        batch_np *= (1.0 / 255.0)
        batch_tensor = torch.from_numpy(batch_np)
        if upsampler.half:
            batch_tensor = batch_tensor.half()
        batch_tensor = batch_tensor.to(upsampler.device, non_blocking=True)
        if hasattr(upsampler, "mod_scale"):
            upsampler.mod_scale = None
        if hasattr(upsampler, "mod_pad_h"):
            upsampler.mod_pad_h = 0
        if hasattr(upsampler, "mod_pad_w"):
            upsampler.mod_pad_w = 0
        if hasattr(upsampler, "img"):
            upsampler.img = batch_tensor
        return batch_tensor

    def _legacy_batch_preprocess() -> object:
        first = frames[0]
        first_img = cv2.cvtColor(first.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_BGR2RGB)
        preprocess(first_img)
        first_tensor = upsampler.img

        import torch

        batch_img = first_tensor.new_empty((
            frame_count,
            int(first_tensor.shape[1]),
            int(first_tensor.shape[2]),
            int(first_tensor.shape[3]),
        ))
        batch_img[0:1].copy_(first_tensor)

        for index in range(1, frame_count):
            frame = frames[index]
            if frame.shape[:2] != (h_input, w_input):
                raise ValueError("TRT frame batch requires equal frame sizes")
            img = cv2.cvtColor(frame.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_BGR2RGB)
            preprocess(img)
            batch_img[index:index + 1].copy_(upsampler.img)
        return batch_img

    t_prep = time.perf_counter()
    for frame in frames:
        if frame.shape[:2] != (h_input, w_input):
            raise ValueError("TRT frame batch requires equal frame sizes")
    if _supports_fast_batch_preprocess():
        upsampler.img = _fast_batch_preprocess()
    else:
        upsampler.img = _legacy_batch_preprocess()
    if tile_stats is not None:
        prep_elapsed_ms = (time.perf_counter() - t_prep) * 1000
        tile_stats["trt_preprocess_ms"] = tile_stats.get("trt_preprocess_ms", 0.0) + prep_elapsed_ms

    batch, channel, height, width = upsampler.img.shape
    output = upsampler.img.new_zeros((batch, channel, height * upsampler.scale, width * upsampler.scale))
    tile_size = int(getattr(upsampler, "tile_size", 0) or 0)
    tile_pad = int(getattr(upsampler, "tile_pad", 0) or 0)
    max_batch = max(1, int(getattr(upsampler.model, "max_batch", 1)))
    if tile_stats is not None:
        tile_stats["engine_max_batch"] = float(max_batch)
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    target_tile_h = min(height, tile_size + tile_pad * 2)
    target_tile_w = min(width, tile_size + tile_pad * 2)

    pending: list[_TrtPendingTile] = []
    pending_hw: tuple[int, int] | None = None
    pending_items = 0

    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            ofs_x = tile_x * tile_size
            ofs_y = tile_y * tile_size
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            input_tile = upsampler.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
            input_tile = _pad_trt_tile_to_shape(input_tile, target_tile_h, target_tile_w)
            tile_hw = (int(input_tile.shape[2]), int(input_tile.shape[3]))
            tile_items = int(input_tile.shape[0])
            if pending_hw is not None and (
                tile_hw != pending_hw or pending_items + tile_items > max_batch
            ):
                _flush_trt_tile_batch(pending, upsampler, output, tile_stats=tile_stats)
                pending = []
                pending_hw = None
                pending_items = 0
            pending_hw = tile_hw

            output_start_x = input_start_x * upsampler.scale
            output_end_x = input_end_x * upsampler.scale
            output_start_y = input_start_y * upsampler.scale
            output_end_y = input_end_y * upsampler.scale

            output_start_x_tile = (input_start_x - input_start_x_pad) * upsampler.scale
            output_end_x_tile = output_start_x_tile + input_tile_width * upsampler.scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * upsampler.scale
            output_end_y_tile = output_start_y_tile + input_tile_height * upsampler.scale
            pending.append((
                input_tile,
                (
                    output_start_x, output_end_x, output_start_y, output_end_y,
                    output_start_x_tile, output_end_x_tile,
                    output_start_y_tile, output_end_y_tile,
                ),
            ))
            pending_items += tile_items
    _flush_trt_tile_batch(pending, upsampler, output, tile_stats=tile_stats)

    return _trt_output_to_frames(
        output, upsampler, outscale, w_input, h_input, tile_stats,
        async_cpu_transfer=async_cpu_transfer,
    )


def _enhance_frame_trt_tiled(
    frame: np.ndarray,
    upsampler: object,
    outscale: float,
    tile_stats: dict[str, float] | None = None,
) -> np.ndarray:
    frames = _resolve_frame_payload(_enhance_frames_trt_tiled([frame], upsampler, outscale, tile_stats=tile_stats))
    return np.ascontiguousarray(frames[0])


def _fetch_enhanced_frames(
    raw_queue: queue.Queue,
    use_batching: bool,
    batch_size: int,
    height: int,
    width: int,
    upsampler: object,
    outscale: float,
    skipper: FrameSkipper | None = None,
    trt_tile_stats: dict[str, float] | None = None,
    async_cpu_transfer: bool = False,
) -> _FramePayload | _PendingTrtFrameBatch | None:
    if use_batching:
        batch_raw = _collect_batch(raw_queue, batch_size, trt_tile_stats)
        if not batch_raw:
            return None
        images = [np.frombuffer(r, dtype=np.uint8).reshape((height, width, 3)) for r in batch_raw]
        return _enhance_frames_batch(images, upsampler, outscale)

    if (
        not (skipper is not None and skipper.active)
        and _is_trt_upsampler(upsampler)
        and getattr(upsampler, "tile_size", 0) > 0
        and batch_size > 1
    ):
        trt_frame_batch = min(batch_size, max(1, int(getattr(upsampler.model, "max_batch", 1))))
        batch_raw = _collect_batch(raw_queue, trt_frame_batch, trt_tile_stats)
        if not batch_raw:
            return None
        images = [np.frombuffer(r, dtype=np.uint8).reshape((height, width, 3)) for r in batch_raw]
        return _enhance_frames_trt_tiled(
            images, upsampler, outscale,
            tile_stats=trt_tile_stats,
            async_cpu_transfer=async_cpu_transfer,
        )

    t_wait = time.perf_counter()
    raw_frame = raw_queue.get()
    if trt_tile_stats is not None:
        trt_tile_stats["fetch_wait_ms"] = trt_tile_stats.get("fetch_wait_ms", 0.0) + (time.perf_counter() - t_wait) * 1000
    if raw_frame is None:
        return None
    image = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

    if skipper is not None and skipper.should_skip(image):
        skipper.skip_count += 1
        return [skipper.get_cached()]

    if _is_trt_upsampler(upsampler) and getattr(upsampler, "tile_size", 0) > 0:
        enhanced = _enhance_frame_trt_tiled(image, upsampler, outscale, tile_stats=trt_tile_stats)
    else:
        enhanced, _ = upsampler.enhance(image, outscale=outscale)
    if skipper is not None:
        skipper.record(image, enhanced)
    return [enhanced]


def _prepare_encoder_frame(frame: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim != 3:
        raise ValueError(f"编码帧维度异常: {frame.shape}")
    elif frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.shape[2] > 3:
        frame = frame[:, :, :3]

    if frame.shape[1] != output_width or frame.shape[0] != output_height:
        raise ValueError(
            f"编码帧尺寸不匹配: got {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {output_width}x{output_height}"
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _write_encoder_frame(
    encoder_stdin: object,
    frame: np.ndarray,
    output_width: int,
    output_height: int,
    write_stats: dict[str, float] | None = None,
) -> None:
    t_prepare = time.perf_counter()
    prepared = _prepare_encoder_frame(frame, output_width, output_height)
    if write_stats is not None:
        write_stats["write_prepare_ms"] = write_stats.get("write_prepare_ms", 0.0) + (time.perf_counter() - t_prepare) * 1000
    t_pipe = time.perf_counter()
    encoder_stdin.write(memoryview(prepared).cast("B"))
    if write_stats is not None:
        write_stats["write_pipe_ms"] = write_stats.get("write_pipe_ms", 0.0) + (time.perf_counter() - t_pipe) * 1000


def _iter_frame_payload(frames: _FramePayload):
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        for index in range(frames.shape[0]):
            yield frames[index]
        return
    yield from frames


def _prepare_encoder_ndarray_batch(
    frames: np.ndarray,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> np.ndarray | None:
    if not (
        frames.ndim == 4
        and frames.shape[3] == 3
        and frames.dtype == np.uint8
        and frames.flags.c_contiguous
    ):
        return None

    count = int(frames.shape[0])
    input_height = int(frames.shape[1])
    input_width = int(frames.shape[2])
    prepared_batch = np.empty((count, output_height, output_width, 3), dtype=np.uint8)

    if target_profile in {TargetProfile.FHD, TargetProfile.UHD4K}:
        scale = min(output_width / input_width, output_height / input_height)
        resized_width = max(1, int(round(input_width * scale)))
        resized_height = max(1, int(round(input_height * scale)))
        offset_x = (output_width - resized_width) // 2
        offset_y = (output_height - resized_height) // 2
        prepared_batch.fill(0)
        for index in range(count):
            roi = prepared_batch[index, offset_y:offset_y + resized_height, offset_x:offset_x + resized_width]
            cv2.resize(
                frames[index],
                (resized_width, resized_height),
                dst=roi,
                interpolation=cv2.INTER_LANCZOS4,
            )
        return prepared_batch

    for index in range(count):
        cv2.resize(
            frames[index],
            (output_width, output_height),
            dst=prepared_batch[index],
            interpolation=cv2.INTER_LANCZOS4,
        )
    return prepared_batch


def _prepare_encoder_frame_batch(
    frames: _FramePayload,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> np.ndarray:
    if (
        isinstance(frames, np.ndarray)
        and frames.ndim == 4
        and frames.dtype == np.uint8
        and frames.shape[1:] == (output_height, output_width, 3)
        and frames.flags.c_contiguous
    ):
        return frames

    if isinstance(frames, list) and frames and all(
        frame.ndim == 3
        and frame.shape == (output_height, output_width, 3)
        and frame.dtype == np.uint8
        and frame.flags.c_contiguous
        for frame in frames
    ):
        return np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)

    if isinstance(frames, np.ndarray):
        prepared_ndarray = _prepare_encoder_ndarray_batch(frames, output_width, output_height, target_profile)
        if prepared_ndarray is not None:
            return prepared_ndarray

    count = _frame_payload_count(frames)
    prepared_batch = np.empty((count, output_height, output_width, 3), dtype=np.uint8)
    for index, frame in enumerate(_iter_frame_payload(frames)):
        resized = _resize_for_target(frame, output_width, output_height, target_profile)
        prepared_batch[index] = _prepare_encoder_frame(resized, output_width, output_height)
    return prepared_batch


def _payload_to_frame_list(payload: _FramePayload) -> list[np.ndarray]:
    if isinstance(payload, np.ndarray) and payload.ndim == 4:
        return [payload[index] for index in range(payload.shape[0])]
    return payload


def _can_pack_frame_list(payload: _FramePayload) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    first = payload[0]
    if first.ndim != 3 or first.shape[2] != 3:
        return False
    target_shape = first.shape
    return all(
        frame.ndim == 3
        and frame.shape == target_shape
        and frame.dtype == np.uint8
        and frame.flags.c_contiguous
        for frame in payload
    )


def _coalesce_finalized_payloads(
    first_item: _FramePayload,
    finalized_queue: queue.Queue,
) -> tuple[_FramePayload, bool]:
    if isinstance(first_item, np.ndarray) and first_item.ndim == 4:
        # Keep already-packed batches as-is to avoid converting ndarray payloads
        # back into lists, which would trigger extra prepare work in writer.
        return first_item, False

    payloads: list[_FramePayload] = [first_item]
    total_frames = _frame_payload_count(first_item)
    saw_sentinel = False

    for _ in range(_WRITE_COALESCE_MAX_ITEMS - 1):
        if total_frames >= _WRITE_COALESCE_MAX_FRAMES:
            break
        try:
            item = finalized_queue.get_nowait()
        except queue.Empty:
            break
        if item is None:
            saw_sentinel = True
            break
        payloads.append(item)
        total_frames += _frame_payload_count(item)

    if len(payloads) == 1:
        return first_item, saw_sentinel

    merged_frames: list[np.ndarray] = []
    for payload in payloads:
        merged_frames.extend(_payload_to_frame_list(payload))
    return merged_frames, saw_sentinel


def _write_encoder_frame_batch(
    encoder_stdin: object,
    frames: _FramePayload,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    write_stats: dict[str, float] | None = None,
) -> int:
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        t_prepare = time.perf_counter()
        prepared_batch = _prepare_encoder_frame_batch(frames, output_width, output_height, target_profile)
        if write_stats is not None:
            write_stats["write_prepare_ms"] = write_stats.get("write_prepare_ms", 0.0) + (time.perf_counter() - t_prepare) * 1000
        t_pipe = time.perf_counter()
        encoder_stdin.write(memoryview(prepared_batch).cast("B"))
        if write_stats is not None:
            write_stats["write_pipe_ms"] = write_stats.get("write_pipe_ms", 0.0) + (time.perf_counter() - t_pipe) * 1000
        return int(prepared_batch.shape[0])

    if len(frames) >= _PACKED_LIST_WRITE_MIN_BATCH:
        t_prepare = time.perf_counter()
        prepared_batch = _prepare_encoder_frame_batch(frames, output_width, output_height, target_profile)
        if write_stats is not None:
            write_stats["write_prepare_ms"] = write_stats.get("write_prepare_ms", 0.0) + (time.perf_counter() - t_prepare) * 1000
        t_pipe = time.perf_counter()
        encoder_stdin.write(memoryview(prepared_batch).cast("B"))
        if write_stats is not None:
            write_stats["write_pipe_ms"] = write_stats.get("write_pipe_ms", 0.0) + (time.perf_counter() - t_pipe) * 1000
        return int(prepared_batch.shape[0])

    count = 0
    for frame in frames:
        t_prepare = time.perf_counter()
        resized = _resize_for_target(frame, output_width, output_height, target_profile)
        prepared = _prepare_encoder_frame(resized, output_width, output_height)
        if write_stats is not None:
            write_stats["write_prepare_ms"] = write_stats.get("write_prepare_ms", 0.0) + (time.perf_counter() - t_prepare) * 1000
        t_pipe = time.perf_counter()
        encoder_stdin.write(memoryview(prepared).cast("B"))
        if write_stats is not None:
            write_stats["write_pipe_ms"] = write_stats.get("write_pipe_ms", 0.0) + (time.perf_counter() - t_pipe) * 1000
        count += 1
    return count


def _write_enhanced_frames(
    enhanced_list: list[np.ndarray],
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    encoder_stdin: object,
    sharpen_strength: float = 0.0,
    write_stats: dict[str, float] | None = None,
) -> int:
    finalized_list: list[np.ndarray] = []
    for enhanced in enhanced_list:
        if codeformer_restorer is not None:
            enhanced = codeformer_restorer.restore_faces(enhanced)
        if stabilizer is not None:
            enhanced = stabilizer.stabilize(enhanced)
        if sharpen_strength > 0:
            enhanced = apply_sharpening(enhanced, sharpen_strength)
        finalized_list.append(enhanced)
    return _write_encoder_frame_batch(
        encoder_stdin, finalized_list, output_width, output_height, target_profile,
        write_stats=write_stats,
    )


def _write_finalized_frames(
    finalized_list: _FramePayload,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    encoder_stdin: object,
    write_stats: dict[str, float] | None = None,
) -> int:
    return _write_encoder_frame_batch(
        encoder_stdin, finalized_list, output_width, output_height, target_profile,
        write_stats=write_stats,
    )


def _record_and_queue_enhanced_payload(
    payload: _FramePayload | _PendingTrtFrameBatch,
    elapsed_ms: float,
    enhanced_queue: queue.Queue,
    safe_put: Callable[[queue.Queue, object], bool],
    diag_infer_ms: list[float],
    diag_batches: list[int],
    diag_frames: list[int],
    diag_batch_sizes: list[int],
) -> bool:
    enhanced_payload = _resolve_frame_payload(payload)
    count = _frame_payload_count(enhanced_payload)
    diag_infer_ms[0] += elapsed_ms
    diag_batches[0] += 1
    diag_frames[0] += count
    if len(diag_batch_sizes) >= 100:
        diag_batch_sizes.pop(0)
    diag_batch_sizes.append(count)
    return safe_put(enhanced_queue, enhanced_payload)


def _process_stream_frames(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    decoder: object,
    encoder: object,
    progress_callback: Callable[[int, str], None] | None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
) -> None:
    frame_size = metadata.width * metadata.height * 3
    total_frames = _estimate_total_frames(metadata, config.preview_seconds)

    assert decoder.stdout is not None
    assert encoder.stdin is not None
    skipper = FrameSkipper(getattr(config, "skip_frame_threshold", 0.0))
    _emit_progress(
        progress_callback, 18,
        f"正在流式处理视频帧 (batch={config.batch_size}, "
        f"{'async' if config.async_pipeline else 'sync'}"
        f"{', 智能跳帧=%d' % int(skipper._threshold) if skipper.active else ''})",
    )

    abort = threading.Event()
    cancel_triggered = [False]

    def _cancel_watch_loop() -> None:
        if control is None:
            return
        while not abort.is_set():
            if control.is_cancelled:
                cancel_triggered[0] = True
                abort.set()
                # Force-stop ffmpeg processes to unblock pipe read/write quickly.
                _cleanup_stream_processes(decoder, encoder)
                return
            time.sleep(0.05)

    cancel_watcher = threading.Thread(
        target=_cancel_watch_loop,
        daemon=True,
        name="cancel-watcher",
    )
    cancel_watcher.start()

    raw_queue, decode_errors, reader = _start_decode_thread(decoder.stdout, frame_size, abort)

    if config.async_pipeline:
        _process_frames_async(
            config, metadata, output_width, output_height, outscale,
            upsampler, codeformer_restorer, stabilizer,
            raw_queue, encoder, total_frames, progress_callback,
            abort=abort, control=control,
            preview_mux_trigger=preview_mux_trigger,
            skipper=skipper,
        )
    else:
        _process_frames_sync(
            config, metadata, output_width, output_height, outscale,
            upsampler, codeformer_restorer, stabilizer,
            raw_queue, encoder, total_frames, progress_callback,
            preview_mux_trigger=preview_mux_trigger,
            skipper=skipper,
        )

    if skipper.skip_count > 0:
        logging.getLogger(__name__).info(
            "智能跳帧: 跳过 %d 帧 (%.1f%%)",
            skipper.skip_count,
            skipper.skip_count / max(total_frames or 1, 1) * 100,
        )

    reader.join()
    cancel_watcher.join(timeout=0.2)
    if cancel_triggered[0] or (control is not None and control.is_cancelled):
        raise ExportCancelled("导出已被用户取消")
    if decode_errors:
        raise decode_errors[0]


def _process_frames_sync(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    raw_queue: queue.Queue,
    encoder: object,
    total_frames: int | None,
    progress_callback: Callable[[int, str], None] | None,
    preview_mux_trigger: Callable[[int], None] | None = None,
    skipper: FrameSkipper | None = None,
) -> None:
    use_batching = getattr(upsampler, "tile_size", 0) == 0 and config.batch_size > 1
    processed_frames = 0
    last_reported_progress = -1
    fps = metadata.fps or 30.0
    preview_interval_frames = int(fps * 60)
    last_preview_at = 0
    start_time = time.perf_counter()

    while True:
        enhanced_list = _fetch_enhanced_frames(
            raw_queue, use_batching, config.batch_size, metadata.height, metadata.width, upsampler, outscale,
            skipper=skipper,
        )
        if enhanced_list is None:
            break
        processed_frames += _write_enhanced_frames(
            enhanced_list, codeformer_restorer, stabilizer, output_width, output_height, config.target_profile, encoder.stdin,
            sharpen_strength=config.sharpen_strength if config.sharpen_enabled else 0.0,
        )
        last_reported_progress = _report_stream_progress(
            processed_frames, total_frames, last_reported_progress, progress_callback,
            start_time=start_time,
        )
        if preview_mux_trigger and processed_frames - last_preview_at >= preview_interval_frames:
            last_preview_at = processed_frames
            preview_mux_trigger(processed_frames)

    encoder.stdin.close()


def _process_frames_async(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | GFPGANRestorer | None,
    stabilizer: TemporalStabilizer | None,
    raw_queue: queue.Queue,
    encoder: object,
    total_frames: int | None,
    progress_callback: Callable[[int, str], None] | None,
    *,
    abort: threading.Event | None = None,
    control: ExportControl | None = None,
    preview_mux_trigger: Callable[[int], None] | None = None,
    skipper: FrameSkipper | None = None,
) -> None:
    if abort is None:
        abort = threading.Event()

    def _safe_put(q: queue.Queue, item: object) -> bool:
        while not abort.is_set():
            try:
                q.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def _safe_get(q: queue.Queue) -> tuple[bool, object]:
        while not abort.is_set():
            try:
                return True, q.get(timeout=1.0)
            except queue.Empty:
                continue
        return False, None

    sharpen_strength = config.sharpen_strength if config.sharpen_enabled else 0.0
    postprocess_needed = (
        codeformer_restorer is not None
        or stabilizer is not None
        or sharpen_strength > 0
    )

    enhanced_item_bytes = output_width * output_height * 3 * max(1, config.batch_size)
    eq_depth = max(4, min(16, 128 * 1024 * 1024 // max(enhanced_item_bytes, 1)))
    if postprocess_needed:
        enhanced_queue: queue.Queue[_FramePayload | None] = queue.Queue(maxsize=eq_depth)
        finalized_queue: queue.Queue[_FramePayload | None] = queue.Queue(maxsize=eq_depth)
    else:
        finalized_queue = queue.Queue(maxsize=eq_depth)
        enhanced_queue = finalized_queue

    enhance_errors: list[BaseException] = []
    write_errors: list[BaseException] = []
    frames_written = [0]
    diag_enhance_batches = [0]
    diag_enhance_frames = [0]
    diag_enhance_infer_ms = [0.0]
    diag_write_frames = [0]
    diag_write_ms = [0.0]
    diag_write_prepare_ms = [0.0]
    diag_write_pipe_ms = [0.0]
    diag_batch_sizes: list[int] = []
    diag_postprocess_ms = [0.0]
    diag_face_ms = [0.0]
    diag_stabilize_ms = [0.0]
    diag_sharpen_ms = [0.0]
    diag_trt_tile_stats: dict[str, float] = {}
    use_batching = getattr(upsampler, "tile_size", 0) == 0 and config.batch_size > 1

    def _enhance_loop() -> None:
        pending_payload: _FramePayload | _PendingTrtFrameBatch | None = None
        pending_elapsed_ms = 0.0
        try:
            while not abort.is_set():
                if control is not None:
                    control.check()
                t_wait = time.perf_counter()
                enhanced_payload = _fetch_enhanced_frames(
                    raw_queue, use_batching, config.batch_size,
                    metadata.height, metadata.width, upsampler, outscale,
                    skipper=skipper,
                    trt_tile_stats=diag_trt_tile_stats,
                    async_cpu_transfer=True,
                )
                t_done = time.perf_counter()
                if enhanced_payload is None:
                    break
                if pending_payload is not None and not _record_and_queue_enhanced_payload(
                    pending_payload, pending_elapsed_ms, enhanced_queue, _safe_put,
                    diag_enhance_infer_ms, diag_enhance_batches,
                    diag_enhance_frames, diag_batch_sizes,
                ):
                    break
                pending_payload = enhanced_payload
                pending_elapsed_ms = (t_done - t_wait) * 1000
            if pending_payload is not None:
                _record_and_queue_enhanced_payload(
                    pending_payload, pending_elapsed_ms, enhanced_queue, _safe_put,
                    diag_enhance_infer_ms, diag_enhance_batches,
                    diag_enhance_frames, diag_batch_sizes,
                )
        except ExportCancelled:
            abort.set()
        except Exception as exc:
            enhance_errors.append(exc)
            abort.set()
        finally:
            while not abort.is_set():
                try:
                    enhanced_queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue

    enhance_thread = threading.Thread(target=_enhance_loop, daemon=True, name="sr-enhance")

    def _write_loop() -> None:
        try:
            while not abort.is_set():
                if control is not None:
                    control.check()
                ok, item = _safe_get(finalized_queue)
                if not ok or item is None:
                    break
                batch_payload, saw_sentinel = _coalesce_finalized_payloads(item, finalized_queue)
                t_w = time.perf_counter()
                write_stats: dict[str, float] = {}
                if control is not None:
                    control.check()
                count = _write_finalized_frames(
                    batch_payload, output_width, output_height,
                    config.target_profile, encoder.stdin,
                    write_stats=write_stats,
                )
                diag_write_ms[0] += (time.perf_counter() - t_w) * 1000
                diag_write_prepare_ms[0] += write_stats.get("write_prepare_ms", 0.0)
                diag_write_pipe_ms[0] += write_stats.get("write_pipe_ms", 0.0)
                diag_write_frames[0] += count
                frames_written[0] += count
                if saw_sentinel:
                    break
        except ExportCancelled:
            abort.set()
        except Exception as exc:
            write_errors.append(exc)
            abort.set()
        finally:
            try:
                encoder.stdin.close()
            except Exception:
                pass

    write_thread = threading.Thread(target=_write_loop, daemon=True, name="frame-writer")
    stage3_errors: list[BaseException] = []
    postprocess_thread: threading.Thread | None = None

    if postprocess_needed:
        def _postprocess_loop() -> None:
            stab_pool: ThreadPoolExecutor | None = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="stabilizer")
                if stabilizer is not None else None
            )
            stab_future: Future[np.ndarray] | None = None

            def _stabilize_and_sharpen(frame: np.ndarray) -> np.ndarray:
                t_stage = time.perf_counter()
                result = frame
                if stabilizer is not None:
                    t_stab = time.perf_counter()
                    result = stabilizer.stabilize(result)
                    diag_stabilize_ms[0] += (time.perf_counter() - t_stab) * 1000
                if sharpen_strength > 0:
                    t_sharp = time.perf_counter()
                    result = apply_sharpening(result, sharpen_strength)
                    diag_sharpen_ms[0] += (time.perf_counter() - t_sharp) * 1000
                diag_postprocess_ms[0] += (time.perf_counter() - t_stage) * 1000
                return result

            try:
                while not abort.is_set():
                    ok, enhanced_list = _safe_get(enhanced_queue)
                    if not ok or enhanced_list is None:
                        break

                    # Fast-path for the common sharpen-only case: keep batch payload
                    # as contiguous 4D ndarray so writer can bypass per-frame prepare.
                    if (
                        codeformer_restorer is None
                        and stab_pool is None
                        and stab_future is None
                        and sharpen_strength > 0
                        and isinstance(enhanced_list, np.ndarray)
                        and enhanced_list.ndim == 4
                        and enhanced_list.dtype == np.uint8
                        and enhanced_list.flags.c_contiguous
                    ):
                        sharpened_batch = np.empty_like(enhanced_list)
                        t_pp = time.perf_counter()
                        for index in range(enhanced_list.shape[0]):
                            if control is not None:
                                control.check()
                            t_sharp = time.perf_counter()
                            sharpened_batch[index] = apply_sharpening(enhanced_list[index], sharpen_strength)
                            diag_sharpen_ms[0] += (time.perf_counter() - t_sharp) * 1000
                        diag_postprocess_ms[0] += (time.perf_counter() - t_pp) * 1000
                        _safe_put(finalized_queue, sharpened_batch)
                        continue

                    finalized_list: list[np.ndarray] = []
                    for enhanced in enhanced_list:
                        if abort.is_set():
                            break
                        if control is not None:
                            control.check()
                        if codeformer_restorer is not None:
                            t_face = time.perf_counter()
                            enhanced = codeformer_restorer.restore_faces(enhanced)
                            face_ms = (time.perf_counter() - t_face) * 1000
                            diag_face_ms[0] += face_ms
                            diag_postprocess_ms[0] += face_ms

                        if stab_future is not None:
                            finalized_list.append(stab_future.result())
                            stab_future = None

                        if stab_pool is not None:
                            stab_future = stab_pool.submit(_stabilize_and_sharpen, enhanced)
                        else:
                            if sharpen_strength > 0:
                                t_pp = time.perf_counter()
                                t_sharp = time.perf_counter()
                                enhanced = apply_sharpening(enhanced, sharpen_strength)
                                diag_sharpen_ms[0] += (time.perf_counter() - t_sharp) * 1000
                                diag_postprocess_ms[0] += (time.perf_counter() - t_pp) * 1000
                            finalized_list.append(enhanced)

                    if finalized_list and not abort.is_set():
                        _safe_put(finalized_queue, finalized_list)

                if stab_future is not None and not abort.is_set():
                    _safe_put(finalized_queue, [stab_future.result()])
            except ExportCancelled:
                abort.set()
            except Exception as exc:
                stage3_errors.append(exc)
                abort.set()
            finally:
                while not abort.is_set():
                    try:
                        finalized_queue.put(None, timeout=1.0)
                        break
                    except queue.Full:
                        continue
                if stab_pool is not None:
                    stab_pool.shutdown(wait=False)

        postprocess_thread = threading.Thread(target=_postprocess_loop, daemon=True, name="postprocess")

    enhance_thread.start()
    if postprocess_thread is not None:
        postprocess_thread.start()
    write_thread.start()

    gpu_sampler = GpuSampler(interval=5.0)
    cpu_tracker = CpuUsageTracker()
    monitor_thread = postprocess_thread if postprocess_thread is not None else enhance_thread
    last_reported_progress = -1
    fps = metadata.fps or 30.0
    preview_interval_frames = int(fps * 60)
    last_preview_at = 0
    start_time = time.perf_counter()
    last_diag_time = start_time
    diag_interval = 10.0
    diag_initial_interval = 3.0   # 前几次报告缩短到 3s，便于短时运行也能采集到数据
    diag_count = 0                # 已发送的诊断报告次数
    while monitor_thread.is_alive():
        monitor_thread.join(timeout=0.5)
        current_written = frames_written[0]
        now = time.perf_counter()
        last_reported_progress = _report_stream_progress(
            current_written, total_frames, last_reported_progress,
            progress_callback, start_time=start_time,
        )
        if preview_mux_trigger and current_written - last_preview_at >= preview_interval_frames:
            last_preview_at = current_written
            preview_mux_trigger(current_written)
        effective_interval = diag_initial_interval if diag_count < 3 else diag_interval
        if now - last_diag_time >= effective_interval and progress_callback:
            last_diag_time = now
            cpu_pct = cpu_tracker.sample_percent(now)
            rq = raw_queue.qsize()
            eq = enhanced_queue.qsize() if enhanced_queue is not finalized_queue else -1
            fq = finalized_queue.qsize()
            avg_batch = (
                sum(diag_batch_sizes[-20:]) / len(diag_batch_sizes)
                if diag_batch_sizes else 0
            )
            elapsed_s = now - start_time
            enh_fps = diag_enhance_frames[0] / elapsed_s if elapsed_s > 0 else 0
            wr_fps = diag_write_frames[0] / elapsed_s if elapsed_s > 0 else 0
            pp_avg = diag_postprocess_ms[0] / diag_write_frames[0] if diag_write_frames[0] > 0 else 0
            avg_infer_ms = diag_enhance_infer_ms[0] / diag_enhance_batches[0] if diag_enhance_batches[0] > 0 else 0
            trt_tile_batches = diag_trt_tile_stats.get("tile_batches", 0.0)
            trt_tiles = diag_trt_tile_stats.get("tiles", 0.0)
            trt_avg_tile_batch = trt_tiles / trt_tile_batches if trt_tile_batches > 0 else 0.0
            queue_info = format_queue_info(rq, eq, fq, _FRAME_QUEUE_DEPTH, _ENHANCED_QUEUE_DEPTH)
            pp_info = f" 后处理={pp_avg:.0f}ms/帧" if pp_avg > 0 else ""
            infer_info = f" 推理={avg_infer_ms:.0f}ms/批" if avg_infer_ms > 0 else ""
            trt_info = f" TRT-tile-batch={trt_avg_tile_batch:.1f}" if trt_tile_batches > 0 else ""
            gpu_info = format_gpu_snapshot(gpu_sampler.snapshot())
            _emit_progress(
                progress_callback,
                last_reported_progress,
                f"[诊断] {queue_info} | 增强 {enh_fps:.1f}fps 写入 {wr_fps:.1f}fps"
                f" | 平均帧batch={avg_batch:.1f}{trt_info}{infer_info}{pp_info} CPU={cpu_pct:.0f}%{gpu_info}",
            )
            diag_count += 1

        if control is not None and control.is_cancelled:
            abort.set()
            break

    _report_stream_progress(
        frames_written[0], total_frames, last_reported_progress,
        progress_callback, start_time=start_time,
    )

    if control is not None and control.is_cancelled:
        abort.set()
        raise ExportCancelled("导出已被用户取消")

    gpu_sampler.stop()
    gpu_summary = gpu_sampler.summary()
    total_elapsed = time.perf_counter() - start_time
    if progress_callback and total_elapsed > 0:
        enh_batches = diag_enhance_batches[0]
        enh_frames = diag_enhance_frames[0]
        wr_frames = diag_write_frames[0]
        frame_denominator = max(wr_frames, 1)
        avg_batch = enh_frames / enh_batches if enh_batches else 0
        avg_infer_ms = diag_enhance_infer_ms[0] / enh_batches if enh_batches else 0
        avg_write_ms = diag_write_ms[0] / wr_frames if wr_frames else 0
        avg_write_prepare_ms = diag_write_prepare_ms[0] / wr_frames if wr_frames else 0
        avg_write_pipe_ms = diag_write_pipe_ms[0] / wr_frames if wr_frames else 0
        avg_pp_ms = diag_postprocess_ms[0] / frame_denominator
        avg_face_ms = diag_face_ms[0] / frame_denominator
        avg_stabilize_ms = diag_stabilize_ms[0] / frame_denominator
        avg_sharpen_ms = diag_sharpen_ms[0] / frame_denominator
        total_fps = enh_frames / total_elapsed
        cpu_total_pct = cpu_tracker.total_percent(total_elapsed)
        skip_frames = skipper.skip_count if skipper is not None else 0
        skip_pct = skip_frames * 100.0 / max(enh_frames + skip_frames, 1)
        trt_tile_batches = diag_trt_tile_stats.get("tile_batches", 0.0)
        trt_tiles = diag_trt_tile_stats.get("tiles", 0.0)
        trt_tile_ms = diag_trt_tile_stats.get("tile_infer_ms", 0.0)
        trt_tile_max_batch = diag_trt_tile_stats.get("tile_batch_max", 0.0)
        trt_engine_max_batch = diag_trt_tile_stats.get("engine_max_batch", 0.0)
        trt_fetch_wait_ms = diag_trt_tile_stats.get("fetch_wait_ms", 0.0)
        trt_preprocess_ms = diag_trt_tile_stats.get("trt_preprocess_ms", 0.0)
        trt_post_ms = diag_trt_tile_stats.get("trt_post_ms", 0.0)
        trt_resize_ms = diag_trt_tile_stats.get("trt_resize_ms", 0.0)
        trt_pack_ms = diag_trt_tile_stats.get("trt_pack_ms", 0.0)
        trt_stitch_ms = diag_trt_tile_stats.get("trt_stitch_ms", 0.0)
        trt_cpu_schedule_ms = diag_trt_tile_stats.get("trt_cpu_schedule_ms", 0.0)
        trt_cpu_ms = diag_trt_tile_stats.get("trt_cpu_ms", 0.0)
        trt_resolve_ms = diag_trt_tile_stats.get("trt_resolve_ms", 0.0)
        trt_avg_tile_batch = trt_tiles / trt_tile_batches if trt_tile_batches > 0 else 0.0
        trt_tile_ms_per_batch = trt_tile_ms / trt_tile_batches if trt_tile_batches > 0 else 0.0
        trt_frames = max(enh_frames, 1)
        infer_per_frame = avg_infer_ms / max(avg_batch, 1)
        trt_fetch_wait_ms_per_frame = trt_fetch_wait_ms / trt_frames
        trt_preprocess_ms_per_frame = trt_preprocess_ms / trt_frames
        trt_kernel_ms_per_frame = trt_tile_ms / trt_frames
        trt_stitch_ms_per_frame = trt_stitch_ms / trt_frames
        trt_post_ms_per_frame = trt_post_ms / trt_frames
        trt_resize_ms_per_frame = trt_resize_ms / trt_frames
        trt_pack_ms_per_frame = trt_pack_ms / trt_frames
        trt_cpu_schedule_ms_per_frame = trt_cpu_schedule_ms / trt_frames
        trt_cpu_ms_per_frame = trt_cpu_ms / trt_frames
        trt_resolve_ms_per_frame = trt_resolve_ms / trt_frames
        trt_other_ms_per_frame = max(
            0.0,
            infer_per_frame
            - trt_fetch_wait_ms_per_frame
            - trt_preprocess_ms_per_frame
            - trt_kernel_ms_per_frame
            - trt_stitch_ms_per_frame
            - trt_post_ms_per_frame
            - trt_resize_ms_per_frame
            - trt_pack_ms_per_frame
            - trt_cpu_schedule_ms_per_frame
            - trt_cpu_ms_per_frame,
            -trt_resolve_ms_per_frame,
        )

        gpu_line = format_gpu_summary(gpu_summary)
        pp_line = (
            f"  后处理:    平均 {avg_pp_ms:.1f}ms/帧  "
            f"face={avg_face_ms:.1f}ms  stabilize={avg_stabilize_ms:.1f}ms  "
            f"sharpen={avg_sharpen_ms:.1f}ms"
        ) if postprocess_needed else "  后处理: 未启用"
        trt_line = (
            f"  TRT tiles: {trt_tiles:.0f}个 / {trt_tile_batches:.0f}批  "
            f"平均 {trt_avg_tile_batch:.2f} tiles/批  峰值 {trt_tile_max_batch:.0f}  "
            f"engine_max_batch={trt_engine_max_batch:.0f}  TRT调用 {trt_tile_ms_per_batch:.1f}ms/批"
        ) if trt_tile_batches > 0 else "  TRT tiles: 未使用 TensorRT tile 批处理"
        trt_post_line = (
            f"  TRT拆分:    wait={trt_fetch_wait_ms_per_frame:.1f}ms/帧  "
            f"prep={trt_preprocess_ms_per_frame:.1f}ms/帧  "
            f"stitch={trt_stitch_ms_per_frame:.1f}ms/帧  "
            f"post={trt_post_ms_per_frame:.1f}ms/帧  "
            f"resize={trt_resize_ms_per_frame:.1f}ms/帧  "
            f"pack={trt_pack_ms_per_frame:.1f}ms/帧  "
            f"sched={trt_cpu_schedule_ms_per_frame:.1f}ms/帧  "
            f"GPU→CPU={trt_cpu_ms_per_frame:.1f}ms/帧"
        ) if trt_tile_batches > 0 else ""
        trt_other_line = (
            f"  TRT其他:    resolve={trt_resolve_ms_per_frame:.1f}ms/帧  "
            f"residual={trt_other_ms_per_frame:.1f}ms/帧"
        ) if trt_tile_batches > 0 else ""

        report_lines = [
            "═" * 46,
            "              [性能报告]",
            "═" * 46,
            f"  输出帧数:  {enh_frames}帧  耗时 {total_elapsed:.1f}s",
            f"  总体吞吐:  {total_fps:.2f} fps",
            f"  推理批次:  {enh_batches}批  平均 {avg_batch:.2f}帧/批",
            f"  推理延迟:  {avg_infer_ms:.1f}ms/批  ({infer_per_frame:.1f}ms/帧)",
            trt_line,
            trt_post_line,
            trt_other_line,
            f"  写入延迟:  {avg_write_ms:.2f}ms/帧  prepare={avg_write_prepare_ms:.2f}ms  pipe={avg_write_pipe_ms:.2f}ms",
            pp_line,
            f"  CPU占用:   进程平均 {cpu_total_pct:.0f}%  (按 {cpu_tracker.cpu_count} 核归一化)",
            f"  跳帧统计:  跳过 {skip_frames}帧 ({skip_pct:.1f}%)",
            "  " + "-" * 42,
            gpu_line,
            "═" * 46,
        ]
        for line in report_lines:
            _emit_progress(progress_callback, last_reported_progress, line)

    for thread in (enhance_thread, postprocess_thread, write_thread):
        if thread is not None:
            thread.join(timeout=10.0)

    if control is not None and control.is_cancelled:
        raise ExportCancelled("导出已被用户取消")

    if enhance_errors:
        raise enhance_errors[0]
    if stage3_errors:
        raise stage3_errors[0]
    if write_errors:
        raise write_errors[0]


def _report_stream_progress(
    processed_frames: int,
    total_frames: int | None,
    last_reported_progress: int,
    progress_callback: Callable[[int, str], None] | None,
    start_time: float = 0.0,
) -> int:
    progress = _map_frame_progress(processed_frames, total_frames)
    if progress > last_reported_progress:
        elapsed = time.perf_counter() - start_time if start_time > 0 else 0
        fps_str = f" ({processed_frames / elapsed:.2f} fps)" if elapsed > 1 else ""
        _emit_progress(
            progress_callback, progress,
            f"正在增强视频帧 {processed_frames}/{total_frames or '?'}{fps_str}",
        )
        return progress
    return last_reported_progress


def _finalize_stream_processes(decoder: object, encoder: object) -> None:
    stream_codec.finalize_stream_processes(decoder, encoder)


def _cleanup_stream_processes(decoder: object, encoder: object) -> None:
    stream_codec.cleanup_stream_processes(decoder, encoder)


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)


__all__ = [name for name in globals() if name.startswith("_")]