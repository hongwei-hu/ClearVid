from __future__ import annotations

import importlib
import queue
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from clearvid.app.models.codeformer_runner import CodeFormerRestorer
from clearvid.app.schemas.models import EnhancementConfig, TargetProfile, VideoMetadata
from clearvid.app.utils.subprocess_utils import run_command

DEFAULT_MODEL_FILENAME = "realesr-general-x4v3.pth"
DEFAULT_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
DEFAULT_MODEL_SCALE = 4


def find_realesrgan_weights(weights_path: Path | None) -> list[Path]:
    if not weights_path or not weights_path.exists():
        return []
    if weights_path.is_file() and weights_path.suffix.lower() == ".pth":
        return [weights_path]
    return sorted(path for path in weights_path.glob("*.pth") if path.is_file())


def inspect_realesrgan_runtime(weights_path: Path | None = None) -> tuple[bool, str, str | None, bool, bool]:
    torch_version: str | None = None
    cuda_available = False
    gpu_compatible = False

    try:
        import torch
    except ImportError as exc:
        return False, f"PyTorch is not installed: {exc}", torch_version, cuda_available, gpu_compatible

    torch_version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        return False, "CUDA is not available in the current PyTorch runtime.", torch_version, False, False

    try:
        capability = torch.cuda.get_device_capability(0)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        arch_list = set(torch.cuda.get_arch_list())
        gpu_compatible = required_arch in arch_list
    except Exception:  # noqa: BLE001
        gpu_compatible = False

    if not gpu_compatible:
        return (
            False,
            "Current PyTorch build does not support this GPU architecture. Install a build that supports RTX 5090.",
            torch_version,
            True,
            False,
        )

    try:
        _load_runtime_components()
    except ImportError as exc:
        return False, f"Real-ESRGAN is not installed: {exc}", torch_version, True, True
    except Exception as exc:  # noqa: BLE001
        return False, f"Real-ESRGAN 运行时初始化失败: {exc}", torch_version, True, True

    discovered_weights = find_realesrgan_weights(weights_path)
    if weights_path and not discovered_weights:
        return (
            True,
            f"Real-ESRGAN 运行时可用，缺少默认权重；首次运行时将自动下载到: {weights_path}",
            torch_version,
            True,
            True,
        )

    if discovered_weights:
        return True, f"Real-ESRGAN ready: {discovered_weights[0].name}", torch_version, True, True
    return True, "Real-ESRGAN runtime is available", torch_version, True, True


def validate_realesrgan_environment(weights_path: Path | None = None) -> tuple[bool, str]:
    available, message, _, _, _ = inspect_realesrgan_runtime(weights_path)
    return available, message


def run_realesrgan_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    progress_callback: Callable[[int, str], None] | None = None,
) -> None:
    _emit_progress(progress_callback, 6, "正在准备 Real-ESRGAN 权重")
    weights_dir = Path.cwd() / "weights" / "realesrgan"
    model_path = ensure_realesrgan_weights(weights_dir)
    _emit_progress(progress_callback, 10, "正在初始化 Real-ESRGAN")
    upsampler = _build_upsampler(config, model_path, metadata.width, metadata.height)
    _emit_progress(progress_callback, 14, "正在初始化人脸修复")
    codeformer_restorer = _build_codeformer_restorer(config, metadata, output_width, output_height)

    temp_root = Path(tempfile.mkdtemp(prefix="clearvid-realesrgan-"))
    temp_video_path = temp_root / "enhanced_video.mp4"

    try:
        outscale = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
        _stream_process_video(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            temp_video_path=temp_video_path,
            progress_callback=progress_callback,
        )
        _emit_progress(progress_callback, 96, "正在封装音频与元数据")
        _mux_output(config, temp_video_path)
        _emit_progress(progress_callback, 100, "视频导出完成")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def ensure_realesrgan_weights(weights_path: Path) -> Path:
    weights_path.mkdir(parents=True, exist_ok=True)
    discovered = find_realesrgan_weights(weights_path)
    if discovered:
        return discovered[0]

    _, _, load_file_from_url = _load_runtime_components()
    try:
        downloaded_path = load_file_from_url(
            url=DEFAULT_MODEL_URL,
            model_dir=str(weights_path),
            progress=True,
            file_name=DEFAULT_MODEL_FILENAME,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"自动下载 Real-ESRGAN 权重失败: {exc}") from exc

    model_path = Path(downloaded_path)
    if not model_path.exists():
        raise RuntimeError(f"Real-ESRGAN 权重下载后未找到: {model_path}")
    return model_path


def _load_runtime_components() -> tuple[type, type, object]:
    realesrgan_utils = importlib.import_module("realesrgan.utils")
    srvgg_arch = importlib.import_module("realesrgan.archs.srvgg_arch")
    download_util = importlib.import_module("basicsr.utils.download_util")
    return realesrgan_utils.RealESRGANer, srvgg_arch.SRVGGNetCompact, download_util.load_file_from_url


def _build_upsampler(
    config: EnhancementConfig, model_path: Path, input_width: int, input_height: int,
) -> object:
    real_esrganer_cls, srvgg_cls, _ = _load_runtime_components()
    model = srvgg_cls(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=32,
        upscale=DEFAULT_MODEL_SCALE,
        act_type="prelu",
    )
    tile = _resolve_tile_size(config.tile_size, input_width, input_height, config.fp16_enabled)
    return real_esrganer_cls(
        scale=DEFAULT_MODEL_SCALE,
        model_path=str(model_path),
        model=model,
        tile=tile,
        tile_pad=config.tile_pad,
        pre_pad=0,
        half=config.fp16_enabled,
    )


def _build_codeformer_restorer(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
) -> CodeFormerRestorer | None:
    if not config.face_restore_enabled:
        return None

    upscale_factor = _resolve_outscale(metadata, output_width, output_height, config.target_profile)
    return CodeFormerRestorer(
        fidelity_weight=config.face_restore_strength,
        upscale_factor=upscale_factor,
        weights_root=Path.cwd() / "weights",
    )


def _stream_process_video(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | None,
    temp_video_path: Path,
    progress_callback: Callable[[int, str], None] | None,
) -> None:
    import subprocess

    decode_command = _build_decode_command(config)
    encode_command = _build_encode_command(config, metadata, output_width, output_height, temp_video_path)
    decoder = subprocess.Popen(decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        _process_stream_frames(
            config=config,
            metadata=metadata,
            output_width=output_width,
            output_height=output_height,
            outscale=outscale,
            upsampler=upsampler,
            codeformer_restorer=codeformer_restorer,
            decoder=decoder,
            encoder=encoder,
            progress_callback=progress_callback,
        )
        _finalize_stream_processes(decoder, encoder)
    finally:
        _cleanup_stream_processes(decoder, encoder)


def _mux_output(config: EnhancementConfig, temp_video_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(temp_video_path),
        "-i",
        str(config.input_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
    ]

    if config.preserve_audio:
        command.extend(["-map", "1:a?", "-c:a", "copy"])
    else:
        command.append("-an")

    if config.preserve_subtitles:
        command.extend(["-map", "1:s?", "-c:s", "copy"])
    else:
        command.append("-sn")

    if config.preserve_metadata:
        command.extend(["-map_metadata", "1", "-map_chapters", "1"])
    else:
        command.extend(["-map_metadata", "-1", "-map_chapters", "-1"])

    command.extend(["-shortest", str(config.output_path)])
    run_command(command)


def _resolve_outscale(
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> float:
    if target_profile == TargetProfile.SOURCE:
        return 1.0
    if target_profile == TargetProfile.SCALE2X:
        return 2.0
    if target_profile == TargetProfile.SCALE4X:
        return 4.0
    return min(output_width / metadata.width, output_height / metadata.height)


def _resize_for_target(
    frame: np.ndarray,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
) -> np.ndarray:
    if target_profile in {TargetProfile.FHD, TargetProfile.UHD4K}:
        return _fit_and_pad_frame(frame, output_width, output_height)
    if frame.shape[1] == output_width and frame.shape[0] == output_height:
        return frame
    return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)


def _fit_and_pad_frame(frame: np.ndarray, output_width: int, output_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(output_width / width, output_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LANCZOS4)

    if resized_width == output_width and resized_height == output_height:
        return resized

    canvas = np.zeros((output_height, output_width, 3), dtype=frame.dtype)
    offset_x = (output_width - resized_width) // 2
    offset_y = (output_height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized
    return canvas


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


def _estimate_total_frames(metadata: VideoMetadata, preview_seconds: int | None) -> int | None:
    duration_seconds = preview_seconds or metadata.duration_seconds
    if duration_seconds <= 0 or metadata.fps <= 0:
        return None
    return max(1, int(round(duration_seconds * metadata.fps)))


def _map_frame_progress(processed_frames: int, total_frames: int | None) -> int:
    if not total_frames:
        return 18
    ratio = min(max(processed_frames / total_frames, 0.0), 1.0)
    return 18 + int(76 * ratio)


def _build_decode_command(config: EnhancementConfig) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "auto",
        "-i",
        str(config.input_path),
    ]
    if config.preview_seconds:
        command.extend(["-t", str(config.preview_seconds)])
    command.extend(["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    return command


def _build_encode_command(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    temp_video_path: Path,
) -> list[str]:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{output_width}x{output_height}",
        "-r",
        f"{metadata.fps:.6f}",
        "-i",
        "pipe:0",
        "-c:v",
        config.encoder,
        "-preset",
        config.encoder_preset,
    ]
    if config.video_bitrate:
        command.extend(["-b:v", config.video_bitrate])
    else:
        command.extend(["-cq", "18"])
    command.extend(["-pix_fmt", "yuv420p", "-an", str(temp_video_path)])
    return command


_FRAME_QUEUE_DEPTH = 8


def _resolve_tile_size(config_tile_size: int, width: int, height: int, fp16: bool) -> int:
    """Return effective tile size. 0 = auto based on VRAM."""
    if config_tile_size > 0:
        return config_tile_size
    try:
        import torch

        if not torch.cuda.is_available():
            return 512
        vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 512
    megapixels = (width * height) / 1_000_000
    estimated_mb = int(megapixels * (2000 if fp16 else 4000))
    if estimated_mb < vram_mb * 0.5:
        return 0
    return 512


def _start_decode_thread(
    decoder_stdout: object,
    frame_size: int,
) -> tuple[queue.Queue, list[BaseException], threading.Thread]:
    raw_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=_FRAME_QUEUE_DEPTH)
    errors: list[BaseException] = []

    def _loop() -> None:
        try:
            while True:
                raw = _read_exact_bytes(decoder_stdout, frame_size)
                if raw is None:
                    break
                raw_queue.put(raw)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            raw_queue.put(None)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return raw_queue, errors, thread


def _collect_batch(raw_queue: queue.Queue, batch_size: int) -> list[bytes]:
    batch: list[bytes] = []
    for _ in range(batch_size):
        raw = raw_queue.get()
        if raw is None:
            raw_queue.put(None)  # preserve sentinel for next caller
            break
        batch.append(raw)
    return batch


def _enhance_frames_batch(
    frames: list[np.ndarray],
    upsampler: object,
    outscale: float,
) -> list[np.ndarray]:
    """Batch-enhance multiple frames bypassing per-frame overhead."""
    import torch

    tensors = []
    for frame in frames:
        img = frame.astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float())

    batch = torch.stack(tensors).to(upsampler.device)
    if upsampler.half:
        batch = batch.half()

    with torch.inference_mode():
        output = upsampler.model(batch)

    results: list[np.ndarray] = []
    factor = outscale / upsampler.scale
    need_resize = abs(factor - 1.0) > 1e-6
    for i in range(output.shape[0]):
        out = output[i].float().cpu().clamp_(0, 1).numpy()
        out = np.transpose(out[[2, 1, 0], :, :], (1, 2, 0))
        if need_resize:
            h, w = out.shape[:2]
            out = cv2.resize(
                out, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_LANCZOS4,
            )
        results.append((out * 255.0).round().astype(np.uint8))
    return results


def _fetch_enhanced_frames(
    raw_queue: queue.Queue,
    use_batching: bool,
    batch_size: int,
    height: int,
    width: int,
    upsampler: object,
    outscale: float,
) -> list[np.ndarray] | None:
    """Fetch and enhance the next frame(s). Returns None at end of stream."""
    if use_batching:
        batch_raw = _collect_batch(raw_queue, batch_size)
        if not batch_raw:
            return None
        images = [np.frombuffer(r, dtype=np.uint8).reshape((height, width, 3)) for r in batch_raw]
        return _enhance_frames_batch(images, upsampler, outscale)

    raw_frame = raw_queue.get()
    if raw_frame is None:
        return None
    image = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))
    enhanced, _ = upsampler.enhance(image, outscale=outscale)
    return [enhanced]


def _write_enhanced_frames(
    enhanced_list: list[np.ndarray],
    codeformer_restorer: CodeFormerRestorer | None,
    output_width: int,
    output_height: int,
    target_profile: TargetProfile,
    encoder_stdin: object,
) -> int:
    """Post-process and write frames to encoder. Returns count written."""
    count = 0
    for enhanced in enhanced_list:
        if codeformer_restorer is not None:
            enhanced = codeformer_restorer.restore_faces(enhanced)
        finalized = _resize_for_target(enhanced, output_width, output_height, target_profile)
        encoder_stdin.write(np.ascontiguousarray(finalized).tobytes())
        count += 1
    return count


def _process_stream_frames(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    outscale: float,
    upsampler: object,
    codeformer_restorer: CodeFormerRestorer | None,
    decoder: object,
    encoder: object,
    progress_callback: Callable[[int, str], None] | None,
) -> None:
    frame_size = metadata.width * metadata.height * 3
    total_frames = _estimate_total_frames(metadata, config.preview_seconds)
    use_batching = getattr(upsampler, "tile_size", 0) == 0 and config.batch_size > 1

    assert decoder.stdout is not None
    assert encoder.stdin is not None
    _emit_progress(progress_callback, 18, "正在流式处理视频帧")

    raw_queue, decode_errors, reader = _start_decode_thread(decoder.stdout, frame_size)
    processed_frames = 0
    last_reported_progress = -1

    while True:
        enhanced_list = _fetch_enhanced_frames(
            raw_queue, use_batching, config.batch_size, metadata.height, metadata.width, upsampler, outscale,
        )
        if enhanced_list is None:
            break
        processed_frames += _write_enhanced_frames(
            enhanced_list, codeformer_restorer, output_width, output_height, config.target_profile, encoder.stdin,
        )
        last_reported_progress = _report_stream_progress(
            processed_frames, total_frames, last_reported_progress, progress_callback,
        )

    encoder.stdin.close()
    reader.join()
    if decode_errors:
        raise decode_errors[0]


def _report_stream_progress(
    processed_frames: int,
    total_frames: int | None,
    last_reported_progress: int,
    progress_callback: Callable[[int, str], None] | None,
) -> int:
    progress = _map_frame_progress(processed_frames, total_frames)
    if progress > last_reported_progress:
        _emit_progress(progress_callback, progress, f"正在增强视频帧 {processed_frames}/{total_frames or '?'}")
        return progress
    return last_reported_progress


def _finalize_stream_processes(decoder: object, encoder: object) -> None:
    decoder_stderr = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
    encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    decoder_return_code = decoder.wait()
    encoder_return_code = encoder.wait()
    if decoder_return_code != 0:
        raise RuntimeError(decoder_stderr.strip() or "FFmpeg 解码失败")
    if encoder_return_code != 0:
        raise RuntimeError(encoder_stderr.strip() or "FFmpeg 编码失败")


def _cleanup_stream_processes(decoder: object, encoder: object) -> None:
    if decoder.stdout:
        decoder.stdout.close()
    if decoder.stderr:
        decoder.stderr.close()
    if encoder.stdin and not encoder.stdin.closed:
        encoder.stdin.close()
    if encoder.stderr:
        encoder.stderr.close()
    if decoder.poll() is None:
        decoder.kill()
    if encoder.poll() is None:
        encoder.kill()


def _emit_progress(
    progress_callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)
