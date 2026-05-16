from __future__ import annotations

import subprocess
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clearvid.app.bootstrap.paths import ffmpeg_path
from clearvid.app.preprocess.filters import build_preprocess_filters
from clearvid.app.schemas.models import EnhancementConfig, VideoMetadata
from clearvid.app.utils.subprocess_utils import run_command

PopenFactory = Callable[..., subprocess.Popen[bytes]]
RunFactory = Callable[..., subprocess.CompletedProcess[bytes]]
CommandRunner = Callable[[list[str]], object]
_STDERR_TAIL_LIMIT = 64 * 1024


class _StderrDrain:
    """Continuously drains a process stderr pipe and keeps the diagnostic tail."""

    def __init__(self, stream: object) -> None:
        self._stream = stream
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._thread = threading.Thread(target=self._drain, daemon=True, name="ffmpeg-stderr")
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.readline()
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="replace")
                self._append(bytes(chunk))
        except Exception:
            return

    def _append(self, chunk: bytes) -> None:
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self._size > _STDERR_TAIL_LIMIT and self._chunks:
            self._size -= len(self._chunks.popleft())

    def text(self) -> str:
        self._thread.join(timeout=1.0)
        return b"".join(self._chunks).decode("utf-8", errors="replace")


def build_decode_command(config: EnhancementConfig, metadata: VideoMetadata) -> list[str]:
    command = [
        ffmpeg_path() or "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "auto",
        "-threads",
        "0",
        "-i",
        str(config.input_path),
    ]
    if config.preview_seconds:
        command.extend(["-t", str(config.preview_seconds)])

    vf_filters = build_preprocess_filters(config, metadata)
    if vf_filters:
        command.extend(["-vf", ",".join(vf_filters)])

    command.extend(["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    return command


def build_encode_command(
    config: EnhancementConfig,
    metadata: VideoMetadata,
    output_width: int,
    output_height: int,
    temp_video_path: Path,
) -> list[str]:
    command = [
        ffmpeg_path() or "ffmpeg",
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
    ]
    if config.encoder == "av1_nvenc":
        command.extend(["-preset", config.encoder_preset])
        command.extend(["-tier", "1"])
    else:
        command.extend(["-preset", config.encoder_preset])

    if config.encoder_crf is not None:
        command.extend(["-cq", str(config.encoder_crf)])
    elif config.video_bitrate:
        command.extend(["-b:v", config.video_bitrate])
    else:
        command.extend(["-cq", "18"])

    command.extend(["-pix_fmt", config.output_pixel_format])
    command.extend(["-movflags", "frag_keyframe+empty_moov"])
    command.extend(["-an", str(temp_video_path)])
    return command


def start_stream_processes(
    decode_command: list[str],
    encode_command: list[str],
    popen_factory: PopenFactory = subprocess.Popen,
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]:
    decoder: subprocess.Popen[bytes] | None = None
    try:
        decoder = popen_factory(decode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = popen_factory(encode_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        if decoder is not None and decoder.poll() is None:
            decoder.kill()
            decoder.wait()
        executable = decode_command[0] if isinstance(exc, FileNotFoundError) else "FFmpeg"
        raise RuntimeError(f"无法启动 {executable}: {exc}") from exc

    if decoder.stdout is None or decoder.stderr is None or encoder.stdin is None or encoder.stderr is None:
        cleanup_stream_processes(decoder, encoder)
        raise RuntimeError("FFmpeg 管道初始化失败")
    _attach_stderr_drain(decoder)
    _attach_stderr_drain(encoder)
    return decoder, encoder


def mux_preview(
    config: EnhancementConfig,
    temp_video_path: Path,
    preview_path: Path,
    duration_sec: float,
    run_factory: RunFactory = subprocess.run,
) -> bool:
    command = [
        ffmpeg_path() or "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(temp_video_path),
        "-i", str(config.input_path),
        "-t", f"{duration_sec:.3f}",
        "-map", "0:v:0",
        "-c:v", "copy",
    ]
    if config.preserve_audio:
        command.extend(["-map", "1:a?", "-c:a", "copy"])
    else:
        command.append("-an")
    command.extend(["-sn", "-map_metadata", "-1", "-shortest", str(preview_path)])
    try:
        run_factory(command, capture_output=True, check=True, timeout=30)
        return True
    except Exception:  # noqa: BLE001
        return False


def mux_output(
    config: EnhancementConfig,
    temp_video_path: Path,
    command_runner: CommandRunner = run_command,
) -> None:
    command = [
        ffmpeg_path() or "ffmpeg",
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
    command_runner(command)


def extract_frame(
    video_path: Path,
    timestamp_sec: float = 0.0,
    width: int = 0,
    height: int = 0,
    run_factory: RunFactory = subprocess.run,
) -> np.ndarray:
    if width <= 0 or height <= 0:
        from clearvid.app.io.probe import probe_video

        meta = probe_video(video_path)
        width, height = meta.width, meta.height

    command = [
        ffmpeg_path() or "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-hwaccel", "auto",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "pipe:1",
    ]
    result = run_factory(command, capture_output=True, check=True, timeout=30)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape((height, width, 3))


def finalize_stream_processes(decoder: object, encoder: object) -> None:
    decoder_return_code = decoder.wait()
    encoder_return_code = encoder.wait()
    decoder_stderr = _stderr_text(decoder)
    encoder_stderr = _stderr_text(encoder)
    if decoder_return_code != 0:
        raise RuntimeError(decoder_stderr.strip() or "FFmpeg 解码失败")
    if encoder_return_code != 0:
        raise RuntimeError(encoder_stderr.strip() or "FFmpeg 编码失败")


def cleanup_stream_processes(decoder: object, encoder: object) -> None:
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


def _attach_stderr_drain(process: object) -> None:
    stream = getattr(process, "stderr", None)
    if stream is None:
        return
    try:
        setattr(process, "_clearvid_stderr_drain", _StderrDrain(stream))
    except Exception:
        return


def _stderr_text(process: object) -> str:
    drain = getattr(process, "_clearvid_stderr_drain", None)
    if drain is not None:
        return drain.text()
    stream = getattr(process, "stderr", None)
    if stream is None:
        return ""
    data = stream.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)
