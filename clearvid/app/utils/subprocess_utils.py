from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path


class ProcessError(RuntimeError):
    pass


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ProcessError(completed.stderr.strip() or completed.stdout.strip() or "Command failed")
    return completed


def run_ffmpeg_with_progress(
    command: list[str],
    duration_seconds: float | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    progress_message: str = "正在处理视频",
    progress_start: int = 0,
    progress_end: int = 100,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if len(command) < 2:
        return run_command(command, cwd=cwd)

    effective_command = _build_ffmpeg_progress_command(command)
    process = subprocess.Popen(
        effective_command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout_chunks: list[str] = []
    if progress_callback:
        progress_callback(progress_start, progress_message)

    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if not line:
            continue

        stdout_chunks.append(line)
        _report_ffmpeg_progress(
            line=line,
            duration_seconds=duration_seconds,
            progress_callback=progress_callback,
            progress_message=progress_message,
            progress_start=progress_start,
            progress_end=progress_end,
        )

    stderr_text = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    stdout_text = "".join(stdout_chunks)
    if return_code != 0:
        raise ProcessError(stderr_text.strip() or stdout_text.strip() or "Command failed")

    if progress_callback:
        progress_callback(progress_end, progress_message)

    return subprocess.CompletedProcess(
        args=effective_command,
        returncode=return_code,
        stdout=stdout_text,
        stderr=stderr_text,
    )


def _build_ffmpeg_progress_command(command: list[str]) -> list[str]:
    return command[:-1] + ["-progress", "pipe:1", "-nostats", command[-1]]


def _report_ffmpeg_progress(
    line: str,
    duration_seconds: float | None,
    progress_callback: Callable[[int, str], None] | None,
    progress_message: str,
    progress_start: int,
    progress_end: int,
) -> None:
    if not progress_callback or not duration_seconds or duration_seconds <= 0:
        return

    key, _, value = line.strip().partition("=")
    if key != "out_time_ms":
        return

    try:
        elapsed_seconds = int(value) / 1_000_000
    except ValueError:
        return

    ratio = min(max(elapsed_seconds / duration_seconds, 0.0), 1.0)
    mapped_progress = progress_start + int((progress_end - progress_start) * ratio)
    progress_callback(mapped_progress, progress_message)
