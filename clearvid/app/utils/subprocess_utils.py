from __future__ import annotations

import subprocess
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
