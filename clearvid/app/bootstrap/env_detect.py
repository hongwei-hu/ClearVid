"""GPU and system environment detection for first-time setup."""

from __future__ import annotations

import re
import subprocess
import shutil
from dataclasses import dataclass


@dataclass
class GpuInfo:
    """Detected GPU information."""

    name: str | None = None
    driver_version: str | None = None
    memory_mb: int | None = None
    cuda_capable: bool = False
    recommended_torch_index: str = ""
    recommended_label: str = ""


def detect_gpu() -> GpuInfo:
    """Detect NVIDIA GPU via nvidia-smi.  Returns GpuInfo with recommendation."""
    info = GpuInfo()

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        info.recommended_torch_index = "https://download.pytorch.org/whl/cpu"
        info.recommended_label = "CPU (无 NVIDIA GPU)"
        return info

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
        line = (result.stdout or "").strip().splitlines()[0] if result.stdout else ""
    except Exception:  # noqa: BLE001
        info.recommended_torch_index = "https://download.pytorch.org/whl/cpu"
        info.recommended_label = "CPU (nvidia-smi 失败)"
        return info

    if not line:
        info.recommended_torch_index = "https://download.pytorch.org/whl/cpu"
        info.recommended_label = "CPU (无法检测 GPU)"
        return info

    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 3:
        info.name = parts[0]
        info.driver_version = parts[1]
        try:
            info.memory_mb = int(float(parts[2]))
        except (ValueError, TypeError):
            pass

    info.cuda_capable = True
    info.recommended_torch_index, info.recommended_label = _pick_torch_variant(
        info.driver_version
    )
    return info


def _pick_torch_variant(driver_version: str | None) -> tuple[str, str]:
    """Choose the best CUDA variant based on the NVIDIA driver version.

    Driver compatibility table (minimum driver for each CUDA):
        CUDA 12.8 → driver ≥ 570
        CUDA 12.6 → driver ≥ 560
        CUDA 12.4 → driver ≥ 550
        CUDA 12.1 → driver ≥ 530
        CUDA 11.8 → driver ≥ 520
    """
    if not driver_version:
        return "https://download.pytorch.org/whl/cu128", "CUDA 12.8 (默认)"

    match = re.match(r"(\d+)", driver_version)
    major = int(match.group(1)) if match else 0

    if major >= 570:
        return "https://download.pytorch.org/whl/cu128", f"CUDA 12.8 (驱动 {driver_version})"
    if major >= 560:
        return "https://download.pytorch.org/whl/cu126", f"CUDA 12.6 (驱动 {driver_version})"
    if major >= 530:
        return "https://download.pytorch.org/whl/cu121", f"CUDA 12.1 (驱动 {driver_version})"
    if major >= 520:
        return "https://download.pytorch.org/whl/cu118", f"CUDA 11.8 (驱动 {driver_version})"

    return (
        "https://download.pytorch.org/whl/cpu",
        f"CPU (驱动 {driver_version} 过旧，建议升级至 ≥535)",
    )


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is reachable."""
    from clearvid.app.bootstrap.paths import ffmpeg_path

    return ffmpeg_path() is not None
