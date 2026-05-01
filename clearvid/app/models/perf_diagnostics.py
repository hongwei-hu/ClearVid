from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any


class CpuUsageTracker:
    """Track process CPU usage normalized by logical CPU count."""

    def __init__(self) -> None:
        self._start_cpu = time.process_time()
        self._last_cpu = self._start_cpu
        self._last_wall = time.perf_counter()
        self._cpu_count = max(os.cpu_count() or 1, 1)

    @property
    def cpu_count(self) -> int:
        return self._cpu_count

    def sample_percent(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        cpu_now = time.process_time()
        wall_delta = max(now - self._last_wall, 1e-6)
        percent = (cpu_now - self._last_cpu) / wall_delta * 100.0 / self._cpu_count
        self._last_cpu = cpu_now
        self._last_wall = now
        return percent

    def total_percent(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            return 0.0
        return (time.process_time() - self._start_cpu) / elapsed_seconds * 100.0 / self._cpu_count


class GpuSampler:
    """Lightweight GPU metrics collector via ``nvidia-smi``.

    The sampler matches the nvidia-smi row to PyTorch's current CUDA device UUID
    when possible. It no-ops quietly on systems without nvidia-smi.
    """

    _QUERY = (
        "index,name,uuid,pci.bus_id,"
        "utilization.gpu,utilization.memory,"
        "memory.used,memory.total,"
        "temperature.gpu,"
        "clocks.current.sm,clocks.max.sm,"
        "power.draw"
    )
    _MAX_SAMPLES = 120

    def __init__(self, interval: float = 5.0) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._samples: list[dict[str, Any]] = []
        self._target_uuid = self._detect_torch_device_uuid()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="gpu-sampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._samples[-1]) if self._samples else None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return {}

        def _agg(key: str) -> tuple[float, float, float]:
            values = [float(sample[key]) for sample in samples if key in sample]
            if not values:
                return 0.0, 0.0, 0.0
            return min(values), sum(values) / len(values), max(values)

        _, gpu_avg, gpu_max = _agg("gpu_util")
        _, mem_avg, mem_max = _agg("mem_used_mb")
        _, temp_avg, temp_max = _agg("temperature")
        _, power_avg, power_max = _agg("power_w")
        _, sm_avg, sm_max = _agg("sm_clock")
        first = samples[0]
        return {
            "gpu_avg": gpu_avg,
            "gpu_max": gpu_max,
            "mem_avg_mb": mem_avg,
            "mem_peak_mb": mem_max,
            "mem_total_mb": first.get("mem_total_mb", 0),
            "temp_avg": temp_avg,
            "temp_max": temp_max,
            "power_avg_w": power_avg,
            "power_max_w": power_max,
            "sm_clock_avg": sm_avg,
            "sm_clock_max": sm_max,
            "sample_count": len(samples),
            "gpu_index": first.get("index", "?"),
            "gpu_name": first.get("name", "?"),
            "gpu_uuid": first.get("uuid", "?"),
            "pci_bus_id": first.get("pci_bus_id", "?"),
        }

    @staticmethod
    def _detect_torch_device_uuid() -> str | None:
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            uuid = getattr(props, "uuid", None)
            return str(uuid) if uuid else None
        except Exception:  # noqa: BLE001
            return None

    def _poll_loop(self) -> None:
        if not self._do_sample():
            return
        while not self._stop.wait(self._interval):
            self._do_sample()

    def _do_sample(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + self._QUERY, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5.0,
            )
            if result.returncode != 0:
                return False
            lines = result.stdout.strip().splitlines()
            if not lines:
                return False
            parts = [part.strip() for part in self._select_gpu_line(lines).split(",")]
            if len(parts) < 12:
                return False
            sample = {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "pci_bus_id": parts[3],
                "gpu_util": float(parts[4]),
                "mem_util": float(parts[5]),
                "mem_used_mb": float(parts[6]),
                "mem_total_mb": float(parts[7]),
                "temperature": float(parts[8]),
                "sm_clock": float(parts[9]),
                "sm_clock_max": float(parts[10]),
                "power_w": float(parts[11]),
            }
            with self._lock:
                if len(self._samples) >= self._MAX_SAMPLES:
                    self._samples.pop(0)
                self._samples.append(sample)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _select_gpu_line(self, lines: list[str]) -> str:
        if self._target_uuid:
            target_uuid = self._target_uuid.replace("GPU-", "")
            for line in lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 3 and parts[2].replace("GPU-", "") == target_uuid:
                    return line
        return lines[0]


def format_queue_info(raw_depth: int, enhanced_depth: int, finalized_depth: int, frame_queue_depth: int, enhanced_queue_depth: int) -> str:
    raw_pct = raw_depth * 100 // max(frame_queue_depth, 1)
    finalized_pct = finalized_depth * 100 // max(enhanced_queue_depth, 1)
    info = f"队列: 解码={raw_depth}/{frame_queue_depth}({raw_pct}%)"
    if enhanced_depth >= 0:
        enhanced_pct = enhanced_depth * 100 // max(enhanced_queue_depth, 1)
        info += f" 增强={enhanced_depth}/{enhanced_queue_depth}({enhanced_pct}%)"
    return info + f" 写入={finalized_depth}/{enhanced_queue_depth}({finalized_pct}%)"


def format_gpu_snapshot(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return ""
    return (
        f" | GPU {float(snapshot['gpu_util']):.0f}%"
        f" 显存 {float(snapshot['mem_used_mb']) / 1024:.1f}/{float(snapshot['mem_total_mb']) / 1024:.0f}GB"
        f" {float(snapshot['temperature']):.0f}°C"
        f" {float(snapshot['power_w']):.0f}W"
        f" SM{float(snapshot['sm_clock']):.0f}MHz"
    )


def format_gpu_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "  GPU: nvidia-smi 未就绪，无GPU指标"
    uuid_short = str(summary["gpu_uuid"]).replace("GPU-", "")[:8]
    return (
        f"  GPU采样:   index={summary['gpu_index']}  {summary['gpu_name']}  "
        f"uuid={uuid_short}  bus={summary['pci_bus_id']}\n"
        f"  GPU利用率: 平均 {float(summary['gpu_avg']):.0f}%  峰值 {float(summary['gpu_max']):.0f}%\n"
        f"  显存占用:  平均 {float(summary['mem_avg_mb']) / 1024:.1f}GB  "
        f"峰值 {float(summary['mem_peak_mb']) / 1024:.1f}GB  "
        f"总计 {float(summary['mem_total_mb']) / 1024:.0f}GB\n"
        f"  GPU温度:   平均 {float(summary['temp_avg']):.0f}°C  峰值 {float(summary['temp_max']):.0f}°C\n"
        f"  功耗:      平均 {float(summary['power_avg_w']):.0f}W  峰值 {float(summary['power_max_w']):.0f}W\n"
        f"  SM时钟:    平均 {float(summary['sm_clock_avg']):.0f}MHz  "
        f"(最大 {float(summary['sm_clock_max']):.0f}MHz)  "
        f"采样次数={summary['sample_count']}"
    )
