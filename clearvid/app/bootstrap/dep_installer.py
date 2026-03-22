"""Dependency installer with real-time progress reporting.

Uses ``pip install`` under the hood and streams output line-by-line to a
caller-supplied callback so a GUI or console can display a progress bar.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from clearvid.app.bootstrap.paths import APP_ROOT, LIB_DIR, write_lib_version

# ---------------------------------------------------------------------------
# Dependency groups — mirroring pyproject.toml optional groups
# ---------------------------------------------------------------------------

_CORE_DEPS = [
    "pydantic>=2.10,<3.0",
    "pyyaml>=6.0.2,<7.0",
    "rich>=13.9,<15.0",
    "typer>=0.15,<1.0",
]

_GUI_DEPS = [
    "PySide6>=6.8,<7.0",
]

_MEDIA_DEPS = [
    "numpy>=2.2,<3.0",
    "opencv-python>=4.10,<5.0",
]

# torch / torchvision are handled separately — their index-url varies by GPU.
_INFERENCE_DEPS = [
    "basicsr>=1.4.2",
    "facexlib>=0.3.0",
    "realesrgan>=0.3.0",
    "gfpgan>=1.3.8",
]

# Current lib stamp version — bump when dependency set changes.
LIB_VERSION = "1.0.0"


@dataclass
class InstallPlan:
    """What needs to be installed and how."""

    torch_index_url: str = "https://download.pytorch.org/whl/cu128"
    torch_label: str = "CUDA 12.8"
    target_dir: Path = field(default_factory=lambda: LIB_DIR)
    python_exe: str = field(default_factory=lambda: sys.executable)


def build_install_steps(plan: InstallPlan) -> list[tuple[str, list[str]]]:
    """Return a list of ``(description, pip_args)`` tuples."""
    target = str(plan.target_dir)
    python = plan.python_exe
    steps: list[tuple[str, list[str]]] = []

    # 1. Core + GUI + media
    steps.append((
        "安装基础依赖 (PySide6, numpy, opencv ...)",
        [
            python, "-m", "pip", "install",
            "--target", target,
            "--no-warn-script-location",
            "--disable-pip-version-check",
            *_CORE_DEPS, *_GUI_DEPS, *_MEDIA_DEPS,
        ],
    ))

    # 2. PyTorch (index-url specific to CUDA version)
    steps.append((
        f"安装 PyTorch ({plan.torch_label})",
        [
            python, "-m", "pip", "install",
            "--target", target,
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--index-url", plan.torch_index_url,
            "torch>=2.6,<3.0",
            "torchvision>=0.21,<1.0",
        ],
    ))

    # 3. AI model frameworks (depend on torch)
    steps.append((
        "安装 AI 模型框架 (Real-ESRGAN, CodeFormer ...)",
        [
            python, "-m", "pip", "install",
            "--target", target,
            "--no-warn-script-location",
            "--disable-pip-version-check",
            *_INFERENCE_DEPS,
        ],
    ))

    # 4. Install ClearVid itself as editable or as package
    pyproject = APP_ROOT / "pyproject.toml"
    if pyproject.exists():
        steps.append((
            "安装 ClearVid 主程序",
            [
                python, "-m", "pip", "install",
                "--target", target,
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "--no-deps",
                str(APP_ROOT),
            ],
        ))

    return steps


def run_install(
    plan: InstallPlan,
    *,
    on_step: callable | None = None,
    on_output: callable | None = None,
    on_error: callable | None = None,
) -> bool:
    """Execute the install plan step by step.

    Callbacks:
        on_step(step_index, total_steps, description)
        on_output(line)   — each stdout/stderr line from pip
        on_error(step_index, description, return_code)

    Returns True if all steps succeeded.
    """
    steps = build_install_steps(plan)
    total = len(steps)

    plan.target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure target dir is on sys.path so subsequent imports work
    target_str = str(plan.target_dir)
    if target_str not in sys.path:
        sys.path.insert(0, target_str)

    for idx, (desc, cmd) in enumerate(steps):
        if on_step:
            on_step(idx, total, desc)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if on_output and line:
                    on_output(line)
            proc.wait()
        except Exception as exc:  # noqa: BLE001
            if on_error:
                on_error(idx, desc, -1)
            if on_output:
                on_output(f"[错误] {exc}")
            return False

        if proc.returncode != 0:
            if on_error:
                on_error(idx, desc, proc.returncode)
            return False

    # Stamp version
    write_lib_version(LIB_VERSION)
    return True
