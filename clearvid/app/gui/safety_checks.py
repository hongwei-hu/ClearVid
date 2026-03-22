"""Safety checks: overwrite protection and disk space pre-check."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget


def check_overwrite(output_path: str | Path, parent: QWidget | None = None) -> bool:
    """Return True if it's safe to proceed (file doesn't exist or user confirmed).

    Shows a confirmation dialog if the output file already exists.
    """
    p = Path(output_path)
    if not p.exists():
        return True
    reply = QMessageBox.question(
        parent,
        "文件已存在",
        f"输出文件已存在:\n{p.name}\n\n是否覆盖？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes


def check_disk_space(
    output_path: str | Path,
    required_mb: float,
    parent: QWidget | None = None,
) -> bool:
    """Return True if there's enough disk space (or user chose to proceed anyway).

    Shows a warning if free space is less than *required_mb*.
    """
    p = Path(output_path)
    target_dir = p.parent if p.parent.exists() else Path.cwd()
    try:
        usage = shutil.disk_usage(str(target_dir))
        free_mb = usage.free / (1024 * 1024)
    except OSError:
        return True  # Can't check → don't block

    if free_mb >= required_mb:
        return True

    reply = QMessageBox.warning(
        parent,
        "磁盘空间不足",
        (
            f"输出目录 ({target_dir.drive or str(target_dir)}) "
            f"剩余 {free_mb:.0f} MB，\n"
            f"预计输出需要 {required_mb:.0f} MB。\n\n"
            "仍然继续？"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return reply == QMessageBox.StandardButton.Yes
