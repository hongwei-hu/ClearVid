"""Processing history: record and browse past export results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui.user_settings import UserSettings

_HISTORY_FILE = Path.home() / ".clearvid" / "history.json"
_MAX_HISTORY = 200


@dataclass
class HistoryRecord:
    """A single processing result entry."""

    timestamp: str
    input_path: str
    output_path: str
    profile: str = ""
    quality_mode: str = ""
    elapsed_sec: float = 0.0
    success: bool = True
    error: str = ""

    @staticmethod
    def now(
        input_path: str,
        output_path: str,
        profile: str = "",
        quality_mode: str = "",
        elapsed_sec: float = 0.0,
        success: bool = True,
        error: str = "",
    ) -> HistoryRecord:
        return HistoryRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_path=input_path,
            output_path=output_path,
            profile=profile,
            quality_mode=quality_mode,
            elapsed_sec=elapsed_sec,
            success=success,
            error=error,
        )


def load_history() -> list[HistoryRecord]:
    if not _HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
        return [HistoryRecord(**r) for r in data[-_MAX_HISTORY:]]
    except Exception:  # noqa: BLE001
        return []


def save_history(records: list[HistoryRecord]) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in records[-_MAX_HISTORY:]]
    _HISTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_history(record: HistoryRecord) -> None:
    records = load_history()
    records.append(record)
    save_history(records)


class HistoryDialog(QDialog):
    """Dialog showing past processing records in a table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("处理历史")
        self.resize(780, 460)
        self.setModal(True)

        layout = QVBoxLayout(self)

        records = load_history()

        if not records:
            layout.addWidget(QLabel("暂无处理记录。"))
        else:
            headers = ["时间", "输入文件", "输出文件", "规格", "质量", "耗时", "状态"]
            table = QTableWidget(len(records), len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)

            for row, rec in enumerate(reversed(records)):
                table.setItem(row, 0, QTableWidgetItem(rec.timestamp))
                table.setItem(row, 1, QTableWidgetItem(Path(rec.input_path).name))
                table.setItem(row, 2, QTableWidgetItem(Path(rec.output_path).name))
                table.setItem(row, 3, QTableWidgetItem(rec.profile))
                table.setItem(row, 4, QTableWidgetItem(rec.quality_mode))
                elapsed_text = f"{rec.elapsed_sec:.1f}s" if rec.elapsed_sec else ""
                table.setItem(row, 5, QTableWidgetItem(elapsed_text))
                status_text = "✅" if rec.success else f"❌ {rec.error}"
                table.setItem(row, 6, QTableWidgetItem(status_text))

            layout.addWidget(table)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        if records:
            clear_btn = QPushButton("清除历史")
            clear_btn.clicked.connect(self._clear_history)
            btn_row.addWidget(clear_btn)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _clear_history(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        answer = QMessageBox.question(
            self,
            "清除历史",
            "确定要清除所有处理记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            save_history([])
            self.accept()
