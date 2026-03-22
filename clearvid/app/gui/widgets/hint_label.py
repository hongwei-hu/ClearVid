"""Info-button widget: small ℹ️ icon that shows a detail popup on click."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolTip, QWidget


class InfoButton(QPushButton):
    """A small ℹ️ button that shows a tooltip-style popup with detailed help text."""

    def __init__(self, detail_text: str, parent: QWidget | None = None) -> None:
        super().__init__("ℹ️", parent)
        self._detail = detail_text
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setToolTip("点击查看详细说明")
        self.setStyleSheet(
            "QPushButton { border: none; background: transparent; font-size: 13px; padding: 0; }"
            "QPushButton:hover { background: rgba(79,195,247,0.2); border-radius: 4px; }"
        )
        self.clicked.connect(self._show_detail)

    def _show_detail(self) -> None:
        pos = self.mapToGlobal(self.rect().bottomLeft())
        QToolTip.showText(pos, self._detail, self, self.rect(), 8000)


def labeled_row_with_info(
    label_text: str,
    widget: QWidget,
    tooltip: str = "",
    detail: str = "",
) -> QHBoxLayout:
    """Create ``Label | Widget | ℹ️`` row.

    * *tooltip*: shown on hover over label and widget (brief one-liner).
    * *detail*: shown when ℹ️ button is clicked (multi-line explanation).
      If *detail* is empty, falls back to *tooltip* for the popup content.
    """
    row = QHBoxLayout()
    lbl = QLabel(label_text)
    lbl.setMinimumWidth(80)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    row.addWidget(lbl)
    row.addWidget(widget, 1)

    if detail or tooltip:
        info_btn = InfoButton(detail or tooltip)
        row.addWidget(info_btn)

    return row
