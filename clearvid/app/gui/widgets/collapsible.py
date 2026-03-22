"""Collapsible section widget for accordion-style settings panels."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """A section with a clickable header that toggles content visibility.

    Usage::

        section = CollapsibleSection("画质增强", name="enhancement", expanded=False)
        section.content_layout.addWidget(my_widget)
    """

    toggled = Signal(str, bool)  # (section_name, is_expanded)

    def __init__(
        self,
        title: str,
        name: str = "",
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name = name or title
        self._expanded = expanded

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setText(f"  {title}")
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setMinimumHeight(32)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setStyleSheet(
            "QToolButton { text-align: left; padding-left: 4px; width: 100%; }"
        )
        self._toggle.setSizePolicy(
            self._toggle.sizePolicy().horizontalPolicy(),
            self._toggle.sizePolicy().verticalPolicy(),
        )
        layout.addWidget(self._toggle)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # Content area
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(6)
        self._content.setVisible(expanded)
        layout.addWidget(self._content)

        self._toggle.clicked.connect(self._on_toggled)

    @property
    def name(self) -> str:
        return self._name

    @property
    def content_layout(self) -> QVBoxLayout:
        """Layout inside the collapsible content area. Add widgets here."""
        return self._content_layout

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._toggle.setChecked(expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _on_toggled(self) -> None:
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        self.toggled.emit(self._name, self._expanded)
