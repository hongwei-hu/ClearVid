"""First-time onboarding overlay — 3-step guide on initial launch."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui.user_settings import UserSettings

_STEPS = [
    ("① 拖入视频", "将视频文件拖到左侧面板，\n或点击「添加文件」按钮浏览选择。"),
    ("② 选择预设", "在右侧面板选择一个快速预设，\n或手动展开各参数面板微调。"),
    ("③ 点击导出", "确认输出路径后点击「开始导出」，\n导出完成后会收到通知。"),
]


class OnboardingOverlay(QWidget):
    """Semi-transparent overlay that displays a 3-step onboarding guide."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background: transparent;")
        self._step = 0
        self._build_ui()
        self._update_step()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Card container
        self._card = QWidget()
        self._card.setFixedSize(420, 260)
        self._card.setStyleSheet(
            "QWidget { background: #232340; border: 2px solid #4fc3f7; "
            "border-radius: 16px; }"
        )
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(32, 24, 32, 24)
        card_layout.setSpacing(12)

        self._title_label = QLabel()
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("color: #4fc3f7; font-size: 22px; font-weight: bold; border: none;")
        card_layout.addWidget(self._title_label)

        self._desc_label = QLabel()
        self._desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #ccc; font-size: 14px; border: none;")
        card_layout.addWidget(self._desc_label)

        card_layout.addStretch()

        # Navigation row
        nav = QHBoxLayout()
        self._skip_btn = QPushButton("跳过引导")
        self._skip_btn.setStyleSheet("color: #888; border: 1px solid #555; padding: 6px 16px; border-radius: 6px;")
        self._skip_btn.clicked.connect(self._finish)
        nav.addWidget(self._skip_btn)

        nav.addStretch()

        # Step dots
        self._dots = QLabel()
        self._dots.setStyleSheet("color: #4fc3f7; font-size: 16px; border: none;")
        nav.addWidget(self._dots)

        nav.addStretch()

        self._next_btn = QPushButton("下一步 →")
        self._next_btn.setStyleSheet(
            "background: #4fc3f7; color: #1a1a2e; font-weight: bold; "
            "padding: 6px 20px; border-radius: 6px; border: none;"
        )
        self._next_btn.clicked.connect(self._next)
        nav.addWidget(self._next_btn)

        card_layout.addLayout(nav)
        layout.addWidget(self._card, alignment=Qt.AlignmentFlag.AlignCenter)

    def _update_step(self) -> None:
        title, desc = _STEPS[self._step]
        self._title_label.setText(title)
        self._desc_label.setText(desc)

        dots = "  ".join(
            "●" if i == self._step else "○" for i in range(len(_STEPS))
        )
        self._dots.setText(dots)

        if self._step == len(_STEPS) - 1:
            self._next_btn.setText("开始使用 ✓")
        else:
            self._next_btn.setText("下一步 →")

    def _next(self) -> None:
        if self._step < len(_STEPS) - 1:
            self._step += 1
            self._update_step()
        else:
            self._finish()

    def _finish(self) -> None:
        settings = UserSettings()
        settings.set_onboarding_shown(True)
        self.hide()
        self.deleteLater()

    # Draw semi-transparent backdrop
    def paintEvent(self, event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
        painter.end()

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        self.setGeometry(self.parent().rect())
