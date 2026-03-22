"""Preset cards widget: clickable cards for quick parameter presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class Preset:
    """Definition of a single preset card."""

    key: str
    icon: str
    name: str
    desc: str
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

BUILTIN_PRESETS: list[Preset] = [
    Preset(
        key="portrait",
        icon="👤",
        name="人像优先",
        desc="优化人脸，适合 Vlog",
        params={
            "quality_mode": "quality",
            "upscale_model": "x4plus",
            "face_restore_enabled": True,
            "face_restore_strength": 0.70,
            "face_restore_model": "codeformer",
            "face_poisson_blend": True,
            "sharpen_enabled": True,
            "sharpen_strength": 0.12,
            "temporal_stabilize_enabled": True,
        },
    ),
    Preset(
        key="landscape",
        icon="🏔️",
        name="风景建筑",
        desc="强化纹理，适合旅拍",
        params={
            "quality_mode": "quality",
            "upscale_model": "x4plus",
            "face_restore_enabled": False,
            "sharpen_enabled": True,
            "sharpen_strength": 0.20,
            "temporal_stabilize_enabled": True,
        },
    ),
    Preset(
        key="vintage",
        icon="📼",
        name="老旧视频",
        desc="全面修复，强降噪去块",
        params={
            "quality_mode": "quality",
            "upscale_model": "x4plus",
            "face_restore_enabled": True,
            "face_restore_strength": 0.55,
            "face_restore_model": "codeformer",
            "sharpen_enabled": True,
            "sharpen_strength": 0.15,
            "temporal_stabilize_enabled": True,
            "temporal_stabilize_strength": 0.80,
            "preprocess_denoise": True,
            "preprocess_deblock": True,
            "preprocess_deinterlace": True,
            "preprocess_colorspace": True,
        },
    ),
    Preset(
        key="fast",
        icon="⚡",
        name="快速预览",
        desc="最快速度确认效果",
        params={
            "quality_mode": "fast",
            "upscale_model": "general_v3",
            "face_restore_enabled": False,
            "sharpen_enabled": False,
            "temporal_stabilize_enabled": False,
            "preprocess_denoise": False,
            "preprocess_deblock": False,
        },
    ),
    Preset(
        key="ultimate",
        icon="💎",
        name="极致画质",
        desc="不计时间追求最佳",
        params={
            "quality_mode": "quality",
            "upscale_model": "x4plus",
            "face_restore_enabled": True,
            "face_restore_strength": 0.65,
            "face_restore_model": "codeformer",
            "face_poisson_blend": True,
            "sharpen_enabled": True,
            "sharpen_strength": 0.15,
            "temporal_stabilize_enabled": True,
            "temporal_stabilize_strength": 0.70,
            "preprocess_denoise": True,
            "preprocess_deblock": True,
            "preprocess_deinterlace": True,
            "preprocess_colorspace": True,
            "encoder_crf": 15,
        },
    ),
    Preset(
        key="custom",
        icon="⚙️",
        name="自定义",
        desc="手动调整每项参数",
        params={},  # empty — does not override anything
    ),
]


# ---------------------------------------------------------------------------
# Single card widget
# ---------------------------------------------------------------------------


class _PresetCard(QWidget):
    """Individual preset card with icon, name, description."""

    clicked = Signal(str)  # preset key

    def __init__(self, preset: Preset, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._key = preset.key
        self._selected = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(preset.desc)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(56)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel(preset.icon)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 20px; background: transparent;")
        layout.addWidget(icon)

        name = QLabel(preset.name)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #e0e0e0; background: transparent;"
        )
        layout.addWidget(name)

        self._apply_style()

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, event: object) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._key)

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "QWidget { background: #1f3a5c; border: 2px solid #4fc3f7; "
                "border-radius: 6px; }"
            )
        else:
            self.setStyleSheet(
                "QWidget { background: #1f2b47; border: 1px solid #2a3a5c; "
                "border-radius: 6px; }"
                "QWidget:hover { border: 1px solid #4fc3f7; background: #243352; }"
            )


# ---------------------------------------------------------------------------
# Preset cards grid
# ---------------------------------------------------------------------------


class PresetCardsWidget(QWidget):
    """Grid of preset cards. Emits *preset_selected* with a :class:`Preset`."""

    preset_selected = Signal(object)  # Preset dataclass

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards: dict[str, _PresetCard] = {}
        self._presets: dict[str, Preset] = {}
        self._current_key: str = ""

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        for idx, preset in enumerate(BUILTIN_PRESETS):
            card = _PresetCard(preset)
            card.clicked.connect(self._on_card_clicked)
            row, col = divmod(idx, 3)
            layout.addWidget(card, row, col)
            self._cards[preset.key] = card
            self._presets[preset.key] = preset

    def _on_card_clicked(self, key: str) -> None:
        # Deselect old
        if self._current_key and self._current_key in self._cards:
            self._cards[self._current_key].set_selected(False)
        # Select new
        self._current_key = key
        self._cards[key].set_selected(True)
        self.preset_selected.emit(self._presets[key])

    def select(self, key: str) -> None:
        """Programmatically select a preset card."""
        if key in self._cards:
            self._on_card_clicked(key)

    def clear_selection(self) -> None:
        if self._current_key in self._cards:
            self._cards[self._current_key].set_selected(False)
        self._current_key = ""
