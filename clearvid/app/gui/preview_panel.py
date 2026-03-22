"""Center panel: video preview with Before/After comparison."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PreviewPanel(QWidget):
    """Center panel with Before/After frame preview and time slider."""

    preview_requested = Signal(float)  # timestamp in seconds

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video_duration: float = 0.0
        self._before_pixmap_full: QPixmap | None = None
        self._after_pixmap_full: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Info card ---
        self._info_label = QLabel("请选择或拖入视频文件")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet(
            "color: #9e9e9e; padding: 8px; background: #1f2b47; "
            "border-radius: 4px; font-size: 12px;"
        )
        self._info_label.setMaximumHeight(40)
        layout.addWidget(self._info_label)

        # --- Before / After images ---
        image_row = QHBoxLayout()

        # Before column
        before_col = QVBoxLayout()
        before_title = QLabel("原始帧")
        before_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        before_title.setStyleSheet(
            "color: #9e9e9e; font-size: 11px; font-weight: bold;"
        )
        before_col.addWidget(before_title)

        self._before_label = QLabel("拖入视频后\n点击「生成预览」")
        self._before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._before_label.setMinimumHeight(300)
        self._before_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._before_label.setStyleSheet(
            "background: #111122; color: #555; border: 1px solid #2a3a5c; "
            "border-radius: 4px;"
        )
        self._before_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._before_label.mouseDoubleClickEvent = (
            lambda e: self._show_full_image(self._before_pixmap_full, "原始帧")
        )
        before_col.addWidget(self._before_label, 1)
        image_row.addLayout(before_col, 1)

        # After column
        after_col = QVBoxLayout()
        after_title = QLabel("增强帧")
        after_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_title.setStyleSheet(
            "color: #9e9e9e; font-size: 11px; font-weight: bold;"
        )
        after_col.addWidget(after_title)

        self._after_label = QLabel("拖入视频后\n点击「生成预览」")
        self._after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._after_label.setMinimumHeight(300)
        self._after_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._after_label.setStyleSheet(
            "background: #111122; color: #555; border: 1px solid #2a3a5c; "
            "border-radius: 4px;"
        )
        self._after_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._after_label.mouseDoubleClickEvent = (
            lambda e: self._show_full_image(self._after_pixmap_full, "增强帧")
        )
        after_col.addWidget(self._after_label, 1)
        image_row.addLayout(after_col, 1)

        layout.addLayout(image_row, 1)

        # --- Time slider row ---
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("时间"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1000)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_slider_moved)
        slider_row.addWidget(self._slider, 1)

        self._time_label = QLabel("0.0 秒")
        self._time_label.setMinimumWidth(60)
        slider_row.addWidget(self._time_label)

        self._preview_btn = QPushButton("生成预览")
        self._preview_btn.clicked.connect(self._request_preview)
        slider_row.addWidget(self._preview_btn)

        layout.addLayout(slider_row)

    # ---- Public API ----

    def set_video_info(self, info_text: str, duration: float) -> None:
        """Update the video information card."""
        self._info_label.setText(info_text)
        self._video_duration = duration

    def set_preview_loading(self, loading: bool) -> None:
        self._preview_btn.setEnabled(not loading)
        self._preview_btn.setText("预览生成中..." if loading else "生成预览")

    def update_preview(
        self, original_bgr: np.ndarray, enhanced_bgr: np.ndarray
    ) -> None:
        """Display new before / after preview images."""
        self._before_pixmap_full = self._numpy_to_pixmap(original_bgr)
        self._after_pixmap_full = self._numpy_to_pixmap(enhanced_bgr)
        self._fit_pixmaps()

    def get_timestamp(self) -> float:
        if self._video_duration > 0:
            return self._slider.value() / 1000.0 * self._video_duration
        return 0.0

    # ---- Internal ----

    def _on_slider_moved(self, value: int) -> None:
        if self._video_duration > 0:
            seconds = value / 1000.0 * self._video_duration
            self._time_label.setText(f"{seconds:.1f} 秒")
        else:
            self._time_label.setText(f"{value / 10.0:.1f}%")

    def _request_preview(self) -> None:
        self.preview_requested.emit(self.get_timestamp())

    def _fit_pixmaps(self) -> None:
        """Scale stored full-res pixmaps to fit the current label size."""
        for pixmap, label in [
            (self._before_pixmap_full, self._before_label),
            (self._after_pixmap_full, self._after_label),
        ]:
            if pixmap is None:
                continue
            w = label.width() - 4
            h = label.height() - 4
            if w > 0 and h > 0:
                label.setPixmap(
                    pixmap.scaled(
                        w,
                        h,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_pixmaps()

    def _show_full_image(self, pixmap: QPixmap | None, title: str) -> None:
        """Open a dialog showing the full-resolution image with scroll support."""
        if pixmap is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"ClearVid — {title} (100%)")
        dlg.resize(
            min(pixmap.width() + 40, 1600),
            min(pixmap.height() + 60, 950),
        )
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(img_label)
        dlg_layout.addWidget(scroll)
        dlg.exec()

    @staticmethod
    def _numpy_to_pixmap(bgr_array: np.ndarray) -> QPixmap:
        rgb = bgr_array[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image)
