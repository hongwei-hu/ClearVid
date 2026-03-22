"""Center panel: video preview with split-line Before/After comparison."""

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

from clearvid.app.gui.widgets.split_preview import SplitCompareWidget


class PreviewPanel(QWidget):
    """Center panel with split-line Before/After comparison and time slider."""

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

        # --- Split-line Before/After comparison ---
        self._split_widget = SplitCompareWidget()
        self._split_widget.setMinimumHeight(300)
        layout.addWidget(self._split_widget, 1)

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
        """Display new before / after preview images via split-line widget."""
        self._before_pixmap_full = self._numpy_to_pixmap(original_bgr)
        self._after_pixmap_full = self._numpy_to_pixmap(enhanced_bgr)
        self._split_widget.set_images(self._before_pixmap_full, self._after_pixmap_full)

    def get_timestamp(self) -> float:
        if self._video_duration > 0:
            return self._slider.value() / 1000.0 * self._video_duration
        return 0.0

    def current_timestamp(self) -> float:
        """Alias for get_timestamp — used by auto-preview and shortcuts."""
        return self.get_timestamp()

    # ---- Internal ----

    def _on_slider_moved(self, value: int) -> None:
        if self._video_duration > 0:
            seconds = value / 1000.0 * self._video_duration
            self._time_label.setText(f"{seconds:.1f} 秒")
        else:
            self._time_label.setText(f"{value / 10.0:.1f}%")

    def _request_preview(self) -> None:
        self.preview_requested.emit(self.get_timestamp())

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
