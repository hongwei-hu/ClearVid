"""Before/After split-line comparison widget (Topaz-style draggable divider)."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

_HANDLE_HALF_W = 16   # half-width of the draggable handle area
_LINE_WIDTH = 2
_ACCENT = QColor("#4fc3f7")
_HANDLE_BG = QColor(30, 40, 60, 200)
_LABEL_FONT_SIZE = 11


class SplitCompareWidget(QWidget):
    """Overlay Before (left) / After (right) with a draggable vertical split line.

    Usage::

        widget = SplitCompareWidget()
        widget.set_images(before_pixmap, after_pixmap)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before: QPixmap | None = None
        self._after: QPixmap | None = None
        self._split: float = 0.5  # 0.0 … 1.0
        self._dragging = False

        self.setMouseTracking(True)
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ---- public API -------------------------------------------------------

    def set_images(self, before: QPixmap | None, after: QPixmap | None) -> None:
        self._before = before
        self._after = after
        self._split = 0.5
        self.update()

    def has_images(self) -> bool:
        return self._before is not None and self._after is not None

    @property
    def split_ratio(self) -> float:
        return self._split

    # ---- painting ---------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if not self.has_images():
            painter.fillRect(0, 0, w, h, QColor("#111122"))
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "拖入视频后\n勾选「自动生成预览」或按 Space")
            painter.end()
            return

        split_x = int(w * self._split)
        image_rect = _fit_rect(self._before.size(), self.size())
        after_source_rect = _center_crop_rect(self._after.size(), self._before.size())

        # -- Draw Before on the left side --
        painter.setClipRect(QRect(0, 0, split_x, h))
        painter.fillRect(0, 0, w, h, QColor("#111122"))
        painter.drawPixmap(image_rect, self._before)

        # -- Draw After on the right side --
        painter.setClipRect(QRect(split_x, 0, w - split_x, h))
        painter.fillRect(0, 0, w, h, QColor("#111122"))
        painter.drawPixmap(image_rect, self._after, after_source_rect)

        painter.setClipping(False)

        # -- Split line --
        pen = QPen(_ACCENT, _LINE_WIDTH)
        painter.setPen(pen)
        painter.drawLine(split_x, 0, split_x, h)

        # -- Handle (centre pill) --
        handle_h = 48
        hy = (h - handle_h) // 2
        handle_rect = QRect(split_x - _HANDLE_HALF_W, hy, _HANDLE_HALF_W * 2, handle_h)
        painter.setBrush(_HANDLE_BG)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(handle_rect, 6, 6)

        # Draw ◀▶ arrows inside handle
        painter.setPen(QPen(_ACCENT, 2))
        cx, cy = split_x, h // 2
        # Left arrow ◀
        painter.drawLine(cx - 8, cy, cx - 3, cy - 5)
        painter.drawLine(cx - 8, cy, cx - 3, cy + 5)
        # Right arrow ▶
        painter.drawLine(cx + 8, cy, cx + 3, cy - 5)
        painter.drawLine(cx + 8, cy, cx + 3, cy + 5)

        # -- Labels --
        painter.setPen(QColor(255, 255, 255, 180))
        font = painter.font()
        font.setPixelSize(_LABEL_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(12, 22, "原始")
        painter.drawText(w - 42, 22, "增强")

        painter.end()

    # ---- mouse interaction ------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_split(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        x = event.position().x()
        w = self.width()
        split_x = w * self._split

        # Change cursor near split line
        if abs(x - split_x) < _HANDLE_HALF_W + 4:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self._dragging:
            self._update_split(x)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def _update_split(self, x: float) -> None:
        ratio = max(0.02, min(0.98, x / max(self.width(), 1)))
        if ratio != self._split:
            self._split = ratio
            self.update()


def _fit_rect(source_size: QSize, target_size: QSize) -> QRect:
    source_w = max(1, source_size.width())
    source_h = max(1, source_size.height())
    target_w = max(1, target_size.width())
    target_h = max(1, target_size.height())
    scale = min(target_w / source_w, target_h / source_h)
    width = max(1, int(round(source_w * scale)))
    height = max(1, int(round(source_h * scale)))
    x = (target_w - width) // 2
    y = (target_h - height) // 2
    return QRect(x, y, width, height)


def _center_crop_rect(source_size: QSize, reference_size: QSize) -> QRect:
    source_w = max(1, source_size.width())
    source_h = max(1, source_size.height())
    reference_w = max(1, reference_size.width())
    reference_h = max(1, reference_size.height())
    source_ratio = source_w / source_h
    reference_ratio = reference_w / reference_h

    if abs(source_ratio - reference_ratio) < 1e-6:
        return QRect(0, 0, source_w, source_h)
    if source_ratio > reference_ratio:
        crop_w = max(1, int(round(source_h * reference_ratio)))
        crop_x = (source_w - crop_w) // 2
        return QRect(crop_x, 0, crop_w, source_h)

    crop_h = max(1, int(round(source_w / reference_ratio)))
    crop_y = (source_h - crop_h) // 2
    return QRect(0, crop_y, source_w, crop_h)
