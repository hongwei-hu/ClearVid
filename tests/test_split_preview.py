from __future__ import annotations

from PySide6.QtCore import QSize

from clearvid.app.gui.widgets.split_preview import _center_crop_rect, _fit_rect


def test_fit_rect_uses_reference_aspect_ratio_centered_in_target() -> None:
    rect = _fit_rect(QSize(408, 720), QSize(600, 500))

    assert rect.width() == 283
    assert rect.height() == 500
    assert rect.x() == 158
    assert rect.y() == 0


def test_center_crop_rect_crops_wide_enhanced_frame_to_original_portrait_ratio() -> None:
    rect = _center_crop_rect(QSize(1920, 1080), QSize(408, 720))

    assert rect.width() == 612
    assert rect.height() == 1080
    assert rect.x() == 654
    assert rect.y() == 0


def test_center_crop_rect_keeps_matching_aspect_ratio() -> None:
    rect = _center_crop_rect(QSize(1632, 2880), QSize(408, 720))

    assert rect.x() == 0
    assert rect.y() == 0
    assert rect.width() == 1632
    assert rect.height() == 2880
