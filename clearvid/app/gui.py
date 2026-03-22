from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.recommend import recommend
from clearvid.app.schemas.models import BackendType, EnhancementConfig, FaceRestoreModel, InferenceAccelerator, QualityMode, TargetProfile, UpscaleModel


BACKEND_LABELS = {
    BackendType.AUTO: "自动",
    BackendType.BASELINE: "基线增强",
    BackendType.REALESRGAN: "Real-ESRGAN",
}

UPSCALE_MODEL_LABELS = {
    UpscaleModel.AUTO: "自动（质量模式决定）",
    UpscaleModel.GENERAL_V3: "General v3 （轻量快速）",
    UpscaleModel.X4PLUS: "x4plus RRDB （高质量）",
}

ACCELERATOR_LABELS = {
    InferenceAccelerator.AUTO: "自动检测",
    InferenceAccelerator.NONE: "无加速",
    InferenceAccelerator.COMPILE: "torch.compile",
    InferenceAccelerator.TENSORRT: "TensorRT",
}

FACE_MODEL_LABELS = {
    FaceRestoreModel.CODEFORMER: "CodeFormer",
    FaceRestoreModel.GFPGAN: "GFPGAN",
}

QUALITY_LABELS = {
    QualityMode.FAST: "快速",
    QualityMode.BALANCED: "平衡",
    QualityMode.QUALITY: "高质量",
}

TARGET_LABELS = {
    TargetProfile.SOURCE: "保持原始分辨率",
    TargetProfile.FHD: "1080p",
    TargetProfile.UHD4K: "4K",
    TargetProfile.SCALE2X: "放大 2 倍",
    TargetProfile.SCALE4X: "放大 4 倍",
}


def _populate_combo(combo: Any, labels: dict[Any, str], values: Iterable[Any], default_value: Any) -> None:
    for item in values:
        combo.addItem(labels[item], item)
    combo.setCurrentText(labels[default_value])


def _coerce_enum(enum_type: type, value: Any, default: Any) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError:
            return default
    return default


def _set_combo_by_value(combo: Any, value: str) -> None:
    """Set a QComboBox to the item whose data (enum value) matches *value*."""
    for i in range(combo.count()):
        data = combo.itemData(i)
        item_value = data.value if hasattr(data, "value") else str(data)
        if item_value == value:
            combo.setCurrentIndex(i)
            return


def main() -> None:
    qt = _load_qt()
    application = qt["QApplication"](sys.argv)
    worker_class = _create_worker_class(qt["QThread"], qt["Signal"])
    preview_worker_class = _create_preview_worker_class(qt["QThread"], qt["Signal"])
    window_class = _create_main_window_class(qt, worker_class, preview_worker_class)
    window = window_class()
    window.show()
    sys.exit(application.exec())


def _load_qt() -> dict[str, object]:
    try:
        from PySide6.QtCore import QThread, Signal, Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QProgressBar,
            QPushButton,
            QSlider,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PySide6 is not installed. Install with the gui extra.") from exc

    return {
        "QApplication": QApplication,
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QDoubleSpinBox": QDoubleSpinBox,
        "QFileDialog": QFileDialog,
        "QGridLayout": QGridLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QImage": QImage,
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPixmap": QPixmap,
        "QPlainTextEdit": QPlainTextEdit,
        "QProgressBar": QProgressBar,
        "QPushButton": QPushButton,
        "QSlider": QSlider,
        "QSpinBox": QSpinBox,
        "QThread": QThread,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "Qt": Qt,
        "Signal": Signal,
    }


def _create_worker_class(q_thread_cls: Any, signal_factory: Any) -> type:
    class Worker(q_thread_cls):  # type: ignore[misc, valid-type]
        completed = signal_factory(str)  # type: ignore[operator]
        failed = signal_factory(str)  # type: ignore[operator]
        progress = signal_factory(int, str)  # type: ignore[operator]

        def __init__(self, config: EnhancementConfig):
            super().__init__()
            self._config = config

        def run(self) -> None:
            try:
                result = Orchestrator().run_single(self._config, progress_callback=self._emit_progress)
                self.completed.emit(result.model_dump_json(indent=2))
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

        def _emit_progress(self, percent: int, message: str) -> None:
            self.progress.emit(percent, message)

    return Worker


def _create_preview_worker_class(q_thread_cls: Any, signal_factory: Any) -> type:
    class PreviewWorker(q_thread_cls):  # type: ignore[misc, valid-type]
        finished = signal_factory(object, object)  # type: ignore[operator]  # (original_bgr, enhanced_bgr)
        failed = signal_factory(str)  # type: ignore[operator]

        def __init__(self, config: EnhancementConfig, timestamp_sec: float):
            super().__init__()
            self._config = config
            self._timestamp_sec = timestamp_sec

        def run(self) -> None:
            try:
                original, enhanced, _ = Orchestrator().preview_frame(self._config, self._timestamp_sec)
                self.finished.emit(original, enhanced)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    return PreviewWorker


def _create_main_window_class(qt: dict[str, object], worker_class: type, preview_worker_class: type) -> type:
    q_check_box = qt["QCheckBox"]
    q_combo_box = qt["QComboBox"]
    q_double_spin_box = qt["QDoubleSpinBox"]
    q_file_dialog = qt["QFileDialog"]
    q_grid_layout = qt["QGridLayout"]
    q_group_box = qt["QGroupBox"]
    q_hbox_layout = qt["QHBoxLayout"]
    q_image = qt["QImage"]
    q_label = qt["QLabel"]
    q_line_edit = qt["QLineEdit"]
    q_main_window = qt["QMainWindow"]
    q_message_box = qt["QMessageBox"]
    q_pixmap = qt["QPixmap"]
    q_plain_text_edit = qt["QPlainTextEdit"]
    q_progress_bar = qt["QProgressBar"]
    q_push_button = qt["QPushButton"]
    q_slider = qt["QSlider"]
    q_spin_box = qt["QSpinBox"]
    q_vbox_layout = qt["QVBoxLayout"]
    q_widget = qt["QWidget"]
    qt_ns = qt["Qt"]

    class MainWindow(q_main_window):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ClearVid 视频清晰度增强")
            self.resize(920, 720)
            self._worker: object | None = None
            self._preview_worker: object | None = None
            self._last_progress_message = ""
            self._environment = collect_environment_info()
            self._video_duration: float = 0.0

            root = q_widget()
            layout = q_vbox_layout(root)

            self.status_box = q_plain_text_edit()
            self.status_box.setReadOnly(True)
            self.status_box.setMaximumHeight(100)
            layout.addWidget(self.status_box)

            form_group = q_group_box("任务设置")
            form_layout = q_grid_layout(form_group)

            self.input_edit = q_line_edit()
            self.output_edit = q_line_edit()

            input_button = q_push_button("选择文件")
            input_button.clicked.connect(self._pick_input)
            output_button = q_push_button("选择位置")
            output_button.clicked.connect(self._pick_output)

            self.target_combo = q_combo_box()
            _populate_combo(self.target_combo, TARGET_LABELS, TargetProfile, TargetProfile.FHD)

            self.quality_combo = q_combo_box()
            _populate_combo(self.quality_combo, QUALITY_LABELS, QualityMode, QualityMode.QUALITY)

            self.backend_combo = q_combo_box()
            _populate_combo(self.backend_combo, BACKEND_LABELS, BackendType, BackendType.AUTO)

            self.upscale_model_combo = q_combo_box()
            _populate_combo(self.upscale_model_combo, UPSCALE_MODEL_LABELS, UpscaleModel, UpscaleModel.AUTO)

            self.accelerator_combo = q_combo_box()
            _populate_combo(self.accelerator_combo, ACCELERATOR_LABELS, InferenceAccelerator, InferenceAccelerator.AUTO)

            self.async_pipeline = q_check_box("异步流水线")
            self.async_pipeline.setChecked(True)

            self.preview_seconds = q_spin_box()
            self.preview_seconds.setMinimum(0)
            self.preview_seconds.setMaximum(24 * 60 * 60)
            self.preview_seconds.setValue(0)

            self.face_restore_enabled = q_check_box("启用人脸修复")
            self.face_restore_enabled.setChecked(True)
            self.face_restore_strength = q_double_spin_box()
            self.face_restore_strength.setDecimals(2)
            self.face_restore_strength.setRange(0.0, 1.0)
            self.face_restore_strength.setSingleStep(0.05)
            self.face_restore_strength.setValue(0.55)

            self.temporal_stabilize_enabled = q_check_box("启用时序稳定")
            self.temporal_stabilize_enabled.setChecked(True)
            self.temporal_stabilize_strength = q_double_spin_box()
            self.temporal_stabilize_strength.setDecimals(2)
            self.temporal_stabilize_strength.setRange(0.0, 1.0)
            self.temporal_stabilize_strength.setSingleStep(0.05)
            self.temporal_stabilize_strength.setValue(0.6)

            self.face_restore_model_combo = q_combo_box()
            _populate_combo(self.face_restore_model_combo, FACE_MODEL_LABELS, FaceRestoreModel, FaceRestoreModel.CODEFORMER)
            self.face_poisson_blend = q_check_box("Poisson 融合")
            self.face_poisson_blend.setChecked(False)

            self.sharpen_enabled = q_check_box("启用锐化")
            self.sharpen_enabled.setChecked(True)
            self.sharpen_strength = q_double_spin_box()
            self.sharpen_strength.setDecimals(2)
            self.sharpen_strength.setRange(0.0, 1.0)
            self.sharpen_strength.setSingleStep(0.05)
            self.sharpen_strength.setValue(0.12)

            self.encoder_crf = q_spin_box()
            self.encoder_crf.setMinimum(0)
            self.encoder_crf.setMaximum(63)
            self.encoder_crf.setValue(18)
            self.encoder_crf.setSpecialValueText("自动")

            self.output_pixel_format = q_combo_box()
            self.output_pixel_format.addItem("yuv420p (8-bit)", "yuv420p")
            self.output_pixel_format.addItem("yuv420p10le (10-bit)", "yuv420p10le")
            self.output_pixel_format.addItem("p010le (10-bit)", "p010le")
            self.output_pixel_format.setCurrentIndex(0)

            self.preprocess_denoise = q_check_box("预处理降噪")
            self.preprocess_denoise.setChecked(True)
            self.preprocess_deblock = q_check_box("预处理去块")
            self.preprocess_deblock.setChecked(True)
            self.preprocess_deinterlace = q_check_box("自动去隔行")
            self.preprocess_deinterlace.setChecked(True)
            self.preprocess_colorspace = q_check_box("色彩空间归一化")
            self.preprocess_colorspace.setChecked(True)

            inspect_button = q_push_button("分析视频")
            inspect_button.clicked.connect(self._inspect_input)

            self.preserve_audio = q_check_box("保留音频")
            self.preserve_audio.setChecked(True)
            self.preserve_subtitles = q_check_box("保留字幕")
            self.preserve_subtitles.setChecked(True)
            self.preserve_metadata = q_check_box("保留元数据")
            self.preserve_metadata.setChecked(True)

            form_layout.addWidget(q_label("输入视频"), 0, 0)
            form_layout.addWidget(self.input_edit, 0, 1)
            form_layout.addWidget(input_button, 0, 2)
            form_layout.addWidget(q_label("输出文件"), 1, 0)
            form_layout.addWidget(self.output_edit, 1, 1)
            form_layout.addWidget(output_button, 1, 2)
            form_layout.addWidget(q_label("输出规格"), 2, 0)
            form_layout.addWidget(self.target_combo, 2, 1)
            form_layout.addWidget(q_label("质量模式"), 3, 0)
            form_layout.addWidget(self.quality_combo, 3, 1)
            form_layout.addWidget(q_label("增强后端"), 4, 0)
            form_layout.addWidget(self.backend_combo, 4, 1)
            form_layout.addWidget(q_label("超分模型"), 5, 0)
            form_layout.addWidget(self.upscale_model_combo, 5, 1)
            form_layout.addWidget(q_label("推理加速"), 6, 0)
            form_layout.addWidget(self.accelerator_combo, 6, 1)
            form_layout.addWidget(self.async_pipeline, 6, 2)
            form_layout.addWidget(q_label("预览秒数"), 7, 0)
            form_layout.addWidget(self.preview_seconds, 7, 1)
            form_layout.addWidget(inspect_button, 7, 2)
            form_layout.addWidget(q_label("人脸修复强度"), 8, 0)
            form_layout.addWidget(self.face_restore_strength, 8, 1)
            form_layout.addWidget(self.face_restore_enabled, 8, 2)
            form_layout.addWidget(q_label("时序稳定强度"), 9, 0)
            form_layout.addWidget(self.temporal_stabilize_strength, 9, 1)
            form_layout.addWidget(self.temporal_stabilize_enabled, 9, 2)

            form_layout.addWidget(q_label("人脸修复模型"), 10, 0)
            form_layout.addWidget(self.face_restore_model_combo, 10, 1)
            form_layout.addWidget(self.face_poisson_blend, 10, 2)

            form_layout.addWidget(q_label("锐化强度"), 11, 0)
            form_layout.addWidget(self.sharpen_strength, 11, 1)
            form_layout.addWidget(self.sharpen_enabled, 11, 2)

            form_layout.addWidget(q_label("编码 CRF"), 12, 0)
            form_layout.addWidget(self.encoder_crf, 12, 1)
            form_layout.addWidget(q_label("像素格式"), 12, 2)

            form_layout.addWidget(self.output_pixel_format, 13, 0, 1, 2)

            preprocess_row = q_hbox_layout()
            preprocess_row.addWidget(self.preprocess_denoise)
            preprocess_row.addWidget(self.preprocess_deblock)
            preprocess_row.addWidget(self.preprocess_deinterlace)
            preprocess_row.addWidget(self.preprocess_colorspace)
            form_layout.addWidget(q_label("预处理选项"), 14, 0)
            form_layout.addLayout(preprocess_row, 14, 1, 1, 2)

            checkbox_row = q_hbox_layout()
            checkbox_row.addWidget(self.preserve_audio)
            checkbox_row.addWidget(self.preserve_subtitles)
            checkbox_row.addWidget(self.preserve_metadata)
            form_layout.addLayout(checkbox_row, 15, 0, 1, 3)

            layout.addWidget(form_group)

            # --- Preview panel ---
            preview_group = q_group_box("帧预览 (Before / After)")
            preview_layout = q_vbox_layout(preview_group)

            # Time slider row
            slider_row = q_hbox_layout()
            slider_row.addWidget(q_label("时间位置"))
            self.preview_slider = q_slider(qt_ns.Horizontal)
            self.preview_slider.setMinimum(0)
            self.preview_slider.setMaximum(1000)
            self.preview_slider.setValue(0)
            slider_row.addWidget(self.preview_slider)
            self.preview_time_label = q_label("0.0 秒")
            self.preview_time_label.setMinimumWidth(60)
            slider_row.addWidget(self.preview_time_label)
            self.preview_button = q_push_button("生成预览")
            self.preview_button.clicked.connect(self._run_preview)
            slider_row.addWidget(self.preview_button)
            preview_layout.addLayout(slider_row)

            self.preview_slider.valueChanged.connect(self._on_slider_moved)

            # Before / After image row
            image_row = q_hbox_layout()
            self.before_label = q_label("原始帧")
            self.before_label.setAlignment(qt_ns.AlignCenter)
            self.before_label.setMinimumHeight(180)
            self.before_label.setStyleSheet("background: #1a1a1a; color: #888; border: 1px solid #333;")
            self.after_label = q_label("增强帧")
            self.after_label.setAlignment(qt_ns.AlignCenter)
            self.after_label.setMinimumHeight(180)
            self.after_label.setStyleSheet("background: #1a1a1a; color: #888; border: 1px solid #333;")
            image_row.addWidget(self.before_label)
            image_row.addWidget(self.after_label)
            preview_layout.addLayout(image_row)

            layout.addWidget(preview_group)

            buttons = q_hbox_layout()
            plan_button = q_push_button("自动生成输出路径")
            plan_button.clicked.connect(self._autofill_output)
            smart_button = q_push_button("一键最佳")
            smart_button.clicked.connect(self._apply_recommendation)
            self.run_button = q_push_button("开始导出")
            self.run_button.clicked.connect(self._run_job)
            buttons.addWidget(plan_button)
            buttons.addWidget(smart_button)
            buttons.addWidget(self.run_button)
            layout.addLayout(buttons)

            self.progress_label = q_label("等待开始")
            self.progress_bar = q_progress_bar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_label)
            layout.addWidget(self.progress_bar)

            self.log = q_plain_text_edit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log)

            self.setCentralWidget(root)
            self._refresh_environment_status()

        def _pick_input(self) -> None:
            selected, _ = q_file_dialog.getOpenFileName(
                self,
                "选择输入视频",
                str(Path.cwd()),
                "视频文件 (*.mp4 *.mkv *.mov *.avi *.m4v)",
            )
            if selected:
                self.input_edit.setText(selected)
                self._autofill_output()

        def _pick_output(self) -> None:
            selected, _ = q_file_dialog.getSaveFileName(
                self,
                "选择输出文件",
                str(Path.cwd() / "outputs" / "clearvid_output.mp4"),
                "MP4 文件 (*.mp4)",
            )
            if selected:
                self.output_edit.setText(selected)

        def _autofill_output(self) -> None:
            if not self.input_edit.text():
                return
            input_path = Path(self.input_edit.text())
            stem = input_path.stem
            selected_profile = _coerce_enum(
                TargetProfile,
                self.target_combo.currentData(),
                TargetProfile.FHD,
            )
            suffix = selected_profile.value if selected_profile else "output"
            self.output_edit.setText(str(Path.cwd() / "outputs" / f"{stem}_{suffix}.mp4"))

        def _inspect_input(self) -> None:
            if not self.input_edit.text():
                q_message_box.information(self, "未选择输入视频", "请先选择一个输入视频文件。")
                return

            try:
                metadata = probe_video(Path(self.input_edit.text()))
            except Exception as exc:  # noqa: BLE001
                q_message_box.critical(self, "视频分析失败", str(exc))
                return

            self.log.appendPlainText(
                f"输入信息: 分辨率 {metadata.width}x{metadata.height}, 帧率 {metadata.fps:.3f} fps, "
                f"视频编码 {metadata.video_codec}, 音频流 {metadata.audio_streams} 条, 时长 {metadata.duration_seconds:.2f} 秒"
            )

        def _on_slider_moved(self, value: int) -> None:
            if self._video_duration > 0:
                seconds = value / 1000.0 * self._video_duration
                self.preview_time_label.setText(f"{seconds:.1f} 秒")
            else:
                self.preview_time_label.setText(f"{value / 10.0:.1f}%")

        def _run_preview(self) -> None:
            if not self.input_edit.text():
                q_message_box.information(self, "未选择输入视频", "请先选择一个输入视频文件。")
                return

            # Probe for duration if not yet known
            if self._video_duration <= 0:
                try:
                    meta = probe_video(Path(self.input_edit.text()))
                    self._video_duration = meta.duration_seconds
                except Exception:  # noqa: BLE001
                    self._video_duration = 60.0  # fallback

            timestamp = self.preview_slider.value() / 1000.0 * max(self._video_duration, 0.1)

            # Build a minimal config for preview
            target_profile = _coerce_enum(TargetProfile, self.target_combo.currentData(), TargetProfile.FHD)
            quality_mode = _coerce_enum(QualityMode, self.quality_combo.currentData(), QualityMode.QUALITY)
            upscale_model = _coerce_enum(UpscaleModel, self.upscale_model_combo.currentData(), UpscaleModel.AUTO)

            config = EnhancementConfig(
                input_path=Path(self.input_edit.text()),
                output_path=Path(self.output_edit.text() or "preview_temp.mp4"),
                target_profile=target_profile,
                quality_mode=quality_mode,
                upscale_model=upscale_model,
                face_restore_enabled=self.face_restore_enabled.isChecked(),
                face_restore_strength=self.face_restore_strength.value(),
                face_restore_model=_coerce_enum(
                    FaceRestoreModel,
                    self.face_restore_model_combo.currentData(),
                    FaceRestoreModel.CODEFORMER,
                ),
                face_poisson_blend=self.face_poisson_blend.isChecked(),
                sharpen_enabled=self.sharpen_enabled.isChecked(),
                sharpen_strength=self.sharpen_strength.value(),
            )

            self.preview_button.setEnabled(False)
            self.preview_button.setText("预览生成中...")
            self.log.appendPlainText(f"正在生成预览帧 (t={timestamp:.1f}s)...")

            self._preview_worker = preview_worker_class(config, timestamp)
            self._preview_worker.finished.connect(self._on_preview_finished)
            self._preview_worker.failed.connect(self._on_preview_failed)
            self._preview_worker.start()

        def _on_preview_finished(self, original: object, enhanced: object) -> None:
            import numpy as np

            self.preview_button.setEnabled(True)
            self.preview_button.setText("生成预览")

            def _numpy_to_pixmap(bgr_array: np.ndarray, max_width: int = 440) -> object:
                rgb = bgr_array[:, :, ::-1].copy()
                h, w, ch = rgb.shape
                image = q_image(rgb.data, w, h, ch * w, q_image.Format.Format_RGB888)
                pix = q_pixmap.fromImage(image)
                if pix.width() > max_width:
                    pix = pix.scaledToWidth(max_width, qt_ns.SmoothTransformation)
                return pix

            self.before_label.setPixmap(_numpy_to_pixmap(original))
            self.after_label.setPixmap(_numpy_to_pixmap(enhanced))
            self.log.appendPlainText("预览帧已生成")

        def _on_preview_failed(self, message: str) -> None:
            self.preview_button.setEnabled(True)
            self.preview_button.setText("生成预览")
            self.log.appendPlainText(f"预览失败: {message}")

        def _apply_recommendation(self) -> None:
            if not self.input_edit.text():
                q_message_box.information(self, "未选择输入视频", "请先选择一个输入视频再使用一键最佳。")
                return

            try:
                metadata = probe_video(Path(self.input_edit.text()))
            except Exception as exc:  # noqa: BLE001
                q_message_box.critical(self, "视频分析失败", str(exc))
                return

            self._video_duration = metadata.duration_seconds
            rec = recommend(metadata, self._environment)

            # Apply recommendation to widgets
            _set_combo_by_value(self.target_combo, rec.target_profile)
            _set_combo_by_value(self.quality_combo, rec.quality_mode)
            _set_combo_by_value(self.upscale_model_combo, rec.upscale_model)
            _set_combo_by_value(self.accelerator_combo, rec.inference_accelerator)
            self.face_restore_enabled.setChecked(rec.face_restore_enabled)
            _set_combo_by_value(self.face_restore_model_combo, rec.face_restore_model)
            self.temporal_stabilize_enabled.setChecked(rec.temporal_stabilize_enabled)
            self.sharpen_enabled.setChecked(rec.sharpen_enabled)
            self.sharpen_strength.setValue(rec.sharpen_strength)
            self.async_pipeline.setChecked(rec.async_pipeline)

            self._autofill_output()

            notes_text = "\n".join(f"  • {n}" for n in rec.notes)
            self.log.appendPlainText(f"一键最佳已应用:\n{notes_text}")

        def _refresh_environment_status(self) -> None:
            preferred_backend_text = {
                "auto": "自动",
                "baseline": "基线后端",
                "realesrgan": "Real-ESRGAN",
            }.get(self._environment.preferred_backend.value, self._environment.preferred_backend.value)
            bool_text = lambda value: "是" if value else "否"
            lines = [
                f"推荐后端: {preferred_backend_text}",
                f"FFmpeg 可用: {bool_text(self._environment.ffmpeg_available)}",
                f"GPU: {self._environment.gpu_name or '未检测到'}",
                f"Torch 版本: {self._environment.torch_version or '未安装'}",
                f"Torch GPU 兼容: {bool_text(self._environment.torch_gpu_compatible)}",
                f"Real-ESRGAN 可用: {bool_text(self._environment.realesrgan_available)}",
                f"模型状态: {self._environment.realesrgan_message or '未检测'}",
            ]
            self.status_box.setPlainText("\n".join(lines))

        def _run_job(self) -> None:
            if not self.input_edit.text() or not self.output_edit.text():
                q_message_box.warning(self, "缺少路径", "请输入输入视频路径和输出文件路径。")
                return

            target_profile = _coerce_enum(
                TargetProfile,
                self.target_combo.currentData(),
                TargetProfile.FHD,
            )
            quality_mode = _coerce_enum(
                QualityMode,
                self.quality_combo.currentData(),
                QualityMode.QUALITY,
            )
            backend = _coerce_enum(
                BackendType,
                self.backend_combo.currentData(),
                BackendType.AUTO,
            )
            upscale_model = _coerce_enum(
                UpscaleModel,
                self.upscale_model_combo.currentData(),
                UpscaleModel.AUTO,
            )
            inference_accelerator = _coerce_enum(
                InferenceAccelerator,
                self.accelerator_combo.currentData(),
                InferenceAccelerator.AUTO,
            )

            config = EnhancementConfig(
                input_path=Path(self.input_edit.text()),
                output_path=Path(self.output_edit.text()),
                target_profile=target_profile,
                quality_mode=quality_mode,
                backend=backend,
                upscale_model=upscale_model,
                inference_accelerator=inference_accelerator,
                async_pipeline=self.async_pipeline.isChecked(),
                face_restore_enabled=self.face_restore_enabled.isChecked(),
                face_restore_strength=self.face_restore_strength.value(),
                face_restore_model=_coerce_enum(
                    FaceRestoreModel,
                    self.face_restore_model_combo.currentData(),
                    FaceRestoreModel.CODEFORMER,
                ),
                face_poisson_blend=self.face_poisson_blend.isChecked(),
                sharpen_enabled=self.sharpen_enabled.isChecked(),
                sharpen_strength=self.sharpen_strength.value(),
                encoder_crf=self.encoder_crf.value() if self.encoder_crf.value() > 0 else None,
                output_pixel_format=self.output_pixel_format.currentData() or "yuv420p",
                temporal_stabilize_enabled=self.temporal_stabilize_enabled.isChecked(),
                temporal_stabilize_strength=self.temporal_stabilize_strength.value(),
                preprocess_denoise=self.preprocess_denoise.isChecked(),
                preprocess_deblock=self.preprocess_deblock.isChecked(),
                preprocess_deinterlace="auto" if self.preprocess_deinterlace.isChecked() else "off",
                preprocess_colorspace_normalize=self.preprocess_colorspace.isChecked(),
                preserve_audio=self.preserve_audio.isChecked(),
                preserve_subtitles=self.preserve_subtitles.isChecked(),
                preserve_metadata=self.preserve_metadata.isChecked(),
                preview_seconds=self.preview_seconds.value() or None,
            )

            self.log.appendPlainText(f"开始处理: {config.input_path}")
            self._set_running_state(True)
            self._on_progress(0, "正在准备任务")
            self._worker = worker_class(config)
            self._worker.progress.connect(self._on_progress)
            self._worker.completed.connect(self._on_completed)
            self._worker.failed.connect(self._on_failed)
            self._worker.start()

        def _on_completed(self, payload: str) -> None:
            self._set_running_state(False)
            self._on_progress(100, "处理完成")
            self.log.appendPlainText("处理完成:\n" + payload)

        def _on_failed(self, message: str) -> None:
            self._set_running_state(False)
            self.log.appendPlainText(f"处理失败: {message}")

        def _on_progress(self, percent: int, message: str) -> None:
            self.progress_bar.setValue(max(0, min(100, percent)))
            self.progress_label.setText(message)
            if message != self._last_progress_message:
                self.log.appendPlainText(message)
                self._last_progress_message = message

        def _set_running_state(self, is_running: bool) -> None:
            self.run_button.setEnabled(not is_running)

    return MainWindow
