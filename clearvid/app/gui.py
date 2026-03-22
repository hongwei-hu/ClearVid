from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import BackendType, EnhancementConfig, QualityMode, TargetProfile, UpscaleModel


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


def main() -> None:
    qt = _load_qt()
    application = qt["QApplication"](sys.argv)
    worker_class = _create_worker_class(qt["QThread"], qt["Signal"])
    window_class = _create_main_window_class(qt, worker_class)
    window = window_class()
    window.show()
    sys.exit(application.exec())


def _load_qt() -> dict[str, object]:
    try:
        from PySide6.QtCore import QThread, Signal
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
        "QLabel": QLabel,
        "QLineEdit": QLineEdit,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPlainTextEdit": QPlainTextEdit,
        "QProgressBar": QProgressBar,
        "QPushButton": QPushButton,
        "QSpinBox": QSpinBox,
        "QThread": QThread,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
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


def _create_main_window_class(qt: dict[str, object], worker_class: type) -> type:
    q_check_box = qt["QCheckBox"]
    q_combo_box = qt["QComboBox"]
    q_double_spin_box = qt["QDoubleSpinBox"]
    q_file_dialog = qt["QFileDialog"]
    q_grid_layout = qt["QGridLayout"]
    q_group_box = qt["QGroupBox"]
    q_hbox_layout = qt["QHBoxLayout"]
    q_label = qt["QLabel"]
    q_line_edit = qt["QLineEdit"]
    q_main_window = qt["QMainWindow"]
    q_message_box = qt["QMessageBox"]
    q_plain_text_edit = qt["QPlainTextEdit"]
    q_progress_bar = qt["QProgressBar"]
    q_push_button = qt["QPushButton"]
    q_spin_box = qt["QSpinBox"]
    q_vbox_layout = qt["QVBoxLayout"]
    q_widget = qt["QWidget"]

    class MainWindow(q_main_window):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ClearVid 视频清晰度增强")
            self.resize(920, 620)
            self._worker: object | None = None
            self._last_progress_message = ""
            self._environment = collect_environment_info()

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
            form_layout.addWidget(q_label("预览秒数"), 6, 0)
            form_layout.addWidget(self.preview_seconds, 6, 1)
            form_layout.addWidget(inspect_button, 6, 2)
            form_layout.addWidget(q_label("人脸修复强度"), 7, 0)
            form_layout.addWidget(self.face_restore_strength, 7, 1)
            form_layout.addWidget(self.face_restore_enabled, 7, 2)
            form_layout.addWidget(q_label("时序稳定强度"), 8, 0)
            form_layout.addWidget(self.temporal_stabilize_strength, 8, 1)
            form_layout.addWidget(self.temporal_stabilize_enabled, 8, 2)

            checkbox_row = q_hbox_layout()
            checkbox_row.addWidget(self.preserve_audio)
            checkbox_row.addWidget(self.preserve_subtitles)
            checkbox_row.addWidget(self.preserve_metadata)
            form_layout.addLayout(checkbox_row, 9, 0, 1, 3)

            layout.addWidget(form_group)

            buttons = q_hbox_layout()
            plan_button = q_push_button("自动生成输出路径")
            plan_button.clicked.connect(self._autofill_output)
            self.run_button = q_push_button("开始导出")
            self.run_button.clicked.connect(self._run_job)
            buttons.addWidget(plan_button)
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

            config = EnhancementConfig(
                input_path=Path(self.input_edit.text()),
                output_path=Path(self.output_edit.text()),
                target_profile=target_profile,
                quality_mode=quality_mode,
                backend=backend,
                upscale_model=upscale_model,
                face_restore_enabled=self.face_restore_enabled.isChecked(),
                face_restore_strength=self.face_restore_strength.value(),
                temporal_stabilize_enabled=self.temporal_stabilize_enabled.isChecked(),
                temporal_stabilize_strength=self.temporal_stabilize_strength.value(),
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
