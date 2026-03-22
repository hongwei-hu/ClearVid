"""Main window with three-column layout, menus, and status bar."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui.export_panel import ExportPanel
from clearvid.app.gui.file_panel import FilePanel
from clearvid.app.gui.preview_panel import PreviewPanel
from clearvid.app.gui.theme import get_dark_theme
from clearvid.app.gui.user_settings import UserSettings
from clearvid.app.gui.workers import PreviewWorker, Worker
from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.recommend import recommend


class MainWindow(QMainWindow):
    """ClearVid main window — three-column layout with dark theme."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ClearVid 视频清晰度增强")
        self.resize(1280, 800)

        self._worker: Worker | None = None
        self._preview_worker: PreviewWorker | None = None
        self._last_progress_message = ""
        self._environment = collect_environment_info()
        self._settings = UserSettings()
        self._video_duration: float = 0.0

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._restore_state()

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("文件(&F)")
        file_menu.addAction("添加文件...\tCtrl+O", self._action_browse_file)
        file_menu.addSeparator()
        self._recent_menu = file_menu.addMenu("最近使用")
        self._refresh_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction("退出\tAlt+F4", self.close)

        # ---- Settings ----
        settings_menu = mb.addMenu("设置(&S)")
        settings_menu.addAction("环境诊断", self._show_env_dialog)
        log_action = settings_menu.addAction("显示日志面板", self._toggle_log)
        log_action.setCheckable(True)
        log_action.setChecked(False)

        # ---- Help ----
        help_menu = mb.addMenu("帮助(&H)")
        help_menu.addAction("关于 ClearVid", self._show_about)

    # ==================================================================
    # Central widget — three-column splitter
    # ==================================================================

    def _build_central(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Three-column splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- Left: file management --
        self._file_panel = FilePanel()
        self._file_panel.setMinimumWidth(200)
        self._file_panel.setMaximumWidth(400)
        self._file_panel.file_selected.connect(self._on_file_selected)
        last_dir = self._settings.last_input_dir()
        if last_dir:
            self._file_panel.set_last_input_dir(last_dir)
        self._file_panel.set_recent_files(self._settings.recent_files())
        self._splitter.addWidget(self._file_panel)

        # -- Center: preview --
        self._preview_panel = PreviewPanel()
        self._preview_panel.preview_requested.connect(self._run_preview)
        self._splitter.addWidget(self._preview_panel)

        # -- Right: export settings --
        self._export_panel = ExportPanel()
        self._export_panel.setMinimumWidth(260)
        self._export_panel.setMaximumWidth(420)
        self._export_panel.export_requested.connect(self._run_job)
        self._export_panel.smart_params_requested.connect(self._apply_recommendation)
        self._export_panel.output_dir_changed.connect(self._settings.set_last_output_dir)
        self._splitter.addWidget(self._export_panel)

        # Default column sizes
        saved_sizes = self._settings.splitter_sizes()
        if saved_sizes and len(saved_sizes) == 3:
            self._splitter.setSizes(saved_sizes)
        else:
            self._splitter.setSizes([220, 660, 300])

        self._splitter.setStretchFactor(0, 0)  # left: fixed
        self._splitter.setStretchFactor(1, 1)  # center: stretch
        self._splitter.setStretchFactor(2, 0)  # right: fixed

        root_layout.addWidget(self._splitter, 1)

        # Log panel (hidden by default, toggle via Settings menu)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        self._log.setVisible(False)
        root_layout.addWidget(self._log)

        self.setCentralWidget(root)

    # ==================================================================
    # Status bar
    # ==================================================================

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        env = self._environment
        gpu_text = env.gpu_name or "无 GPU"

        if env.torch_gpu_compatible and env.realesrgan_available:
            icon, text = "\U0001f7e2", "GPU 就绪"
        elif env.ffmpeg_available:
            icon, text = "\U0001f7e1", "仅 CPU 模式"
        else:
            icon, text = "\U0001f534", "环境异常"

        sb.addWidget(QLabel(f"  {icon} {text}"))
        sb.addWidget(QLabel(f"  GPU: {gpu_text}"))
        sb.addPermanentWidget(QLabel("ClearVid v0.1  "))

    # ==================================================================
    # File handling
    # ==================================================================

    def _on_file_selected(self, path: str) -> None:
        """Called when a file is selected/dropped in the file panel."""
        self._settings.add_recent_file(path)
        self._settings.set_last_input_dir(str(Path(path).parent))
        self._refresh_recent_menu()
        self._file_panel.set_recent_files(self._settings.recent_files())

        # Auto-probe
        try:
            metadata = probe_video(Path(path))
            self._video_duration = metadata.duration_seconds
            info_text = (
                f"{metadata.width}\u00d7{metadata.height}  |  "
                f"{metadata.fps:.1f} fps  |  {metadata.video_codec}  |  "
                f"{metadata.duration_seconds:.1f} 秒"
            )
            self._preview_panel.set_video_info(info_text, metadata.duration_seconds)
            self._log_message(
                f"已加载: {Path(path).name}  "
                f"({metadata.width}x{metadata.height}, {metadata.fps:.1f} fps, "
                f"{metadata.video_codec}, {metadata.duration_seconds:.1f}s)"
            )
        except Exception as exc:  # noqa: BLE001
            self._preview_panel.set_video_info(f"\u26a0 无法解析: {exc}", 0)
            self._log_message(f"视频探测失败: {exc}")

        self._export_panel.autofill_output(path)

    def _action_browse_file(self) -> None:
        """Menu action: trigger the file panel's browse dialog."""
        self._file_panel._browse_file()  # noqa: SLF001

    # ==================================================================
    # Preview
    # ==================================================================

    def _run_preview(self, timestamp: float) -> None:
        input_path = self._file_panel.input_path
        if not input_path:
            QMessageBox.information(self, "未选择视频", "请先选择一个输入视频文件。")
            return

        config = self._export_panel.build_preview_config(input_path)

        self._preview_panel.set_preview_loading(True)
        self._log_message(f"正在生成预览帧 (t={timestamp:.1f}s)...")

        self._preview_worker = PreviewWorker(config, timestamp)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _on_preview_finished(self, original: object, enhanced: object) -> None:
        self._preview_panel.set_preview_loading(False)
        self._preview_panel.update_preview(original, enhanced)
        self._log_message("预览帧已生成 — 双击图片可放大查看")

    def _on_preview_failed(self, message: str) -> None:
        self._preview_panel.set_preview_loading(False)
        self._log_message(f"预览失败: {message}")

    # ==================================================================
    # Smart recommendation
    # ==================================================================

    def _apply_recommendation(self) -> None:
        input_path = self._file_panel.input_path
        if not input_path:
            QMessageBox.information(
                self, "未选择视频", "请先选择一个输入视频再使用一键最佳。"
            )
            return

        try:
            metadata = probe_video(Path(input_path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "视频分析失败", str(exc))
            return

        self._video_duration = metadata.duration_seconds
        rec = recommend(metadata, self._environment)
        self._export_panel.apply_recommendation(rec)
        self._export_panel.autofill_output(input_path)

        notes_text = "\n".join(f"  \u2022 {n}" for n in rec.notes)
        self._log_message(f"一键最佳已应用:\n{notes_text}")

    # ==================================================================
    # Export job
    # ==================================================================

    def _run_job(self) -> None:
        input_path = self._file_panel.input_path
        output_path = self._export_panel.output_edit.text()

        if not input_path or not output_path:
            QMessageBox.warning(self, "缺少路径", "请设置输入视频路径和输出文件路径。")
            return

        self._export_panel.hide_post_export()
        config = self._export_panel.build_config(input_path)

        self._log_message(f"开始处理: {config.input_path}")
        self._export_panel.set_export_enabled(False)
        self._export_panel.set_progress(0, "正在准备任务...")

        self._worker = Worker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, percent: int, message: str) -> None:
        self._export_panel.set_progress(percent, message)
        if message != self._last_progress_message:
            self._log_message(message)
            self._last_progress_message = message

    def _on_completed(self, payload: str) -> None:
        self._export_panel.set_export_enabled(True)
        self._export_panel.set_progress(100, "\u2705 处理完成")
        self._export_panel.show_post_export(self._export_panel.output_edit.text())
        self._log_message("处理完成:\n" + payload)

    def _on_failed(self, message: str) -> None:
        self._export_panel.set_export_enabled(True)
        self._export_panel.set_progress(0, "\u274c 处理失败")
        self._log_message(f"处理失败: {message}")

    # ==================================================================
    # Log
    # ==================================================================

    def _log_message(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _toggle_log(self) -> None:
        self._log.setVisible(not self._log.isVisible())

    # ==================================================================
    # Dialogs
    # ==================================================================

    def _show_env_dialog(self) -> None:
        env = self._environment
        _b = lambda v: "是" if v else "否"  # noqa: E731
        lines = [
            f"FFmpeg 可用: {_b(env.ffmpeg_available)}",
            f"GPU: {env.gpu_name or '未检测到'}",
            f"GPU 显存: {env.gpu_memory_mb or 'N/A'} MB",
            f"Torch 版本: {env.torch_version or '未安装'}",
            f"Torch GPU 兼容: {_b(env.torch_gpu_compatible)}",
            f"Real-ESRGAN 可用: {_b(env.realesrgan_available)}",
            f"模型状态: {env.realesrgan_message or '未检测'}",
        ]
        QMessageBox.information(self, "环境诊断", "\n".join(lines))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "关于 ClearVid",
            "ClearVid 视频清晰度增强工具 v0.1\n\n"
            "基于 Real-ESRGAN + CodeFormer 的\n"
            "AI 视频超分辨率与人脸修复工具。",
        )

    # ==================================================================
    # Recent files menu
    # ==================================================================

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        recent = self._settings.recent_files()
        for path in recent:
            name = Path(path).name
            action = self._recent_menu.addAction(name)
            action.setData(path)
            # Use default arg to capture `path` in the lambda
            action.triggered.connect(
                lambda checked=False, p=path: self._open_recent(p)
            )
        if not recent:
            self._recent_menu.addAction("（无最近文件）").setEnabled(False)

    def _open_recent(self, path: str) -> None:
        if Path(path).exists():
            self._file_panel.input_path = path
            self._file_panel.add_file(path)
            self._on_file_selected(path)

    # ==================================================================
    # State persistence
    # ==================================================================

    def _restore_state(self) -> None:
        geo = self._settings.window_geometry()
        if geo is not None:
            self.restoreGeometry(geo)
        state = self._settings.window_state()
        if state is not None:
            self.restoreState(state)

        # Restore collapsible panel states
        panel_states = self._settings.panel_states()
        for section in self._export_panel.get_sections().values():
            if section.name in panel_states:
                section.set_expanded(panel_states[section.name])
            section.toggled.connect(self._settings.save_panel_state)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self._settings.save_window_geometry(self.saveGeometry())
        self._settings.save_window_state(self.saveState())
        self._settings.save_splitter_sizes(self._splitter.sizes())
        super().closeEvent(event)


# ======================================================================
# Application entry point
# ======================================================================


def launch() -> None:
    """Create QApplication, apply theme, and show the main window."""
    app = QApplication(sys.argv)
    app.setStyleSheet(get_dark_theme())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
