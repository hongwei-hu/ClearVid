"""Main window with three-column layout, menus, and status bar."""

from __future__ import annotations

import logging
import sys
import time
import yaml
from pathlib import Path

from clearvid.app.bootstrap.paths import APP_ROOT, OUTPUTS_DIR
from clearvid.app.bootstrap.weight_manager import (
    download_weight,
    missing_weights_for_export,
)

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui._helpers import coerce_enum
from clearvid.app.gui.estimation import format_duration
from clearvid.app.gui.export_panel import ExportPanel
from clearvid.app.gui.file_panel import FilePanel
from clearvid.app.gui.history_dialog import HistoryRecord, append_history, HistoryDialog
from clearvid.app.gui.naming import DEFAULT_TEMPLATE, render_output_name
from clearvid.app.gui.preset_cards import BUILTIN_PRESETS
from clearvid.app.gui.preview_panel import PreviewPanel
from clearvid.app.gui.queue_worker import ExportJob, JobStatus, QueueWorker
from clearvid.app.gui.safety_checks import check_disk_space, check_overwrite
from clearvid.app.gui.settings_dialog import SettingsDialog
from clearvid.app.gui.theme import get_dark_theme
from clearvid.app.gui.user_settings import UserSettings
from clearvid.app.gui.workers import PreviewWorker, Worker
from clearvid.app.io.probe import collect_environment_info, probe_video
from clearvid.app.recommend import recommend
from clearvid.app.schemas.models import TargetProfile


class MainWindow(QMainWindow):
    """ClearVid main window — three-column layout with dark theme."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ClearVid 视频清晰度增强")
        self.resize(1280, 800)

        self._worker: Worker | None = None
        self._queue_worker: QueueWorker | None = None
        self._preview_worker: PreviewWorker | None = None
        self._last_progress_message = ""
        self._environment = collect_environment_info()
        self._settings = UserSettings()
        self._video_duration: float = 0.0
        self._video_frames: int = 0
        self._video_size_bytes: int = 0
        self._export_start_time: float = 0.0

        # Debounce timer for auto-preview refresh (#19)
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(500)
        self._preview_debounce.timeout.connect(self._auto_refresh_preview)

        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._connect_auto_preview()
        self._bind_shortcuts()
        self._restore_state()
        self._maybe_show_onboarding()

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _build_menu_bar(self) -> None:
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("文件(&F)")
        file_menu.addAction("添加文件...\tCtrl+O", self._action_browse_file)
        file_menu.addAction("添加文件夹...\tCtrl+Shift+O", self._action_browse_folder)
        file_menu.addSeparator()
        self._recent_menu = file_menu.addMenu("最近使用")
        self._refresh_recent_menu()
        file_menu.addSeparator()
        file_menu.addAction("导出配置...\tCtrl+E", self._export_config_yaml)
        file_menu.addAction("导入配置...\tCtrl+I", self._import_config_yaml)
        file_menu.addSeparator()
        file_menu.addAction("退出\tAlt+F4", self.close)

        # ---- Settings ----
        settings_menu = mb.addMenu("设置(&S)")
        settings_menu.addAction("全局设置...", self._show_settings_dialog)
        settings_menu.addSeparator()
        settings_menu.addAction("环境诊断", self._show_env_dialog)
        self._log_action = settings_menu.addAction("显示日志面板", self._toggle_log)
        self._log_action.setCheckable(True)
        self._log_action.setChecked(False)

        # ---- Presets ---
        preset_menu = mb.addMenu("预设(&P)")
        builtin_menu = preset_menu.addMenu("内置预设")
        for preset in BUILTIN_PRESETS:
            if preset.key == "custom":
                continue
            action = builtin_menu.addAction(f"{preset.icon} {preset.name}")
            action.setData(preset)
            action.triggered.connect(
                lambda checked=False, p=preset: self._export_panel._on_preset_selected(p)  # noqa: SLF001
            )

        # ---- Help ----
        help_menu = mb.addMenu("帮助(&H)")
        help_menu.addAction("处理历史", self._show_history_dialog)
        help_menu.addAction("快捷键一览\tF1", self._show_shortcuts_help)
        help_menu.addSeparator()
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
        self._export_panel.export_all_requested.connect(self._run_queue)
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

        # Log panel (hidden by default, toggle via Settings menu or status bar button)
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

        # Log toggle button in status bar (#22)
        self._log_toggle_btn = QPushButton("📋 日志")
        self._log_toggle_btn.setCheckable(True)
        self._log_toggle_btn.setChecked(False)
        self._log_toggle_btn.setFixedHeight(22)
        self._log_toggle_btn.setStyleSheet(
            "QPushButton { border: 1px solid #555; border-radius: 3px; padding: 0 8px; font-size: 11px; }"
            "QPushButton:checked { background: #4fc3f7; color: #1a1a2e; }"
        )
        self._log_toggle_btn.clicked.connect(self._toggle_log)
        sb.addPermanentWidget(self._log_toggle_btn)
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
            self._video_frames = int(metadata.duration_seconds * metadata.fps) if metadata.fps else 0
            try:
                self._video_size_bytes = Path(path).stat().st_size
            except OSError:
                self._video_size_bytes = 0
            info_text = (
                f"{metadata.width}\u00d7{metadata.height}  |  "
                f"{metadata.fps:.1f} fps  |  {metadata.video_codec}  |  "
                f"{format_duration(metadata.duration_seconds)}"
            )
            self._preview_panel.set_video_info(info_text, metadata.duration_seconds)
            self._log_message(
                f"已加载: {Path(path).name}  "
                f"({metadata.width}x{metadata.height}, {metadata.fps:.1f} fps, "
                f"{metadata.video_codec}, {format_duration(metadata.duration_seconds)})"
            )
            # Update estimation
            self._export_panel.update_estimation(
                self._video_duration, self._video_frames, self._video_size_bytes,
            )
            # Auto-preview at slider position when enabled
            if self._preview_panel.is_auto_preview():
                # Use singleShot so the file-selection flow finishes first.
                QTimer.singleShot(
                    100,
                    lambda: self._run_preview(
                        self._preview_panel.current_timestamp(),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            self._preview_panel.set_video_info(f"\u26a0 无法解析: {exc}", 0)
            self._log_message(f"视频探测失败: {exc}")

        self._export_panel.autofill_output(path, self._settings.last_output_dir())

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

        # Cancel any in-flight preview to avoid dangling QThread crash.
        self._cancel_preview_worker()

        config = self._export_panel.build_preview_config(input_path)

        self._preview_panel.set_preview_loading(True)
        self._log_message(f"正在生成预览帧 (t={timestamp:.1f}s)...")

        self._preview_worker = PreviewWorker(config, timestamp)
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _cancel_preview_worker(self) -> None:
        """Safely stop and dispose of any running PreviewWorker."""
        worker = self._preview_worker
        if worker is None:
            return
        try:
            worker.finished.disconnect()
            worker.failed.disconnect()
        except (RuntimeError, TypeError):
            pass
        if worker.isRunning():
            worker.quit()
            worker.wait(3000)  # wait up to 3 s
            if worker.isRunning():
                worker.terminate()
                worker.wait(1000)
        self._preview_worker = None

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
        self._export_panel.autofill_output(input_path, self._settings.last_output_dir())

        notes_text = "\n".join(f"  \u2022 {n}" for n in rec.notes)
        self._log_message(f"一键最佳已应用:\n{notes_text}")

    # ==================================================================
    # Export job (single file)
    # ==================================================================

    def _ensure_weights(self, config) -> bool:
        """Check for missing weights and offer to download them. Returns True if ready."""
        missing = missing_weights_for_export(
            face_restore_enabled=config.face_restore_enabled,
            face_restore_model=config.face_restore_model.value if hasattr(config.face_restore_model, "value") else str(config.face_restore_model),
            upscale_model=config.upscale_model.value if hasattr(config.upscale_model, "value") else str(config.upscale_model),
        )
        if not missing:
            return True

        names = "\n".join(f"  • {s.name} ({s.size_mb} MB)" for s in missing)
        total_mb = sum(s.size_mb for s in missing)
        reply = QMessageBox.question(
            self,
            "需要下载模型权重",
            f"以下模型权重文件缺失，需要下载后才能处理:\n\n{names}\n\n"
            f"总计约 {total_mb} MB，是否立即下载？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        self._log_message(f"开始下载 {len(missing)} 个模型权重...")
        for spec in missing:
            self._log_message(f"  下载: {spec.name} ({spec.size_mb} MB)...")
            QApplication.processEvents()
            ok = download_weight(spec)
            if not ok:
                QMessageBox.critical(self, "下载失败", f"未能下载 {spec.name}。\n请检查网络连接后重试。")
                return False
            self._log_message(f"  ✅ {spec.name} 下载完成")

        self._log_message("所有权重文件准备就绪")
        return True

    def _run_job(self) -> None:
        input_path = self._file_panel.input_path
        output_path = self._export_panel.output_edit.text()

        if not input_path or not output_path:
            QMessageBox.warning(self, "缺少路径", "请设置输入视频路径和输出文件路径。")
            return

        # Safety checks
        if not check_overwrite(output_path, self):
            return
        est = getattr(self._export_panel, "_last_estimate", None)
        required_mb = est.estimated_size_mb * 1.2 if est else 500
        if not check_disk_space(output_path, required_mb, self):
            return

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self._export_panel.hide_post_export()
        config = self._export_panel.build_config(input_path)

        # Check and download missing weights before starting
        if not self._ensure_weights(config):
            return

        self._log_message(f"开始处理: {config.input_path}")
        self._export_panel.set_export_enabled(False)
        self._export_panel.set_progress(0, "正在准备任务...")
        self._export_start_time = time.monotonic()

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
        elapsed = time.monotonic() - self._export_start_time
        self._export_panel.set_export_enabled(True)
        self._export_panel.set_progress(100, "\u2705 处理完成")
        self._export_panel.show_post_export(self._export_panel.output_edit.text())
        self._log_message("处理完成:\n" + payload)
        self._notify_desktop("导出完成", f"{Path(self._file_panel.input_path).name} 已处理完毕")
        # Record history
        append_history(HistoryRecord.now(
            input_path=self._file_panel.input_path or "",
            output_path=self._export_panel.output_edit.text(),
            profile=self._export_panel.target_combo.currentText(),
            quality_mode=self._export_panel.quality_combo.currentText(),
            elapsed_sec=elapsed,
        ))

    def _on_failed(self, message: str) -> None:
        elapsed = time.monotonic() - self._export_start_time
        self._export_panel.set_export_enabled(True)
        self._export_panel.set_progress(0, "\u274c 处理失败")
        self._log_message(f"处理失败: {message}")
        append_history(HistoryRecord.now(
            input_path=self._file_panel.input_path or "",
            output_path=self._export_panel.output_edit.text(),
            elapsed_sec=elapsed,
            success=False,
            error=message[:200],
        ))

    # ==================================================================
    # Export queue (all files in file list)
    # ==================================================================

    def _run_queue(self) -> None:
        """Build jobs from all files in the file list and start queue processing."""
        file_list = self._file_panel._file_list  # noqa: SLF001
        if file_list.count() == 0:
            QMessageBox.information(self, "队列为空", "文件列表中没有视频文件。\n请先拖入或添加视频。")
            return

        template = self._export_panel.naming_edit.text() or DEFAULT_TEMPLATE
        out_dir = self._settings.last_output_dir() or str(OUTPUTS_DIR)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        jobs: list[ExportJob] = []
        for i in range(file_list.count()):
            item = file_list.item(i)
            input_path = item.data(Qt.ItemDataRole.UserRole)
            if not input_path:
                continue

            profile = coerce_enum(
                TargetProfile,
                self._export_panel.target_combo.currentData(),
                TargetProfile.FHD,
            )
            out_name = render_output_name(template, input_path, profile.value if profile else "fhd")
            out_path = str(Path(out_dir) / out_name)

            config = self._export_panel.build_config(input_path)
            # Override output path for queue item
            config = config.model_copy(update={"output_path": Path(out_path)})
            jobs.append(ExportJob(id=i, config=config))

        if not jobs:
            QMessageBox.information(self, "队列为空", "未找到有效的视频文件。")
            return

        # Check weights using first job's config (all jobs share the same settings)
        if not self._ensure_weights(jobs[0].config):
            return

        self._log_message(f"开始队列导出: {len(jobs)} 个文件")
        self._export_panel.set_export_enabled(False)
        self._export_panel.export_all_btn.setEnabled(False)

        self._queue_worker = QueueWorker(jobs)
        self._queue_worker.job_started.connect(self._on_queue_job_started)
        self._queue_worker.job_progress.connect(self._on_queue_job_progress)
        self._queue_worker.job_completed.connect(self._on_queue_job_completed)
        self._queue_worker.job_failed.connect(self._on_queue_job_failed)
        self._queue_worker.queue_finished.connect(self._on_queue_finished)
        self._queue_worker.start()

    def _on_queue_job_started(self, job_id: int) -> None:
        file_list = self._file_panel._file_list  # noqa: SLF001
        if job_id < file_list.count():
            name = Path(file_list.item(job_id).data(Qt.ItemDataRole.UserRole) or "").name
            self._log_message(f"队列 [{job_id + 1}]: 开始处理 {name}")
            self._export_panel.set_progress(0, f"队列 [{job_id + 1}]: {name}")

    def _on_queue_job_progress(self, job_id: int, percent: int, message: str) -> None:
        file_list = self._file_panel._file_list  # noqa: SLF001
        total = file_list.count()
        overall = int((job_id * 100 + percent) / max(total, 1))
        self._export_panel.set_progress(overall, f"[{job_id + 1}/{total}] {message}")

    def _on_queue_job_completed(self, job_id: int, result_json: str) -> None:
        file_list = self._file_panel._file_list  # noqa: SLF001
        if job_id < file_list.count():
            name = Path(file_list.item(job_id).data(Qt.ItemDataRole.UserRole) or "").name
            self._log_message(f"队列 [{job_id + 1}]: {name} ✅ 完成")

    def _on_queue_job_failed(self, job_id: int, error: str) -> None:
        file_list = self._file_panel._file_list  # noqa: SLF001
        if job_id < file_list.count():
            name = Path(file_list.item(job_id).data(Qt.ItemDataRole.UserRole) or "").name
            self._log_message(f"队列 [{job_id + 1}]: {name} ❌ 失败 — {error}")

    def _on_queue_finished(self) -> None:
        self._export_panel.set_export_enabled(True)
        self._export_panel.export_all_btn.setEnabled(True)
        self._export_panel.set_progress(100, "✅ 队列处理完成")
        self._log_message("所有队列任务已完成")
        self._notify_desktop("队列导出完成", "所有视频已处理完毕")

    # ==================================================================
    # Desktop notification
    # ==================================================================

    def _notify_desktop(self, title: str, message: str) -> None:
        """Show a desktop notification if enabled in settings."""
        if not self._settings.notify_on_complete():
            return
        try:
            from PySide6.QtWidgets import QSystemTrayIcon
            from PySide6.QtGui import QIcon

            if not hasattr(self, "_tray_icon"):
                self._tray_icon = QSystemTrayIcon(self)
                self._tray_icon.setIcon(QIcon())
                self._tray_icon.show()
            self._tray_icon.showMessage(title, message, QSystemTrayIcon.MessageIcon.Information, 5000)
        except Exception:  # noqa: BLE001
            pass  # silently skip if tray is not available

    # ==================================================================
    # Log
    # ==================================================================

    def _log_message(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _toggle_log(self) -> None:
        visible = not self._log.isVisible()
        self._log.setVisible(visible)
        self._log_action.setChecked(visible)
        self._log_toggle_btn.setChecked(visible)

    # ==================================================================
    # File menu actions
    # ==================================================================

    def _action_browse_folder(self) -> None:
        """Menu action: browse for a folder and add all video files."""
        folder = QFileDialog.getExistingDirectory(
            self, "选择视频文件夹", self._settings.last_input_dir() or ""
        )
        if not folder:
            return
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".ts"}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in video_exts:
                self._file_panel.add_file(str(p))
        self._settings.set_last_input_dir(folder)

    def _export_config_yaml(self) -> None:
        """Export current parameter set to a YAML file."""
        input_path = self._file_panel.input_path or "dummy.mp4"
        config = self._export_panel.build_config(input_path)
        data = config.model_dump(mode="json")
        # Remove path fields — those are per-session
        data.pop("input_path", None)
        data.pop("output_path", None)

        path, _ = QFileDialog.getSaveFileName(
            self, "导出配置", str(APP_ROOT / "clearvid_config.yaml"),
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if path:
            Path(path).write_text(
                yaml.dump(data, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            self._log_message(f"配置已导出: {path}")

    def _import_config_yaml(self) -> None:
        """Import parameters from a YAML file and apply to widgets."""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "",
            "YAML 文件 (*.yaml *.yml);;所有文件 (*)",
        )
        if not path:
            return
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", f"无法解析 YAML:\n{exc}")
            return
        if not isinstance(data, dict):
            QMessageBox.critical(self, "导入失败", "YAML 内容格式不正确。")
            return
        # Apply to export panel using preset mechanism (same param key mapping)
        from clearvid.app.gui.preset_cards import Preset
        imported = Preset(key="imported", icon="📥", name="导入", desc="", params=data)
        self._export_panel._on_preset_selected(imported)  # noqa: SLF001
        self._log_message(f"配置已导入: {Path(path).name}")

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

    def _show_settings_dialog(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec():
            # Sync naming template back to export panel
            self._export_panel.naming_edit.setText(self._settings.naming_template())

    def _show_history_dialog(self) -> None:
        HistoryDialog(self).exec()

    def _show_shortcuts_help(self) -> None:
        text = (
            "快捷键一览\n"
            "─────────────────────\n"
            "Ctrl+O          添加文件\n"
            "Ctrl+Shift+O    添加文件夹\n"
            "Space            生成预览\n"
            "Ctrl+Enter     开始导出\n"
            "Escape           取消当前任务\n"
            "Ctrl+E          导出配置\n"
            "Ctrl+I           导入配置\n"
            "Ctrl+L          切换日志面板\n"
            "F1                快捷键一览\n"
        )
        QMessageBox.information(self, "快捷键一览", text)

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
    # Auto-preview debounce (#19)
    # ==================================================================

    def _connect_auto_preview(self) -> None:
        """Connect export panel parameter widgets to debounced preview refresh."""
        ep = self._export_panel
        # Combos
        for combo in (ep.target_combo, ep.quality_combo, ep.backend_combo,
                       ep.upscale_model_combo, ep.accelerator_combo,
                       ep.face_model_combo, ep.pixel_format_combo):
            combo.currentIndexChanged.connect(self._schedule_preview_refresh)
        # Checkboxes
        for cb in (ep.face_restore_enabled, ep.face_poisson_blend,
                    ep.sharpen_enabled, ep.temporal_enabled,
                    ep.preprocess_denoise, ep.preprocess_deblock,
                    ep.preprocess_deinterlace, ep.preprocess_colorspace):
            cb.toggled.connect(self._schedule_preview_refresh)
        # Spinboxes
        for spin in (ep.face_restore_strength, ep.sharpen_strength,
                      ep.temporal_strength, ep.encoder_crf):
            spin.valueChanged.connect(self._schedule_preview_refresh)

    def _schedule_preview_refresh(self) -> None:
        """Restart the 500 ms debounce timer on any parameter change."""
        if self._file_panel.input_path and self._preview_panel.is_auto_preview():
            self._preview_debounce.start()

    def _auto_refresh_preview(self) -> None:
        """Fired after 500 ms of no parameter changes — refresh preview at current timestamp."""
        if not self._preview_panel.is_auto_preview():
            return
        ts = self._preview_panel.current_timestamp()
        if ts >= 0 and self._file_panel.input_path:
            self._run_preview(ts)

    # ==================================================================
    # Keyboard shortcuts (#21)
    # ==================================================================

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Space"), self).activated.connect(
            lambda: self._run_preview(self._preview_panel.current_timestamp())
            if self._file_panel.input_path else None
        )
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self._run_job)
        QShortcut(QKeySequence("Escape"), self).activated.connect(self._cancel_current)
        QShortcut(QKeySequence("Ctrl+L"), self).activated.connect(self._toggle_log)
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self._action_browse_file)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self).activated.connect(self._action_browse_folder)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(self._export_config_yaml)
        QShortcut(QKeySequence("Ctrl+I"), self).activated.connect(self._import_config_yaml)
        QShortcut(QKeySequence("F1"), self).activated.connect(self._show_shortcuts_help)

    def _cancel_current(self) -> None:
        """Cancel the currently running worker or queue."""
        if self._queue_worker and self._queue_worker.isRunning():
            self._queue_worker.cancel()
            self._log_message("队列取消中…")
        elif self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._log_message("任务取消中…")
        elif self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._log_message("预览取消中…")

    # ==================================================================
    # Onboarding (#20)
    # ==================================================================

    def _maybe_show_onboarding(self) -> None:
        if not self._settings.onboarding_shown():
            from clearvid.app.gui.onboarding import OnboardingOverlay
            self._onboarding = OnboardingOverlay(self)
            self._onboarding.show()
            self._onboarding.raise_()

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

        # Restore naming template
        saved_template = self._settings.naming_template()
        if saved_template:
            self._export_panel.naming_edit.setText(saved_template)

    def closeEvent(self, event: object) -> None:  # noqa: N802
        self._settings.save_window_geometry(self.saveGeometry())
        self._settings.save_window_state(self.saveState())
        self._settings.save_splitter_sizes(self._splitter.sizes())
        self._settings.set_naming_template(self._export_panel.naming_edit.text())
        super().closeEvent(event)


# ======================================================================
# Application entry point
# ======================================================================


def launch() -> None:
    """Create QApplication, apply theme, and show the main window."""
    # Global exception hook — log crashes instead of silent exit.
    _logger = logging.getLogger("clearvid.gui")

    def _exception_hook(exc_type, exc_value, exc_tb):
        _logger.critical(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb),
        )
        # Also print to stderr so the console shows it.
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _exception_hook

    app = QApplication(sys.argv)

    # Set a default font with explicit point size to prevent
    # "QFont::setPointSize: Point size <= 0" warnings from px-based stylesheets.
    default_font = QFont("Segoe UI", 10)
    default_font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(default_font)

    app.setStyleSheet(get_dark_theme())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
