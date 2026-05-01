"""TensorRT engine cache manager dialog."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.bootstrap.paths import REALESRGAN_WEIGHTS_DIR, TRT_CACHE_DIR
from clearvid.app.gui.workers import TrtWarmupWorker
from clearvid.app.models.realesrgan_runner import _MODEL_REGISTRY
from clearvid.app.models.tensorrt_engine import (
    TrtProfileCacheEntry,
    list_trt_profile_cache,
    select_compatible_engine_for_video,
)

_MODEL_LABELS = {
    "general_v3": "General v3（速度优先）",
    "x4plus": "x4plus（质量优先）",
}

_RECOMMENDED_PROFILES = {
    "general_v3": [(1024, 16), (512, 8), (512, 4)],
    "x4plus": [(512, 4)],
}


class _WeightBackedModel:
    def parameters(self):
        return iter(())


@dataclass(frozen=True)
class _ProfileRow:
    model_key: str
    entry: TrtProfileCacheEntry


class TrtManagerDialog(QDialog):
    """Manage cached TensorRT engines in a user-friendly way."""

    log_message = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        current_width: int = 0,
        current_height: int = 0,
        current_model_key: str = "general_v3",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("TensorRT 引擎管理")
        self.resize(880, 560)
        self.setModal(False)
        self._current_width = max(0, int(current_width))
        self._current_height = max(0, int(current_height))
        self._worker: TrtWarmupWorker | None = None
        self._build_queue: list[tuple[str, int, int]] = []
        self._rows: list[_ProfileRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        intro = QLabel(
            "TensorRT 引擎是按模型、分块大小和并行量预先编译的加速文件。"
            "一般不需要理解细节：优先使用“自动选择”，需要补引擎时再在这里构建。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #b0bec5; font-size: 12px;")
        layout.addWidget(intro)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet(
            "color: #90caf9; font-size: 12px; padding: 6px; background: #16213e; border-radius: 4px;"
        )
        layout.addWidget(self.recommendation_label)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "模型", "分块", "并行量", "状态", "适合场景", "大小", "时间/原因", "文件",
        ])
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._apply_selected_profile_to_controls)
        layout.addWidget(self.table, 1)

        build_group = QGroupBox("手动构建")
        build_layout = QVBoxLayout(build_group)
        build_layout.setSpacing(6)

        build_hint = QLabel(
            "分块越大，较大分辨率视频通常越快；并行量越大，GPU 利用率越高。"
            "如果不确定，点击“构建推荐组合”。"
        )
        build_hint.setWordWrap(True)
        build_hint.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        build_layout.addWidget(build_hint)

        control_row = QHBoxLayout()
        self.model_combo = QComboBox()
        for model_key in ("general_v3", "x4plus"):
            self.model_combo.addItem(_MODEL_LABELS.get(model_key, model_key), model_key)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(current_model_key)))
        self.model_combo.currentIndexChanged.connect(self._update_recommendation)
        control_row.addWidget(QLabel("模型"))
        control_row.addWidget(self.model_combo, 1)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(128, 2048)
        self.tile_spin.setSingleStep(128)
        self.tile_spin.setValue(1024 if current_model_key == "general_v3" else 512)
        control_row.addWidget(QLabel("分块"))
        control_row.addWidget(self.tile_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 16)
        self.batch_spin.setValue(16 if current_model_key == "general_v3" else 4)
        control_row.addWidget(QLabel("并行量"))
        control_row.addWidget(self.batch_spin)

        self.fallback_check = QCheckBox("失败后自动尝试更稳的组合")
        self.fallback_check.setChecked(True)
        self.fallback_check.setToolTip(
            "开启后，构建失败会自动降到更保守的 profile。关闭则只构建你选择的这一组参数。"
        )
        control_row.addWidget(self.fallback_check)
        build_layout.addLayout(control_row)

        action_row = QHBoxLayout()
        self.build_btn = QPushButton("构建选中组合")
        self.build_btn.clicked.connect(self._start_manual_build)
        action_row.addWidget(self.build_btn)

        self.build_recommended_btn = QPushButton("构建推荐组合")
        self.build_recommended_btn.clicked.connect(self._start_recommended_builds)
        action_row.addWidget(self.build_recommended_btn)

        self.delete_btn = QPushButton("删除/清除选中项")
        self.delete_btn.clicked.connect(self._delete_selected)
        action_row.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        action_row.addWidget(self.refresh_btn)

        self.open_cache_btn = QPushButton("打开缓存目录")
        self.open_cache_btn.clicked.connect(self._open_cache_dir)
        action_row.addWidget(self.open_cache_btn)
        action_row.addStretch()
        build_layout.addLayout(action_row)
        layout.addWidget(build_group)

        bottom_row = QHBoxLayout()
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #b0bec5; font-size: 11px;")
        bottom_row.addWidget(self.status_label, 1)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        bottom_row.addWidget(close_btn)
        layout.addLayout(bottom_row)

        self.refresh()

    def refresh(self) -> None:
        self._rows = []
        self.table.setRowCount(0)
        for model_key in ("general_v3", "x4plus"):
            weight_path = self._weight_path(model_key)
            if not weight_path.exists():
                continue
            entries = list_trt_profile_cache(
                _WeightBackedModel(),
                fp16=True,
                cache_dir=TRT_CACHE_DIR,
                weight_path=weight_path,
            )
            for entry in entries:
                if entry.state == "missing":
                    continue
                self._add_profile_row(model_key, entry)
        self._update_recommendation()
        self.status_label.setText("已刷新 TensorRT 引擎池。绿色=可直接使用，红色=曾经构建失败。")

    def _add_profile_row(self, model_key: str, entry: TrtProfileCacheEntry) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._rows.append(_ProfileRow(model_key=model_key, entry=entry))

        values = [
            _MODEL_LABELS.get(model_key, model_key),
            str(entry.tile_size),
            str(entry.batch_size),
            self._state_text(entry.state),
            self._usage_hint(entry.tile_size, entry.batch_size),
            f"{entry.size_mb:.0f} MB" if entry.size_mb > 0 else "-",
            self._time_or_reason(entry),
            Path(entry.engine_path).name if entry.state == "ready" else Path(entry.failed_path).name,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setToolTip(value)
            if entry.state == "ready":
                item.setForeground(QColor("#66bb6a"))
            elif entry.state == "failed":
                item.setForeground(QColor("#ef5350"))
            self.table.setItem(row, column, item)

    def _update_recommendation(self) -> None:
        if self._current_width <= 0 or self._current_height <= 0:
            self.recommendation_label.setText(
                "当前还没有选中视频。选择视频后，这里会显示自动导出预计会使用哪个 TensorRT 引擎。"
            )
            return
        model_key = str(self.model_combo.currentData() or "general_v3")
        weight_path = self._weight_path(model_key)
        if not weight_path.exists():
            self.recommendation_label.setText("当前模型权重尚未下载，暂时无法预测会使用哪个引擎。")
            return
        selected = select_compatible_engine_for_video(
            _WeightBackedModel(),
            width=self._current_width,
            height=self._current_height,
            fp16=True,
            cache_dir=TRT_CACHE_DIR,
            weight_path=weight_path,
        )
        if selected is None:
            self.recommendation_label.setText(
                f"当前视频 {self._current_width}×{self._current_height} 暂无可复用 TensorRT 引擎，"
                "导出时会回退到 torch.compile 或标准推理。"
            )
            return
        self.recommendation_label.setText(
            f"当前视频 {self._current_width}×{self._current_height} 预计使用: "
            f"{_MODEL_LABELS.get(model_key, model_key)} / 分块 {selected.tile_size} / 并行量 {selected.batch_size} "
            f"({selected.tiles_per_frame} 个分块/帧)。"
        )

    def _start_manual_build(self) -> None:
        model_key = str(self.model_combo.currentData() or "general_v3")
        self._build_queue = [(model_key, self.tile_spin.value(), self.batch_spin.value())]
        self._start_next_build()

    def _start_recommended_builds(self) -> None:
        model_key = str(self.model_combo.currentData() or "general_v3")
        ready = {
            (row.entry.tile_size, row.entry.batch_size)
            for row in self._rows
            if row.model_key == model_key and row.entry.state == "ready"
        }
        self._build_queue = [
            (model_key, tile, batch)
            for tile, batch in _RECOMMENDED_PROFILES.get(model_key, [])
            if (tile, batch) not in ready
        ]
        if not self._build_queue:
            QMessageBox.information(self, "推荐组合已就绪", "当前模型的推荐 TensorRT 引擎已经构建完成。")
            return
        self._start_next_build()

    def _start_next_build(self) -> None:
        if not self._build_queue:
            self._set_building(False)
            self.refresh()
            return
        if self._worker is not None and self._worker.isRunning():
            return
        model_key, tile, batch = self._build_queue.pop(0)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(model_key)))
        self.tile_spin.setValue(tile)
        self.batch_spin.setValue(batch)
        self.status_label.setText(f"正在构建: {_MODEL_LABELS.get(model_key, model_key)} / 分块 {tile} / 并行量 {batch}")
        self.log_message.emit(f"[TRT] 管理器开始构建: model={model_key}, tile={tile}, batch={batch}")
        self._set_building(True)
        self._worker = TrtWarmupWorker(
            model_key=model_key,
            tile_size=tile,
            batch_size=batch,
            low_load=False,
            allow_fallbacks=self.fallback_check.isChecked(),
        )
        self._worker.progress.connect(self._on_build_progress)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.done.connect(self._on_build_done)
        self._worker.start()

    def _on_build_progress(self, pct: int, msg: str) -> None:
        self.status_label.setText(f"构建中 {pct}%: {msg}")
        self.log_message.emit(f"[TRT 管理 {pct:3d}%] {msg}")

    def _on_build_failed(self, err: str) -> None:
        first_line = err.splitlines()[0] if err else "未知错误"
        self.status_label.setText(f"构建失败: {first_line}")
        self.log_message.emit(f"[TRT 管理失败]\n{err}")

    def _on_build_done(self) -> None:
        self._worker = None
        self.refresh()
        self._start_next_build()

    def _delete_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            QMessageBox.information(self, "未选择项目", "请先在表格中选择一个引擎或失败记录。")
            return
        profile = self._rows[row]
        entry = profile.entry
        title = "删除引擎" if entry.state == "ready" else "清除失败记录"
        answer = QMessageBox.question(
            self,
            title,
            f"确定处理 {_MODEL_LABELS.get(profile.model_key, profile.model_key)} / "
            f"分块 {entry.tile_size} / 并行量 {entry.batch_size} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        for path_text in (entry.engine_path, entry.failed_path, entry.engine_path.replace(".engine", ".onnx")):
            path = Path(path_text)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                QMessageBox.warning(self, "删除失败", f"无法删除 {path.name}: {exc}")
                return
        self.log_message.emit(
            f"[TRT] 已清理 profile: model={profile.model_key}, tile={entry.tile_size}, batch={entry.batch_size}"
        )
        self.refresh()

    def _apply_selected_profile_to_controls(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return
        profile = self._rows[row]
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(profile.model_key)))
        self.tile_spin.setValue(profile.entry.tile_size)
        self.batch_spin.setValue(profile.entry.batch_size)

    def _open_cache_dir(self) -> None:
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(TRT_CACHE_DIR))

    def _set_building(self, building: bool) -> None:
        self.build_btn.setEnabled(not building)
        self.build_recommended_btn.setEnabled(not building)
        self.delete_btn.setEnabled(not building)
        self.refresh_btn.setEnabled(not building)
        self.model_combo.setEnabled(not building)
        self.tile_spin.setEnabled(not building)
        self.batch_spin.setEnabled(not building)
        self.fallback_check.setEnabled(not building)

    @staticmethod
    def _weight_path(model_key: str) -> Path:
        entry = _MODEL_REGISTRY.get(model_key)
        if entry is None:
            return REALESRGAN_WEIGHTS_DIR / "missing.pth"
        return REALESRGAN_WEIGHTS_DIR / str(entry["filename"])

    @staticmethod
    def _state_text(state: str) -> str:
        return {
            "ready": "已就绪",
            "failed": "失败记录",
            "missing": "未构建",
        }.get(state, state)

    @staticmethod
    def _usage_hint(tile: int, batch: int) -> str:
        if tile >= 1024:
            return "720p/1080p 大画面优先"
        if tile >= 768:
            return "中等分辨率折中方案"
        if batch >= 8:
            return "小到中等分辨率高速"
        return "保守兜底，稳定优先"

    @staticmethod
    def _time_or_reason(entry: TrtProfileCacheEntry) -> str:
        if entry.state == "failed" and entry.failure_reason:
            return entry.failure_reason[:120]
        if entry.modified_at > 0:
            return datetime.fromtimestamp(entry.modified_at).strftime("%Y-%m-%d %H:%M")
        return "-"
