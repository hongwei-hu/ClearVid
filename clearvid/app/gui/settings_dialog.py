"""Global settings dialog — output directory, naming, theme, notifications, reset."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.gui.naming import DEFAULT_TEMPLATE
from clearvid.app.gui.user_settings import UserSettings


class SettingsDialog(QDialog):
    """Modal dialog for editing global application settings."""

    def __init__(self, settings: UserSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("全局设置")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # ---- Default output directory ----
        dir_row = QHBoxLayout()
        self._output_dir_edit = QLineEdit(settings.last_output_dir())
        self._output_dir_edit.setPlaceholderText("默认输出目录…")
        dir_row.addWidget(self._output_dir_edit, 1)
        browse_btn = QPushButton("📁")
        browse_btn.setFixedWidth(32)
        browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(browse_btn)
        form.addRow("默认输出目录", dir_row)

        # ---- Naming template ----
        self._naming_edit = QLineEdit(settings.naming_template())
        self._naming_edit.setPlaceholderText(DEFAULT_TEMPLATE)
        hint = QLabel("变量: {name} {profile} {date} {time} {ext}")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow("命名规则", self._naming_edit)
        form.addRow("", hint)

        # ---- Theme ----
        self._theme_combo = QComboBox()
        self._theme_combo.addItem("暗色", "dark")
        self._theme_combo.addItem("亮色 (暂不支持)", "light")
        idx = self._theme_combo.findData(settings.theme())
        if idx >= 0:
            self._theme_combo.setCurrentIndex(idx)
        form.addRow("主题", self._theme_combo)

        # ---- Notifications ----
        self._notify_check = QCheckBox("导出完成后显示桌面通知")
        self._notify_check.setChecked(settings.notify_on_complete())
        form.addRow("通知", self._notify_check)

        layout.addLayout(form)

        # ---- Reset ----
        reset_btn = QPushButton("🔄 重置所有设置")
        reset_btn.clicked.connect(self._confirm_reset)
        layout.addWidget(reset_btn)

        layout.addSpacing(12)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok_btn = QPushButton("确定")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _browse_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "选择默认输出目录", self._output_dir_edit.text()
        )
        if d:
            self._output_dir_edit.setText(d)

    def _accept(self) -> None:
        self._settings.set_last_output_dir(self._output_dir_edit.text())
        self._settings.set_naming_template(self._naming_edit.text() or DEFAULT_TEMPLATE)
        self._settings.set_theme(self._theme_combo.currentData() or "dark")
        self._settings.set_notify_on_complete(self._notify_check.isChecked())
        self.accept()

    def _confirm_reset(self) -> None:
        answer = QMessageBox.question(
            self,
            "重置设置",
            "确定要恢复所有设置为默认值吗？\n此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._settings._s.clear()  # noqa: SLF001
            self._output_dir_edit.clear()
            self._naming_edit.setText(DEFAULT_TEMPLATE)
            self._theme_combo.setCurrentIndex(0)
            self._notify_check.setChecked(True)
            QMessageBox.information(self, "已重置", "所有设置已恢复默认值。")
