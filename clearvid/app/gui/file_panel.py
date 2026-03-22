"""Left sidebar: file management panel with drag-drop and recent files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from clearvid.app.bootstrap.paths import APP_ROOT

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".m4v"}


class FilePanel(QWidget):
    """Left sidebar with file input, drag-drop zone, and recent files list."""

    file_selected = Signal(str)  # Emitted when a video file is selected/activated

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_input_dir = str(APP_ROOT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Drag-drop zone ---
        self._drop_zone = QLabel("\U0001f4c2\n拖入视频文件\n或文件夹")
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setMinimumHeight(100)
        self._reset_drop_style()
        layout.addWidget(self._drop_zone)

        # --- File input row ---
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("输入视频路径...")
        self._input_edit.returnPressed.connect(self._on_path_entered)
        input_row.addWidget(self._input_edit, 1)

        browse_btn = QPushButton("浏览")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(self._browse_file)
        input_row.addWidget(browse_btn)
        layout.addLayout(input_row)

        # --- File list ---
        list_label = QLabel("文件列表")
        list_label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 12px;")
        layout.addWidget(list_label)

        self._file_list = QListWidget()
        self._file_list.setAlternatingRowColors(False)
        self._file_list.currentItemChanged.connect(self._on_list_selection)
        layout.addWidget(self._file_list, 1)

        # --- Recent files ---
        recent_label = QLabel("最近使用")
        recent_label.setStyleSheet("color: #4fc3f7; font-weight: bold; font-size: 12px;")
        layout.addWidget(recent_label)

        self._recent_list = QListWidget()
        self._recent_list.setMaximumHeight(150)
        self._recent_list.itemDoubleClicked.connect(self._on_recent_clicked)
        layout.addWidget(self._recent_list)

        # Enable drag-drop on the whole panel
        self.setAcceptDrops(True)

    # ---- Public API ----

    @property
    def input_path(self) -> str:
        return self._input_edit.text()

    @input_path.setter
    def input_path(self, path: str) -> None:
        self._input_edit.setText(path)

    def set_last_input_dir(self, d: str) -> None:
        if d:
            self._last_input_dir = d

    def set_recent_files(self, paths: list[str]) -> None:
        self._recent_list.clear()
        for p in paths:
            name = Path(p).name
            item = QListWidgetItem(f"\U0001f4c4 {name}")
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self._recent_list.addItem(item)

    def add_file(self, path: str) -> None:
        """Add a file to the file list (avoids duplicates)."""
        for i in range(self._file_list.count()):
            if self._file_list.item(i).data(Qt.ItemDataRole.UserRole) == path:
                self._file_list.setCurrentRow(i)
                return
        name = Path(path).name
        item = QListWidgetItem(f"\U0001f3ac {name}")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self._file_list.addItem(item)
        self._file_list.setCurrentItem(item)

    # ---- Drag & Drop ----

    def dragEnterEvent(self, event: object) -> None:  # noqa: N802
        mime = event.mimeData()
        if mime.hasUrls():
            for url in mime.urls():
                p = Path(url.toLocalFile())
                if p.is_dir() or p.suffix.lower() in VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    self._drop_zone.setStyleSheet(
                        "QLabel { border: 2px solid #4fc3f7; border-radius: 8px; "
                        "color: #4fc3f7; font-size: 13px; padding: 16px; "
                        "background: #1f2b47; }"
                    )
                    return
        event.ignore()

    def dragLeaveEvent(self, event: object) -> None:  # noqa: N802
        self._reset_drop_style()

    def dropEvent(self, event: object) -> None:  # noqa: N802
        self._reset_drop_style()
        paths: list[str] = []
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                for ext in VIDEO_EXTENSIONS:
                    for f in sorted(p.rglob(f"*{ext}")):
                        paths.append(str(f))
            elif p.suffix.lower() in VIDEO_EXTENSIONS:
                paths.append(str(p))

        seen: set[str] = set()
        for path in paths:
            if path not in seen:
                seen.add(path)
                self.add_file(path)

        if paths:
            first = paths[0]
            self._input_edit.setText(first)
            self.file_selected.emit(first)

    # ---- Slots ----

    def _browse_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择输入视频",
            self._last_input_dir,
            "视频文件 (*.mp4 *.mkv *.mov *.avi *.m4v)",
        )
        if selected:
            self._last_input_dir = str(Path(selected).parent)
            self._input_edit.setText(selected)
            self.add_file(selected)
            self.file_selected.emit(selected)

    def _on_path_entered(self) -> None:
        path = self._input_edit.text().strip()
        if path and Path(path).exists():
            self.add_file(path)
            self.file_selected.emit(path)

    def _on_list_selection(
        self, current: QListWidgetItem | None, _prev: QListWidgetItem | None
    ) -> None:
        if current:
            path = current.data(Qt.ItemDataRole.UserRole)
            if path:
                self._input_edit.setText(path)
                self.file_selected.emit(path)

    def _on_recent_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).exists():
            self._input_edit.setText(path)
            self.add_file(path)
            self.file_selected.emit(path)

    # ---- Internal helpers ----

    def _reset_drop_style(self) -> None:
        self._drop_zone.setStyleSheet(
            "QLabel { border: 2px dashed #2a3a5c; border-radius: 8px; "
            "color: #9e9e9e; font-size: 13px; padding: 16px; background: #16213e; }"
        )
