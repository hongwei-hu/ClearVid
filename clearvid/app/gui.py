from __future__ import annotations

import sys
from pathlib import Path

from clearvid.app.orchestrator import Orchestrator
from clearvid.app.schemas.models import BackendType, EnhancementConfig, QualityMode, TargetProfile


def main() -> None:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PySide6 is not installed. Install with the gui extra.") from exc

    class Worker(QThread):
        completed = Signal(str)
        failed = Signal(str)

        def __init__(self, config: EnhancementConfig):
            super().__init__()
            self._config = config

        def run(self) -> None:
            try:
                result = Orchestrator().run_single(self._config)
                self.completed.emit(result.model_dump_json(indent=2))
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ClearVid")
            self.resize(860, 520)
            self._worker: Worker | None = None

            root = QWidget()
            layout = QVBoxLayout(root)

            form_group = QGroupBox("Job")
            form_layout = QGridLayout(form_group)

            self.input_edit = QLineEdit()
            self.output_edit = QLineEdit()

            input_button = QPushButton("Browse")
            input_button.clicked.connect(self._pick_input)
            output_button = QPushButton("Browse")
            output_button.clicked.connect(self._pick_output)

            self.target_combo = QComboBox()
            for item in TargetProfile:
                self.target_combo.addItem(item.value, item)
            self.target_combo.setCurrentText(TargetProfile.FHD.value)

            self.quality_combo = QComboBox()
            for item in QualityMode:
                self.quality_combo.addItem(item.value, item)
            self.quality_combo.setCurrentText(QualityMode.QUALITY.value)

            self.backend_combo = QComboBox()
            for item in BackendType:
                self.backend_combo.addItem(item.value, item)

            self.preserve_audio = QCheckBox("Preserve audio")
            self.preserve_audio.setChecked(True)
            self.preserve_subtitles = QCheckBox("Preserve subtitles")
            self.preserve_subtitles.setChecked(True)
            self.preserve_metadata = QCheckBox("Preserve metadata")
            self.preserve_metadata.setChecked(True)

            form_layout.addWidget(QLabel("Input"), 0, 0)
            form_layout.addWidget(self.input_edit, 0, 1)
            form_layout.addWidget(input_button, 0, 2)
            form_layout.addWidget(QLabel("Output"), 1, 0)
            form_layout.addWidget(self.output_edit, 1, 1)
            form_layout.addWidget(output_button, 1, 2)
            form_layout.addWidget(QLabel("Target profile"), 2, 0)
            form_layout.addWidget(self.target_combo, 2, 1)
            form_layout.addWidget(QLabel("Quality mode"), 3, 0)
            form_layout.addWidget(self.quality_combo, 3, 1)
            form_layout.addWidget(QLabel("Backend"), 4, 0)
            form_layout.addWidget(self.backend_combo, 4, 1)

            checkbox_row = QHBoxLayout()
            checkbox_row.addWidget(self.preserve_audio)
            checkbox_row.addWidget(self.preserve_subtitles)
            checkbox_row.addWidget(self.preserve_metadata)
            form_layout.addLayout(checkbox_row, 5, 0, 1, 3)

            layout.addWidget(form_group)

            buttons = QHBoxLayout()
            plan_button = QPushButton("Autofill output")
            plan_button.clicked.connect(self._autofill_output)
            run_button = QPushButton("Run")
            run_button.clicked.connect(self._run_job)
            buttons.addWidget(plan_button)
            buttons.addWidget(run_button)
            layout.addLayout(buttons)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            layout.addWidget(self.log)

            self.setCentralWidget(root)

        def _pick_input(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select input video",
                str(Path.cwd()),
                "Video Files (*.mp4 *.mkv *.mov *.avi *.m4v)",
            )
            if selected:
                self.input_edit.setText(selected)
                self._autofill_output()

        def _pick_output(self) -> None:
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Select output video",
                str(Path.cwd() / "outputs" / "clearvid_output.mp4"),
                "MP4 Files (*.mp4)",
            )
            if selected:
                self.output_edit.setText(selected)

        def _autofill_output(self) -> None:
            if not self.input_edit.text():
                return
            input_path = Path(self.input_edit.text())
            stem = input_path.stem
            target = self.target_combo.currentText()
            self.output_edit.setText(str(Path.cwd() / "outputs" / f"{stem}_{target}.mp4"))

        def _run_job(self) -> None:
            if not self.input_edit.text() or not self.output_edit.text():
                QMessageBox.warning(self, "Missing paths", "Input and output paths are required.")
                return

            config = EnhancementConfig(
                input_path=Path(self.input_edit.text()),
                output_path=Path(self.output_edit.text()),
                target_profile=TargetProfile(self.target_combo.currentText()),
                quality_mode=QualityMode(self.quality_combo.currentText()),
                backend=BackendType(self.backend_combo.currentText()),
                preserve_audio=self.preserve_audio.isChecked(),
                preserve_subtitles=self.preserve_subtitles.isChecked(),
                preserve_metadata=self.preserve_metadata.isChecked(),
            )

            self.log.appendPlainText(f"Starting job for {config.input_path}")
            self._worker = Worker(config)
            self._worker.completed.connect(self._on_completed)
            self._worker.failed.connect(self._on_failed)
            self._worker.start()

        def _on_completed(self, payload: str) -> None:
            self.log.appendPlainText(payload)

        def _on_failed(self, message: str) -> None:
            self.log.appendPlainText(f"Error: {message}")

    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(application.exec())
