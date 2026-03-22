"""ClearVid GUI dark / light theme definitions."""

DARK_THEME = """
/* === ClearVid Dark Theme === */

/* Global */
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
}

/* Menu bar */
QMenuBar {
    background-color: #16213e;
    color: #e0e0e0;
    border-bottom: 1px solid #2a3a5c;
    padding: 2px;
}
QMenuBar::item { padding: 4px 10px; }
QMenuBar::item:selected { background-color: #2a3a5c; border-radius: 3px; }
QMenu {
    background-color: #1f2b47;
    border: 1px solid #2a3a5c;
    color: #e0e0e0;
    padding: 4px;
}
QMenu::item { padding: 5px 24px 5px 12px; }
QMenu::item:selected { background-color: #2a3a5c; }
QMenu::separator { background-color: #2a3a5c; height: 1px; margin: 4px 8px; }

/* Group boxes */
QGroupBox {
    background-color: #1f2b47;
    border: 1px solid #2a3a5c;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}
QGroupBox::title {
    color: #4fc3f7;
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* Input controls */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #16213e;
    border: 1px solid #2a3a5c;
    border-radius: 4px;
    padding: 5px 8px;
    color: #e0e0e0;
    min-height: 22px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #4fc3f7;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #9e9e9e;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #1f2b47;
    border: 1px solid #2a3a5c;
    color: #e0e0e0;
    selection-background-color: #2a3a5c;
}

/* Buttons */
QPushButton {
    background-color: #2a3a5c;
    border: 1px solid #3a4a6c;
    border-radius: 4px;
    padding: 6px 16px;
    color: #e0e0e0;
    min-height: 24px;
}
QPushButton:hover { background-color: #3a4a6c; }
QPushButton:pressed { background-color: #1f2b47; }
QPushButton:disabled { color: #555; background-color: #1a1a2e; border-color: #222; }

/* Export button (primary action) */
QPushButton#exportButton {
    background-color: #4fc3f7;
    color: #1a1a2e;
    font-weight: bold;
    font-size: 14px;
    padding: 10px 24px;
    border-radius: 6px;
    border: none;
}
QPushButton#exportButton:hover { background-color: #29b6f6; }
QPushButton#exportButton:pressed { background-color: #0288d1; }
QPushButton#exportButton:disabled { background-color: #2a3a5c; color: #555; }

/* Smart button */
QPushButton#smartButton {
    background-color: #81c784;
    color: #1a1a2e;
    font-weight: bold;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
}
QPushButton#smartButton:hover { background-color: #66bb6a; }
QPushButton#smartButton:pressed { background-color: #4caf50; }

/* Progress bar */
QProgressBar {
    background-color: #16213e;
    border: 1px solid #2a3a5c;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, x2:1, stop:0 #4fc3f7, stop:1 #81c784);
    border-radius: 3px;
}

/* Sliders */
QSlider::groove:horizontal {
    background-color: #2a3a5c;
    height: 4px;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background-color: #4fc3f7;
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}
QSlider::handle:horizontal:hover { background-color: #29b6f6; }

/* Checkboxes */
QCheckBox { spacing: 6px; color: #e0e0e0; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #2a3a5c;
    background-color: #16213e;
}
QCheckBox::indicator:checked {
    background-color: #4fc3f7;
    border-color: #4fc3f7;
}
QCheckBox::indicator:disabled { border-color: #1f2b47; background-color: #1a1a2e; }

/* Scroll areas */
QScrollArea { background-color: transparent; border: none; }
QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background-color: #2a3a5c;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background-color: #3a4a6c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #1a1a2e;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background-color: #2a3a5c;
    border-radius: 4px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background-color: #3a4a6c; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* Text edit (log) */
QPlainTextEdit {
    background-color: #0f0f1e;
    border: 1px solid #2a3a5c;
    border-radius: 4px;
    color: #b0b0b0;
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}

/* Status bar */
QStatusBar {
    background-color: #16213e;
    border-top: 1px solid #2a3a5c;
    color: #9e9e9e;
    font-size: 12px;
}
QStatusBar QLabel { color: #9e9e9e; margin: 0 4px; }

/* Splitter handle */
QSplitter::handle { background-color: #2a3a5c; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:hover { background-color: #4fc3f7; }

/* Tool buttons (collapsible headers) */
QToolButton {
    background-color: transparent;
    border: none;
    color: #e0e0e0;
    font-weight: bold;
    font-size: 13px;
    padding: 6px 4px;
    text-align: left;
}
QToolButton:hover { color: #4fc3f7; }

/* List widget (file list, recent files) */
QListWidget {
    background-color: #16213e;
    border: 1px solid #2a3a5c;
    border-radius: 4px;
    color: #e0e0e0;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #1f2b47;
}
QListWidget::item:selected {
    background-color: #2a3a5c;
    color: #4fc3f7;
}
QListWidget::item:hover { background-color: #1f2b47; }

/* Frame separators */
QFrame[frameShape="4"] { color: #2a3a5c; }

/* Tooltips */
QToolTip {
    background-color: #1f2b47;
    color: #e0e0e0;
    border: 1px solid #4fc3f7;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}
"""


def get_dark_theme() -> str:
    """Return the dark theme QSS string."""
    return DARK_THEME


def get_light_theme() -> str:
    """Return a light theme QSS string (placeholder for future)."""
    return ""
