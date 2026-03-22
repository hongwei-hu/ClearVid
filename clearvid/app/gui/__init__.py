"""ClearVid GUI package."""


def main() -> None:
    """Launch the ClearVid GUI application."""
    try:
        from clearvid.app.gui.main_window import launch
    except ImportError as exc:
        raise RuntimeError(
            "PySide6 is not installed. Install with: pip install pyside6"
        ) from exc
    launch()
