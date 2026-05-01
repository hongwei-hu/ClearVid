"""Crash and diagnostic logging for the GUI process."""

from __future__ import annotations

import atexit
import faulthandler
import logging
import logging.handlers
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import TextIO

LOG_DIR = Path.home() / ".clearvid" / "logs"
_GUI_LOG_PATH = LOG_DIR / "clearvid_gui.log"
_configured = False
_fault_file: TextIO | None = None


def setup_gui_crash_logging() -> Path:
    """Install file logging, Python exception hooks, Qt message logging, and faulthandler."""
    global _configured, _fault_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if _configured:
        return LOG_DIR

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        _GUI_LOG_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    crash_path = LOG_DIR / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    _fault_file = crash_path.open("a", encoding="utf-8", buffering=1)
    _write_session_header(_fault_file)
    faulthandler.enable(file=_fault_file, all_threads=True)

    _install_exception_hooks(_fault_file)
    _install_qt_message_handler()
    atexit.register(_shutdown_crash_logging)

    logging.getLogger("clearvid.gui").info("GUI logging initialized: %s", LOG_DIR)
    _configured = True
    return LOG_DIR


def open_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(str(LOG_DIR))
        return
    import subprocess

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(LOG_DIR)])  # noqa: S603,S607


def _write_session_header(file_obj: TextIO) -> None:
    file_obj.write("=" * 80 + "\n")
    file_obj.write(f"ClearVid GUI crash log started: {datetime.now().isoformat(timespec='seconds')}\n")
    file_obj.write(f"PID: {os.getpid()}\n")
    file_obj.write(f"Python: {sys.version}\n")
    file_obj.write(f"Executable: {sys.executable}\n")
    file_obj.write(f"Platform: {platform.platform()}\n")
    file_obj.write(f"Command: {' '.join(sys.argv)}\n")
    file_obj.write("=" * 80 + "\n")


def _install_exception_hooks(file_obj: TextIO) -> None:
    logger = logging.getLogger("clearvid.gui.crash")
    original_excepthook = sys.excepthook
    original_threading_hook = getattr(threading, "excepthook", None)

    def _exception_hook(exc_type, exc_value, exc_tb):
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        file_obj.write("\n[Unhandled Python exception]\n")
        traceback.print_exception(exc_type, exc_value, exc_tb, file=file_obj)
        file_obj.flush()
        original_excepthook(exc_type, exc_value, exc_tb)

    def _thread_exception_hook(args):
        logger.critical(
            "Unhandled thread exception in %s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        file_obj.write(f"\n[Unhandled thread exception: {getattr(args.thread, 'name', 'unknown')}]\n")
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=file_obj)
        file_obj.flush()
        if original_threading_hook is not None:
            original_threading_hook(args)

    sys.excepthook = _exception_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_exception_hook


def _install_qt_message_handler() -> None:
    try:
        from PySide6.QtCore import qInstallMessageHandler
    except Exception:
        return

    logger = logging.getLogger("clearvid.qt")

    def _qt_message_handler(mode, context, message):  # noqa: ANN001
        mode_name = getattr(mode, "name", str(mode))
        level = logging.INFO
        if "Warning" in mode_name:
            level = logging.WARNING
        elif "Critical" in mode_name or "Fatal" in mode_name:
            level = logging.ERROR
        location = ""
        file_name = getattr(context, "file", "") or ""
        line = getattr(context, "line", 0) or 0
        if file_name or line:
            location = f" ({file_name}:{line})"
        logger.log(level, "Qt %s%s: %s", mode_name, location, message)

    qInstallMessageHandler(_qt_message_handler)


def _shutdown_crash_logging() -> None:
    global _fault_file
    logging.getLogger("clearvid.gui").info("GUI process exiting")
    try:
        faulthandler.disable()
    except Exception:
        pass
    if _fault_file is not None:
        try:
            _fault_file.write(f"\nProcess exited normally: {datetime.now().isoformat(timespec='seconds')}\n")
            _fault_file.close()
        finally:
            _fault_file = None
