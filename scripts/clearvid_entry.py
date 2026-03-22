"""PyInstaller entry point for ClearVid.exe.

This is a thin wrapper that bootstraps sys.path then delegates to the
launcher module.  It is designed to be frozen with PyInstaller — only
stdlib and the clearvid.app.bootstrap package are needed at freeze time.
"""

import sys
import os
from pathlib import Path


def _bootstrap_paths() -> None:
    """Set up sys.path so that the bundled clearvid package and lib/ are importable."""
    if getattr(sys, "frozen", False):
        # PyInstaller one-dir: sys._MEIPASS == temp dir (one-file) or exe dir (one-dir)
        app_root = Path(sys.executable).resolve().parent
    else:
        app_root = Path(__file__).resolve().parent.parent

    os.environ["CLEARVID_ROOT"] = str(app_root)

    # Add app root so `import clearvid` works
    root_str = str(app_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Add lib/ for runtime-installed dependencies
    lib_dir = app_root / "lib"
    lib_str = str(lib_dir)
    if lib_str not in sys.path:
        sys.path.insert(0, lib_str)
    os.environ["PYTHONPATH"] = lib_str + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Add vendor/ for bundled basicsr
    vendor_dir = app_root / "vendor" / "basicsr-1.4.2"
    if vendor_dir.is_dir():
        vendor_str = str(vendor_dir)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)


if __name__ == "__main__":
    _bootstrap_paths()
    from clearvid.app.bootstrap.launcher import main
    main()
