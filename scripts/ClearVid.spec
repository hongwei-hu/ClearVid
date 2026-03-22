# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ClearVid.exe — lightweight launcher.

Only bundles the bootstrap/launcher code and stdlib.  Heavy dependencies
(PyTorch, PySide6, opencv, etc.) are NOT bundled — they are installed
into lib/ at first run.

Build:
    pyinstaller scripts/ClearVid.spec --noconfirm

Output:
    dist/ClearVid/ClearVid.exe   (one-dir mode)
"""

import os
import sys
from pathlib import Path

repo_root = Path(SPECPATH).parent  # SPECPATH == scripts/
clearvid_src = repo_root / "clearvid"

# Collect only the bootstrap and schema modules (stdlib-compatible)
# Everything else is loaded dynamically from lib/ at runtime.
bootstrap_modules = [
    "clearvid",
    "clearvid.app",
    "clearvid.app.bootstrap",
    "clearvid.app.bootstrap.paths",
    "clearvid.app.bootstrap.env_detect",
    "clearvid.app.bootstrap.dep_installer",
    "clearvid.app.bootstrap.weight_manager",
    "clearvid.app.bootstrap.launcher",
]

a = Analysis(
    [str(repo_root / "scripts" / "clearvid_entry.py")],
    pathex=[str(repo_root)],
    binaries=[],
    datas=[
        # Include clearvid source so launcher can import it
        (str(clearvid_src), "clearvid"),
    ],
    hiddenimports=bootstrap_modules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude heavy packages — they live in lib/ and are loaded at runtime
    excludes=[
        "torch", "torchvision", "torchaudio",
        "PySide6", "shiboken6",
        "cv2", "numpy", "PIL",
        "basicsr", "realesrgan", "gfpgan", "facexlib",
        "tensorrt", "onnxruntime",
        "tkinter", "unittest", "test",
        "matplotlib", "scipy", "pandas",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # one-dir mode
    name="ClearVid",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console needed for first-time setup wizard output
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=None,  # TODO: add icon when available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ClearVid",
)
