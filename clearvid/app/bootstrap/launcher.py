"""ClearVid launcher — orchestrates first-time setup and daily startup.

Flow
----
1. Check if ``lib/`` exists with correct version stamp.
2. If not → run first-time setup wizard (console-based, no PySide6 needed).
3. Ensure ``lib/`` is on ``sys.path``.
4. Check FFmpeg reachability.
5. Launch main GUI (or CLI).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _fix_console_encoding() -> None:
    """Ensure console can handle Unicode on Windows."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass


def _ensure_lib_on_path() -> None:
    """Add ``lib/`` to sys.path so pip-installed packages are importable."""
    from clearvid.app.bootstrap.paths import LIB_DIR

    lib_str = str(LIB_DIR)
    if lib_str not in sys.path:
        sys.path.insert(0, lib_str)
    # Also set PYTHONPATH so child processes (pip) inherit it
    existing = os.environ.get("PYTHONPATH", "")
    if lib_str not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = lib_str + os.pathsep + existing


def _needs_install() -> bool:
    from clearvid.app.bootstrap.dep_installer import LIB_VERSION
    from clearvid.app.bootstrap.paths import installed_lib_version

    return installed_lib_version() != LIB_VERSION


def _run_first_time_setup() -> bool:
    """Console-based first-time setup.  Returns True if successful."""
    from clearvid.app.bootstrap.env_detect import detect_gpu, check_ffmpeg
    from clearvid.app.bootstrap.dep_installer import InstallPlan, run_install
    from clearvid.app.bootstrap.paths import LIB_DIR

    print()
    print("=" * 60)
    print("  ClearVid 首次配置")
    print("=" * 60)
    print()

    # GPU detection
    print("[1/3] 检测硬件环境...")
    gpu = detect_gpu()
    if gpu.cuda_capable:
        print(f"  ✅ 检测到 GPU: {gpu.name}")
        print(f"     驱动: {gpu.driver_version}  |  显存: {gpu.memory_mb} MB")
        print(f"     推荐: {gpu.recommended_label}")
    else:
        print(f"  ⚠️  {gpu.recommended_label}")
        print("     将使用 CPU 模式（处理速度较慢）")

    if not check_ffmpeg():
        print()
        print("  ❌ 未检测到 FFmpeg！")
        print("     请将 ffmpeg.exe 放置到 ClearVid 目录或添加到系统 PATH。")
        print("     下载地址: https://www.gyan.dev/ffmpeg/builds/")
        # Don't block — user might add it later
        print()

    print()
    print(f"[2/3] 安装依赖 → {LIB_DIR}")
    print(f"       PyTorch 版本: {gpu.recommended_label}")
    print()

    plan = InstallPlan(
        torch_index_url=gpu.recommended_torch_index,
        torch_label=gpu.recommended_label,
    )

    current_step_name = ""

    def on_step(idx: int, total: int, desc: str) -> None:
        nonlocal current_step_name
        current_step_name = desc
        print(f"  [{idx + 1}/{total}] {desc}")

    def on_output(line: str) -> None:
        # Show pip output with indent, filter noise
        if line.startswith("Requirement already") or line.startswith("  "):
            return
        if "Successfully installed" in line or "Downloading" in line or "Installing" in line:
            print(f"    {line}")

    def on_error(idx: int, desc: str, rc: int) -> None:
        print(f"  ❌ 安装失败: {desc} (退出码 {rc})")
        print("     请检查网络连接后重试。")

    success = run_install(plan, on_step=on_step, on_output=on_output, on_error=on_error)

    if success:
        print()
        print("[3/3] ✅ 配置完成！正在启动 ClearVid...")
        print()
    else:
        print()
        print("  安装未完成。请检查错误信息后重试。")
        print()

    return success


def _verify_import(module_name: str, display_name: str) -> bool:
    """Try importing a module and print result."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        print(f"  ⚠️  {display_name} 未安装 — 相关功能不可用")
        return False


def main() -> None:
    """Main launcher entry point."""
    _fix_console_encoding()
    _ensure_lib_on_path()

    if _needs_install():
        success = _run_first_time_setup()
        if not success:
            input("按 Enter 退出...")
            sys.exit(1)
        # Refresh path after install
        _ensure_lib_on_path()

    # Quick verification (non-blocking)
    _verify_import("PySide6", "PySide6 (GUI)")
    _verify_import("torch", "PyTorch (推理)")

    # Launch the actual application
    from clearvid.app.gui import main as gui_main

    gui_main()


if __name__ == "__main__":
    main()
