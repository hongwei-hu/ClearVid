@echo off
setlocal
set CLEARVID_ROOT=%~dp0
set CLEARVID_ROOT=%CLEARVID_ROOT:~0,-1%

rem If portable lib/ exists, use launcher mode (handles first-time setup)
if exist "%~dp0lib\" (
    set "PYTHONPATH=%~dp0lib;%PYTHONPATH%"
    if exist "%~dp0python\python.exe" (
        "%~dp0python\python.exe" -m clearvid.app.bootstrap.launcher
    ) else (
        powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start_clearvid.ps1" -Mode launcher
    )
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start_clearvid.ps1" -Mode gui
)
