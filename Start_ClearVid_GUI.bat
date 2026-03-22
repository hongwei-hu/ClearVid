@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start_clearvid.ps1" -Mode gui
