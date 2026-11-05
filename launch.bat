@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if exist ".venv\Scripts\stealzalut-gui.exe" (
    start "" ".venv\Scripts\stealzalut-gui.exe" %*
) else (
    uv run stealzalut-gui %*
)

