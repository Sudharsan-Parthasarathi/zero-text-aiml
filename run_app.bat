@echo off
title ZERO-TEXT 2.0 Visual Galaxy & Search System
echo Starting ZERO-TEXT 2.0...
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py
) else (
    python main.py
)
pause
