@echo off
title ZERO-TEXT 2.0 Visual Galaxy & Search System
echo Starting ZERO-TEXT 2.0...
cd /d "%~dp0"
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set KMP_DUPLICATE_LIB_OK=TRUE
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py
) else (
    python main.py
)
pause
