@echo off
title ZERO-ARCADE Live Public Web Publisher
echo =======================================================
echo    ZERO-ARCADE - Instant Live Public Website Publisher
echo =======================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment (.venv) not found.
    pause
    exit /b 1
)

netstat -ano | findstr :8000 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Backend server already running on port 8000.
) else (
    echo [1/2] Starting ZERO-ARCADE AI Backend...
    start "ZERO-ARCADE Server" /min .venv\Scripts\python.exe main.py --no-browser
    timeout /t 4 /nobreak >nul
)

echo [2/2] Connecting to public HTTPS tunnel...
echo.
echo =======================================================
echo  YOUR LIVE PUBLIC URL WILL APPEAR BELOW IN SECONDS:
echo  (Share this link with anyone - works on any phone/PC)
echo  Keep this terminal window open while sharing!
echo =======================================================
echo.

ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:8000 a.pinggy.io

if %ERRORLEVEL% neq 0 (
    echo.
    echo Trying fallback tunnel (localhost.run)...
    ssh -o StrictHostKeyChecking=no -R 80:localhost:8000 nokey@localhost.run
)

pause
