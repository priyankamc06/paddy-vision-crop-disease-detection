@echo off
title Crop Vision - AI Crop Disease Detection (Flask Web Portal)
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "APP_DIR=%PROJECT_DIR%dataset\balanced_dataset"

echo ========================================================
echo   🌾 Crop Vision - Smart Crop Disease Detection
echo   Powered by InceptionV3 Deep Learning + Flask
echo ========================================================
echo.

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment python not found at:
    echo "%VENV_PYTHON%"
    echo Please make sure the setup completed properly.
    pause
    exit /b 1
)

echo Starting Flask server...
echo URL: http://127.0.0.1:5000
echo.
echo Opening browser in 3 seconds...
start /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5000"

cd /d "%APP_DIR%"
"%VENV_PYTHON%" app_flask.py

pause
