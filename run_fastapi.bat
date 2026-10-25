@echo off
title Paddy Vision - FastAPI REST Service
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "VENV_UVICORN=%PROJECT_DIR%.venv\Scripts\uvicorn.exe"
set "APP_DIR=%PROJECT_DIR%dataset\balanced_dataset"

echo ========================================================
echo   🌾 Paddy Vision - FastAPI REST Service
echo   Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo.

if not exist "%VENV_UVICORN%" (
    echo [ERROR] Uvicorn executable not found at:
    echo "%VENV_UVICORN%"
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
"%VENV_UVICORN%" backend_api:app --reload --port 8000

pause
