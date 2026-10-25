@echo off
title MediScan - Plant Disease Diagnosis (Streamlit UI)
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "VENV_STREAMLIT=%PROJECT_DIR%.venv\Scripts\streamlit.exe"
set "APP_DIR=%PROJECT_DIR%dataset\balanced_dataset"

echo ========================================================
echo   🌾 MediScan - Plant Disease Diagnosis (Streamlit)
echo ========================================================
echo.

if not exist "%VENV_STREAMLIT%" (
    echo [ERROR] Streamlit executable not found at:
    echo "%VENV_STREAMLIT%"
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
"%VENV_STREAMLIT%" run App.py

pause
