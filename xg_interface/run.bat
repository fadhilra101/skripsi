@echo off
setlocal

rem ========================================
rem xG Prediction Interface - Run Script
rem ========================================

rem Always run from this script's folder
cd /d "%~dp0"

echo ========================================
echo xG Prediction Interface
echo ========================================
echo.

rem Prefer virtual environment if available
set "PY="
set "ENV_DESC="
if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    set "ENV_DESC=virtual environment"
) else (
    set "PY=python"
    set "ENV_DESC=system Python"
)

echo Using %ENV_DESC%: %PY%
%PY% --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please run setup.bat first or install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo.
echo Checking required packages...
%PY% -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Packages missing. Installing requirements...
    if "%PY%"=="python" (
        %PY% -m pip install --user --no-warn-script-location -r requirements.txt
    ) else (
        %PY% -m pip install -r requirements.txt
    )
    if errorlevel 1 (
        echo ERROR: Failed to install requirements. Run setup.bat or fix issues and retry.
        pause
        exit /b 1
    )
)

echo.
echo Checking for model file...
if exist "xg_model.joblib" (
    echo Model file found: xg_model.joblib
) else (
    echo No model file found. A demo model will be created automatically.
    echo See MODEL_PLACEMENT.md for placing your trained model.
)

echo.
echo Starting xG Prediction Interface...
echo The application will open in your default browser.
echo Press Ctrl+C here to stop the app.
echo.

%PY% -m streamlit run app.py

pause
