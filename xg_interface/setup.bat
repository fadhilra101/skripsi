@echo off
setlocal enableextensions enabledelayedexpansion

rem ========================================
rem xG Prediction Interface - Setup Script
rem ========================================

rem Always run from this script's folder
cd /d "%~dp0"

echo ========================================
echo xG Prediction Interface - Setup
echo ========================================
echo.
echo This script will:
echo  - Check Python installation
echo  - Optionally create a virtual environment (recommended)
echo  - Install required Python packages
echo  - Verify installation
echo.

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://www.python.org/downloads/windows/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do set PYVER=%%v
echo Found %PYVER%

echo.
echo [2/4] Virtual environment (recommended)
if exist "venv\Scripts\python.exe" (
    echo Existing virtual environment detected.
) else (
    choice /m "Create a virtual environment now?"
    if errorlevel 2 (
        echo Skipping virtual environment creation.
    ) else (
        echo Creating virtual environment...
        python -m venv venv
        if errorlevel 1 (
            echo ERROR: Failed to create virtual environment.
            echo You can install packages to the user site instead.
        ) else (
            echo Virtual environment created.
        )
    )
)

set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
echo Using Python: %PY%

echo.
echo [3/4] Upgrading pip and installing requirements...
%PY% -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip. Continuing...
)

if "%PY%"=="python" (
    echo Installing to user site (no admin required)...
    %PY% -m pip install --user --no-warn-script-location -r requirements.txt
) else (
    echo Installing into virtual environment...
    %PY% -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    echo Try running again, or use the alternative script setup_venv.bat.
    pause
    exit /b 1
)

echo.
echo [4/4] Verifying installation...
set FAIL=0
call :checkpkg streamlit
call :checkpkg pandas
call :checkpkg numpy
call :checkpkg matplotlib
call :checkpkg plotly
call :checkpkg lightgbm
call :checkpkg scikit-learn
call :checkpkg mplsoccer
call :checkpkg kaleido

if %FAIL% NEQ 0 (
    echo.
    echo One or more packages failed to import. Please review errors above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo  - Place your trained model file (xg_model.joblib) in this folder, or
echo    let the app create a demo model automatically.
echo  - Start the app with: run.bat

pause
exit /b 0

:checkpkg
set PKG=%~1
%PY% -c "import importlib,sys; m='%PKG%'; mod=importlib.import_module(m); v=getattr(mod,'__version__','unknown'); print(f'  \u2713 {m}: {v}')" 1>nul 2>nul
if errorlevel 1 (
    echo   x %PKG% failed to import
    set FAIL=1
) else (
    %PY% -c "import importlib; m='%PKG%'; mod=importlib.import_module(m); v=getattr(mod,'__version__','unknown'); print(f'  ^✓ {m}: {v}')"
)
exit /b 0
