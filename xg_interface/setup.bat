@echo off
echo ========================================
echo xG Prediction Interface - Setup Script
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/4] Checking pip installation...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not installed
    echo Please install pip
    pause
    exit /b 1
)
echo pip found!

echo.
echo [3/4] Installing requirements...
echo Installing Python packages (including LightGBM) from requirements.txt...
pip install --user --no-warn-script-location -r requirements.txt
if errorlevel 1 (
    echo Installation failed, trying with --break-system-packages...
    pip install --user --break-system-packages --no-warn-script-location -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        echo Please try running as Administrator or use virtual environment (setup_venv.bat)
        pause
        exit /b 1
    )
)

echo.
echo [4/4] Checking installation...
echo Verifying Streamlit installation...
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
if errorlevel 1 (
    echo ERROR: Streamlit installation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Place your trained model file (xg_model.joblib) in the root directory
echo 2. Run 'run.bat' to start the application
echo.
echo If you don't have a model file, the app will create a dummy model for demonstration.
echo.
pause
