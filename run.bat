@echo off
echo.
echo  ==================================
echo   WasteWise - Setup ^& Launch
echo  ==================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b
)

:: Create venv if not exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)

:: Activate
echo [2/4] Activating virtual environment...
call venv\Scripts\activate

:: Install deps
echo [3/4] Installing dependencies...
pip install -r requirements.txt -q

echo.
echo  Choose an option:
echo  1. Train the model (first time setup)
echo  2. Run the web app (after training)
echo.
set /p choice="Enter 1 or 2: "

if "%choice%"=="1" (
    echo.
    set /p data_dir="Enter full path to your dataset folder: "
    echo.
    echo [4/4] Starting training... (this may take 30-90 minutes)
    python model_training/train.py --data_dir "%data_dir%"
    echo.
    echo Training done! Now run this script again and choose option 2.
) else (
    echo [4/4] Starting WasteWise web app...
    echo  Open your browser at: http://localhost:5000
    echo  Press Ctrl+C to stop.
    echo.
    python app.py
)

pause
