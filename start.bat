@echo off
echo Starting Agentic Data Chat System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if .env file exists
if not exist ".env" (
    echo Warning: .env file not found
    echo Please create .env file with your GOOGLE_API_KEY
    echo Example: GOOGLE_API_KEY=your_api_key_here
    echo.
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install requirements
    pause
    exit /b 1
)

REM Create necessary directories
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs

REM Start the application
echo.
echo Starting Agentic Data Chat API...
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

python main.py

pause
