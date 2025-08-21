@echo off
echo Starting Agentic Data Chat System (Full Stack)
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed or not in PATH
    echo Please install Node.js and try again
    pause
    exit /b 1
)

REM Start backend
echo [1/3] Setting up Python backend...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

if not exist ".env" (
    echo Warning: .env file not found
    echo Please create .env file with your GOOGLE_API_KEY
    echo.
)

echo Installing Python requirements...
pip install -r requirements.txt >nul 2>&1

REM Create directories
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs

REM Start backend in background
echo [2/3] Starting Python backend server...
start /B python main.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Setup and start frontend
echo [3/3] Setting up React frontend...
cd frontend

if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
)

echo Starting React frontend...
echo.
echo ================================================
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:3000  
echo API Docs: http://localhost:8000/docs
echo ================================================
echo.

REM Start frontend (this will keep the window open)
call npm start

pause
