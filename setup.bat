@echo off
echo ============================================================
echo    Agentic Data Chat System - Quick Setup
echo ============================================================
echo.

REM Check if .env exists
if exist ".env" (
    echo .env file already exists!
    echo Current contents:
    type .env
    echo.
    set /p overwrite="Do you want to overwrite it? (y/n): "
    if /i not "%overwrite%"=="y" goto :skip_env
)

echo Creating .env file...
echo # Agentic Data Chat System Configuration > .env
echo # >> .env
echo # Get your API key from: https://aistudio.google.com/app/apikey >> .env
echo GOOGLE_API_KEY=your_api_key_here >> .env
echo # >> .env
echo # Optional settings >> .env
echo LOG_LEVEL=INFO >> .env
echo SESSION_TIMEOUT_MINUTES=60 >> .env
echo MAX_FILE_SIZE_MB=100 >> .env
echo # >> .env

echo ✅ .env file created successfully!
echo.
echo ⚠️  IMPORTANT: Please edit .env and add your Google API key!
echo    1. Get API key from: https://aistudio.google.com/app/apikey
echo    2. Replace 'your_api_key_here' with your actual API key
echo.

:skip_env

REM Create directories
echo Creating required directories...
if not exist "temp" mkdir temp
if not exist "logs" mkdir logs
if not exist "agents" mkdir agents
if not exist "config" mkdir config
if not exist "utils" mkdir utils

echo ✅ Directories created!
echo.

echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo    1. Edit .env file and add your Google API key
echo    2. Run: start.bat (to install dependencies and start server)
echo    3. Open: http://localhost:8000/docs (for API documentation)
echo.

pause
