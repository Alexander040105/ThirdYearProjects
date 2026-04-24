@echo off
echo =============================================
echo Geopolitical Risk Dashboard Setup
echo =============================================
echo.

cd /d "%~dp0"

echo Installing dependencies...
pip install -r requirements_dashboard.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo =============================================
echo Starting Dashboard...
echo =============================================
echo Opening: http://localhost:8050/
echo.
echo Press Ctrl+C to stop the dashboard
echo.

python dashboard.py

pause
