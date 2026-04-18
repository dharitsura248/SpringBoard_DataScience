@echo off
:: ============================================================
::  AI Stock Analyzer — Windows Launcher
::  Runs the full portfolio analysis on Current_Holding_Apr26.csv
:: ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   AI STOCK ANALYZER
echo ============================================================
echo.

:: Check for Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add it to PATH.
    pause
    exit /b 1
)

:: Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Installing / verifying dependencies ...
    pip install -r requirements.txt --quiet
    echo.
)

:: Determine analysis mode
set MODE=basic
if "%1"=="--10k"     set MODE=with10k
if "%1"=="-h"        goto :usage
if "%1"=="--help"    goto :usage
if "%1"=="notebook"  goto :notebook

:: ── Run CLI analysis ────────────────────────────────────────
echo Running portfolio analysis on: data\Current_Holding_Apr26.csv
echo.
if "%MODE%"=="with10k" (
    python stock_analyzer.py --holdings data\Current_Holding_Apr26.csv --10k
) else (
    python stock_analyzer.py --holdings data\Current_Holding_Apr26.csv
)
goto :end

:: ── Launch Jupyter notebook ─────────────────────────────────
:notebook
echo Launching Jupyter notebook ...
jupyter notebook AI_Stock_Analyzer.ipynb
goto :end

:usage
echo.
echo Usage:  start.bat [option]
echo.
echo Options:
echo   (none)       Run portfolio analysis (CLI mode, no 10-K)
echo   --10k        Also fetch SEC 10-K filings for each holding
echo   notebook     Open the Jupyter notebook in the browser
echo   -h / --help  Show this help message
echo.

:end
endlocal
