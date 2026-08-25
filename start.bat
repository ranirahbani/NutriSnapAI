@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM NutriSnap AI — Startup Script (Windows)
REM ─────────────────────────────────────────────────────────────────────────────
REM Usage:
REM   start.bat              Normal mode (Gradio)
REM   start.bat --fallback   Force fallback HTML interface
REM ─────────────────────────────────────────────────────────────────────────────
setlocal EnableDelayedExpansion

REM ── Color tokens (ANSI via escape char) ─────────────────────────────────────
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
set "RESET=%ESC%[0m"
set "GREEN=%ESC%[38;2;44;122;74m"
set "GREEN_BOLD=%ESC%[1;38;2;44;122;74m"
set "ACCENT=%ESC%[38;2;76;175;80m"
set "YELLOW=%ESC%[33m"
set "RED=%ESC%[38;2;231;76;60m"
set "WHITE=%ESC%[38;2;232;240;236m"
set "BOLD=%ESC%[1m"
set "FAINT=%ESC%[2m"

REM ── Banner ──────────────────────────────────────────────────────────────────
cls
echo.
echo   %GREEN_BOLD%+-----------------------------------------------------------+%RESET%
echo   %GREEN_BOLD%^|                                                           ^|%RESET%
echo   %GREEN_BOLD%^|%RESET%   %BOLD%%WHITE%NutriSnap AI%RESET%                                       %GREEN_BOLD%^|%RESET%
echo   %GREEN_BOLD%^|%RESET%   %FAINT%%ACCENT%Snap a photo. Know your nutrition.%RESET%                 %GREEN_BOLD%^|%RESET%
echo   %GREEN_BOLD%^|                                                           ^|%RESET%
echo   %GREEN_BOLD%+-----------------------------------------------------------+%RESET%
echo.

REM ── Resolve script directory ─────────────────────────────────────────────────
cd /d "%~dp0"

REM ── Step 1 — Check Python 3 ─────────────────────────────────────────────────
echo   %ACCENT%[check]%RESET% Checking for Python 3...

set "PYTHON_BIN="
where python3 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_BIN=python3"
    goto :found_python
)
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYV=%%v"
    echo !PYV! | findstr /B "3" >nul
    if !ERRORLEVEL! EQU 0 (
        set "PYTHON_BIN=python"
        goto :found_python
    )
)

echo   %RED%[error]%RESET% Python 3 is not installed or not on your PATH.
echo.
echo   %YELLOW%Install it from:%RESET%   https://www.python.org/downloads/
echo   %YELLOW%winget:%RESET%            winget install Python.Python.3.11
echo   %YELLOW%Chocolatey:%RESET%        choco install python3
echo.
exit /b 1

:found_python
for /f "tokens=2 delims= " %%v in ('%PYTHON_BIN% --version 2^>^&1') do set "PY_VERSION=%%v"
echo   %GREEN%[ ok ]%RESET% Found Python %BOLD%%PY_VERSION%%RESET% (%PYTHON_BIN%)

REM ── Step 2 — Virtual environment ────────────────────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo   %GREEN%[ ok ]%RESET% Virtual environment already exists at %BOLD%venv\%RESET%
    goto :activate_venv
)

echo   %ACCENT%[info]%RESET% No virtual environment found — creating one...
%PYTHON_BIN% -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo   %RED%[error]%RESET% Could not create virtual environment.
    echo   %YELLOW%Make sure Python 3.8+ is installed and try again.%RESET%
    exit /b 1
)
echo   %GREEN%[ ok ]%RESET% Virtual environment created at %BOLD%venv\%RESET%

:activate_venv
REM ── Step 3 — Activate virtual environment ───────────────────────────────────
echo   %ACCENT%[info]%RESET% Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo   %RED%[error]%RESET% Failed to activate virtual environment.
    echo   %YELLOW%Try running manually:%RESET%  venv\Scripts\activate
    exit /b 1
)
echo   %GREEN%[ ok ]%RESET% Virtual environment activated

REM ── Step 4 — Upgrade pip ────────────────────────────────────────────────────
echo   %ACCENT%[info]%RESET% Upgrading pip...
pip install --upgrade pip --quiet 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   %YELLOW%[warn]%RESET%  pip upgrade failed ^(non-fatal, continuing...^)
) else (
    echo   %GREEN%[ ok ]%RESET% pip upgraded
)

REM ── Step 5 — Install requirements ───────────────────────────────────────────
echo.
echo   %ACCENT%[info]%RESET% Installing dependencies from requirements.txt...
echo   %FAINT%-------------------------------------------------------------%RESET%

pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   %RED%[error]%RESET% Dependency installation failed.
    echo.
    echo   %YELLOW%Common fixes:%RESET%
    echo     * Update pip:         %FAINT%python -m pip install --upgrade pip%RESET%
    echo     * For torch issues:   %FAINT%pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu%RESET%
    echo     * Check Python:       %FAINT%python --version%RESET%  ^(need 3.8+^)
    echo     * Use a fresh venv:   %FAINT%rmdir /s /q venv%RESET%  then re-run this script
    echo.
    exit /b 1
)

echo   %FAINT%-------------------------------------------------------------%RESET%
echo   %GREEN%[ ok ]%RESET% All dependencies installed successfully

REM Show installed packages
echo.
echo   %ACCENT%[info]%RESET% Installed packages:
for /f "usebackq delims=" %%L in ("requirements.txt") do (
    set "line=%%L"
    if not "!line:~0,1!"=="#" if not "!line!"=="" (
        for /f "tokens=1 delims=>=<!" %%p in ("!line!") do set "pkg=%%p"
        for /f "tokens=2 delims=: " %%v in ('pip show !pkg! 2^>nul ^| findstr "^Version:"') do set "pkgver=%%v"
        if defined pkgver (
            echo     %ACCENT%*%RESET% !pkg! %FAINT%!pkgver!%RESET%
        ) else (
            echo     %ACCENT%*%RESET% !pkg! %FAINT%(installed)%RESET%
        )
        set "pkgver="
    )
)

REM ── Step 6 — Launch the app ────────────────────────────────────────────────
echo.
echo   %GREEN_BOLD%---------------------------------------------------------------%RESET%
echo   %GREEN_BOLD%  Launching NutriSnap AI...%RESET%
echo   %GREEN_BOLD%---------------------------------------------------------------%RESET%
echo.

REM Pass through all CLI arguments (e.g. --fallback)
python app.py %*

endlocal
