@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion
set "PYTHONIOENCODING=utf-8"
set "LOGFILE=logs\desktop.log"

rem --- 日志目录(实时黑匣子日志,持续输出) ---
if not exist logs mkdir logs
echo. > "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"
echo TingFeng Hermes Desktop - %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"

goto :main

:tee
echo %~1
echo [%time%] %~1 >> "%LOGFILE%"
goto :eof

:main
call :tee "========================================"
call :tee "  TingFeng Hermes - Desktop (Electron)"
call :tee "  v10.3.1"
call :tee "========================================"
call :tee ""
call :tee "Starting Electron desktop shell..."
call :tee ""

rem --- 1. check node ---
where node >nul 2>&1
if errorlevel 1 (
    call :tee "[ERROR] Node.js not found!"
    call :tee "Desktop needs Node.js 16+ LTS. Download from https://nodejs.org/"
    call :tee ""
    pause
    exit /b 1
)

rem --- 2. check npm ---
where npm >nul 2>&1
if errorlevel 1 (
    call :tee "[ERROR] npm not found!"
    call :tee "Node.js install usually ships npm, check install."
    call :tee ""
    pause
    exit /b 2
)

call :tee "[INFO] Node.js:"
for /f "delims=" %%v in ('node -v') do call :tee "  %%v"
call :tee "[INFO] npm:"
for /f "delims=" %%v in ('call npm -v') do call :tee "  %%v"
call :tee ""

rem --- 3. cleanup stale electron + port-occupying python ---
call :tee "[INFO] cleaning stale processes..."
taskkill /F /IM electron.exe /T >nul 2>&1
node desktop\cleanup-stale.js
call :tee "[INFO] cleanup done."
call :tee ""

rem --- 4. install desktop deps on first run ---
if not exist "desktop\node_modules" (
    call :tee "[INFO] first run, installing desktop deps..."
    call :tee "[INFO] this may take a few minutes..."
    call :tee ""
    pushd desktop
    call npm install
    set NPM_RC=!errorlevel!
    popd
    if not "!NPM_RC!"=="0" (
        call :tee "[ERROR] npm install failed ^(exit !NPM_RC!^)."
        call :tee "try: npm config set registry https://registry.npmmirror.com"
        call :tee ""
        pause
        exit /b 3
    )
    call :tee "[INFO] deps installed."
    call :tee ""
)

rem --- 5. launch electron (前台,stdout/stderr 实时 tee 到日志+控制台) ---
call :tee "[INFO] starting electron..."
call :tee "[INFO] main.js spawns python serve.py and loadURL."
call :tee "[INFO] close window to quit, python service ends with it."
call :tee "[INFO] real-time log: %LOGFILE%"
call :tee ""

pushd desktop
call npm start 2>&1
set START_RC=!errorlevel!
popd

call :tee ""
if !START_RC! neq 0 (
    call :tee "[ERROR] electron exited with code !START_RC!"
    color 0C
) else (
    call :tee "app closed."
)

call :tee ""
call :tee "=== log saved to %LOGFILE% ==="
echo.
pause
exit /b !START_RC!
