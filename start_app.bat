@echo off
setlocal
cd /d "%~dp0"
if not exist logs mkdir logs

echo ========================================
echo   Video Analysis Pro (视频分析专业版)
echo ========================================
echo.
echo 正在启动 (详细日志: logs/startup_bat.log)...
echo.

call :main

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 启动脚本执行出错!
    echo 请查看 logs/startup_bat.log 获取详情。
    echo.
    echo === 错误日志预览 (最后 20 行) ===
    powershell -command "Get-Content logs/startup_bat.log -Tail 20"
    echo ===============================
    color 0C
) else (
    echo.
    echo 应用已关闭.
)

echo.
pause
exit /b

:main
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo [INFO] Batch script started at %date% %time%
echo [INFO] Working Directory: %cd%

rem --- 1. Prefer existing venv ---
if exist "venv\Scripts\python.exe" (
    echo [INFO] Detected Virtual Environment. Using venv python.
    powershell -Command "& 'venv\Scripts\python.exe' launcher.py 2>&1 | Tee-Object -FilePath logs\startup_bat.log"
    goto :eof
)

rem --- 2. Probe Python 3.10/3.11/3.12 via the py launcher (newest first) ---
echo [INFO] Venv not found, probing for Python 3.10-3.12...

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set FOUND_PY=-3.12
    goto :found
)

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set FOUND_PY=-3.11
    goto :found
)

py -3.10 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set FOUND_PY=-3.10
    goto :found
)

rem --- 3. Default python on PATH with acceptable version? (plain `python` often
rem        maps to a Microsoft Store stub or 3.13+, so we gate explicitly) ---
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 (
    set FOUND_PY=-3
    goto :found
)

echo [ERROR] 未找到 Python 3.10 - 3.12！
echo.
echo 本软件需要 Python 3.10 / 3.11 / 3.12（暂不支持 3.13+，部分 AI 依赖没有对应版本）。
echo 请从以下地址下载安装，并勾选 "Add python.exe to PATH":
echo     https://www.python.org/downloads/release/python-31011/
echo.
start "" "https://www.python.org/downloads/release/python-31011/"
exit /b 1

:found
echo [INFO] Using py %FOUND_PY%
powershell -Command "& py %FOUND_PY% launcher.py 2>&1 | Tee-Object -FilePath logs\startup_bat.log"
goto :eof
