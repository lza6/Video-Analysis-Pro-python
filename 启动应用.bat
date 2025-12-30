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

if exist "venv\Scripts\python.exe" goto :found_venv

rem --- Fallback to System Python ---
echo [INFO] Venv not found, checking system python...
python --version >nul 2>&1
if errorlevel 1 goto :no_python

echo [INFO] Using System Python.
powershell -Command "python launcher.py | Tee-Object -FilePath logs\startup_bat.log"
goto :eof

:found_venv
echo [INFO] Detected Virtual Environment. Using venv python.
powershell -Command ".'venv\Scripts\python.exe' launcher.py | Tee-Object -FilePath logs\startup_bat.log"
goto :eof

:no_python
echo [ERROR] Python not found!
echo Please install Python 3.8+ and add to PATH.
exit /b 1
