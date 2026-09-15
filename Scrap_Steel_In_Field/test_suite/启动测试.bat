
@echo off
chcp 65001 >nul
echo ========================================
echo   废钢装箱系统测试套件
echo ========================================
echo.
echo 正在启动...
cd /d "%~dp0"
python main.py
if errorlevel 1 pause
