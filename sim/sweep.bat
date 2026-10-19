@echo off
REM ---------------------------------------------------------------------------
REM sweep.bat - run every array size (1x1 .. 16x16) and print the scaling table
REM   sweep.bat              64 images per size
REM   sweep.bat --images 160
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
call env.bat
if errorlevel 1 exit /b 1
python sweep.py %*
