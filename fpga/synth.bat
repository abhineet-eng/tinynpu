@echo off
REM ---------------------------------------------------------------------------
REM synth.bat - synthesize and place-and-route TinyNPU for an Artix-7 FPGA
REM   synth.bat                         8x8 array on xc7a35t (Basys 3 / Arty A7-35T)
REM   synth.bat 16 xc7a100tcsg324-1     16x16 array on xc7a100t (Nexys A7 / Arty A7-100T)
REM Takes 5-15 minutes. Results: fpga\build_N<n>\reports\summary.txt
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
call ..\sim\env.bat
if errorlevel 1 goto :fail

set "N=%~1"
if "%N%"=="" set "N=8"
set "PART=%~2"
if "%PART%"=="" set "PART=xc7a35tcpg236-1"
set "CLK=%~3"
if "%CLK%"=="" set "CLK=10.0"

if not exist build_N%N% mkdir build_N%N%
cd build_N%N%
copy /y ..\..\mem\N%N%\weights.mem . >nul
copy /y ..\..\mem\N%N%\bias.mem . >nul

call vivado -mode batch -nojournal -log vivado.log -source ..\synth.tcl -tclargs %N% %PART% %CLK%
if errorlevel 1 goto :fail
cd ..
echo.
echo Summary saved to fpga\build_N%N%\reports\summary.txt
goto :eof

:fail
echo.
echo ***** FAILED - read the error messages above (or fpga\build_N%N%\vivado.log) *****
exit /b 1
