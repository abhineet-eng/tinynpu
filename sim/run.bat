@echo off
REM ---------------------------------------------------------------------------
REM run.bat - simulate TinyNPU in the Vivado simulator (xsim)
REM   run.bat                 8x8 array, all 10,000 test images
REM   run.bat 8 1000          8x8 array, first 1,000 images
REM   run.bat 16 10000        16x16 array
REM Array sizes: 1, 2, 4, 8, 16
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
call env.bat
if errorlevel 1 goto :fail

set "N=%~1"
if "%N%"=="" set "N=8"
set "NUM=%~2"
if "%NUM%"=="" set "NUM=10000"
set "EXTRA=%~3"

if not exist ..\mem\N%N%\weights.mem (
  echo ERROR: no weights for N=%N%. Use 1, 2, 4, 8 or 16.
  goto :fail
)

if not exist work_N%N% mkdir work_N%N%
cd work_N%N%
copy /y ..\..\mem\N%N%\weights.mem . >nul
copy /y ..\..\mem\N%N%\bias.mem . >nul

echo.
echo ===== Compiling (N=%N%) =====
call xvlog -sv -d NPU_SVA -i ..\..\rtl ^
  ..\..\rtl\npu_delay.sv ..\..\rtl\npu_pe.sv ..\..\rtl\npu_array.sv ..\..\rtl\npu_gemm.sv ^
  ..\..\rtl\npu_requant.sv ..\..\rtl\npu_ram.sv ..\..\rtl\npu_rom.sv ..\..\rtl\npu_core.sv ^
  ..\..\rtl\npu_axil_regs.sv ..\..\rtl\npu_top.sv ..\..\tb\npu_sva.sv ..\..\tb\tb_npu.sv
if errorlevel 1 goto :fail

echo.
echo ===== Elaborating =====
REM Options containing "=" go in files: Windows batch files split arguments at "="
>xelab_args.txt echo -generic_top N=%N%
call xelab tb_npu -f xelab_args.txt -s tb_snap -timescale 1ns/1ps
if errorlevel 1 goto :fail

echo.
echo ===== Running %NUM% images =====
>xsim_args.txt echo -testplusarg NUM_IMAGES=%NUM% -testplusarg MEM_DIR=../../mem
if not "%EXTRA%"=="" >>xsim_args.txt echo -testplusarg %EXTRA%
call xsim tb_snap -R -f xsim_args.txt -log run.log
cd ..
echo.
echo Log saved to sim\work_N%N%\run.log
goto :eof

:fail
echo.
echo ***** FAILED - read the error messages above *****
exit /b 1
