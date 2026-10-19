@echo off
REM ---------------------------------------------------------------------------
REM wave.bat - open one batch of 8 images in the Vivado simulator GUI
REM When the window opens: in the Scope panel right-click "dut" (or "u_gemm")
REM -> Add to Wave Window, then click Run All (the blue play button).
REM If a box asks whether to quit at the end, click No.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"
call env.bat
if errorlevel 1 goto :fail

if not exist work_wave mkdir work_wave
cd work_wave
copy /y ..\..\mem\N8\weights.mem . >nul
copy /y ..\..\mem\N8\bias.mem . >nul

call xvlog -sv -i ..\..\rtl ^
  ..\..\rtl\npu_delay.sv ..\..\rtl\npu_pe.sv ..\..\rtl\npu_array.sv ..\..\rtl\npu_gemm.sv ^
  ..\..\rtl\npu_requant.sv ..\..\rtl\npu_ram.sv ..\..\rtl\npu_rom.sv ..\..\rtl\npu_core.sv ^
  ..\..\rtl\npu_axil_regs.sv ..\..\rtl\npu_top.sv ..\..\tb\tb_npu.sv
if errorlevel 1 goto :fail
>xelab_args.txt echo -generic_top N=8
call xelab tb_npu -f xelab_args.txt -s tb_wave -timescale 1ns/1ps -debug typical
if errorlevel 1 goto :fail
>xsim_args.txt echo -testplusarg NUM_IMAGES=8 -testplusarg MEM_DIR=../../mem
call xsim tb_wave -gui -f xsim_args.txt
cd ..
goto :eof

:fail
echo.
echo ***** FAILED - read the error messages above *****
exit /b 1
