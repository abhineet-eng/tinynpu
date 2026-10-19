@echo off
REM ---------------------------------------------------------------------------
REM env.bat - finds Vivado and loads its settings (called by the other scripts)
REM If Vivado is somewhere else, change the path on the next line to the full
REM path of its settings64.bat.
REM ---------------------------------------------------------------------------
set "VIVADO_SETTINGS=C:\Users\CBI\Desktop\VIVADO\Vivado\2023.2\settings64.bat"

where xvlog >nul 2>nul
if not errorlevel 1 exit /b 0

if defined VIVADO_SETTINGS if not exist "%VIVADO_SETTINGS%" set "VIVADO_SETTINGS="
if defined VIVADO_SETTINGS goto :load

for /d %%D in ("C:\Xilinx\Vivado\*" "D:\Xilinx\Vivado\*") do if exist "%%D\settings64.bat" set "VIVADO_SETTINGS=%%D\settings64.bat"
for /d %%D in ("C:\Xilinx\*" "D:\Xilinx\*" "C:\AMDDesignTools\*" "D:\AMDDesignTools\*") do if exist "%%D\Vivado\settings64.bat" set "VIVADO_SETTINGS=%%D\Vivado\settings64.bat"
for /d %%D in ("%USERPROFILE%\Desktop\VIVADO\Vivado\*") do if exist "%%D\settings64.bat" set "VIVADO_SETTINGS=%%D\settings64.bat"

if not defined VIVADO_SETTINGS (
  echo.
  echo ERROR: Could not find Vivado. Open sim\env.bat and set VIVADO_SETTINGS.
  echo.
  exit /b 1
)

:load
echo Using Vivado settings: %VIVADO_SETTINGS%
call "%VIVADO_SETTINGS%"
exit /b 0
