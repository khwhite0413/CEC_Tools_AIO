@echo off
title CEC_AllInOne Pack

set "ROOT=%~dp0"
set "OUT=%ROOT%CEC_AllInOne.zip"

if exist "%OUT%" del /f /q "%OUT%"

echo Packing CEC_AllInOne (this can take a few minutes)...

powershell -NoProfile -Command ^
    "$src='%ROOT%'.TrimEnd('\\'); $dst='%OUT%';" ^
    "$items = Get-ChildItem -LiteralPath $src ^| Where-Object { $_.Name -notin @('.git','__pycache__','CEC_AllInOne.zip') };" ^
    "Compress-Archive -Path ($items.FullName) -DestinationPath $dst -Force"

if errorlevel 1 (
    echo ERROR: pack failed
    pause
    exit /b 1
)

for %%A in ("%OUT%") do set "ZSIZE=%%~zA"
echo Done: %OUT% (%ZSIZE% bytes)
pause
