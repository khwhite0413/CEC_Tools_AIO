@echo off
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PATH=%ROOT%python;%ROOT%python\Scripts;%ROOT%ffmpeg;%PATH%"
set "PYTHONHOME="
set "PYTHONPATH="
set "HF_HOME=%ROOT%weights\memfof"
REM Note: no HF_HUB_OFFLINE=1 here. Letting huggingface_hub do its
REM normal HEAD request on MEMFOF.from_pretrained costs one round-trip
REM per session and is the safer path — forcing offline mode meant any
REM cache-layout mismatch (hub version skew, partial download) failed
REM the load instead of falling back to network.

if "%1"=="launch" goto :run

REM First invocation: spawn ourselves minimized via `cmd /c` so the spawned
REM shell auto-closes when the script ends, then this cmd exits.
start /min "" cmd /c "%~f0" launch
exit /b 0

:run
"%ROOT%python\python.exe" "%ROOT%main.py"
set "RC=!ERRORLEVEL!"
REM Auto-close on success; pause only on crash so traceback stays readable.
if not "!RC!"=="0" (
    echo.
    echo === CEC All-In-One exited with code !RC! ===
    pause
)
REM `exit` (not `exit /b`) terminates the cmd shell itself, so the minimized
REM window goes away regardless of how `cmd` was invoked.
exit !RC!
