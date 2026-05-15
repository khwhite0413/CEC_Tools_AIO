@echo off
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PATH=%ROOT%python;%ROOT%python\Scripts;%ROOT%ffmpeg;%PATH%"
set "PYTHONHOME="
set "PYTHONPATH="
set "HF_HOME=%ROOT%weights\memfof"
REM Skip every huggingface_hub network call — MEMFOF.from_pretrained
REM otherwise sends a HEAD request to HF Hub on each load to verify the
REM cached snapshot, which hangs or rate-limits (429) on slow networks
REM and froze the batch tab at "0%" on the denoise stage. setup.bat
REM has already populated the cache; offline-mode just trusts what's
REM on disk and skips the network round-trip entirely.
set "HF_HUB_OFFLINE=1"

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
