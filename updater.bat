@echo off
setlocal enableextensions enabledelayedexpansion
title CEC All-In-One Updater

REM ============================================================
REM  CEC All-In-One auto-updater helper
REM
REM  Spawned by app/updater.py after a release ZIP is downloaded
REM  and extracted. This script waits for the running app to
REM  exit, overwrites source files in place (skipping big/local
REM  state like weights, python, ffmpeg, cache, output, logs),
REM  runs setup.bat to sync any new deps, then relaunches the
REM  app via run.bat.
REM
REM  Arguments (all positional, quoted):
REM    %1  PID to wait for (the running python.exe)
REM    %2  source directory (extracted zipball root)
REM    %3  destination directory (project root — where this .bat lives)
REM ============================================================

set "WAIT_PID=%~1"
set "SRC=%~2"
set "DST=%~3"

if "%WAIT_PID%"=="" (
    echo ERROR: updater.bat requires 3 args: PID, source, dest
    pause
    exit /b 1
)
if not exist "%SRC%" (
    echo ERROR: source directory not found: %SRC%
    pause
    exit /b 1
)
if not exist "%DST%\setup.bat" (
    echo ERROR: destination doesn't look like a CEC install: %DST%
    pause
    exit /b 1
)

set "LOG=%DST%\logs\updater.log"
if not exist "%DST%\logs" mkdir "%DST%\logs"
> "%LOG%" echo === CEC updater start %date% %time% ===
>> "%LOG%" echo Source: %SRC%
>> "%LOG%" echo Dest:   %DST%
>> "%LOG%" echo PID:    %WAIT_PID%

echo.
echo ============================================
echo   CEC All-In-One Auto-Updater
echo ============================================
echo.
echo 앱이 종료되기를 기다리는 중... (PID: %WAIT_PID%)
echo.

REM Wait for the app process to exit. Cap at 60s so a hung app
REM doesn't leave the updater spinning forever.
set /a WAIT_COUNT=0
:waitloop
tasklist /FI "PID eq %WAIT_PID%" 2>NUL | find /I "%WAIT_PID%" >NUL
if errorlevel 1 goto :do_update
set /a WAIT_COUNT+=1
if !WAIT_COUNT! GEQ 60 (
    echo.
    echo TIMEOUT: 앱이 60초 안에 종료되지 않았습니다. 업데이트 중단.
    >> "%LOG%" echo TIMEOUT waiting for PID %WAIT_PID%
    pause
    goto :end
)
timeout /t 1 /nobreak >NUL
goto :waitloop

:do_update
echo.
echo [1/3] 소스 파일 교체 중...
>> "%LOG%" echo App exited, replacing source files...

REM robocopy from staging → project root.
REM   /E                 include subdirs (even empty)
REM   /XD ...            skip user data + generated dirs. Anything in
REM                      this list stays untouched on the user's disk.
REM   /XF updater.bat    the file we're running from is Windows-locked
REM                      right now; robocopy would skip it anyway but
REM                      being explicit avoids a warning line in the log.
REM   /NFL /NDL /NP      quiet output (no per-file/per-dir/no-progress)
REM   /R:1 /W:1          minimal retry — we don't want a permission-denied
REM                      loop stalling the whole update
REM Exit code:  0..7 = success, 8+ = errors
robocopy "%SRC%" "%DST%" /E ^
    /XD weights python ffmpeg cache output logs debug .git __pycache__ ^
    /XF updater.bat ^
    /NFL /NDL /NP /R:1 /W:1 >> "%LOG%"

if errorlevel 8 (
    echo.
    echo 파일 교체 실패. 자세한 내용은 로그 확인:
    echo   %LOG%
    >> "%LOG%" echo robocopy FAILED with exit code %errorlevel%
    pause
    goto :end
)

REM Belt-and-braces CRLF fix. GitHub zipballs may ship .bat files with
REM LF-only line endings depending on .gitattributes, and cmd.exe's
REM parser chokes on them mid-script — `setlocal enabledelayedexpansion`
REM gets read as the mysterious command `tlocal`, multi-line `if (`
REM blocks fragment into isolated "not recognized" errors, etc. This
REM step runs a python one-liner to normalize every .bat we just
REM copied. Uses the embedded python we know is installed at
REM %DST%\python\python.exe. Safe to no-op if python is missing —
REM setup.bat will just fail the same way it would have anyway.
>> "%LOG%" echo Normalizing .bat line endings (LF -^> CRLF) after robocopy...
"%DST%\python\python.exe" -c "import glob,os;[open(f,'wb').write(open(f,'rb').read().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')) for f in glob.glob(os.path.join(r'%DST%','*.bat'))]" >> "%LOG%" 2>&1

echo [2/3] 신규 의존성 확인 (setup.bat)...
echo.
>> "%LOG%" echo Running setup.bat with CEC_AUTO_INSTALL=1...

REM CEC_AUTO_INSTALL suppresses setup.bat's final `pause` so it
REM doesn't halt our chain waiting for a keypress. setup.bat is
REM idempotent — python/weights/torch already-installed cases
REM short-circuit fast, so this is usually a few seconds.
set "CEC_AUTO_INSTALL=1"
call "%DST%\setup.bat"
set "SETUP_RC=%errorlevel%"

if not "%SETUP_RC%"=="0" (
    echo.
    echo setup.bat 실패 rc=%SETUP_RC%. 로그: %LOG%
    >> "%LOG%" echo setup.bat FAILED rc=%SETUP_RC%
    pause
    goto :end
)

echo.
echo [3/3] 앱 재시작...
>> "%LOG%" echo Relaunching app via run.bat...

REM start "" spawns run.bat detached in a new window. The empty ""
REM is start's window-title arg — omitting it makes start treat the
REM .bat path as the title, which is silently wrong.
start "" "%DST%\run.bat"

REM Clean up the staging temp folder (best effort; ignore failures
REM since Python's tempfile.mkdtemp leaves it under %TEMP% which
REM Windows sweeps periodically anyway).
>> "%LOG%" echo Cleaning up staging: %SRC%
rmdir /s /q "%SRC%" >NUL 2>&1

>> "%LOG%" echo === updater end %date% %time% ===

echo.
echo 업데이트 완료. 이 창은 자동으로 닫힙니다.
timeout /t 2 /nobreak >NUL

:end
endlocal
exit /b 0
