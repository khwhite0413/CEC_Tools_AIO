@echo off
setlocal enabledelayedexpansion
title CEC All-In-One Setup

echo ============================================
echo   CEC All-In-One Auto Setup
echo ============================================
echo.

set "ROOT=%~dp0"
set "PYTHON_DIR=%ROOT%python"
set "PYTHON=%PYTHON_DIR%\python.exe"

REM Isolate from global Python (keep system PATH appended)
set "PYTHONHOME="
set "PYTHONPATH="
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"

REM === Step 1: Python embeddable ===
if exist "%PYTHON%" (
    echo [1/10] Python embedded - already installed
) else (
    echo [1/10] Downloading Python 3.12 embedded...
    mkdir "%PYTHON_DIR%" 2>nul
    curl -sL -o "%ROOT%python-embed.zip" "https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip"
    if errorlevel 1 (
        echo    ERROR: download failed
        goto :fail
    )
    for %%A in ("%ROOT%python-embed.zip") do set "PYSIZE=%%~zA"
    if !PYSIZE! LSS 5000000 (
        echo    ERROR: python embed zip too small ^(!PYSIZE! bytes^)
        del /f /q "%ROOT%python-embed.zip"
        goto :fail
    )
    powershell -NoProfile -Command "Expand-Archive -Path '%ROOT%python-embed.zip' -DestinationPath '%PYTHON_DIR%' -Force"
    del "%ROOT%python-embed.zip"
    powershell -NoProfile -Command "(Get-Content '%PYTHON_DIR%\python312._pth') -replace '#import site','import site' | Set-Content '%PYTHON_DIR%\python312._pth'"
    echo ..>> "%PYTHON_DIR%\python312._pth"
    curl -sL -o "%PYTHON_DIR%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py"
    "%PYTHON%" "%PYTHON_DIR%\get-pip.py" --no-warn-script-location -q
    del "%PYTHON_DIR%\get-pip.py"
    echo    Done
)

REM === Step 2: Detect GPU and install PyTorch ===
echo [2/10] Detecting GPU...

"%PYTHON%" -c "import torch" >nul 2>&1
if not errorlevel 1 (
    echo    PyTorch already installed
    goto :step3
)

set "CUDA_INDEX=cpu"
set "CUDA_MAJOR=0"

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo    No NVIDIA GPU - CPU mode
    goto :install_torch
)

for /f "tokens=9 delims= " %%a in ('nvidia-smi ^| findstr "CUDA Version"') do set "CUDA_VER=%%a"
for /f "tokens=1 delims=." %%a in ("%CUDA_VER%") do set "CUDA_MAJOR=%%a"

echo    CUDA Version: %CUDA_VER% ^(major: %CUDA_MAJOR%^)

if "%CUDA_MAJOR%"=="13" set "CUDA_INDEX=cu128" & goto :install_torch
if "%CUDA_MAJOR%"=="12" set "CUDA_INDEX=cu124" & goto :install_torch
if "%CUDA_MAJOR%"=="11" set "CUDA_INDEX=cu118" & goto :install_torch
set "CUDA_INDEX=cu121"

:install_torch
echo    Installing PyTorch ^(%CUDA_INDEX%^)...
if "%CUDA_INDEX%"=="cpu" (
    "%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location -q
) else if "%CUDA_INDEX%"=="cu128" (
    "%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --no-warn-script-location -q
) else (
    "%PYTHON%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_INDEX% --no-warn-script-location -q
)

"%PYTHON%" -c "import torch; print('    PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo    ERROR: PyTorch import failed
    goto :fail
)

:step3
REM === Step 3: Python dependencies ===
echo [3/10] Installing dependencies...
"%PYTHON%" -m pip install --no-warn-script-location -q -r "%ROOT%requirements.txt"
if errorlevel 1 (
    echo    ERROR: pip install failed
    goto :fail
)
echo    Done

REM === Step 4: MEMFOF (vendored or git) ===
echo [4/10] Installing MEMFOF...
"%PYTHON%" -c "import memfof" >nul 2>&1
if not errorlevel 1 (
    echo    MEMFOF already installed
    goto :step5
)

if exist "%ROOT%third_party\memfof\setup.py" (
    "%PYTHON%" -m pip install --no-warn-script-location -q "%ROOT%third_party\memfof"
    goto :verify_memfof
)
if exist "%ROOT%third_party\memfof\pyproject.toml" (
    "%PYTHON%" -m pip install --no-warn-script-location -q "%ROOT%third_party\memfof"
    goto :verify_memfof
)

echo    Installing MEMFOF from GitHub ^(no vendored copy found^)...
"%PYTHON%" -m pip install --no-warn-script-location -q "git+https://github.com/msu-video-group/memfof.git"

:verify_memfof
"%PYTHON%" -c "import memfof; print('    MEMFOF OK')"
if errorlevel 1 (
    echo    ERROR: MEMFOF install failed
    goto :fail
)

:step5
REM === Step 5: Pre-fetch MEMFOF weights ===
echo [5/10] Fetching MEMFOF weights ^(first run only^)...
set "HF_HOME=%ROOT%weights\memfof"
"%PYTHON%" -c "import os; os.environ['HF_HOME']=r'%ROOT%weights\memfof'; from memfof import MEMFOF; m=MEMFOF.from_pretrained('egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH'); print('    weights OK')"
if errorlevel 1 (
    echo    WARNING: MEMFOF weights download failed - will retry on first run
)

REM === Step 6: ProPainter source (vendored) ===
echo [6/10] Verifying ProPainter source...
if exist "%ROOT%third_party\ProPainter\model\propainter.py" (
    echo    ProPainter source OK
    goto :step7
)
echo    ProPainter source missing - downloading...
if exist "%ROOT%third_party\propainter.zip" del /f /q "%ROOT%third_party\propainter.zip"
if exist "%ROOT%third_party\ProPainter-main" rmdir /s /q "%ROOT%third_party\ProPainter-main"
mkdir "%ROOT%third_party" 2>nul
curl -L -# -o "%ROOT%third_party\propainter.zip" "https://github.com/sczhou/ProPainter/archive/refs/heads/main.zip"
if errorlevel 1 (
    echo    ERROR: ProPainter download failed
    goto :fail
)
for %%A in ("%ROOT%third_party\propainter.zip") do set "ZIPSIZE=%%~zA"
if !ZIPSIZE! LSS 5000000 (
    echo    ERROR: ProPainter zip too small ^(!ZIPSIZE! bytes^)
    del /f /q "%ROOT%third_party\propainter.zip"
    goto :fail
)
powershell -NoProfile -Command "Expand-Archive -Path '%ROOT%third_party\propainter.zip' -DestinationPath '%ROOT%third_party' -Force"
if errorlevel 1 (
    echo    ERROR: ProPainter extraction failed
    goto :fail
)
if exist "%ROOT%third_party\ProPainter-main" (
    move "%ROOT%third_party\ProPainter-main" "%ROOT%third_party\ProPainter" >nul
)
del /f /q "%ROOT%third_party\propainter.zip" 2>nul
if not exist "%ROOT%third_party\ProPainter\model\propainter.py" (
    echo    ERROR: ProPainter files missing after extraction
    goto :fail
)
echo    ProPainter source OK

:step7
REM === Step 7: ProPainter weights ===
echo [7/10] Fetching ProPainter weights ^(~580MB, first run only^)...
"%PYTHON%" -c "from vdcore.model_downloader import download_all_weights; download_all_weights(progress_callback=print)"
if errorlevel 1 (
    echo    WARNING: ProPainter weights download failed - retry from the Inpaint tab
)

REM === Step 8: OpenProteus upscale weight ===
echo [8/10] Fetching OpenProteus upscale weight ^(~2.4MB^)...
set "UPS_DIR=%ROOT%weights\openproteus"
mkdir "%UPS_DIR%" 2>nul
set "UPS_FILE=%UPS_DIR%\2x_OpenProteus_Compact_i2_70K.pth"
if exist "%UPS_FILE%" (
    echo    OpenProteus weight already present
) else (
    curl -sL -o "%UPS_FILE%" "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth"
    if errorlevel 1 (
        echo    WARNING: OpenProteus download failed - retry from the Upscale tab
    ) else (
        for %%A in ("%UPS_FILE%") do set "USIZE=%%~zA"
        if !USIZE! LSS 1000000 (
            echo    WARNING: OpenProteus file too small ^(!USIZE! bytes^), removing
            del /f /q "%UPS_FILE%"
        ) else (
            echo    OpenProteus weight OK
        )
    )
)

REM === Step 9: Pretendard fonts (bundled at runtime, OFL) ===
echo [9/10] Fetching Pretendard fonts...
set "FONT_DIR=%ROOT%assets\fonts"
mkdir "%FONT_DIR%" 2>nul
set "FONT_BASE=https://cdn.jsdelivr.net/gh/orioncactus/pretendard@latest/packages/pretendard/dist/public/static"
for %%W in (Regular Medium SemiBold Bold) do (
    if not exist "%FONT_DIR%\Pretendard-%%W.otf" (
        curl -sL -o "%FONT_DIR%\Pretendard-%%W.otf" "%FONT_BASE%/Pretendard-%%W.otf"
        for %%A in ("%FONT_DIR%\Pretendard-%%W.otf") do set "FSIZE=%%~zA"
        if !FSIZE! LSS 500000 (
            echo    WARNING: Pretendard-%%W.otf too small ^(!FSIZE! bytes^), removing
            del /f /q "%FONT_DIR%\Pretendard-%%W.otf"
        )
    )
)
if exist "%FONT_DIR%\Pretendard-Regular.otf" (
    echo    Pretendard fonts OK
) else (
    echo    WARNING: Pretendard download failed - app will use system font
)

REM === Step 10: ffmpeg + ffprobe (bundled together) ===
REM imageio_ffmpeg only ships ffmpeg.exe; ffprobe.exe is on no PATH on a
REM fresh install, so the batch tab's "no audio" probe used to fall back
REM to the slower ffmpeg-stderr scan that occasionally failed under
REM non-UTF-8 locales / slow drives. Bundling a full ffmpeg+ffprobe pair
REM (BtbN GPL build, ~80MB) into ./ffmpeg lets _get_ffmpeg/_get_ffprobe
REM in fdcore find both binaries side-by-side and the probe never trips.
echo [10/10] Bundling ffmpeg + ffprobe ^(~80MB, first run only^)...
set "FFMPEG_DIR=%ROOT%ffmpeg"
if exist "%FFMPEG_DIR%\ffmpeg.exe" if exist "%FFMPEG_DIR%\ffprobe.exe" (
    echo    ffmpeg + ffprobe already bundled
    goto :ffmpeg_done
)
mkdir "%FFMPEG_DIR%" 2>nul
if exist "%ROOT%ffmpeg.zip" del /f /q "%ROOT%ffmpeg.zip"
if exist "%ROOT%ffmpeg_tmp" rmdir /s /q "%ROOT%ffmpeg_tmp"
curl -L -# -o "%ROOT%ffmpeg.zip" "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-win64-gpl-7.1.zip"
if errorlevel 1 (
    echo    WARNING: ffmpeg download failed - app falls back to imageio_ffmpeg
    goto :ffmpeg_done
)
for %%A in ("%ROOT%ffmpeg.zip") do set "FFSIZE=%%~zA"
if !FFSIZE! LSS 10000000 (
    echo    WARNING: ffmpeg zip too small ^(!FFSIZE! bytes^), removing
    del /f /q "%ROOT%ffmpeg.zip"
    goto :ffmpeg_done
)
powershell -NoProfile -Command "Expand-Archive -Path '%ROOT%ffmpeg.zip' -DestinationPath '%ROOT%ffmpeg_tmp' -Force"
if errorlevel 1 (
    echo    WARNING: ffmpeg extraction failed
    del /f /q "%ROOT%ffmpeg.zip" 2>nul
    goto :ffmpeg_done
)
REM Zip's top-level is a single versioned folder (ffmpeg-n7.1-...); pull the
REM two exes out of its bin/ regardless of the exact suffix.
for /d %%D in ("%ROOT%ffmpeg_tmp\*") do (
    if exist "%%D\bin\ffmpeg.exe" (
        move /y "%%D\bin\ffmpeg.exe" "%FFMPEG_DIR%\ffmpeg.exe" >nul
        move /y "%%D\bin\ffprobe.exe" "%FFMPEG_DIR%\ffprobe.exe" >nul
    )
)
rmdir /s /q "%ROOT%ffmpeg_tmp" 2>nul
del /f /q "%ROOT%ffmpeg.zip" 2>nul
if exist "%FFMPEG_DIR%\ffmpeg.exe" if exist "%FFMPEG_DIR%\ffprobe.exe" (
    echo    ffmpeg + ffprobe bundled OK
) else (
    echo    WARNING: ffmpeg extraction incomplete - app falls back to imageio_ffmpeg
)
:ffmpeg_done

echo.
echo ============================================
echo   Setup complete! Run: run.bat
echo ============================================
pause
exit /b 0

:fail
echo.
echo ============================================
echo   Setup FAILED. See errors above.
echo ============================================
pause
exit /b 1
