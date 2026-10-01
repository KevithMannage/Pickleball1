@echo off
setlocal enabledelayedexpansion

:: ===========================================================================
:: infer_remote.bat - push code+weights+video to a remote GPU server, run
:: infer_video.py there, pull predictions/<stem>/ back.
::
:: Usage:
::   infer_remote.bat vidoes\1_3.mp4
::   infer_remote.bat vidoes\1_3.mp4 --no-coast --search-radius 60
::
:: Requires: ssh and scp on PATH (Windows OpenSSH Client, or Git's).
:: First run installs torch/opencv/etc into a venv on the server; later runs
:: reuse it, so only the first run is slow.
:: ===========================================================================

:: ---- edit these for your server ----
set REMOTE_USER=cse_g4
set REMOTE_HOST=10.8.100.22
set REMOTE_DIR=~/pickleball-infer
set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
:: -------------------------------------

if "%~1"=="" (
    echo Usage: %~nx0 vidoes\video_name.mp4 [extra infer_video.py args...]
    exit /b 1
)
if not exist "%~1" (
    echo ERROR: video not found: %~1
    exit /b 1
)

set VIDEO_PATH=%~1
set VIDEO_NAME=%~nx1
for %%F in ("%VIDEO_NAME%") do set STEM=%%~nF

:: remaining args after the video path are passed through to infer_video.py
shift
set EXTRA_ARGS=
:collect_args
if "%~1"=="" goto args_done
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto collect_args
:args_done

echo === target: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%
echo === video : %VIDEO_PATH%  (stem: %STEM%)
echo.

echo [1/5] Ensuring remote directories exist...
ssh %REMOTE_USER%@%REMOTE_HOST% "mkdir -p %REMOTE_DIR%/vidoes"
if errorlevel 1 goto :fail

echo [2/5] Uploading code + weights...
scp infer_video.py reqirement.txt wasb_pickleball_final.pth.zip %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/
if errorlevel 1 goto :fail

echo [3/5] Uploading video: %VIDEO_NAME%
scp "%VIDEO_PATH%" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/vidoes/%VIDEO_NAME%
if errorlevel 1 goto :fail

echo [4/5] Setting up environment (first run only) and running inference on the GPU...
ssh %REMOTE_USER%@%REMOTE_HOST% "cd %REMOTE_DIR% && (test -d venv || python3 -m venv venv) && source venv/bin/activate && pip install -q --upgrade pip && pip install -q torch torchvision --index-url %TORCH_INDEX_URL% && pip install -q opencv-python pandas numpy tqdm pillow gdown && python infer_video.py vidoes/%VIDEO_NAME%%EXTRA_ARGS%"
if errorlevel 1 goto :fail

echo [5/5] Downloading results to predictions\%STEM%\ ...
if not exist "predictions" mkdir "predictions"
scp -r %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/predictions/%STEM% predictions\
if errorlevel 1 goto :fail

echo.
echo Done. Results in predictions\%STEM%\
exit /b 0

:fail
echo.
echo FAILED - see the error above.
exit /b 1
