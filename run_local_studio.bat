@echo off
title CineCut AI Studio - Local Server
cls

echo ===================================================================
echo   CineCut AI Pro - Local AI Backend Installer & Runner
echo ===================================================================
echo.
echo Installing python dependencies...
echo.

:: 1. Check Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10+ and select "Add Python to PATH" during installation.
    echo Download link: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist ".venv" (
    echo Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate Environment
echo Activating environment...
call .venv\Scripts\activate.bat

:: 4. Ask for GPU/CUDA Acceleration
echo.
echo ===================================================================
echo   GPU/CUDA Acceleration Setup
echo ===================================================================
echo Do you have an Nvidia GPU and want to enable GPU acceleration? (Highly recommended!)
set /p gpu_choice="Enter Y for Yes, N for No (Y/N): "

echo Updating pip...
python -m pip install --upgrade pip

if /i "%gpu_choice%"=="Y" (
    echo Installing PyTorch with Nvidia CUDA support (this might take a few minutes)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing standard CPU-only PyTorch...
    pip install torch torchaudio
)

:: 5. Install other dependencies
echo Installing AI libraries (FastAPI, Whisper, Demucs, etc.)...
pip install fastapi uvicorn python-multipart soundfile scipy numpy faster-whisper demucs edge-tts librosa SpeechRecognition requests

:: 6. Pre-cache Models
echo.
echo ===================================================================
echo   Pre-downloading AI Models
echo ===================================================================
echo Pre-downloading model weights to prevent delays during first run...
echo.
echo Downloading Meta Demucs separation model...
python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

echo Downloading OpenAI Whisper speech model...
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

echo.
echo ===================================================================
echo   Setup Complete! Launching Server...
echo ===================================================================
echo.
echo Launching local server on http://127.0.0.1:5000
echo Please keep this window open while using the website.
echo.
start https://cinecut-ai-studio.vercel.app

uvicorn server:app --host 127.0.0.1 --port 5000
pause
