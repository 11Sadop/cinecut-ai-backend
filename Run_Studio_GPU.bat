@echo off
title CineCut AI Studio - NVIDIA RTX GPU Server
color 0A
cls
echo =======================================================
echo    CineCut AI Studio Pro - NVIDIA RTX 4060 Accelerated
echo =======================================================
echo.
echo Starting local AI server with PyTorch CUDA GPU support...
echo.
cd /d "%~dp0"
start "" "http://127.0.0.1:5000"
python -m uvicorn server:app --host 127.0.0.1 --port 5000
pause
