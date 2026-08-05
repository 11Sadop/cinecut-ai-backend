@echo off
cd /d "C:\Users\FSOS\.gemini\antigravity\scratch\capcut-ai-studio"
echo.
echo ======================================
echo  CineCut AI - Installing packages...
echo ======================================
echo.
pip install fastapi uvicorn python-multipart soundfile scipy numpy demucs faster-whisper edge-tts torch torchaudio
echo.
echo ======================================
echo  Starting server on port 5000...
echo  Keep this window open!
echo ======================================
echo.
python -m uvicorn server:app --host 127.0.0.1 --port 5000
pause
