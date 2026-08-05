@echo off
title CineCut AI Studio - Local GPU Server Installer
chcp 65001 > nul
cls

echo ===================================================================
echo   🎬 استوديو CineCut AI Pro - برنامج التثبيت والتشغيل المحلي الذكي
echo ===================================================================
echo.
echo سيقوم هذا السكربت بتجهيز خادم الذكاء الاصطناعي بالكامل على جهازك
echo وتفعيل معالجة النماذج العملاقة على كرت الشاشة (GPU) للحصول على أقصى سرعة.
echo.
echo -------------------------------------------------------------------

:: 1. Check Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [❌ خطأ] لم يتم العثور على بايثون (Python) مثبت في نظامك.
    echo يرجى تحميل بايثون 3.10 أو أحدث وتفعيله في متغيرات البيئة (PATH).
    echo رابط التحميل: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 2. Create Virtual Environment
if not exist ".venv" (
    echo [⚙️] جاري إنشاء بيئة عمل برمجية معزولة (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [❌ خطأ] تعذر إنشاء بيئة العمل.
        pause
        exit /b 1
    )
    echo [✅] تم إنشاء بيئة العمل بنجاح.
)

:: 3. Activate Environment
echo [⚙️] جاري تفعيل بيئة العمل...
call .venv\Scripts\activate.bat

:: 4. Ask for GPU/CUDA Acceleration
echo.
echo ===================================================================
echo   ⚙️ خيارات تسريع الذكاء الاصطناعي (GPU Acceleration)
echo ===================================================================
echo هل تمتلك كرت شاشة من شركة انفيديا (Nvidia GPU) وتريد تفعيل تسريع المعالجة؟
set /p gpu_choice="أدخل Y للموافقة أو N للاكتفاء بالمعالج العادي (Y/N): "

echo [⚙️] جاري تحديث أداة التثبيت (pip)...
python -m pip install --upgrade pip

if /i "%gpu_choice%"=="Y" (
    echo [🚀] جاري تثبيت PyTorch مع دعم كرت الشاشة Nvidia CUDA (قد يستغرق دقائق)...
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [⚙️] جاري تثبيت PyTorch العادي المعتمد على المعالج (CPU)...
    pip install torch torchaudio
)

:: 5. Install standard requirements
echo [⚙️] جاري تثبيت باقي مكتبات الذكاء الاصطناعي وعزل الصوت (FastAPI, Whisper, Demucs)...
pip install fastapi uvicorn python-multipart soundfile scipy numpy faster-whisper demucs edge-tts librosa SpeechRecognition requests

:: 6. Pre-cache Models
echo.
echo ===================================================================
echo   📥 تحميل وتخزين نماذج الذكاء الاصطناعي (Pre-caching Models)
echo ===================================================================
echo سيتم الآن تحميل أوزان النماذج وتخزينها محلياً لتجنب التعليق أثناء المعالجة الأولى.
echo.
echo [📥] جاري تحميل نموذج فصل الصوت رباعي المسارات (Meta Demucs)...
python -c "from demucs.pretrained import get_model; get_model('htdemucs')"

echo [📥] جاري تحميل نموذج استخراج الكلمات (OpenAI Whisper Medium)...
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

echo.
echo [✅] تم التثبيت والتجهيز بنجاح!
echo -------------------------------------------------------------------
echo.
echo [🌐] جاري فتح موقع CineCut AI Studio في المتصفح...
start https://cinecut-ai-studio.vercel.app

echo [🚀] جاري تشغيل خادم الذكاء الاصطناعي المحلي على المنفذ 5000...
echo (لا تغلق هذه النافذة طالما أنك تستخدم الموقع)
echo.
uvicorn server:app --host 127.0.0.1 --port 5000

pause
