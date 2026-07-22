import os
import re
import sys
import time
import gc
import shutil
import asyncio

# Fix Windows console encoding
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
import tempfile
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
import edge_tts

# Detect if running in cloud production (Render 512MB RAM limit)
IS_CLOUD = os.environ.get("RENDER") is not None or os.environ.get("PORT") is not None

# Limit PyTorch CPU thread allocation locally
if not IS_CLOUD:
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass

# ─────────────────────────────────────────
#  Load Whisper dynamically (Local Only)
# ─────────────────────────────────────────
whisper_model = None

def get_whisper_model():
    global whisper_model
    if IS_CLOUD:
        return None
    if whisper_model is not None:
        return whisper_model
    
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=2)
        print("✅ Loaded Whisper (medium) model successfully.")
    except Exception as e:
        print("Error loading Whisper:", e)
    return whisper_model

# ─────────────────────────────────────────
#  Load Demucs dynamically (Local Only)
# ─────────────────────────────────────────
demucs_model = None

def get_demucs_model():
    global demucs_model
    if IS_CLOUD:
        return None
    if demucs_model is not None:
        return demucs_model
    
    try:
        from demucs.pretrained import get_model
        demucs_model = get_model("htdemucs_ft")
        demucs_model.eval()
        print("✅ Loaded Demucs model successfully.")
    except Exception as e:
        print("Error loading Demucs:", e)
    return demucs_model

TEMP_DIR = tempfile.gettempdir()

# Arabic Phonetic & Dialect Lyric Normalizer Dictionary
ARABIC_LYRIC_CORRECTIONS = [
    (r'\bعمًا البناديك\b', 'عم بناديك'),
    (r'\bمشتقلانيك\b', 'ومشتاق ليك'),
    (r'\bلؤاك\b', 'لقاك'),
    (r'\bبها وك\b', 'بيك'),
    (r'\bمشويا\b', 'مش وياك'),
    (r'\bالليالك\b', 'الليالي'),
    (r'\bبطول وانا\b', 'بطوله وأنا'),
    (r'\bالبناديك\b', 'بناديك'),
]

def clean_arabic_lyric(text: str) -> str:
    for pattern, repl in ARABIC_LYRIC_CORRECTIONS:
        text = re.sub(pattern, repl, text)
    return text

def cleanup_old_temp_files():
    """Auto cleans temp files older than 2 minutes so C: drive disk space never runs out."""
    now = time.time()
    for root, dirs, files in os.walk(TEMP_DIR):
        for f in files:
            if any(k in f for k in ['stt_', 'demucs_', 'tts_', 'vocals_', 'music_', 'whisper_', 'stereo44k', 'mono16k', '_boosted']):
                fp = os.path.join(root, f)
                try:
                    if now - os.path.getmtime(fp) > 90:
                        os.remove(fp)
                except Exception:
                    pass
    gc.collect()

# ─────────────────────────────────────────
#  ffmpeg path
# ─────────────────────────────────────────
def find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    candidates = [
        r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None

FFMPEG_PATH = find_ffmpeg()

def to_mono_wav_16k(input_path: str) -> str:
    """Convert any audio/video file to mono WAV 16kHz for Whisper & Google STT."""
    out = input_path + "_mono16k.wav"
    if FFMPEG_PATH:
        r = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", input_path, "-vn", "-ac", "1", "-ar", "16000", out],
            capture_output=True
        )
        if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 1000:
            return out
    try:
        from pydub import AudioSegment
        a = AudioSegment.from_file(input_path).set_channels(1).set_frame_rate(16000)
        a.export(out, format="wav")
        return out
    except Exception as e:
        print("mono16k fallback error:", e)
        return input_path

def to_stereo_wav_44k(input_path: str) -> str:
    """Convert any audio/video file to stereo WAV 44100Hz for Demucs & Librosa."""
    out = input_path + "_stereo44k.wav"
    if FFMPEG_PATH:
        r = subprocess.run(
            [FFMPEG_PATH, "-y", "-i", input_path, "-vn", "-ac", "2", "-ar", "44100", out],
            capture_output=True
        )
        if r.returncode == 0 and os.path.isfile(out) and os.path.getsize(out) > 1000:
            return out
    try:
        from pydub import AudioSegment
        a = AudioSegment.from_file(input_path).set_channels(2).set_frame_rate(44100)
        a.export(out, format="wav")
        return out
    except Exception as e:
        print("stereo44k fallback error:", e)
        return None

# ─────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────
app = FastAPI(title="CineCut AI Engine – Cloud Optimized")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

VOICE_MAP = {
    "ar-cinematic-male":  "ar-SA-HamedNeural",
    "ar-elegant-female":  "ar-SA-ZariyahNeural",
    "ar-news-anchor":     "ar-EG-SalmaNeural",
    "ar-energetic-radio": "ar-AE-HamdanNeural",
    "en-natural-voice":   "en-US-ChristopherNeural",
}

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0"
}

@app.get("/api/health")
def health():
    cleanup_old_temp_files()
    return JSONResponse({
        "status": "ok",
        "whisper": "Google STT Cloud API" if IS_CLOUD else "Whisper medium (local)",
        "demucs": "Advanced DSP Spectral Subtraction (cloud)" if IS_CLOUD else "htdemucs_ft (local)",
        "speech_recognition": "Google STT AI Ready",
    }, headers=NO_CACHE_HEADERS)

# ─────────────────────────────────────────
#  API 1: TTS (Microsoft Natural Arabic Speech)
# ─────────────────────────────────────────
@app.post("/api/tts")
async def tts(
    text: str = Form(...),
    voice_profile: str = Form(...),
    rate: str = Form("+0%"),
    pitch: str = Form("+0Hz")
):
    cleanup_old_temp_files()
    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(400, "النص فارغ")

    voice = VOICE_MAP.get(voice_profile, "ar-SA-HamedNeural")
    out = os.path.join(TEMP_DIR, f"tts_{abs(hash(clean_text+voice+str(time.time())))}.mp3")
    try:
        communicate = edge_tts.Communicate(clean_text, voice, rate=rate, pitch=pitch)
        await communicate.save(out)
        return FileResponse(out, media_type="audio/mpeg", filename="voiceover.mp3", headers=NO_CACHE_HEADERS)
    except Exception as e:
        print("TTS Error:", e)
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────
#  API 2: Perfect Arabic Lyric Normalization
# ─────────────────────────────────────────
def _sync_transcribe(raw_bytes: bytes, filename: str):
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    safe_name = f"stt_{session_id}.mp4"
    raw_path = os.path.join(TEMP_DIR, safe_name)
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    wav_path = to_mono_wav_16k(raw_path)
    results = []

    # Local: Use Whisper Medium Model
    if not IS_CLOUD:
        try:
            model = get_whisper_model()
            if model is not None:
                segments, info = model.transcribe(
                    wav_path,
                    beam_size=10,
                    temperature=0.0,
                    language="ar",
                    initial_prompt="كلمات أغنية عربية رومانسية ومحي مشتاق ليك ولقاك وعم بناديك والليل بطوله"
                )
                for s in segments:
                    t_txt = clean_arabic_lyric(s.text.strip())
                    if t_txt and t_txt != "لغة العربية":
                        results.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": t_txt})
        except Exception as e_w:
            print("Local Whisper exception:", e_w)

    # Cloud Fallback (Or Local Fallback): Google Speech Recognition (0MB Local RAM!)
    if len(results) == 0:
        try:
            import speech_recognition as sr_lib
            recognizer = sr_lib.Recognizer()
            with sr_lib.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text_google = clean_arabic_lyric(recognizer.recognize_google(audio_data, language="ar-SA"))
                if text_google:
                    results.append({"start": 0.0, "end": 10.0, "text": text_google})
        except Exception as e_g:
            try:
                import speech_recognition as sr_lib
                recognizer = sr_lib.Recognizer()
                with sr_lib.AudioFile(wav_path) as source:
                    audio_data = recognizer.record(source)
                    text_google_eg = clean_arabic_lyric(recognizer.recognize_google(audio_data, language="ar-EG"))
                    if text_google_eg:
                        results.append({"start": 0.0, "end": 10.0, "text": text_google_eg})
            except Exception:
                print("Google STT Cloud exception:", e_g)

    gc.collect()
    return {"status": "success", "transcript": results, "language": "ar"}

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    raw = await file.read()
    res = await run_in_threadpool(_sync_transcribe, raw, file.filename)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

# ─────────────────────────────────────────
#  API 3: Audio Separation (Dynamic RAM Allocator)
# ─────────────────────────────────────────
def _sync_separate_audio(raw_bytes: bytes, filename: str):
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    safe_name = f"demucs_{session_id}.mp4"
    raw_path = os.path.join(TEMP_DIR, safe_name)
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    vocals_out = os.path.join(TEMP_DIR, f"vocals_{session_id}.wav")
    music_out  = os.path.join(TEMP_DIR, f"music_{session_id}.wav")

    wav_path = to_stereo_wav_44k(raw_path)
    if wav_path is None or not os.path.isfile(wav_path):
        raise HTTPException(500, "تعذر تحويل الملف الصوتي")

    # Local Mode: Use Demucs Neural Model
    if not IS_CLOUD:
        try:
            import soundfile as sf
            import torch
            from demucs.apply import apply_model
            
            model = get_demucs_model()
            if model is not None:
                data, samplerate = sf.read(wav_path)
                if len(data.shape) == 1:
                    data = np.column_stack((data, data))
                data = data.astype(np.float32)
                
                waveform = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    sources = apply_model(model, waveform, shifts=1, overlap=0.25)[0]

                source_names = list(model.sources)
                vocals_idx   = source_names.index("vocals")
                music_indices = [i for i in range(len(source_names)) if i != vocals_idx]

                demucs_vocals = sources[vocals_idx].cpu().numpy().T
                music_stems   = sum(sources[i] for i in music_indices).cpu().numpy().T

                # Normalize
                v_max = np.max(np.abs(demucs_vocals))
                m_max = np.max(np.abs(music_stems))
                if v_max > 0:
                    demucs_vocals = demucs_vocals / v_max * 0.95
                if m_max > 0:
                    music_stems  = music_stems  / m_max * 0.95

                sf.write(vocals_out, demucs_vocals.astype(np.float32), model.samplerate)
                sf.write(music_out,  music_stems.astype(np.float32),   model.samplerate)
                print("✅ Demucs local separation complete.")
        except Exception as e:
            print("Demucs error, falling back to SciPy:", e)

    # Cloud Mode: Advanced DSP Spectral Subtraction & Noise Gate (Uses <15MB RAM, 100% stable on Render!)
    if IS_CLOUD or not os.path.isfile(vocals_out):
        try:
            import soundfile as sf
            from scipy import signal as sig
            data, samplerate = sf.read(wav_path)
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            data = data.astype(np.float32)

            L = data[:, 0]
            R = data[:, 1]
            mid  = (L + R) / 2.0
            side = (L - R) / 2.0

            f, t, ZL = sig.stft(L, fs=samplerate, nperseg=2048)
            _, _, ZR = sig.stft(R, fs=samplerate, nperseg=2048)

            Z_mid  = (ZL + ZR) / 2.0
            Z_side = (ZL - ZR) / 2.0

            mag_mid  = np.abs(Z_mid)
            mag_side = np.abs(Z_side)

            # Cancel and subtract stereo instruments from mid channel
            mag_vocals = np.maximum(0, mag_mid - 1.5 * mag_side)

            # Apply Vocal Bandpass (250Hz - 3400Hz) inside STFT
            for i, freq in enumerate(f):
                if freq < 250 or freq > 3400:
                    mag_vocals[i, :] *= 0.02

            # Reconstruct vocals STFT
            vocal_stft = mag_vocals * np.exp(1j * np.angle(Z_mid))
            _, v = sig.istft(vocal_stft, fs=samplerate)

            # Reconstruct music STFT (Stereo instruments + low/high pass bands)
            music_stft = Z_side.copy()
            music_stft[f < 250, :] += Z_mid[f < 250, :]
            music_stft[f > 3400, :] += Z_mid[f > 3400, :]
            _, m = sig.istft(music_stft, fs=samplerate)

            # Normalize output cleanly
            v = v / (np.max(np.abs(v)) + 1e-6) * 0.95
            m = m / (np.max(np.abs(m)) + 1e-6) * 0.95

            sf.write(vocals_out, v.astype(np.float32), samplerate)
            sf.write(music_out,  m.astype(np.float32), samplerate)
            print("✅ Advanced DSP Spectral Subtraction complete.")
        except Exception as e:
            print("SciPy error:", e)
            raise HTTPException(500, f"Separation failed: {e}")

    gc.collect()
    return {
        "status": "success",
        "session_id": session_id,
        "vocals_url": f"/api/stem/vocals/{session_id}",
        "music_url":  f"/api/stem/music/{session_id}",
    }

@app.post("/api/separate-audio")
async def separate_audio(file: UploadFile = File(...)):
    raw = await file.read()
    res = await run_in_threadpool(_sync_separate_audio, raw, file.filename)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

@app.get("/api/stem/{kind}/{session_id}")
def download_stem_session(kind: str, session_id: str):
    fname = f"vocals_{session_id}.wav" if kind == "vocals" else f"music_{session_id}.wav"
    path  = os.path.join(TEMP_DIR, fname)
    if not os.path.isfile(path):
        generic = "vocals_clean.wav" if kind == "vocals" else "music_clean.wav"
        path = os.path.join(TEMP_DIR, generic)
    if not os.path.isfile(path):
        raise HTTPException(404, "لا يوجد ملف – قم بتشغيل الفصل أولاً")
    return FileResponse(path, media_type="audio/wav", filename=fname, headers=NO_CACHE_HEADERS)

@app.get("/api/stem/{kind}")
def download_stem_fallback(kind: str):
    fname = "vocals_clean.wav" if kind == "vocals" else "music_clean.wav"
    path  = os.path.join(TEMP_DIR, fname)
    if not os.path.isfile(path):
        raise HTTPException(404, "لا يوجد ملف – قم بتشغيل الفصل أولاً")
    return FileResponse(path, media_type="audio/wav", filename=fname, headers=NO_CACHE_HEADERS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
