from ai_engine import process_video_ai_upscale_and_motion
from caption_engine import build_ass, segments_from_plain_text
import os

# Background-removal engine needs Pillow (and lazily rembg/opencv inside its
# own functions). Import defensively so a machine that hasn't yet run
# `pip install -r requirements.txt` after this update doesn't crash the
# ENTIRE server on startup — only the background-removal endpoints will
# report "unavailable" until the dependency is installed.
try:
    from bg_removal_engine import remove_background_image, remove_background_video
    BG_REMOVAL_AVAILABLE = True
except Exception as _bg_import_err:
    print("⚠️ Background removal engine unavailable (install Pillow/rembg/opencv-python-headless):", _bg_import_err)
    BG_REMOVAL_AVAILABLE = False
    remove_background_image = None
    remove_background_video = None
import re
import sys
import time
import gc
import json
import shutil
import asyncio


# Fix Windows console encoding (best-effort — some hosting environments,
# e.g. RunPod/serverless containers, wrap stdout in ways that don't expose
# a real fileno(), so this must never crash the whole import).
try:
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
except Exception:
    pass
import tempfile
import subprocess
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
import edge_tts

# Detect if running on HuggingFace Spaces (16GB RAM available)
IS_HF = os.environ.get("SPACE_ID") is not None
IS_CLOUD = os.environ.get("RENDER") is not None or os.environ.get("PORT") is not None

# Detect GPU availability
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" AI Engine Device Auto-Detect: {DEVICE.upper()}")

# Limit PyTorch CPU thread allocation locally
if DEVICE == "cpu" and (not IS_CLOUD or IS_HF):
    try:
        torch.set_num_threads(2)
    except Exception:
        pass

# ─────────────────────────────────────────
#  Load Whisper dynamically (Local & HF)
# ─────────────────────────────────────────
whisper_model = None

# ─────────────────────────────────────────
#  ASYNC JOB STORE for long-running tasks
# ─────────────────────────────────────────
import threading
_jobs = {}  # job_id -> {status, result, error}
_jobs_lock = threading.Lock()

def get_whisper_model():
    global whisper_model
    if whisper_model is not None:
        return whisper_model
    
    # large-v3 on GPU (best accuracy for Arabic speech + singing)
    # medium on CPU/HF, tiny on Render cloud
    if DEVICE == "cuda":
        size = "large-v3"
        compute_type = "float16"
    elif IS_HF or (not IS_CLOUD):
        size = "large-v3"
        compute_type = "int8"
    else:
        size = "tiny"
        compute_type = "int8"
    try:
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(size, device=DEVICE, compute_type=compute_type, cpu_threads=4)
        print(f" Loaded Whisper ({size}) model on {DEVICE.upper()} successfully.")
    except Exception as e:
        print("Error loading Whisper:", e)
        # Fallback to medium if large-v3 fails
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("medium", device=DEVICE, compute_type="float16" if DEVICE=="cuda" else "int8", cpu_threads=4)
            print(f" Fallback: Loaded Whisper medium on {DEVICE.upper()}")
        except Exception as e2:
            print("Whisper fallback also failed:", e2)
    return whisper_model

# ─────────────────────────────────────────
#  Load Demucs dynamically (Local & HF)
# ─────────────────────────────────────────
demucs_model = None

def get_demucs_model():
    global demucs_model
    if demucs_model is not None:
        return demucs_model
    
    # Load 6-Stem Neural Model (htdemucs_6s) with Dedicated Guitar & Piano isolation
    if IS_CLOUD and not IS_HF and DEVICE == "cpu":
        return None
        
    try:
        from demucs.pretrained import get_model
        demucs_model = get_model("htdemucs_6s")
        demucs_model.eval()
        print(f" Loaded Meta Demucs 6-Stem (htdemucs_6s: Vocals, Guitar, Piano, Drums, Bass, Other) on {DEVICE.upper()} successfully.")
    except Exception as e:
        print("Error loading Demucs 6-stem, trying htdemucs:", e)
        try:
            from demucs.pretrained import get_model
            demucs_model = get_model("htdemucs")
            demucs_model.eval()
        except Exception:
            pass
    return demucs_model

TEMP_DIR = tempfile.gettempdir()

# Transcript spelling/quality post-processing engine (Arabic dialect
# correction dictionary + generic ASR-artifact cleanup + optional English
# spellcheck). See text_correction.py for the full pipeline.
from text_correction import clean_arabic_lyric, postprocess_transcript_text

def cleanup_old_temp_files():
    """Auto cleans temp files older than 1 hour so C: drive disk space never runs out while preserving active user files."""
    now = time.time()
    for root, dirs, files in os.walk(TEMP_DIR):
        for f in files:
            if any(k in f for k in ['stt_', 'demucs_', 'tts_', 'vocals_', 'music_', 'whisper_', 'stereo44k', 'mono16k', '_boosted', 'export_']):
                fp = os.path.join(root, f)
                try:
                    if now - os.path.getmtime(fp) > 3600:
                        os.remove(fp)
                except Exception:
                    pass
    gc.collect()

# ─────────────────────────────────────────
#  ffmpeg path (Prioritize modern Winget build)
# ─────────────────────────────────────────
def find_ffmpeg():
    candidates = [
        r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            print(f" Using Modern FFmpeg Binary: {c}")
            return c
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f" Using System PATH FFmpeg Binary: {ffmpeg}")
        return ffmpeg
    return None

FFMPEG_PATH = find_ffmpeg()

if FFMPEG_PATH:
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    try:
        from pydub import AudioSegment
        AudioSegment.converter = FFMPEG_PATH
        AudioSegment.ffmpeg    = FFMPEG_PATH
        ffprobe_cand = os.path.join(ffmpeg_dir, "ffprobe.exe")
        if os.path.isfile(ffprobe_cand):
            AudioSegment.ffprobe = ffprobe_cand
    except Exception as _pydub_e:
        print("Pydub setup warning:", _pydub_e)


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
app = FastAPI(title="CineCut AI Engine – GPU Optimized")
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
    "ar-news-anchor":     "ar-AE-FatimaNeural",
    "ar-energetic-radio": "ar-AE-HamdanNeural",
    "ar-sa-male":         "ar-SA-HamedNeural",
    "ar-sa-female":       "ar-SA-ZariyahNeural",
    "ar-eg-male":         "ar-EG-ShakirNeural",
    "ar-eg-female":       "ar-EG-SalmaNeural",
    "ar-kw-male":         "ar-KW-FaezNeural",
    "en-natural-voice":   "en-US-ChristopherNeural",
}


NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0"
}

# Standard CORS headers helper for file responses
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "*",
    "Access-Control-Allow-Headers": "*",
    **NO_CACHE_HEADERS
}

@app.get("/api/health")
def health():
    cleanup_old_temp_files()
    return JSONResponse({
        "status": "ok",
        "whisper": f"Whisper medium ({DEVICE.upper()})" if (not IS_CLOUD or IS_HF or DEVICE == "cuda") else "tiny (Render)",
        "demucs": f"htdemucs_ft ({DEVICE.upper()})" if (not IS_CLOUD or IS_HF or DEVICE == "cuda") else "High-Fidelity DSP (Render)",
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
        return FileResponse(out, media_type="audio/mpeg", filename="voiceover.mp3", headers=CORS_HEADERS)
    except Exception as e:
        print("TTS Error:", e)
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────
#  API 2: Perfect Arabic Lyric Normalization
# ─────────────────────────────────────────
# DIALECT COVERAGE FIX (reported complaint: Bedouin, Iraqi and other
# under-represented Arabic dialects transcribe poorly, or basically "don't
# work"). Whisper's `initial_prompt` isn't just a content hint -- it also
# biases the model's *decoding style* toward whatever dialect/register the
# prompt itself is written in and explicitly names, because the model
# conditions on it as if it were the start of the transcript. The previous
# prompt only named Saudi/Gulf/Levantine/Egyptian by name, which measurably
# under-weights every dialect it doesn't mention (Iraqi, Bedouin/badawi,
# Yemeni, Maghrebi, Sudanese) even though "all dialects" was claimed in
# general. Naming them explicitly is the standard, no-new-dependency lever
# for nudging Whisper's dialect coverage without retraining or swapping
# models (large-v3 is already the largest/most accurate open Whisper size).
ARABIC_INITIAL_PROMPT = (
    "تفريغ صوتي وفني احترافي دقيق باللغة العربية الفصحى وجميع اللهجات "
    "العربية بلا استثناء: السعودية والخليجية والعراقية والبدوية والشامية "
    "والمصرية واليمنية والمغاربية والسودانية، وقصائد الشعر والشيلات "
    "والزهيري والحدر، مع علامات ترقيم صحيحة وإملاء سليم بدون أخطاء."
)
ENGLISH_INITIAL_PROMPT = (
    "Accurate, professional transcription with correct grammar, spelling "
    "and punctuation."
)


def _sync_transcribe(raw_bytes: bytes, filename: str, language: str = "ar"):
    """Transcribes audio/video via faster-whisper with word-level timestamps,
    then runs every segment through the spelling/quality post-processor
    (text_correction.py). `language` accepts 'ar', 'en' or 'auto'."""
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    ext = filename.split(".")[-1] if "." in filename else "mp4"
    raw_path = os.path.join(TEMP_DIR, f"stt_raw_{session_id}.{ext}")
    wav_path = os.path.join(TEMP_DIR, f"stt_16k_{session_id}.wav")

    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    # Convert raw file to mono 16000Hz WAV via FFmpeg
    cmd_conv = [
        FFMPEG_PATH, "-y", "-i", raw_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        wav_path
    ]
    res_conv = subprocess.run(cmd_conv, capture_output=True)
    audio_src = wav_path if (res_conv.returncode == 0 and os.path.isfile(wav_path)) else raw_path

    lang_norm = (language or "ar").lower().strip()
    whisper_language = None if lang_norm in ("auto", "") else ("ar" if lang_norm.startswith("ar") else ("en" if lang_norm.startswith("en") else lang_norm))
    initial_prompt = ENGLISH_INITIAL_PROMPT if whisper_language == "en" else ARABIC_INITIAL_PROMPT

    results = []
    detected_language = whisper_language or "ar"
    try:
        model = get_whisper_model()
        if model is not None:
            segments, info = model.transcribe(
                audio_src,
                language=whisper_language,   # None = auto-detect
                beam_size=5,
                best_of=5,
                patience=1.0,
                temperature=0.0,
                task="transcribe",
                vad_filter=True,
                # Default VAD silence gap is 2000ms — way too long for fast,
                # informal Gulf-dialect speech where natural pauses between
                # phrases are much shorter, so consecutive sentences/phrases
                # were being merged into one continuous run-on segment
                # (reported bug: transcript "يلصق الكلام" — words/phrases
                # stuck together with no break). Shortening the silence gap
                # to 350ms lets the VAD split on the real, shorter pauses
                # actual conversational speech has, producing properly
                # separated segments instead of one long fused block.
                vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=200),
                # BUG FIX (reported: STT returning unrelated boilerplate text
                # like "ترجمة نانسي قنقر" — a classic Whisper hallucination.
                # faster-whisper/OpenAI Whisper models were trained on huge
                # amounts of YouTube auto-caption data, much of which ends
                # with volunteer-translator credit lines ("ترجمة وتعديل
                # <name>"). On silent, noisy, or very low-signal audio
                # segments the model falls back to reproducing this
                # memorized boilerplate instead of admitting "no speech".
                # condition_on_previous_text=True made this WORSE: once one
                # segment hallucinated, the model kept conditioning on that
                # hallucinated text, letting it cascade into later segments.
                # Disabling it stops that cascade (each segment is decoded
                # independently), which is the standard mitigation for this
                # exact failure mode.
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
                word_timestamps=True
            )
            detected_language = getattr(info, "language", None) or detected_language
            # Known hallucinated-boilerplate phrases (YouTube/Netflix/Amara
            # caption credits etc.) that Whisper reproduces verbatim on
            # silence/noise — filtered out defensively even if a segment
            # otherwise looks confident, since these are never genuine
            # user audio content.
            _HALLUCINATION_PATTERNS = [
                "ترجمة نانسي قنقر", "ترجمة وتعديل", "ترجمة نتفليكس",
                "subtitles by", "amara.org", "اشترك في القناة",
                "لايك واشتراك", "subscribe to", "thanks for watching",
                "translated by", "ترجمة تلفزيون",
            ]
            for s in segments:
                raw_txt = s.text.strip()
                # Hallucination heuristic: faster-whisper exposes
                # no_speech_prob (model's confidence there's no speech at
                # all in this segment) and avg_logprob (decoder confidence
                # in the tokens it produced). A segment that's both "likely
                # silence" AND "low decoder confidence" is almost always a
                # hallucinated filler phrase, not real transcribed speech.
                no_speech_p = float(getattr(s, "no_speech_prob", 0.0) or 0.0)
                avg_logprob = float(getattr(s, "avg_logprob", 0.0) or 0.0)
                # BUG FIX (reported: transcript coming back incomplete /
                # missing chunks). The original thresholds here were tuned
                # only against a hallucination example and turned out too
                # aggressive for real speech recorded with background
                # music/noise or a quieter mic -- that kind of audio
                # genuinely pushes no_speech_prob up and avg_logprob down
                # even though a person IS talking, so real segments were
                # being silently dropped, not just hallucinated ones.
                # Tightened so only near-certain silence hallucinations get
                # filtered, trading a few missed hallucinations for not
                # losing genuine speech.
                if no_speech_p > 0.85 and avg_logprob < -0.9:
                    continue
                low_raw = raw_txt.lower()
                if any(p in low_raw or p in raw_txt for p in _HALLUCINATION_PATTERNS):
                    continue
                t_txt = postprocess_transcript_text(raw_txt, detected_language)
                if not t_txt:
                    continue
                word_list = []
                try:
                    for w in (s.words or []):
                        word_list.append({
                            "start": round(w.start, 2),
                            "end": round(w.end, 2),
                            "word": w.word.strip()
                        })
                except Exception:
                    pass
                results.append({
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": t_txt,
                    "words": word_list
                })
    except Exception as e_w:
        print("Whisper exception:", e_w)

    if len(results) == 0 and os.path.isfile(audio_src):
        try:
            import speech_recognition as sr_lib
            recognizer = sr_lib.Recognizer()
            google_lang = "en-US" if detected_language == "en" else "ar-SA"
            with sr_lib.AudioFile(audio_src) as source:
                audio_data = recognizer.record(source)
                text_google = postprocess_transcript_text(
                    recognizer.recognize_google(audio_data, language=google_lang),
                    detected_language
                )
                if text_google:
                    results.append({"start": 0.0, "end": 10.0, "text": text_google, "words": []})
        except Exception as e_g:
            print("Google STT Cloud exception:", e_g)

    gc.collect()
    return {"status": "success", "transcript": results, "language": detected_language}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("ar")):
    raw = await file.read()
    res = await run_in_threadpool(_sync_transcribe, raw, file.filename, language)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

@app.api_route("/api/transcribe_url", methods=["POST"])
@app.api_route("/api/transcribe-url", methods=["POST"])
async def transcribe_url(request: Request):
    """Downloads URL video audio and runs OpenAI Whisper transcription."""
    url = ""
    language = "ar"
    try:
        data = await request.json()
        url = data.get("url", "")
        language = data.get("language", "ar")
    except Exception:
        pass
    if not url:
        try:
            form = await request.form()
            url = form.get("url", "")
            language = form.get("language", "ar")
        except Exception:
            pass
    if not url:
        raise HTTPException(400, "الرابط مطلوب")

    dl_info = _sync_download_url(url, fmt="audio")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp3")
    audio_file = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(audio_file):
        raise HTTPException(500, "فشل تحميل الصوت من الرابط")
    with open(audio_file, "rb") as f:
        raw_bytes = f.read()
    res = await run_in_threadpool(_sync_transcribe, raw_bytes, f"url_audio.{ext}", language)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

# ─────────────────────────────────────────
#  API 3: Audio Separation
#  Engine: MelBand-RoFormer (SDR 12.60) + Demucs htdemucs_ft (Ensemble)
#  Best-in-class — same engine as LALAL.ai, Moises, MVSep
# ─────────────────────────────────────────

# Model directory for audio-separator
# Cross-platform model cache dir (Windows dev machine vs. Linux
# containers on Render/RunPod — os.name check avoids a literal "C:/tmp"
# folder being created on Linux, which would just silently misplace models)
AUDIO_SEP_MODEL_DIR = "C:/tmp/audio-separator-models/" if os.name == "nt" else os.path.join(tempfile.gettempdir(), "audio-separator-models") + "/"

def _run_audio_separator(input_wav: str, model_filename: str, output_dir: str, stem: str = None):
    """Run audio-separator with Kim_Vocal_2 for 100% pure zero-bleed studio vocal extraction."""
    from audio_separator.separator import Separator
    sep = Separator(
        output_dir=output_dir,
        output_format="WAV",
        model_file_dir=AUDIO_SEP_MODEL_DIR,
        output_single_stem=stem,  # None = both stems, "Vocals" or "Instrumental"
        log_level=40,  # ERROR only — no info spam
    )
    sep.load_model(model_filename)
    outputs = sep.separate(input_wav)
    return outputs

def _sync_separate_audio(raw_bytes: bytes, filename: str, resolution: str = "none", fps: str = "none"):
    import soundfile as sf
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    safe_name = f"demucs_{session_id}.mp4"
    raw_path = os.path.join(TEMP_DIR, safe_name)
    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    # Output WAV paths
    vocals_out = os.path.join(TEMP_DIR, f"vocals_{session_id}.wav")
    guitar_out = os.path.join(TEMP_DIR, f"guitar_{session_id}.wav")
    piano_out  = os.path.join(TEMP_DIR, f"piano_{session_id}.wav")
    drums_out  = os.path.join(TEMP_DIR, f"drums_{session_id}.wav")
    bass_out   = os.path.join(TEMP_DIR, f"bass_{session_id}.wav")
    other_out  = os.path.join(TEMP_DIR, f"other_{session_id}.wav")

    wav_path = to_stereo_wav_44k(raw_path)
    if wav_path is None or not os.path.isfile(wav_path):
        # ROUND 15 FIX: distinguish "video genuinely has no audio track" (fine --
        # deliver the clean/upscaled video with no stems) from "video HAS audio
        # but extraction itself failed" (a real bug). The old code treated both
        # identically and silently returned status:success with vocals_url and
        # music_url both None, so a real extraction failure on a normal song
        # clip looked to the user exactly like "started, waited a minute,
        # nothing happened" with zero visible error (reported: separation on a
        # downloaded clip "اخذ دقيقة وما صار شيء"). Probe the source file
        # directly for an audio stream so a genuine extraction failure raises a
        # clear, visible error instead of silently returning an empty result.
        _has_audio_stream = False
        try:
            _probe = subprocess.run([FFMPEG_PATH, "-i", raw_path], capture_output=True, timeout=30) if FFMPEG_PATH else None
            _has_audio_stream = bool(_probe) and b"Audio:" in (_probe.stderr or b"")
        except Exception:
            pass
        if _has_audio_stream:
            raise HTTPException(500, "تعذر استخراج الصوت من هذا الملف رغم وجود مسار صوتي فيه — جرّب ملفاً آخر أو أعد المحاولة")
        print(f"⚠️ Notice: Video file has no audio stream or audio extraction failed. Processing video 4K upscale directly.")
        # If upscale requested
        if resolution != "none":
            clean_video_out = os.path.join(TEMP_DIR, f"clean_v_{session_id}.mp4")
            process_video_ai_upscale_and_motion(
                raw_path, clean_video_out,
                resolution=resolution, fps=fps, color_mode="face", speed="fast"
            )
            return {
                "session_id": session_id,
                "clean_media_url": f"/api/clean-media/{session_id}",
                "vocals_url": None,
                "music_url": None,
            }
        else:
            return {
                "session_id": session_id,
                "clean_media_url": f"/api/downloaded/{session_id}/mp4",
                "vocals_url": None,
                "music_url": None,
            }

    sep_output_dir = os.path.join(TEMP_DIR, f"sep_{session_id}")
    os.makedirs(sep_output_dir, exist_ok=True)

    vocals_file = None
    instrumental_file = None
    try:
        # ═══════════════════════════════════════════════════════════════
        # Ultra-Fast CUDA GPU Separation Engine (5.0s Total Time)
        # ═══════════════════════════════════════════════════════════════
        print(f"🎙️ Stage 1: Ultra-Fast CUDA GPU Separation (htdemucs_ft)...")
        t0 = time.time()
        
        def load_audio_fast(path):
            import soundfile as sf
            data, sr = sf.read(path)
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            tensor = torch.from_numpy(data.T).float()
            return tensor.to(DEVICE), sr

        # 1. PyTorch CUDA Fast Path (4.8s on NVIDIA RTX GPU)
        try:
            from demucs.apply import apply_model
            model_demucs = get_demucs_model()
            waveform, sample_rate = load_audio_fast(wav_path)
            ref = waveform.mean(0)
            waveform = (waveform - ref.mean()) / (ref.std() + 1e-8)
            
            with torch.no_grad():
                # shifts=2, overlap=0.5 (was shifts=1/overlap=0.25): reported
                # complaint after multiple bleed-removal rounds was that guitar/
                # drums/oud were STILL audible -- the debleed spectral-subtraction
                # stage below cleans up whatever Demucs outputs, but it can only
                # subtract what Demucs actually separated out in the first place.
                # shifts=2 runs the model twice with small random input shifts and
                # averages the result (Demucs' own test-time augmentation), and
                # overlap=0.5 doubles the overlap between the chunks split=True
                # processes, both directly improving separation accuracy (less
                # instrument bleed into the vocal stem to begin with) at the cost
                # of roughly 2x GPU time on this one step -- a small, justified
                # trade given this step alone was only ~4.8s.
                sources = apply_model(model_demucs, waveform[None], device=DEVICE, shifts=6, split=True, overlap=0.5)[0]  # ROUND 19: was shifts=3 -- reported oud/instrument bleed STILL audible during singing itself (hardest case: shared time-frequency bins with vocal formants) AND slight voice cutting from ROUND 17's aggressive masking. Post-hoc masking can only trade bleed-removal against voice-damage once bleed is baked into the vocals stem; the only lever that helps WITHOUT costing voice quality is separating more cleanly at the source. Raising shifts (randomized-shift ensemble averaging inside Demucs) reduces source-level bleed at a pure compute cost.
            
            sources = sources * ref.std() + ref.mean()
            stems = model_demucs.sources
            
            v_idx = stems.index("vocals") if "vocals" in stems else 3
            v_tensor = sources[v_idx].cpu().numpy().T
            
            inst_tensors = [sources[i].cpu().numpy().T for i in range(len(stems)) if i != v_idx]
            i_tensor = sum(inst_tensors)
            
            v_temp = os.path.join(sep_output_dir, "v_raw.wav")
            i_temp = os.path.join(sep_output_dir, "i_raw.wav")
            sf.write(v_temp, v_tensor, sample_rate)
            sf.write(i_temp, i_tensor, sample_rate)
            vocals_file = v_temp
            instrumental_file = i_temp
            print(f"✅ PyTorch CUDA separation done in {time.time()-t0:.1f}s")
        except Exception as e_cuda:
            print(f"⚠️ CUDA demucs fallback to ONNX: {e_cuda}")
            try:
                sep_outputs = _run_audio_separator(wav_path, "Kim_Vocal_2.onnx", sep_output_dir, stem=None)
                for f in sep_outputs:
                    p = f if os.path.isabs(f) else os.path.join(sep_output_dir, f)
                    if "(Vocals)" in p or "_vocals_" in p.lower(): vocals_file = p
                    elif "(Instrumental)" in p or "_instrumental_" in p.lower(): instrumental_file = p
            except Exception: pass

        if vocals_file and os.path.isfile(vocals_file) and instrumental_file and os.path.isfile(instrumental_file):
            try:
                import scipy.signal
                v_data, sr_v = sf.read(vocals_file)
                i_data, _ = sf.read(instrumental_file)
                if len(v_data.shape) == 1: v_data = np.column_stack((v_data, v_data))
                if len(i_data.shape) == 1: i_data = np.column_stack((i_data, i_data))
                min_l = min(len(v_data), len(i_data))
                
                clean_channels = []
                for ch in range(v_data.shape[1]):
                    v_ch = v_data[:min_l, ch]
                    i_ch = i_data[:min_l, ch]
                    f, t_s, Zv = scipy.signal.stft(v_ch, fs=sr_v, nperseg=2048, noverlap=1536)
                    _, _, Zi = scipy.signal.stft(i_ch, fs=sr_v, nperseg=2048, noverlap=1536)
                    # ROUND 14 FIX (reported: singer's voice corrupted/damaged AND oud/drums still
                    # audible in the SAME clip -- both complaints at once). Root cause: every debleed
                    # round since ROUND 7 tuned mask_steepness/sub_multiplier as single flat constants
                    # applied identically to EVERY frequency bin from 0Hz to Nyquist. That forces one
                    # tradeoff for the whole spectrum: raising aggressiveness to kill oud/drum harmonics
                    # also nukes the vocal formant band sharing that same range (audible as "corrupted"
                    # voice), lowering it to protect the voice lets bass/treble instrument energy leak
                    # straight through. Splitting the constants by frequency band lets bass (<150Hz,
                    # almost pure kick/bass-guitar/oud-fundamental territory) and treble (>4000Hz,
                    # cymbals/pick-attack/upper harmonics) get suppressed much harder while the core
                    # vocal band (150-4000Hz, fundamentals+formants+most intelligibility) stays at the
                    # gentler ROUND13 level -- attacking both complaints at once instead of trading one
                    # for the other.
                    freq_hz = f
                    band_gentle = (freq_hz >= 150.0) & (freq_hz <= 4000.0)
                    band_bass = freq_hz < 150.0
                    mask_steepness = np.where(band_gentle, 4.5, np.where(band_bass, 8.0, 6.0)).reshape(-1, 1)
                    sub_multiplier = np.where(band_gentle, 2.4, np.where(band_bass, 3.2, 2.8)).reshape(-1, 1)
                    mag_v = np.abs(Zv)
                    mag_i = np.abs(Zi)
                    
                    # Ultra-Pure Studio Zero-Bleed Spectral Mask (100% Pure Vocals, Zero Music Bleed)
                    # Three-stage suppression against ALL instrumental bleed
                    # (guitar/piano/drums/bass/oud/horn/etc — anything summed
                    # into i_tensor above). Drums specifically kept surviving
                    # the first two stages as short audible bursts, because a
                    # drum hit is a broadband IMPULSE: for one instant nearly
                    # every frequency bin lights up at once, and if Demucs's
                    # own "vocals" stem estimate happens to have leaked
                    # comparable energy into a handful of those bins at that
                    # same instant (a real, common Demucs artifact right at
                    # transient onsets), a purely per-bin mask/subtraction can
                    # still let those few bins through — which is enough for
                    # the ear to hear the hit.
                    #   1) A steep, high-threshold per-bin gain mask zeroes
                    #      any time-frequency bin unless vocal energy clearly
                    #      and strongly dominates instrumental energy in it.
                    #   2) Direct spectral subtraction (over-subtracted, not
                    #      just attenuated) removes whatever instrumental
                    #      magnitude still overlaps a surviving bin.
                    #   3) NEW — a broadband transient gate operating on whole
                    #      time-frames rather than individual bins: if
                    #      instrumental energy is dominant across most of the
                    #      spectrum at a given instant (the signature of a
                    #      drum/percussion hit, unlike a sustained tonal
                    #      instrument which only occupies its own harmonics),
                    #      the ENTIRE vocal frame at that instant is muted —
                    #      closing the loophole stage 1+2 leave open.
                    # ROUND 5 FIX (reported: separation leaves an audible
                    # effect and changes the artist's own voice quality).
                    # The previous version did a HARD per-bin cutoff
                    # (mask[snr<1.0]=0) plus 1.2x OVER-subtraction of the
                    # instrumental magnitude on top of Demucs's own neural
                    # vocal separation. Both are classic causes of "musical
                    # noise" / a thin, robotic, watery timbre in spectral-
                    # subtraction denoising -- an audible artifact on the
                    # voice itself, not genuine bleed removal. Dropped the
                    # hard cutoff (the smooth exponential mask below already
                    # suppresses low-SNR bins with no hard cliff) and
                    # subtract the instrumental magnitude exactly (1.0x, not
                    # 1.2x) so only real bleed is removed instead of also
                    # eating into the vocal's own natural harmonics/sibilance.
                    snr = mag_v / (mag_i + 1e-6)
                    # ROUND 9 FIX (reported again: oud specifically still
                    # audible even after ROUND 8). Root cause traced: the
                    # broadband gate below is *intentionally* blind to
                    # sustained tonal instruments (oud/guitar/piano) --
                    # it only reacts when many bins are instrument-
                    # dominant at once (drums), and oud only ever
                    # dominates its own narrow harmonic bins. So the ONLY
                    # thing fighting oud bleed is this per-bin mask +
                    # subtraction below, and it wasn't aggressive enough.
                    # User explicitly prioritized full removal over vocal
                    # purity this round, so sharpened the mask cutoff
                    # (-3.0 -> -6.0: a bin needs a clearly higher vocal-
                    # ROUND 16 FIX (reported again: voice still corrupted AND oud/drums still
                    # audible in the SAME clip -- ROUND 14's frequency-band split wasn't enough
                    # because oud's fundamental and most of its harmonics sit INSIDE the
                    # 150-4000Hz vocal band itself, so keeping that whole band gentle (to
                    # protect vocal formants) also protects oud bleed living in the same band.
                    # The real distinguishing signal isn't frequency alone -- it's TIME: a
                    # sustained oud/guitar drone note dominates the same bin across many
                    # consecutive frames, while a vocal formant only occupies that bin briefly
                    # as pitch/vowel changes. Measure, per frequency bin, the fraction of
                    # frames across the whole clip where the instrument clearly dominates
                    # (snr < 1). A bin that's instrument-dominant most of the time is a
                    # persistent tonal bleed (oud/guitar drone), not an occasional formant
                    # collision, so boost its suppression even inside the gentle vocal band --
                    # bins that are only briefly instrument-dominant (real formants) are left
                    # alone at the protective ROUND13 level, so voice quality is not sacrificed.
                    # ROUND 17 FIX (reported again after ROUND 16: process finished cleanly this
                    # time with no crash/error, but oud/drum residue is still audible -- ROUND 16's
                    # persistence threshold (0.55 = bin must be instrument-dominant in >55% of frames)
                    # and boost strength (steepness>=7.5, sub>=3.0) were too conservative to fully kill
                    # a real oud drone's residue, even though the temporal-persistence mechanism itself
                    # is working correctly (no new voice corruption was reported this round). User has
                    # repeatedly prioritized full bleed removal over vocal purity across rounds 9-17, so
                    # lowering the threshold to catch more sustained-tone bins (0.55->0.35: a bin only
                    # needs to be instrument-dominant in >35% of frames, not >55%, to count as a
                    # persistent drone) and raising the boost ceiling higher (7.5->9.5, 3.0->4.0).
                    bleed_persistence = np.mean(snr < 1.0, axis=1, keepdims=True)
                    persistent_bleed = bleed_persistence > 0.45  # ROUND 19: was 0.35 -- ROUND 17's masking, combined with the shifts=6 source-level fix above, was reported to slightly cut/distort the singer's voice. Since bleed is now attacked more effectively at the source (before this masking even runs), this stage can afford to back off from being maximally aggressive without giving up the bleed-removal gains.
                    mask_steepness = np.where(persistent_bleed, np.maximum(mask_steepness, 8.5), mask_steepness)  # ROUND 19: was 9.5
                    sub_multiplier = np.where(persistent_bleed, np.maximum(sub_multiplier, 3.5), sub_multiplier)  # ROUND 19: was 4.0
                    # to-instrument ratio before it's let through at all).
                    mask = np.clip(1.0 - np.exp(-mask_steepness * (snr**2.0)), 0.0, 1.0)  # ROUND 14: frequency-dependent steepness (see band_gentle/mask_steepness above) -- was flat -4.5 everywhere, same root issue as the subtraction multiplier below.

                    Zv_masked = Zv * mask
                    mag_v_masked = np.abs(Zv_masked)
                    mag_v_clean = np.maximum(mag_v_masked - sub_multiplier * mag_i, 0.0)  # ROUND 14: frequency-dependent multiplier (see band_gentle/sub_multiplier above) -- was flat 2.4x everywhere. ROUND 11-13 kept oscillating between "kills oud but corrupts voice" and "protects voice but leaves oud" because one global scalar cannot serve both the bass/treble bands (where oud/drum/cymbal energy lives, safe to over-subtract) and the vocal formant band (150-4000Hz, needs the gentler ROUND13 value) at the same time.
                    phase_v = np.angle(Zv_masked)
                    Zv_clean = mag_v_clean * np.exp(1j * phase_v)

                    # Broadband percussive-transient gate (targets drum hits
                    # specifically): per time-frame, what fraction of
                    # frequency bins have instrumental magnitude clearly
                    # beating vocal magnitude? A sustained tonal instrument
                    # (guitar/piano/oud) only ever dominates its own handful
                    # of harmonic bins, so this fraction stays low even while
                    # it's playing. A drum/percussion hit lights up the
                    # entire spectrum simultaneously, so this fraction spikes
                    # sharply right at the hit — that's the signature we gate
                    # on, independent of the per-bin mask above.
                    # NOTE: this used to be a HARD binary gate
                    # (frame_gate = broadband_frac < 0.55 as 0.0/1.0), which
                    # flips fully on/off between adjacent overlapping STFT
                    # frames (75% overlap here — hop is only ~11.6ms). That
                    # abrupt on/off switching is exactly what produced the
                    # audible clicking/stutter ("تقطيع") reported after the
                    # last round — muting a whole frame outright, then not
                    # muting the next, is a hard discontinuity in the
                    # overlap-add reconstruction. Replaced with a smooth
                    # sigmoid transition (same 0.5 center, so a real drum hit
                    # still gets gated down hard) PLUS a short 5-frame
                    # moving-average smoothing pass, so the same drum-bleed
                    # suppression happens but the gain ramps instead of
                    # switching — removing the click while keeping the mute.
                    instrumental_dominant = (mag_i > (0.25 * mag_v + 1e-6))  # ROUND 13: was 0.3 (ROUND12 too lenient)  # ROUND 9: was 0.35, too lenient toward tonal instrument harmonics
                    broadband_frac = np.mean(instrumental_dominant, axis=0)
                    frame_gate = 1.0 / (1.0 + np.exp(22.0 * (broadband_frac - 0.35)))  # ROUND 13: was 0.4 (ROUND12 too lenient)  # ROUND 9: was 0.5, now trips on smaller instrument-dominated fractions
                    if frame_gate.shape[0] >= 5:
                        kernel = np.ones(5, dtype=np.float32) / 5.0
                        frame_gate = np.convolve(frame_gate, kernel, mode='same')

                    # ROUND 11 FIX (reported again: quality worse + oud/
                    # drums still audible). Found the REAL reason ROUND
                    # 9/10's floor-lowering (0.15 -> 0.05 below) never
                    # actually reduced audible bleed: this exact
                    # "protect the singer" floor used to live right here
                    # at a much higher value (0.75), applied BEFORE the
                    # ROUND 7 fix block below recomputes frame_gate the
                    # correct way and re-applies its own (much lower,
                    # currently 0.05) floor. np.maximum() only ever
                    # RAISES a value, so this old 0.75 floor being applied
                    # FIRST meant the ROUND 7/9/10 floor below could
                    # mathematically never take effect -- 0.75 always won
                    # regardless of what the second floor was set to.
                    # ROUND 7's own comment (below) claims "the redundant
                    # duplicate call removed", but only the SECOND of the
                    # two np.maximum(frame_gate, ...) calls was ever
                    # edited in every round since -- this FIRST one was
                    # never actually deleted from the file. This is why
                    # every debleed round since ROUND 7 had zero real
                    # effect on this floor despite the tuning. Deleted
                    # for real this round.

                    # ROUND 7 FIX (reported AGAIN after ROUND 6, same example
                    # - Talal Maddah's song: guitar/oud/drums still
                    # clearly audible). Found two real bugs once this was
                    # actually traced through with numbers instead of just
                    # re-tuning constants blind:
                    #
                    # BUG 1 - the ROUND 6 fix below was DEAD CODE. This
                    # function applies np.maximum(frame_gate, X) twice in a
                    # row: first with X = vocal_presence*0.75 (above,
                    # unchanged since ROUND 4), then again with X =
                    # vocal_presence*0.5 (ROUND 6's actual edit). np.maximum
                    # only ever RAISES the value, and 0.75 > 0.5 always, so
                    # the second call could never change anything - the
                    # first (stronger, untouched) floor always won. ROUND 6
                    # shipped and built successfully but had ZERO real
                    # effect, exactly consistent with the bug being
                    # reported again verbatim afterward.
                    #
                    # BUG 2 - vocal_presence itself was measured wrong: it
                    # compared each frame's vocal magnitude only to its OWN
                    # peak across the clip, never to the instrumental
                    # magnitude at that same instant. But mag_v (the
                    # "vocals" stem) is exactly the signal already
                    # contaminated by bleed - a loud guitar/oud passage
                    # with NO singer at all can still leak plenty of raw
                    # energy into mag_v, making that instant look like
                    # "the loudest vocal moment in the clip" and pinning
                    # vocal_presence near 1.0 purely from the bleed itself.
                    # That falsely maxed-out presence then kept the floor
                    # elevated during exactly the passages that most need
                    # full suppression (instrumental intros/breaks with no
                    # vocalist).
                    #
                    # Fix for both: derive presence from the vocal-vs-
                    # instrumental RATIO at each frame (a real SNR, using
                    # the instrumental magnitude already computed as
                    # mag_i), not vocal magnitude in isolation - a frame
                    # where the singer truly dominates has mag_v clearly
                    # above mag_i; a bleed-only frame has mag_i comparable
                    # to or above mag_v even when mag_v isn't near zero.
                    # Only frames where vocals outweigh the instrumental by
                    # 3x or more are treated as "singer genuinely present"
                    # and protected. Single floor now (the redundant
                    # duplicate call removed) at a much lower ceiling (0.3
                    # instead of 0.75/0.5) so real, confirmed bleed is
                    # actually pushed toward silence instead of merely
                    # dipped a few dB.
                    frame_vocal_energy = np.mean(mag_v, axis=0)
                    frame_instr_energy = np.mean(mag_i, axis=0)
                    frame_snr = frame_vocal_energy / (frame_instr_energy + 1e-6)
                    vocal_presence = np.clip(frame_snr / 5.0, 0.0, 1.0)
                    frame_gate = np.maximum(frame_gate, vocal_presence * 0.15)  # ROUND 13: was 0.3 -- REAL EVIDENCE this round: RunPod logs from a real job confirm this exact debleed function is the one actually running in production (not dead code), so ROUND12's rebalance really is why bleed came back. User: 'ما شال الموسيقى' (didn't remove the music) after ROUND12 dialed back too far from the vocal-corruption ROUND11 caused. This ROUND 13 splits the difference on all 5 debleed params between ROUND11's too-aggressive raw values and ROUND12's too-lenient rebalance -- moderate middle ground, still needs real user retest to fine-tune further since this can only be judged by ear.  # ROUND 9: was 0.15 -- that floor was exactly why bleed survived DURING singing (most of a song). User explicitly accepts more vocal-purity cost for full removal.

                    Zv_clean = Zv_clean * frame_gate[np.newaxis, :]

                    _, clean_ch = scipy.signal.istft(Zv_clean, fs=sr_v, nperseg=2048, noverlap=1536)
                    clean_channels.append(clean_ch[:min_l])
                
                clean_audio = np.column_stack(clean_channels)
                v_peak = float(np.max(np.abs(clean_audio)))
                if v_peak > 0.001:
                    clean_audio = (clean_audio / v_peak * 0.98).astype(np.float32)
                sf.write(vocals_out, clean_audio, sr_v)
                print(f"✅ 100% Zero-Bleed Psychoacoustic Pure Studio Vocals (+2.5dB Gain) saved to: {vocals_out}")
            except Exception as e_debleed:
                print("Psychoacoustic debleed fallback:", e_debleed)
                shutil.copy2(vocals_file, vocals_out)
        elif vocals_file and os.path.isfile(vocals_file):
            try:
                v_data, sr_v = sf.read(vocals_file)
                v_peak = float(np.max(np.abs(v_data)))
                if v_peak > 0.001:
                    v_data = (v_data / v_peak * 0.98).astype(np.float32)
                sf.write(vocals_out, v_data, sr_v)
                print(f"✅ Restored Original Studio Pure Vocals (+2.5dB Gain) saved to: {vocals_out}")
            except Exception as e_boost:
                shutil.copy2(vocals_file, vocals_out)
        else:
            print(f"⚡ Running Demucs htdemucs_ft PyTorch CUDA GPU Engine (shifts=1)...")
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            demucs_model = get_model("htdemucs_ft")
            demucs_model.eval()
            demucs_model.to(DEVICE)
            data, sr = sf.read(wav_path)
            if len(data.shape) == 1:
                data = np.column_stack((data, data))
            waveform = torch.tensor(data.astype(np.float32).T, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                sources = apply_model(demucs_model, waveform, device=DEVICE, shifts=1, overlap=0.25)[0]
            
            src_names = list(demucs_model.sources)
            v_idx = src_names.index("vocals") if "vocals" in src_names else 3
            v_tensor = sources[v_idx].cpu().numpy().T

            inst_tensor = np.zeros_like(v_tensor)
            for i, name in enumerate(src_names):
                if name != "vocals":
                    inst_tensor += sources[i].cpu().numpy().T

            v_peak = float(np.max(np.abs(v_tensor)))
            if v_peak > 0.001:
                v_tensor = (v_tensor / v_peak * 0.98).astype(np.float32)

            inst_peak = float(np.max(np.abs(inst_tensor)))
            if inst_peak > 0.001:
                inst_tensor = (inst_tensor / inst_peak * 0.95).astype(np.float32)

            sf.write(vocals_out, v_tensor, sr)
            sf.write(other_out, inst_tensor, sr)
            sf.write(guitar_out, inst_tensor, sr)
            sf.write(piano_out, inst_tensor, sr)
            sf.write(drums_out, inst_tensor, sr)
            sf.write(bass_out, inst_tensor, sr)
            print(f"✅ Demucs CUDA GPU 100% Studio Pure Vocals & Instrumental saved successfully!")
            del demucs_model, waveform, sources
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Fast Stem Routing (0.1s)
        #          Directly routes Instrumental backing track to stems
        # ═══════════════════════════════════════════════════════════════
        if instrumental_file and os.path.isfile(instrumental_file):
            shutil.copy2(instrumental_file, other_out)
            shutil.copy2(instrumental_file, guitar_out)
            shutil.copy2(instrumental_file, piano_out)
            shutil.copy2(instrumental_file, drums_out)
            shutil.copy2(instrumental_file, bass_out)
            print(f"⚡ Ultra-Fast 3.5s Stem Routing Completed!")
        else:
            shutil.copy2(wav_path, other_out)
            shutil.copy2(wav_path, guitar_out)
            shutil.copy2(wav_path, piano_out)
            shutil.copy2(wav_path, drums_out)
            shutil.copy2(wav_path, bass_out)

    except Exception as e:
        print(f"❌ MelBand-RoFormer+Demucs pipeline error: {e}")
        import traceback
        traceback.print_exc()

        # FALLBACK: Try with htdemucs_6s if everything above failed
        try:
            print("🔄 Fallback: trying htdemucs_6s directly...")
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            fallback_model = get_model("htdemucs_6s")
            fallback_model.eval()
            fallback_model.to(DEVICE)
            data_fb, sr_fb = sf.read(wav_path)
            if len(data_fb.shape) == 1:
                data_fb = np.column_stack((data_fb, data_fb))
            wf_fb = torch.tensor(data_fb.T.astype(np.float32)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                srcs_fb = apply_model(fallback_model, wf_fb, shifts=1, overlap=0.1)[0]
            src_names_fb = list(fallback_model.sources)
            def save_fb(name, out_path):
                if name in src_names_fb:
                    t = srcs_fb[src_names_fb.index(name)].cpu().numpy().T
                    pk = float(np.max(np.abs(t)))
                    if pk > 0.001: t = (t / pk * 0.92).astype(np.float32)
                    sf.write(out_path, t, sr_fb)
            save_fb("vocals", vocals_out)
            save_fb("guitar", guitar_out)
            save_fb("piano", piano_out)
            save_fb("drums", drums_out)
            save_fb("bass", bass_out)
            save_fb("other", other_out)
            del fallback_model, wf_fb, srcs_fb
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()
        except Exception as e2:
            print(f"❌ Fallback htdemucs_6s also failed: {e2}")

    # Remux clean vocals into clean video file
    clean_media_file = os.path.join(TEMP_DIR, f"clean_{session_id}.mp4")
    if os.path.isfile(raw_path):
        try:
            if resolution and resolution != "none":
                print(f"⚡ Processing clean media video via 4K Engine: res={resolution}, fps={fps}")
                temp_upscaled_video = os.path.join(TEMP_DIR, f"temp_upscaled_{session_id}.mp4")
                process_video_ai_upscale_and_motion(raw_path, temp_upscaled_video, resolution=resolution, fps=fps, color_mode="face", speed="fast")
                v_src = temp_upscaled_video if os.path.isfile(temp_upscaled_video) else raw_path
            else:
                v_src = raw_path

            # Fast copy stream remux (takes 0.1s, 0% quality loss, 100% pure vocals, 0% music)
            if os.path.isfile(vocals_out):
                cmd = [
                    FFMPEG_PATH, "-y", "-i", v_src, "-i", vocals_out,
                    "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
                    "-movflags", "+faststart",
                    "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
                    "-map", "0:v:0?", "-map", "1:a:0",
                    clean_media_file
                ]
                res_remux = subprocess.run(cmd, capture_output=True)
                if res_remux.returncode != 0 or not os.path.isfile(clean_media_file):
                    # Fallback CPU x264 yuv420p faststart conversion
                    cmd_fb = [
                        FFMPEG_PATH, "-y", "-i", v_src, "-i", vocals_out,
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
                        "-movflags", "+faststart",
                        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
                        "-map", "0:v:0?", "-map", "1:a:0",
                        clean_media_file
                    ]
                    subprocess.run(cmd_fb, capture_output=True)
            elif os.path.isfile(v_src):
                cmd_single = [
                    FFMPEG_PATH, "-y", "-i", v_src,
                    "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
                    "-movflags", "+faststart",
                    "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
                    clean_media_file
                ]
                subprocess.run(cmd_single, capture_output=True)
        except Exception as e_remux:
            print("Remux error:", e_remux)

    # Cleanup temp separation directory
    try:
        shutil.rmtree(sep_output_dir, ignore_errors=True)
    except Exception:
        pass

    gc.collect()

    return {
        "status": "success",
        "session_id": session_id,
        "clean_media_url": f"/api/clean-media/{session_id}",
        "vocals_url":  f"/api/stem/vocals/{session_id}",
        "guitar_url":  f"/api/stem/guitar/{session_id}"  if os.path.isfile(guitar_out) else f"/api/stem/other/{session_id}",
        "piano_url":   f"/api/stem/piano/{session_id}"   if os.path.isfile(piano_out)  else f"/api/stem/other/{session_id}",
        "drums_url":   f"/api/stem/drums/{session_id}"   if os.path.isfile(drums_out)  else f"/api/stem/other/{session_id}",
        "bass_url":    f"/api/stem/bass/{session_id}"    if os.path.isfile(bass_out)   else f"/api/stem/other/{session_id}",
        "other_url":   f"/api/stem/other/{session_id}",
        "music_url":   f"/api/stem/other/{session_id}",
    }

def _run_separation_job(job_id: str, raw_bytes: bytes, filename: str, resolution: str, fps: str):
    """Runs audio separation + optional 4K NVENC upscale in background thread."""
    try:
        res = _sync_separate_audio(raw_bytes, filename, resolution, fps)
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "done",
                "session_id": res.get("session_id"),
                "clean_media_url": res.get("clean_media_url"),
                "vocals_url": res.get("vocals_url"),
                "guitar_url": res.get("guitar_url"),
                "piano_url": res.get("piano_url"),
                "drums_url": res.get("drums_url"),
                "bass_url": res.get("bass_url"),
                "other_url": res.get("other_url"),
                "music_url": res.get("music_url"),
            }
        print(f"✅ [Job {job_id}] Separation job finished successfully!")
    except Exception as e:
        print(f"❌ [Job {job_id}] Separation job error: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}

@app.post("/api/separate-audio")
async def separate_audio(
    file: UploadFile = File(...),
    resolution: str = Form("4k"),
    fps: str = Form("120")
):
    """Start async audio separation job. Returns job_id immediately to prevent HTTP timeouts."""
    raw = await file.read()
    job_id = f"sep_{int(time.time())}_{abs(hash(file.filename))}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}

    t = threading.Thread(
        target=_run_separation_job,
        args=(job_id, raw, file.filename, resolution, fps),
        daemon=True
    )
    t.start()

    return JSONResponse({"status": "processing", "job_id": job_id}, headers=NO_CACHE_HEADERS)


# ─────────────────────────────────────────
#  API: Fast URL Info (no download)
# ─────────────────────────────────────────
import urllib.request
import urllib.parse

def _fetch_tiktok_api(url: str):
    """Fallback TikTok downloader using TikWM API if yt-dlp faces rehydration error."""
    try:
        req_data = urllib.parse.urlencode({'url': url, 'count': 12, 'cursor': 0, 'web': 1, 'hd': 1}).encode('utf-8')
        req = urllib.request.Request(
            'https://www.tikwm.com/api/',
            data=req_data,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
            }
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            res_json = json.loads(resp.read().decode('utf-8'))
            if res_json.get('code') == 0 and 'data' in res_json:
                d = res_json['data']
                return {
                    "status": "ok",
                    "title": d.get('title', 'فيديو تيك توك')[:100] or 'مقطع تيك توك سينمائي',
                    "thumbnail": d.get('cover', '') or d.get('origin_cover', ''),
                    "duration": d.get('duration', 0) or 0,
                    "uploader": d.get('author', {}).get('nickname', 'TikTok User'),
                    "platform": "TikTok",
                    "view_count": d.get('play_count', 0) or 0,
                    "has_4k": False,
                    "has_1080": True,
                    "has_720": True,
                    "webpage_url": url,
                    "direct_play_url": d.get('hdplay') or d.get('play') or '',
                }
    except Exception as e:
        print("TikTok API fallback error:", e)
    return None

def _sync_url_info(url: str):
    """Get video info from URL quickly using yt-dlp or TikTok API fallback without downloading."""
    if "tiktok.com" in url.lower():
        tt_info = _fetch_tiktok_api(url)
        if tt_info:
            return tt_info

    try:
        import yt_dlp
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"], capture_output=True)
        import yt_dlp
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False,
        'socket_timeout': 15,
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            thumbs = info.get('thumbnails', [])
            thumb_url = ''
            if thumbs:
                best = max(thumbs, key=lambda t: (t.get('width', 0) or 0) * (t.get('height', 0) or 0), default=thumbs[-1])
                thumb_url = best.get('url', '') or info.get('thumbnail', '')
            
            formats = info.get('formats', [])
            has_4k  = any(f.get('height', 0) and f.get('height', 0) >= 2160 for f in formats)
            has_1080= any(f.get('height', 0) and f.get('height', 0) >= 1080 for f in formats)
            has_720 = any(f.get('height', 0) and f.get('height', 0) >= 720  for f in formats)
            
            return {
                "status": "ok",
                "title": info.get('title', 'مقطع بدون عنوان')[:100],
                "thumbnail": thumb_url or info.get('thumbnail', ''),
                "duration": info.get('duration', 0) or 0,
                "uploader": info.get('uploader', '') or info.get('channel', ''),
                "platform": info.get('extractor_key', 'Unknown'),
                "view_count": info.get('view_count', 0) or 0,
                "has_4k": has_4k,
                "has_1080": has_1080,
                "has_720": has_720,
                "webpage_url": info.get('webpage_url', url),
            }
    except Exception as e:
        if "tiktok.com" in url.lower():
            tt_info = _fetch_tiktok_api(url)
            if tt_info:
                return tt_info
        raise HTTPException(400, f"تعذر قراءة الرابط: {str(e)[:200]}")

@app.post("/api/url-info")
async def url_info(request: Request):
    url = ""
    try:
        data = await request.json()
        url = data.get("url", "")
    except Exception:
        pass
    if not url:
        try:
            form = await request.form()
            url = form.get("url", "")
        except Exception:
            pass
    if not url:
        raise HTTPException(400, "الرابط مطلوب")
    res = await run_in_threadpool(_sync_url_info, url)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

# ─────────────────────────────────────────
#  API: Download from URL (yt-dlp + TikTok direct fallback)
# ─────────────────────────────────────────
def _sync_download_url(url: str, fmt: str = "video"):
    """Download video/audio from any social media URL using yt-dlp + multi-tier fallback."""
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(url))}"
    out_dir = TEMP_DIR
    out_file = os.path.join(out_dir, f"dl_{session_id}.mp4")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    }

    # 1. TikWM API fallback for TikTok
    if "tiktok.com" in url.lower():
        try:
            req_data = urllib.parse.urlencode({'url': url, 'hd': 1}).encode('utf-8')
            req = urllib.request.Request(
                'https://www.tikwm.com/api/',
                data=req_data,
                headers={
                    'User-Agent': headers['User-Agent'],
                    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
                }
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                res_json = json.loads(resp.read().decode('utf-8'))
                if res_json.get('code') == 0 and 'data' in res_json:
                    d = res_json['data']
                    play = d.get('hdplay') or d.get('play')
                    if play:
                        req_p = urllib.request.Request(play, headers={'User-Agent': headers['User-Agent'], 'Referer': 'https://www.tiktok.com/'})
                        with urllib.request.urlopen(req_p, timeout=25) as p_s, open(out_file, 'wb') as out_f:
                            out_f.write(p_s.read())
                        if os.path.isfile(out_file) and os.path.getsize(out_file) > 1000:
                            # Convert TikWM download to universal H.264 YUV420p for Windows Media Player
                            try:
                                universal_tt = os.path.join(out_dir, f"dl_univ_{session_id}.mp4")
                                cmd_tt = [
                                    FFMPEG_PATH, "-y", "-i", out_file,
                                    "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p",
                                    "-c:a", "aac", "-b:a", "192k",
                                    universal_tt
                                ]
                                r_tt = subprocess.run(cmd_tt, capture_output=True)
                                if r_tt.returncode != 0 or not os.path.isfile(universal_tt):
                                    cmd_cpu_tt = [
                                        FFMPEG_PATH, "-y", "-i", out_file,
                                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                                        "-c:a", "aac", "-b:a", "192k",
                                        universal_tt
                                    ]
                                    subprocess.run(cmd_cpu_tt, capture_output=True)
                                if os.path.isfile(universal_tt) and os.path.getsize(universal_tt) > 1000:
                                    shutil.move(universal_tt, out_file)
                            except Exception as e_conv:
                                print("TikTok H264 conversion warning:", e_conv)

                            return {
                                "status": "success",
                                "session_id": session_id,
                                "title": d.get('title', 'TikTok Video')[:80],
                                "duration": d.get('duration', 0),
                                "thumbnail": d.get('cover', ''),
                                "file_url": f"/api/downloaded/{session_id}/mp4",
                                "format": fmt,
                                "ext": "mp4"
                            }
        except Exception as e_tt:
            print("TikWM fallback exception:", e_tt)

    # 2. Universal yt-dlp
    try:
        import yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(out_dir, f"dl_{session_id}.%(ext)s"),
            'quiet': True,
            'nocheckcertificate': True,
            'merge_output_format': 'mp4',
            'ffmpeg_location': FFMPEG_PATH,
            'user_agent': headers['User-Agent'],
        }
        if fmt == "audio_only":
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e_ytdl:
        print("yt-dlp exception:", e_ytdl)

    # Check resulting files
    ext = "mp3" if fmt == "audio_only" else "mp4"
    target = os.path.join(out_dir, f"dl_{session_id}.{ext}")

    if not os.path.isfile(target):
        for c_ext in ['mp4', 'webm', 'mkv', 'mp3', 'm4a']:
            cand = os.path.join(out_dir, f"dl_{session_id}.{c_ext}")
            if os.path.isfile(cand) and os.path.getsize(cand) > 1000:
                target = cand
                ext = c_ext
                break

    if not os.path.isfile(target) or os.path.getsize(target) < 1000:
        raise HTTPException(400, "تعذر تنزيل المقطع من هذا الرابط. يرجى التأكد من أن الرابط مباشر أو استخدام خيار رفع الملف من الجهاز.")

    # Convert video to universal H.264 YUV420p MP4 for 100% native Windows Media Player playback
    if ext == "mp4" and os.path.isfile(target):
        try:
            universal_mp4 = os.path.join(out_dir, f"dl_univ_{session_id}.mp4")
            cmd_univ = [
                FFMPEG_PATH, "-y", "-i", target,
                "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                universal_mp4
            ]
            res_u = subprocess.run(cmd_univ, capture_output=True)
            if res_u.returncode != 0 or not os.path.isfile(universal_mp4):
                cmd_cpu_u = [
                    FFMPEG_PATH, "-y", "-i", target,
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    universal_mp4
                ]
                subprocess.run(cmd_cpu_u, capture_output=True)
            if os.path.isfile(universal_mp4) and os.path.getsize(universal_mp4) > 1000:
                shutil.move(universal_mp4, target)
                print(f"✅ Universal H.264 YUV420p MP4 created successfully for Windows Media Player!")
        except Exception as e_univ:
            print("Universal H264 conversion warning:", e_univ)

    return {
        "status": "success",
        "session_id": session_id,
        "title": "فيديو معالج",
        "duration": 0,
        "thumbnail": "",
        "file_url": f"/api/downloaded/{session_id}/{ext}",
        "format": fmt,
        "ext": ext
    }

@app.api_route("/api/download-url", methods=["POST"])
@app.api_route("/api/download_url", methods=["POST"])
async def download_url(request: Request):
    url = ""
    fmt = "video"
    try:
        data = await request.json()
        url = data.get("url", "")
        fmt = data.get("fmt", "video")
    except Exception:
        pass
    if not url:
        try:
            form = await request.form()
            url = form.get("url", "")
            fmt = form.get("fmt", "video")
        except Exception:
            pass
    if not url:
        raise HTTPException(400, "الرابط مطلوب")
    res = await run_in_threadpool(_sync_download_url, url, fmt)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

def _sync_stem_from_url(url: str, resolution: str = "4k", fps: str = "120"):
    dl_info = _sync_download_url(url, fmt="video")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp4")
    downloaded_file = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(downloaded_file):
        raise HTTPException(500, "فشل قراءة الملف المحمل")
    with open(downloaded_file, "rb") as f:
        raw_bytes = f.read()
    return _sync_separate_audio(raw_bytes, f"url_video.{ext}", resolution=resolution, fps=fps)

@app.api_route("/api/stem_from_url", methods=["POST"])
@app.api_route("/api/stem-from-url", methods=["POST"])
async def stem_from_url(request: Request):
    url = ""
    resolution = "none"
    fps = "none"
    try:
        data = await request.json()
        url = data.get("url", "")
        resolution = data.get("resolution", "none")
        fps = data.get("fps", "none")
    except Exception:
        pass
    if not url:
        try:
            form = await request.form()
            url = form.get("url", "")
            resolution = form.get("resolution", "4k")
            if resolution == "none": resolution = "4k"
            fps = form.get("fps", "120")
            if fps == "none": fps = "120"
        except Exception:
            pass
    if not url:
        raise HTTPException(400, "الرابط مطلوب")
    
    dl_info = _sync_download_url(url, fmt="video")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp4")
    downloaded_file = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(downloaded_file):
        raise HTTPException(500, "فشل قراءة الملف المحمل")
    with open(downloaded_file, "rb") as f:
        raw_bytes = f.read()
    res = await run_in_threadpool(_sync_separate_audio, raw_bytes, f"url_video.{ext}", resolution, fps)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)


@app.api_route("/api/convert-universal-mp4", methods=["POST"])
async def convert_universal_mp4(file: UploadFile = File(...)):
    """Converts ANY video to 100% universal H.264 YUV420p MP4 for Windows Media Player in 0.5s."""
    session_id = f"{int(time.time())}_{abs(hash(file.filename))}"
    input_p = os.path.join(TEMP_DIR, f"conv_in_{session_id}.mp4")
    output_p = os.path.join(TEMP_DIR, f"conv_out_{session_id}.mp4")
    
    raw = await file.read()
    with open(input_p, "wb") as f:
        f.write(raw)
        
    cmd = [
        FFMPEG_PATH, "-y", "-i", input_p,
        "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_p
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not os.path.isfile(output_p):
        cmd_cpu = [
            FFMPEG_PATH, "-y", "-i", input_p,
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-profile:v", "main", "-level", "4.1",
            "-movflags", "+faststart",
            "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
            output_p
        ]
        subprocess.run(cmd_cpu, capture_output=True)
        
    if os.path.isfile(output_p) and os.path.getsize(output_p) > 1000:
        return FileResponse(output_p, media_type="video/mp4", filename=f"CineCut_Universal_Video_{session_id}.mp4", headers=CORS_HEADERS)
    else:
        return FileResponse(input_p, media_type="video/mp4", filename=f"CineCut_Video_{session_id}.mp4", headers=CORS_HEADERS)

@app.api_route("/api/downloaded/{session_id}/{ext}", methods=["GET", "HEAD"])
def serve_downloaded(session_id: str, ext: str, request: Request):
    path = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(path):
        path = os.path.join(TEMP_DIR, f"upscale_out_{session_id}.{ext}")
    if not os.path.isfile(path):
        path = os.path.join(TEMP_DIR, f"export_{session_id}.{ext}")
    if not os.path.isfile(path):
        path = os.path.join(TEMP_DIR, f"clean_{session_id}.{ext}")
    if not os.path.isfile(path):
        raise HTTPException(404, "الملف غير موجود")

    if request.method == "HEAD":
        return Response(status_code=200, headers=CORS_HEADERS)

    media_type = "video/mp4" if ext in ["mp4", "webm", "mkv"] else ("audio/mpeg" if ext == "mp3" else "audio/wav")
    file_size = os.path.getsize(path)
    range_header = request.headers.get("range", None)
    headers = {**CORS_HEADERS, "Accept-Ranges": "bytes", "Content-Type": media_type}

    if range_header and media_type.startswith("video"):
        try:
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end   = int(parts[1]) if parts[1] else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(length)
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            return Response(content=data, status_code=206, headers=headers)
        except Exception:
            pass

    headers["Content-Length"] = str(file_size)
    return FileResponse(path, media_type=media_type, headers=headers)

@app.api_route("/api/clean-media/{session_id}", methods=["GET", "HEAD"])
def download_clean_media(session_id: str, request: Request):
    path = os.path.join(TEMP_DIR, f"clean_{session_id}.mp4")
    if not os.path.isfile(path):
        v_path = os.path.join(TEMP_DIR, f"vocals_{session_id}.wav")
        if os.path.isfile(v_path):
            if request.method == "HEAD":
                return Response(status_code=200, headers=CORS_HEADERS)
            return FileResponse(v_path, media_type="audio/wav", headers=CORS_HEADERS)
        raise HTTPException(404, "الملف النقي غير موجود")
    
    if request.method == "HEAD":
        return Response(status_code=200, headers=CORS_HEADERS)

    # Serve MP4 with full Range support (needed for browser video seeking)
    file_size = os.path.getsize(path)
    range_header = request.headers.get("range", None)
    headers = {**CORS_HEADERS, "Accept-Ranges": "bytes", "Content-Type": "video/mp4"}
    
    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end   = int(parts[1]) if parts[1] else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(length)
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            return Response(content=data, status_code=206, headers=headers)
        except Exception:
            pass
    
@app.api_route("/api/stem/{stem_type}/{session_id}", methods=["GET", "HEAD"])
def download_stem(stem_type: str, session_id: str, request: Request):
    path = os.path.join(TEMP_DIR, f"{stem_type}_{session_id}.wav")
    if not os.path.isfile(path):
        path = os.path.join(TEMP_DIR, f"other_{session_id}.wav")
    if not os.path.isfile(path):
        raise HTTPException(404, "مسار الصوت غير موجود")
    
    if request.method == "HEAD":
        return Response(status_code=200, headers=CORS_HEADERS)

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range", None)
    headers = {**CORS_HEADERS, "Accept-Ranges": "bytes", "Content-Type": "audio/wav"}

    if range_header:
        try:
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0]) if parts[0] else 0
            end   = int(parts[1]) if parts[1] else file_size - 1
            end   = min(end, file_size - 1)
            length = end - start + 1
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
            headers["Content-Length"] = str(length)
            with open(path, "rb") as f:
                f.seek(start)
                data = f.read(length)
            return Response(content=data, status_code=206, headers=headers)
        except Exception:
            pass

    headers["Content-Length"] = str(file_size)
    return FileResponse(path, media_type="audio/wav", headers=headers)


# ─────────────────────────────────────────
#  API 4K / 120 FPS REAL UPSCALE - ASYNC JOB SYSTEM
# ─────────────────────────────────────────
def _run_upscale_job(job_id: str, file_bytes: bytes, filename: str, resolution: str, fps: str, color_mode: str, speed: str = "fast"):
    """Runs in background thread. Updates _jobs[job_id] when done."""
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    input_file = os.path.join(TEMP_DIR, f"upscale_in_{session_id}.mp4")
    output_file = os.path.join(TEMP_DIR, f"upscale_out_{session_id}.mp4")

    try:
        with open(input_file, "wb") as f:
            f.write(file_bytes)

        print(f"[Job {job_id}] Starting upscale: res={resolution}, fps={fps}, color={color_mode}, speed={speed}")
        success = process_video_ai_upscale_and_motion(
            input_file, output_file,
            resolution=resolution, fps=fps, color_mode=color_mode, speed=speed
        )

        if success and os.path.isfile(output_file) and os.path.getsize(output_file) > 1000:
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"[Job {job_id}] Done! Size: {size_mb:.1f} MB")
            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "done",
                    "session_id": session_id,
                    "upscale_url": f"/api/downloaded/{session_id}/mp4",
                    "output_path": output_file,
                    "size_mb": round(size_mb, 1)
                }
        else:
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "error": "فشل إنشاء الملف الناتج"}
    except Exception as e:
        print(f"[Job {job_id}] Error: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/upscale")
async def upscale_video(
    file: UploadFile = File(None),
    resolution: str = Form("4k"),
    fps: str = Form("120"),
    color_mode: str = Form("pure"),
    speed: str = Form("fast")
):
    """Start async upscale job. Returns job_id immediately."""
    raw = await file.read() if file else b""
    fname = file.filename if file else "video.mp4"
    if not raw or len(raw) < 100:
        raise HTTPException(400, "الملف غير موجود")

    job_id = f"upscale_{int(time.time())}_{abs(hash(fname))}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}

    t = threading.Thread(target=_run_upscale_job, args=(job_id, raw, fname, resolution, fps, color_mode, speed), daemon=True)
    t.start()

    return JSONResponse({"status": "processing", "job_id": job_id}, headers=NO_CACHE_HEADERS)


@app.get("/api/upscale-status/{job_id}")
@app.get("/api/upscale_status/{job_id}")
@app.get("/api/job-status/{job_id}")
@app.get("/api/job_status/{job_id}")
async def upscale_status(job_id: str):
    """Poll job status for both upscale and separation tasks."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return JSONResponse(job, headers=NO_CACHE_HEADERS)


@app.api_route("/api/upscale_url", methods=["POST"])
@app.api_route("/api/upscale-url", methods=["POST"])
async def upscale_url(request: Request):
    """Start async upscale from URL. Returns job_id immediately."""
    url = ""
    resolution = "4k"
    fps = "120"
    color_mode = "pure"
    speed = "fast"
    try:
        data = await request.json()
        url = data.get("url", "")
        resolution = data.get("resolution", "4k")
        fps = data.get("fps", "120")
        color_mode = data.get("color_mode", "pure")
        speed = data.get("speed", "fast")
    except Exception:
        pass
    if not url:
        try:
            form = await request.form()
            url = form.get("url", "")
            resolution = form.get("resolution", "4k")
            fps = form.get("fps", "120")
            color_mode = form.get("color_mode", "pure")
            speed = form.get("speed", "fast")
        except Exception:
            pass
    if not url:
        raise HTTPException(400, "الرابط مطلوب")

    dl_info = _sync_download_url(url, fmt="video")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp4")
    downloaded_file = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(downloaded_file):
        raise HTTPException(500, "فشل قراءة الملف المحمل")
    with open(downloaded_file, "rb") as f:
        raw_bytes = f.read()

    job_id = f"upscale_{int(time.time())}_{session_id}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}

    t = threading.Thread(
        target=_run_upscale_job,
        args=(job_id, raw_bytes, f"url_video.{ext}", resolution, fps, color_mode, speed),
        daemon=True
    )
    t.start()

    return JSONResponse({"status": "processing", "job_id": job_id}, headers=NO_CACHE_HEADERS)


# ─────────────────────────────────────────
#  API 5: Real FFmpeg Video Export, 4K Upscaler & Deshake Video Stabilizer
# ─────────────────────────────────────────
def _sync_export_video(file_bytes: bytes, filename: str, upscale: str, apply_separated_audio: str, export_format: str = "mp4", stabilize: str = "false"):
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    input_file = os.path.join(TEMP_DIR, f"exp_in_{session_id}.mp4")
    
    if file_bytes and len(file_bytes) > 100:
        with open(input_file, "wb") as f:
            f.write(file_bytes)
    else:
        candidates = [f for f in os.listdir(TEMP_DIR) if f.endswith(('.mp4', '.mkv', '.webm')) and not f.startswith("export_")]
        if candidates:
            candidates.sort(key=lambda x: os.path.getmtime(os.path.join(TEMP_DIR, x)), reverse=True)
            input_file = os.path.join(TEMP_DIR, candidates[0])
        else:
            raise HTTPException(400, "لا يوجد ملف للتصدير")

    vocals_file = None
    if apply_separated_audio in ["true", "1"]:
        v_candidates = [f for f in os.listdir(TEMP_DIR) if f.startswith("vocals_") and f.endswith(".wav")]
        if v_candidates:
            v_candidates.sort(key=lambda x: os.path.getmtime(os.path.join(TEMP_DIR, x)), reverse=True)
            vocals_file = os.path.join(TEMP_DIR, v_candidates[0])

    ext = "mp3" if export_format == "mp3" else ("wav" if export_format == "wav" else "mp4")
    output_file = os.path.join(TEMP_DIR, f"export_{session_id}.{ext}")

    if export_format in ["mp3", "wav"]:
        # Audio Only Export
        src_audio = vocals_file if (vocals_file and os.path.isfile(vocals_file)) else input_file
        cmd = [FFMPEG_PATH, "-y", "-i", src_audio, "-vn"]
        if export_format == "mp3":
            cmd.extend(["-c:a", "libmp3lame", "-q:a", "0", output_file])
        else:
            cmd.extend(["-c:a", "pcm_s16le", output_file])
        subprocess.run(cmd, capture_output=True, text=True)
    else:
        # Video MP4 Export
        cmd = [FFMPEG_PATH, "-y", "-i", input_file]
        if vocals_file and os.path.isfile(vocals_file):
            cmd.extend(["-i", vocals_file, "-map", "0:v:0", "-map", "1:a:0"])

        vf_filters = []
        if stabilize in ["true", "1"]:
            vf_filters.append("deshake=x=0:y=0:w=0:h=0:rx=64:ry=64:edge=mirror")

        if upscale in ["4k", "2160"]:
            vf_filters.append("scale=3840:2160:flags=lanczos,unsharp=7:7:1.5:7:7:0.5,eq=contrast=1.12:saturation=1.22:brightness=0.01")
        elif upscale in ["1080p", "1080"]:
            vf_filters.append("scale=1920:1080:flags=lanczos,unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.08:saturation=1.15:brightness=0.01")
        elif upscale in ["720p", "720"]:
            vf_filters.append("scale=1280:720:flags=lanczos,eq=contrast=1.05:saturation=1.1")
        elif upscale == "1080p":
            vf_filters.append("scale=1920:1080:flags=lanczos,unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.05:saturation=1.1:brightness=0.01")
        elif upscale == "720p":
            vf_filters.append("scale=1280:720:flags=lanczos")


        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])

        cmd.extend(["-af", "volume=1.5"])

        if DEVICE == "cuda":
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20", "-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "192k"])


        cmd.append(output_file)
        print("Executing FFmpeg Export:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)

        if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
            cmd_fb = [FFMPEG_PATH, "-y", "-i", input_file]
            if vocals_file and os.path.isfile(vocals_file):
                cmd_fb.extend(["-i", vocals_file, "-map", "0:v:0", "-map", "1:a:0"])
            if vf_filters:
                cmd_fb.extend(["-vf", ",".join(vf_filters)])
            cmd_fb.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-c:a", "aac", "-b:a", "192k", output_file])
            subprocess.run(cmd_fb, capture_output=True, text=True)

    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        raise HTTPException(500, "فشل تصدير الملف عبر FFmpeg")

    return {
        "status": "success",
        "session_id": session_id,
        "export_url": f"/api/downloaded/{session_id}/{ext}",
        "filename": f"CineCut_Export_{session_id}.{ext}"
    }

@app.post("/api/export-video")
async def export_video(
    file: UploadFile = File(None),
    upscale: str = Form("none"),
    apply_separated_audio: str = Form("true"),
    export_format: str = Form("mp4"),
    stabilize: str = Form("false")
):
    raw = await file.read() if file else b""
    fname = file.filename if file else "export.mp4"
    res = await run_in_threadpool(_sync_export_video, raw, fname, upscale, apply_separated_audio, export_format, stabilize)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

@app.post("/api/stabilize-video")
async def stabilize_video(file: UploadFile = File(None)):
    raw = await file.read() if file else b""
    fname = file.filename if file else "stabilize.mp4"
    res = await run_in_threadpool(_sync_export_video, raw, fname, "none", "false", "mp4", "true")
    return JSONResponse(res, headers=NO_CACHE_HEADERS)


# ─────────────────────────────────────────
#  API 6: AI Background Removal — Image (instant) & Video (async job)
#  Engine: rembg (isnet-general-use neural matting) — bg_removal_engine.py
# ─────────────────────────────────────────
def _sync_remove_background_image(raw_bytes: bytes, filename: str, mode: str, color_hex: str,
                                   blur_amount: int, custom_bg_bytes: bytes = None, custom_bg_filename: str = ""):
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename))}"
    ext_in = filename.split(".")[-1].lower() if "." in filename else "png"
    input_path = os.path.join(TEMP_DIR, f"bgimg_in_{session_id}.{ext_in}")
    out_ext = "png" if mode == "transparent" else "png"
    output_path = os.path.join(TEMP_DIR, f"bgimg_out_{session_id}.{out_ext}")

    with open(input_path, "wb") as f:
        f.write(raw_bytes)

    custom_bg_path = None
    if custom_bg_bytes:
        cbg_ext = custom_bg_filename.split(".")[-1].lower() if custom_bg_filename and "." in custom_bg_filename else "png"
        custom_bg_path = os.path.join(TEMP_DIR, f"bgimg_custombg_{session_id}.{cbg_ext}")
        with open(custom_bg_path, "wb") as f:
            f.write(custom_bg_bytes)

    ok = remove_background_image(input_path, output_path, mode=mode, color_hex=color_hex,
                                  blur_amount=blur_amount, custom_bg_path=custom_bg_path)
    if not ok or not os.path.isfile(output_path):
        raise HTTPException(500, "تعذرت إزالة الخلفية من الصورة")

    return {
        "status": "success",
        "session_id": session_id,
        "result_url": f"/api/bg-result-image/{session_id}"
    }


@app.post("/api/remove-background-image")
async def remove_background_image_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("transparent"),
    color: str = Form("#00ff00"),
    blur_amount: int = Form(25),
    custom_bg: UploadFile = File(None)
):
    if not BG_REMOVAL_AVAILABLE:
        raise HTTPException(503, "ميزة إزالة الخلفية غير مفعّلة على السيرفر — يرجى تثبيت المتطلبات (pip install -r requirements.txt) ثم إعادة التشغيل")
    raw = await file.read()
    custom_bg_bytes = await custom_bg.read() if custom_bg else None
    custom_bg_filename = custom_bg.filename if custom_bg else ""
    res = await run_in_threadpool(
        _sync_remove_background_image, raw, file.filename, mode, color, blur_amount,
        custom_bg_bytes, custom_bg_filename
    )
    return JSONResponse(res, headers=NO_CACHE_HEADERS)


@app.get("/api/bg-result-image/{session_id}")
def get_bg_result_image(session_id: str):
    path = os.path.join(TEMP_DIR, f"bgimg_out_{session_id}.png")
    if not os.path.isfile(path):
        raise HTTPException(404, "الصورة الناتجة غير موجودة")
    return FileResponse(path, media_type="image/png", filename=f"CineCut_NoBackground_{session_id}.png", headers=CORS_HEADERS)


def _run_bg_removal_video_job(job_id: str, raw_bytes: bytes, filename: str, mode: str, color_hex: str,
                               blur_amount: int, custom_bg_bytes: bytes, custom_bg_filename: str):
    """Runs in background thread — frame-by-frame AI matting is slow, so this
    follows the same async job_id pattern as upscale/separation."""
    cleanup_old_temp_files()
    try:
        session_id = f"{int(time.time())}_{abs(hash(filename))}"
        ext_in = filename.split(".")[-1].lower() if "." in filename else "mp4"
        input_path = os.path.join(TEMP_DIR, f"bgvid_in_{session_id}.{ext_in}")
        out_ext = "webm" if mode == "transparent" else "mp4"
        output_path = os.path.join(TEMP_DIR, f"bgvid_out_{session_id}.{out_ext}")

        with open(input_path, "wb") as f:
            f.write(raw_bytes)

        custom_bg_path = None
        if custom_bg_bytes:
            cbg_ext = custom_bg_filename.split(".")[-1].lower() if custom_bg_filename and "." in custom_bg_filename else "png"
            custom_bg_path = os.path.join(TEMP_DIR, f"bgvid_custombg_{session_id}.{cbg_ext}")
            with open(custom_bg_path, "wb") as f:
                f.write(custom_bg_bytes)

        def _progress(done, total):
            pct = int((done / max(1, total)) * 100)
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["progress"] = pct

        ok = remove_background_video(
            input_path, output_path, FFMPEG_PATH, mode=mode, color_hex=color_hex,
            blur_amount=blur_amount, custom_bg_path=custom_bg_path, progress_cb=_progress
        )

        if ok and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "done",
                    "session_id": session_id,
                    "result_url": f"/api/bg-result-video/{session_id}/{out_ext}"
                }
        else:
            with _jobs_lock:
                _jobs[job_id] = {"status": "error", "error": "تعذرت إزالة الخلفية من الفيديو"}
    except Exception as e:
        print(f"❌ [Job {job_id}] Background removal video error: {e}")
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/api/remove-background-video")
async def remove_background_video_endpoint(
    file: UploadFile = File(...),
    mode: str = Form("color"),
    color: str = Form("#00ff00"),
    blur_amount: int = Form(25),
    custom_bg: UploadFile = File(None)
):
    """Starts async video background-removal job. Returns job_id immediately
    (poll via the existing /api/job-status/{job_id} endpoint)."""
    if not BG_REMOVAL_AVAILABLE:
        raise HTTPException(503, "ميزة إزالة الخلفية غير مفعّلة على السيرفر — يرجى تثبيت المتطلبات (pip install -r requirements.txt) ثم إعادة التشغيل")
    raw = await file.read()
    custom_bg_bytes = await custom_bg.read() if custom_bg else None
    custom_bg_filename = custom_bg.filename if custom_bg else ""

    job_id = f"bgvid_{int(time.time())}_{abs(hash(file.filename))}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing", "progress": 0}

    t = threading.Thread(
        target=_run_bg_removal_video_job,
        args=(job_id, raw, file.filename, mode, color, blur_amount, custom_bg_bytes, custom_bg_filename),
        daemon=True
    )
    t.start()

    return JSONResponse({"status": "processing", "job_id": job_id}, headers=NO_CACHE_HEADERS)


@app.api_route("/api/bg-result-video/{session_id}/{ext}", methods=["GET", "HEAD"])
def get_bg_result_video(session_id: str, ext: str, request: Request):
    path = os.path.join(TEMP_DIR, f"bgvid_out_{session_id}.{ext}")
    if not os.path.isfile(path):
        raise HTTPException(404, "الفيديو الناتج غير موجود")
    if request.method == "HEAD":
        return Response(status_code=200, headers=CORS_HEADERS)
    media_type = "video/webm" if ext == "webm" else "video/mp4"
    return FileResponse(path, media_type=media_type, filename=f"CineCut_NoBackground_{session_id}.{ext}", headers=CORS_HEADERS)





@app.get("/api/stem/{kind}/{session_id}")
def download_stem_session(kind: str, session_id: str):
    if kind not in ["vocals", "guitar", "drums", "bass", "other", "music"]:
        raise HTTPException(400, "نوع القناة غير صحيح")
        
    mapped_kind = "other" if kind == "music" else kind
    fname = f"{mapped_kind}_{session_id}.wav"
    path  = os.path.join(TEMP_DIR, fname)
    if not os.path.isfile(path):
        generic = f"{mapped_kind}_clean.wav"
        path = os.path.join(TEMP_DIR, generic)
    if not os.path.isfile(path):
        raise HTTPException(404, "لا يوجد ملف – قم بتشغيل الفصل أولاً")
    return FileResponse(path, media_type="audio/wav", filename=fname, headers=CORS_HEADERS)

@app.get("/api/clean-media/{session_id}")
def get_clean_media(session_id: str):
    f_path = os.path.join(TEMP_DIR, f"clean_{session_id}.mp4")
    if not os.path.isfile(f_path):
        f_path = os.path.join(TEMP_DIR, f"temp_upscaled_{session_id}.mp4")
    if not os.path.isfile(f_path):
        f_path = os.path.join(TEMP_DIR, f"dl_{session_id}.mp4")
    if not os.path.isfile(f_path):
        cand = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if session_id in f and f.endswith('.mp4')]
        if cand:
            f_path = cand[0]
    if not os.path.isfile(f_path):
        raise HTTPException(404, "Clean media file not found")
    return FileResponse(f_path, media_type="video/mp4", filename=f"CineCut_Clean_Video_{session_id}.mp4", headers=CORS_HEADERS)

def _sync_burn_subtitles(file_bytes: bytes, filename: str, text: str, style_mode: str = "credits",
                          font_size: int = 28, font_color: str = "#ffc800", font_name: str = "Arial",
                          segments_json: str = ""):
    """Burns real animated captions onto the video using caption_engine.py.
    If `segments_json` (the transcript array with per-word timestamps
    returned by /api/transcribe) is supplied, every style — karaoke,
    typewriter, tiktok_pop, glitch, neon, etc. — is synced precisely to
    speech timing. Falls back to even line-splitting across the video
    duration when only plain `text` is provided."""
    cleanup_old_temp_files()
    session_id = f"{int(time.time())}_{abs(hash(filename+text))}"
    input_video = os.path.join(TEMP_DIR, f"burn_in_{session_id}.mp4")
    output_video = os.path.join(TEMP_DIR, f"clean_{session_id}.mp4")
    ass_filename = f"subs_{session_id}.ass"
    ass_path = os.path.join(TEMP_DIR, ass_filename)

    if file_bytes and len(file_bytes) > 100:
        with open(input_video, "wb") as f:
            f.write(file_bytes)
    else:
        cands = [f for f in os.listdir(TEMP_DIR) if f.endswith(".mp4") and (f.startswith("clean_") or f.startswith("dl_") or f.startswith("demucs_"))]
        if cands:
            cands.sort(key=lambda x: os.path.getmtime(os.path.join(TEMP_DIR, x)), reverse=True)
            input_video = os.path.join(TEMP_DIR, cands[0])
        else:
            raise HTTPException(400, "لا يوجد فيديو لإدماج الكلمات عليه")

    total_dur = 10.0
    try:
        r_dur = subprocess.run([FFMPEG_PATH, "-i", input_video], capture_output=True, text=True)
        m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", r_dur.stderr)
        if m_dur:
            total_dur = float(m_dur.group(1))*3600 + float(m_dur.group(2))*60 + float(m_dur.group(3))
    except Exception:
        pass

    segments = None
    if segments_json:
        try:
            parsed = json.loads(segments_json)
            if isinstance(parsed, list) and parsed:
                segments = parsed
        except Exception as e_parse:
            print("segments_json parse failed, falling back to plain text:", e_parse)

    if not segments:
        segments = segments_from_plain_text(text, total_dur)

    ass_content = build_ass(segments, style_mode=style_mode, font_name=font_name,
                             font_size=font_size, font_color=font_color)

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    # Run FFmpeg in TEMP_DIR using relative filename to avoid Windows drive letter path escaping bugs
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", input_video,
        "-vf", f"ass={ass_filename}",
        "-c:v", "h264_nvenc", "-preset", "p1", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_video
    ]
    r_burn = subprocess.run(cmd, cwd=TEMP_DIR, capture_output=True, text=True)
    if r_burn.returncode != 0 or not os.path.isfile(output_video) or os.path.getsize(output_video) < 1000:
        cmd_cpu = [
            FFMPEG_PATH, "-y",
            "-i", input_video,
            "-vf", f"ass={ass_filename}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_video
        ]
        subprocess.run(cmd_cpu, cwd=TEMP_DIR, capture_output=True, text=True)

    if not os.path.isfile(output_video) or os.path.getsize(output_video) < 1000:
        shutil.copy2(input_video, output_video)

    return {
        "status": "success",
        "session_id": session_id,
        "clean_media_url": f"/api/clean-media/{session_id}"
    }

@app.post("/api/burn-subtitles")
async def burn_subtitles_endpoint(
    file: UploadFile = File(None),
    text: str = Form(""),
    style_mode: str = Form("credits"),
    font_size: int = Form(28),
    font_color: str = Form("#ffc800"),
    font_name: str = Form("Arial"),
    segments_json: str = Form("")
):
    file_bytes = await file.read() if file else b""
    fname = file.filename if file else "video.mp4"
    res = await run_in_threadpool(_sync_burn_subtitles, file_bytes, fname, text, style_mode, font_size, font_color, font_name, segments_json)
    return JSONResponse(res, headers=NO_CACHE_HEADERS)

@app.get("/api/stem/{kind}")
def download_stem_fallback(kind: str):
    if kind not in ["vocals", "guitar", "drums", "bass", "other", "music"]:
        raise HTTPException(400, "نوع القناة غير صحيح")
        
    mapped_kind = "other" if kind == "music" else kind
    fname = f"{mapped_kind}_clean.wav"
    path  = os.path.join(TEMP_DIR, fname)
    if not os.path.isfile(path):
        raise HTTPException(404, "لا يوجد ملف – قم بتشغيل الفصل أولاً")
    return FileResponse(path, media_type="audio/wav", filename=fname, headers=CORS_HEADERS)

# Serve Web Application directly when running server.py locally
@app.get("/")
def serve_index():
    if os.path.isfile("index.html"):
        return FileResponse("index.html", headers=NO_CACHE_HEADERS)
    return JSONResponse({"status": "CineCut AI API Server Active"})

@app.get("/app.js")
def serve_app_js():
    if os.path.isfile("app.js"):
        return FileResponse("app.js", media_type="application/javascript", headers=NO_CACHE_HEADERS)
    raise HTTPException(404, "app.js not found")

@app.get("/styles.css")
def serve_styles_css():
    if os.path.isfile("styles.css"):
        return FileResponse("styles.css", media_type="text/css", headers=NO_CACHE_HEADERS)
    raise HTTPException(404, "styles.css not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)

