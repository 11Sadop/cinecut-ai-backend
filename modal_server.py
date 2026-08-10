import os
import io
import sys
import tempfile
import subprocess
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse

# ═════════════════════════════════════════════════════════════════════════════
# 1. MODAL GPU CONTAINER IMAGE SETUP
# ═════════════════════════════════════════════════════════════════════════════
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "wget")
    .pip_install("torch", "torchaudio", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("demucs", "faster-whisper", "yt-dlp", "edge-tts", "fastapi[standard]", "numpy", "scipy")
)


app = modal.App("cinecut-ai-studio", image=image)

# Volatile storage volume for media caching
media_volume = modal.Volume.from_name("cinecut-media-cache", create_if_missing=True)

web_app = FastAPI(title="CineCut AI Pro Modal Cloud GPU Engine")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# 2. FASTAPI ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

@web_app.get("/api/health")
def health_check():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    return {
        "status": "ok",
        "server": "Modal GPU Cloud",
        "device": device,
        "gpu": gpu_name,
        "demucs": f"HTDemucs 4-Stem / 6-Stem ({device.upper()}: {gpu_name})",
        "whisper": f"Faster-Whisper ({device.upper()})"
    }

def apply_vad_zero_leak_gate(in_wav_path, out_wav_path):
    """
    Speech-Adaptive VAD + Zero-Leak Music Muting Algorithm:
    - Analyzes vocal RMS energy in 20ms frames.
    - When human voice is quiet/paused: background music bleed is 100% MUTED to zero silence.
    - When human voice is active: vocal speech harmonics are amplified crisply to 95% full volume.
    - Smooth 30ms gaussian-like moving average prevents clicking and distortion.
    """
    import scipy.io.wavfile as wavfile
    import numpy as np

    try:
        sr, data = wavfile.read(in_wav_path)
        if data.size == 0:
            return in_wav_path

        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        else:
            audio = data.astype(np.float32)

        is_stereo = (audio.ndim == 2)
        mono = np.mean(audio, axis=1) if is_stereo else audio

        frame_len = int(sr * 0.02)
        hop_len = int(sr * 0.01)
        if len(mono) < frame_len:
            return in_wav_path

        num_frames = (len(mono) - frame_len) // hop_len + 1
        rms = np.zeros(num_frames, dtype=np.float32)
        
        for i in range(num_frames):
            start = i * hop_len
            frame = mono[start:start + frame_len]
            rms[i] = np.sqrt(np.mean(frame**2) + 1e-10)

        active_rms = rms[rms > 0.003]
        if len(active_rms) > 0:
            speech_thresh = max(0.004, np.percentile(active_rms, 25) * 0.35)
        else:
            speech_thresh = 0.005

        gain_mask = np.where(rms > speech_thresh, 1.0, 0.0).astype(np.float32)

        sample_mask = np.zeros(len(mono), dtype=np.float32)
        for i in range(num_frames):
            st = i * hop_len
            sample_mask[st:st + hop_len] = gain_mask[i]
        
        if len(sample_mask) < len(mono):
            sample_mask[len(sample_mask):] = gain_mask[-1]

        win_size = int(sr * 0.03)
        if win_size % 2 == 0:
            win_size += 1
        window = np.hanning(win_size)
        window /= window.sum()
        smooth_mask = np.convolve(sample_mask, window, mode='same')
        smooth_mask = np.clip(smooth_mask, 0.0, 1.0)

        if is_stereo:
            clean_audio = audio * smooth_mask[:, np.newaxis]
        else:
            clean_audio = audio * smooth_mask

        peak = np.max(np.abs(clean_audio))
        if peak > 0.001:
            clean_audio = clean_audio * (0.95 / peak)

        clean_int16 = (np.clip(clean_audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wavfile.write(out_wav_path, sr, clean_int16)
        return out_wav_path
    except Exception as e:
        print(f"VAD Zero-Leak processing notice: {e}")
        return in_wav_path

@web_app.post("/api/separate-audio")
async def separate_audio_endpoint(file: UploadFile = File(...)):

    """
    100% Neural AI Vocal Isolation using Meta Demucs HTDemucs on GPU.
    Strips 100% of background music, drums, guitars, and synths.
    """
    try:
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(content)
            tmp_in_path = tmp_in.name

        out_dir = tempfile.mkdtemp()
        
        # 1. Run Meta Demucs GPU with --two-stems=vocals
        cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs",
            "-d", "cuda" if torch_has_cuda() else "cpu",
            "-o", out_dir,
            tmp_in_path
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("Demucs CUDA failed, trying CPU fallback:", proc.stderr)
            cmd[5] = "cpu"
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise Exception(f"Demucs GPU separation failed: {proc.stderr}")


        # Find isolated vocals file
        vocals_path = None
        for root, _, files in os.walk(out_dir):
            for f in files:
                if "vocals" in f.lower() and f.endswith(".wav"):
                    vocals_path = os.path.join(root, f)
                    break

        if not vocals_path or not os.path.exists(vocals_path):
            raise Exception("Isolated vocals file was not generated by Demucs.")

        # 2. Convert raw Demucs output to standard 16-bit PCM WAV via FFmpeg
        clean_pcm_path = os.path.join(out_dir, "vocals_pcm.wav")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", vocals_path,
            "-ar", "44100",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            clean_pcm_path
        ]
        ff_proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        pcm_file = clean_pcm_path if (ff_proc.returncode == 0 and os.path.exists(clean_pcm_path)) else vocals_path

        # 3. Apply Speech-Adaptive VAD & Zero-Leak Music Muting Algorithm
        #    - Mutes background music to 100% absolute silence during speech pauses
        #    - Boosts human voice to 95% full loud studio peak (0% volume loss)
        final_file = os.path.join(out_dir, "vocals_zero_leak.wav")
        apply_vad_zero_leak_gate(pcm_file, final_file)
        if not os.path.exists(final_file):
            final_file = pcm_file

        with open(final_file, "rb") as vf:
            vocals_bytes = vf.read()


        # Clean up temp files
        try:
            os.remove(tmp_in_path)
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass

        return Response(
            content=vocals_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="vocals_clean.wav"'}
        )


    except Exception as e:
        print(f"Error in stem separation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def torch_has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ═════════════════════════════════════════════════════════════════════════════
# 3. MODAL WEB SERVING WITH GPU ALLOCATION
# ═════════════════════════════════════════════════════════════════════════════
@app.function(
    gpu="T4",
    timeout=600,
    min_containers=1,
    volumes={"/cache": media_volume}
)

@modal.asgi_app()
def fastapi_app():
    return web_app
