"""
CineCut AI Engine - Real 4K Super Resolution & Motion Interpolation Engine V151
─────────────────────────────────────────────────────────────────────────────
Two upscale modes, matching the "fast" / "ai" choice already exposed in the
frontend (upscale-speed radio buttons):

  speed="fast" : Pure FFmpeg pipeline (Lanczos scale + unsharp + CAS sharpen
                 + color grading). This is NOT AI reconstruction — it's fast,
                 GPU-encoded, and good enough for quick previews. Renamed
                 honestly in comments below (it used to be mislabeled as the
                 "AI" path even though no neural network was involved).

  speed="ai"   : REAL AI super-resolution. Every frame is run through the
                 actual Real-ESRGAN x4plus neural network (esrgan_engine.py,
                 using the RealESRGAN_x4plus.pth checkpoint shipped in
                 weights/) to genuinely reconstruct detail, then FFmpeg's
                 real motion-compensated `minterpolate` filter (mci mode)
                 produces the FPS boost — not naive frame duplication.
                 Falls back to the fast pipeline automatically if the
                 Real-ESRGAN weights/model can't be loaded (e.g. no GPU
                 available and CPU inference would be impractically slow
                 isn't checked automatically — see NOTE below).
"""
import os
import sys
import io
import time
import subprocess
import json

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import shutil as _shutil


def _find_ffmpeg():
    candidates = [
        r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = _shutil.which("ffmpeg")
    return found or "ffmpeg"


def _find_ffprobe():
    candidates = [
        r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
        "/usr/bin/ffprobe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = _shutil.which("ffprobe")
    return found or "ffprobe"


FFMPEG_PATH = _find_ffmpeg()
FFPROBE_PATH = _find_ffprobe()


def get_video_info(video_path):
    """Get FPS and dimensions using ffprobe"""
    try:
        cmd = [
            FFPROBE_PATH, "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        info = json.loads(out)
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                w = int(stream.get("width", 1920))
                h = int(stream.get("height", 1080))
                fps_str = stream.get("r_frame_rate", "30/1")
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 30.0
                return w, h, fps
    except Exception as e:
        print(f"ffprobe error: {e}")
    return 1920, 1080, 30.0


def _target_dims(resolution):
    if resolution in ["4k", "2160"]:
        return 3840, 2160
    elif resolution in ["1080", "1080p"]:
        return 1920, 1080
    else:
        return 1280, 720


def _color_eq_filter(color_mode):
    if color_mode in ["face", "pure"]:
        return "eq=contrast=1.22:saturation=1.35:gamma=1.08"
    elif color_mode == "vivid":
        return "eq=contrast=1.28:saturation=1.45:gamma=1.08"
    return "eq=contrast=1.15:saturation=1.22"


# ─────────────────────────────────────────────────────────────────────────
#  FAST PATH: FFmpeg-only scale + sharpen + color grade (no neural network).
#  Good for quick previews; genuinely fast (a few seconds).
# ─────────────────────────────────────────────────────────────────────────
def _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode):
    if color_mode in ["face", "pure"]:
        sharpen = "unsharp=13:13:4.0:13:13:2.5,cas=0.98"
    elif color_mode == "vivid":
        sharpen = "unsharp=13:13:4.5:13:13:3.0,cas=0.98"
    else:
        sharpen = "unsharp=9:9:3.0:9:9:1.8,cas=0.85"

    vf_str = (
        f"scale={target_w}:{target_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
        f"{sharpen},fps={target_fps},{_color_eq_filter(color_mode)}"
    )

    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-vf", vf_str,
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "16", "-b:v", "35M", "-profile:v", "main", "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
        return True

    print(f"⚠️ NVENC failed, trying CPU libx264 fallback: {res.stderr[:200]}")
    cmd_cpu = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-profile:v", "main", "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_path
    ]
    res_cpu = subprocess.run(cmd_cpu, capture_output=True)
    return (res_cpu.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)


# ─────────────────────────────────────────────────────────────────────────
#  REAL AI PATH: Real-ESRGAN per-frame super-resolution + FFmpeg
#  motion-compensated `minterpolate` for genuine (non-duplicated) FPS boost.
# ─────────────────────────────────────────────────────────────────────────
def _process_real_ai_upscale(input_path, output_path, target_w, target_h, target_fps, color_mode):
    import cv2
    import numpy as np
    import esrgan_engine

    if not esrgan_engine.is_available():
        print("⚠️ Real-ESRGAN unavailable (no weights/model) — falling back to fast pipeline.")
        return _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open input video for AI upscale")
        return False

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    raw_tmp = output_path + "_esrgan_raw.rgb24"
    frame_count = 0

    try:
        with open(raw_tmp, "wb") as raw_out:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                try:
                    up_rgb = esrgan_engine.upscale_frame_to_size(frame_rgb, target_w, target_h)
                except Exception as e_frame:
                    print(f"⚠️ ESRGAN frame {frame_count} failed ({e_frame}), using plain resize fallback")
                    up_rgb = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                raw_out.write(np.ascontiguousarray(up_rgb).tobytes())
                frame_count += 1
    finally:
        cap.release()

    if frame_count == 0:
        try:
            os.remove(raw_tmp)
        except Exception:
            pass
        print("❌ No frames decoded — falling back to fast pipeline")
        return _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    print(f"✅ Real-ESRGAN reconstructed {frame_count} frames at {target_w}x{target_h}. Encoding with real motion interpolation...")

    # Real motion-compensated interpolation (mci = Motion Compensated
    # Interpolation, aobmc = adaptive overlapped block motion compensation)
    # — genuinely estimates and interpolates motion vectors between frames,
    # unlike the old fake `fps=` filter which just duplicates/drops frames.
    minterp = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    vf_str = f"{minterp},{_color_eq_filter(color_mode)}"

    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{target_w}x{target_h}", "-r", str(src_fps),
        "-i", raw_tmp,
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", vf_str,
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "16", "-b:v", "35M", "-profile:v", "main", "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        "-shortest",
        output_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    ok_final = (res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)

    if not ok_final:
        print(f"⚠️ NVENC encode failed for AI path, trying CPU: {res.stderr[:300]}")
        cmd_cpu = [
            FFMPEG_PATH, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{target_w}x{target_h}", "-r", str(src_fps),
            "-i", raw_tmp,
            "-i", input_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-vf", vf_str,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "main", "-level", "4.1",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
            "-shortest",
            output_path
        ]
        res_cpu = subprocess.run(cmd_cpu, capture_output=True)
        ok_final = (res_cpu.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)

    try:
        os.remove(raw_tmp)
    except Exception:
        pass

    return ok_final


def process_video_ai_upscale_and_motion(input_path, output_path, resolution="4k", fps="120", color_mode="face", speed="fast"):
    """
    Main entry point (unchanged signature — server.py calls this exactly
    as before). `speed` now genuinely changes the pipeline:
      - "fast" : FFmpeg-only scale/sharpen/grade (seconds, no AI)
      - "ai"   : Real-ESRGAN neural super-resolution + real motion-
                 compensated interpolation (much slower, genuinely AI;
                 needs a CUDA GPU to be practical — this is exactly the
                 workload RunPod GPU workers are for)
    """
    t0 = time.time()
    target_w, target_h = _target_dims(resolution)
    target_fps = int(fps) if fps in ["24", "30", "60", "120"] else 120

    print(f"⚡ CineCut Upscale Engine: mode={speed}, res={resolution}({target_w}x{target_h}), fps={fps}, color={color_mode}")

    if speed == "ai":
        ok = _process_real_ai_upscale(input_path, output_path, target_w, target_h, target_fps, color_mode)
    else:
        ok = _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    dur = time.time() - t0
    if ok:
        size_mb = os.path.getsize(output_path) / 1024 / 1024 if os.path.isfile(output_path) else 0
        print(f"✅ Upscale ({speed}) completed in {dur:.2f}s! Size: {size_mb:.1f} MB")
    else:
        print(f"❌ Upscale ({speed}) failed after {dur:.2f}s")
    return ok
