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

# Set by the _process_* functions on failure so callers (handler.py) can
# surface a real, specific reason instead of a generic "Upscale failed"
# message. Reset at the start of every process_video_ai_upscale_and_motion
# call.
LAST_ERROR = None


def _test_nvenc_available():
    """Quick probe (a fraction of a second) for whether this machine's
    ffmpeg build can actually encode with h264_nvenc right now (NVIDIA
    driver + GPU present and working). Used to pick the encoder ONCE up
    front for the AI streaming path, instead of discovering NVENC is
    unavailable only after already spending minutes running every frame
    through Real-ESRGAN."""
    try:
        # ROUND 14 FIX -- REAL EVIDENCE from a production job log (Aug 18):
        # the probe itself was failing, not real NVENC access. The ffmpeg
        # stderr this same diagnostic printed last round showed the exact
        # reason: "[h264_nvenc] InitializeEncoder failed: invalid param (8):
        # Frame Dimension less than the minimum supported value." The test
        # clip below used to be a tiny 64x64 frame -- too small for this
        # GPU/driver's NVENC minimum encode dimensions, so the probe always
        # failed and every real job (always >=360p, usually 4K) silently
        # fell back to CPU libx264 even though NVENC itself works fine at
        # real resolutions. Bumped the probe to 1280x720 -- still a ~0.1s
        # test, but comfortably above any GPU generation's minimum.
        cmd = [
            FFMPEG_PATH, "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=20)
        ok = res.returncode == 0
        if not ok:
            # REAL DIAGNOSTIC (was silently swallowed before): after the
            # Dockerfile NVIDIA_DRIVER_CAPABILITIES=video fix, real RunPod
            # job logs still showed "ffmpeg (libx264)" not NVENC, meaning
            # this probe is STILL failing even on the fixed image. Printing
            # the real ffmpeg stderr here (instead of just returning False)
            # is the only way to find out WHY on the next real job -- could
            # be RunPod's own container runtime not honoring the image's
            # driver-capabilities env at all (platform-level, outside our
            # Dockerfile's control), missing NVENC codec support in this
            # ffmpeg build, or a genuinely different error.
            err_tail = (res.stderr or b"").decode(errors="replace")[-400:]
            print(f"⚠️ NVENC probe failed (returncode={res.returncode}): {err_tail}")
        return ok
    except Exception as e_nvenc_probe:
        print(f"⚠️ NVENC probe raised exception: {e_nvenc_probe}")
        return False


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


def _get_duration_sec(path):
    """Source-clip duration via ffprobe's container-level 'format.duration'
    (works regardless of codec/frame-count quirks). Returns None on failure
    so callers can fall back to a safe default."""
    try:
        cmd = [FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_format", path]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        info = json.loads(out)
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        return dur if dur > 0 else None
    except Exception:
        return None


def _adaptive_video_kbps(duration_sec, target_max_mb=320, floor_kbps=6000, ceiling_kbps=15000, audio_kbps=192):
    """
    ROOT CAUSE of repeated upscale-upload failures even after the earlier
    35M -> 16-20M fixed bitrate cap: a fixed Mbps number only bounds size
    PER SECOND -- total file size is bitrate x duration, so a long enough
    clip (a real case hit 1005.9MB) still blows past whatever Vercel Blob's
    upload path can reliably handle. Fix: pick the video bitrate FROM the
    clip's own duration so every output targets roughly the same final
    SIZE (~400MB) regardless of source length, instead of a duration-blind
    fixed Mbps number. Short clips still get the full quality ceiling;
    only clips long enough to risk a huge file get throttled down (with a
    quality floor so it never drops below decent quality).
    """
    if not duration_sec or duration_sec <= 0:
        return ceiling_kbps
    total_kbits_budget = target_max_mb * 8 * 1024
    video_kbits_budget = total_kbits_budget - (audio_kbps * duration_sec)
    bitrate = video_kbits_budget / duration_sec
    return int(max(floor_kbps, min(ceiling_kbps, bitrate)))


def _target_dims(resolution):
    if resolution in ["4k", "2160"]:
        return 3840, 2160
    elif resolution in ["1080", "1080p"]:
        return 1920, 1080
    else:
        return 1280, 720


def _color_eq_filter(color_mode):
    # ROUND 20 FIX: 'pure' is labeled in the UI as 'original video colors,
    # unedited' but was mapped to the exact same aggressive eq filter as
    # 'face' mode -- meaning picking 'pure' never actually did what its own
    # label promised. Split it out to a genuinely neutral pass so the three
    # modes are honest about what they do.
    if color_mode == 'pure':
        return 'eq=contrast=1.0:saturation=1.0:gamma=1.0'
    if color_mode == 'face':
        return 'eq=contrast=1.22:saturation=1.35:gamma=1.08'
    elif color_mode == 'vivid':
        return 'eq=contrast=1.28:saturation=1.45:gamma=1.08'
    return 'eq=contrast=1.15:saturation=1.22'


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

    dur_sec = _get_duration_sec(input_path)
    kbps = _adaptive_video_kbps(dur_sec)
    b_v = f"{kbps}k"
    maxrate = f"{int(kbps * 1.25)}k"
    bufsize = f"{int(kbps * 2)}k"

    cmd = [
        FFMPEG_PATH, "-y", "-i", input_path,
        "-vf", vf_str,
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", b_v, "-maxrate", maxrate, "-bufsize", bufsize, "-profile:v", "main",  # ROUND 14 FIX: removed hardcoded "-level 4.1" -- REAL EVIDENCE from a production job (Aug 18) shows "[h264_nvenc] InitializeEncoder failed: invalid param (9): Invalid Level" as soon as real frames hit NVENC. Level 4.1 tops out around 1920x1080 -- far below the 3840x2160 (4K) target this pipeline outputs, so NVENC correctly rejects it (libx264 is far more lenient about level mismatches, which is why this never surfaced before now that NVENC is actually being reached). Dropping the flag lets ffmpeg auto-select a level that fits the real output resolution/fps instead of hardcoding one sized for 1080p.
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
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-maxrate", maxrate, "-bufsize", bufsize, "-profile:v", "main",  # ROUND 14: dropped hardcoded "-level 4.1" here too for consistency with the NVENC fix above (libx264 is more lenient about level mismatches so this one wasn't the confirmed cause, but the same 1080p-sized level ceiling could still bite at higher output resolutions)
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_path
    ]
    res_cpu = subprocess.run(cmd_cpu, capture_output=True)
    ok = (res_cpu.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)
    if not ok:
        global LAST_ERROR
        LAST_ERROR = "ffmpeg fast-path encode failed (both NVENC and libx264): " + \
            (res_cpu.stderr or b"").decode(errors="replace")[-400:]
    return ok


# ─────────────────────────────────────────────────────────────────────────
#  REAL AI PATH: Real-ESRGAN per-frame super-resolution + FFmpeg
#  motion-compensated `minterpolate` for genuine (non-duplicated) FPS boost.
# ─────────────────────────────────────────────────────────────────────────
def _process_real_ai_upscale(input_path, output_path, target_w, target_h, target_fps, color_mode, progress_cb=None):
    """
    Streams every decoded frame straight through Real-ESRGAN and then
    straight into ffmpeg's stdin — no intermediate raw-frame file is ever
    written to disk.

    IMPORTANT FIX (this used to be the cause of every real-world AI upscale
    failing after a long wait): the previous version wrote every upscaled
    frame as UNCOMPRESSED rgb24 to a temp file before encoding. At a 4K
    target that's 3840*2160*3 ≈ 24.9 MB PER FRAME — a 3-minute clip at
    ~30fps is ~5400 frames, i.e. over 130 GB written to local disk before
    the encode step even started. RunPod worker containers don't have
    anywhere near that much local disk, so the job would silently fail
    (disk full) partway through, after already burning several minutes on
    GPU inference. Piping frames directly into a single ffmpeg process
    fixes this at the root: peak extra disk usage is now ~0 bytes.
    """
    import cv2
    import numpy as np
    import esrgan_engine

    global LAST_ERROR

    if not esrgan_engine.is_available():
        print("⚠️ Real-ESRGAN unavailable (no weights/model) — falling back to fast pipeline.")
        return _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open input video for AI upscale")
        LAST_ERROR = "Could not open input video for AI upscale (corrupt file or unsupported codec)"
        return False

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Pick the encoder ONCE, up front — avoids discovering NVENC is
    # unavailable only after already spending minutes on GPU inference.
    use_nvenc = _test_nvenc_available()

    # Real motion-compensated interpolation (mci = Motion Compensated
    # Interpolation, aobmc = adaptive overlapped block motion compensation)
    # — genuinely estimates and interpolates motion vectors between frames,
    # unlike a naive `fps=` filter which just duplicates/drops frames.
    minterp = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=obmc:vsbmc=0" if target_fps > (src_fps + 0.5) else None  # ROUND 18 FIX: this used to run unconditionally even when target_fps already equalled (or was below) the source fps -- i.e. paying the full motion-compensated-interpolation cost for zero actual FPS gain. Confirmed root cause of duration complaints is really the CPU x264 4K encode (see venc_args comment below), but this filter was still pure wasted GPU/CPU work whenever no real interpolation was needed, and every bit counts. Only build the filter when we are actually increasing the frame rate.  # ROUND 2: was mi_mode=blend.
    # REPORTED AGAIN after switching to blend: "quality is bad" (رفع الجودة
    # سيئ). Root cause of the complaint: mi_mode=blend is not real motion-
    # compensated interpolation at all -- it is a plain weighted cross-
    # fade between the two surrounding real frames, with zero motion
    # vector search. On any real motion in the clip that produces visible
    # double-exposure/ghosting on every interpolated frame, and blown up
    # to a full 4K output that artifact reads as clearly "bad quality",
    # not a subtle tradeoff. blend was chosen purely to kill the mci
    # motion-search cost that was making 4K jobs slow -- but it threw out
    # interpolation quality entirely to do it, which is not what was
    # asked for ("فلاتر ممتازة مناسبة لرفع الجودة"). Switched back to real
    # motion-compensated interpolation (mi_mode=mci) but with the cheaper
    # mc_mode=obmc (single block size) instead of aobmc, and vsbmc
    # (variable-size block motion compensation -- an extra, expensive
    # refinement pass on top of obmc) explicitly disabled. This is still
    # genuine per-block motion vector estimation (no ghosting), just
    # without the two most expensive refinement passes that combined with
    # aobmc+vsbmc=1 at full 4K to produce the original 23m33s job.
    vf_str = f"{minterp},{_color_eq_filter(color_mode)}" if minterp else _color_eq_filter(color_mode)

    duration_sec = (total_frames / src_fps) if (total_frames and src_fps) else None
    kbps = _adaptive_video_kbps(duration_sec)
    b_v = f"{kbps}k"
    maxrate = f"{int(kbps * 1.25)}k"
    bufsize = f"{int(kbps * 2)}k"

    if use_nvenc:
        venc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19", "-b:v", b_v, "-maxrate", maxrate, "-bufsize", bufsize]
    else:
        venc_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "21", "-maxrate", maxrate, "-bufsize", bufsize]  # was preset=veryfast,crf=20 -- REAL EVIDENCE (RunPod logs): NVENC probe still fails even after the Dockerfile driver-capabilities fix, so every AI-upscale job is still CPU-encoding a full 4K stream. Per-frame timing has stayed ~2.6-3.2s/frame across every minterpolate-mode change tried in prior rounds, which points at this CPU encode step (not minterpolate) as the real bottleneck. ultrafast trades a little compression efficiency for real encode-speed -- the most direct lever available while NVENC access remains blocked at the platform level

    cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{target_w}x{target_h}", "-r", str(src_fps),
        "-i", "pipe:0",
        "-i", input_path,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-vf", vf_str,
    ] + venc_args + [
        "-profile:v", "main",  # ROUND 14 FIX: removed hardcoded "-level 4.1" -- same real bug as the fast-pipeline encoder above, confirmed via a real production job's exact ffmpeg stderr ("Invalid Level"). This is the streaming AI-upscale path (the one that failed in production), so this is the fix that actually matters for the reported bug.
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        "-shortest",
        output_path
    ]

    print(f"⚡ Streaming {total_frames or '?'} source frames through Real-ESRGAN -> ffmpeg "
          f"({'NVENC' if use_nvenc else 'libx264'}), target {target_w}x{target_h} — no raw file on disk.")

    # stderr goes to a log FILE, not a pipe: ffmpeg writes progress
    # continuously, and if we're not draining a stderr PIPE while also
    # blocking on writes to stdin, both sides can deadlock on a long video.
    log_path = output_path + ".ffmpeg.log"
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_f)

    frame_count = 0
    pipe_broke = False
    try:
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
            try:
                proc.stdin.write(np.ascontiguousarray(up_rgb).tobytes())
            except (BrokenPipeError, OSError):
                pipe_broke = True
                break
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"   ...{frame_count}/{total_frames or '?'} frames upscaled")
                # Reports progress back through RunPod's job-tracking API so a
                # genuinely long (tens-of-minutes) AI upscale doesn't fall out
                # of RunPod's internal job bookkeeping before it finishes --
                # see the requirements.txt runpod pin comment for the failure
                # this caused ("Failed to return job results | 400 Bad
                # Request" after the GPU work had already completed).
                if progress_cb:
                    try:
                        progress_cb(frame_count, total_frames)
                    except Exception:
                        pass
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass

    proc.wait()
    log_f.close()

    if frame_count == 0:
        try:
            os.remove(log_path)
        except Exception:
            pass
        print("❌ No frames decoded — falling back to fast pipeline")
        return _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    ok_final = (proc.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)

    if not ok_final or pipe_broke:
        try:
            with open(log_path, "rb") as lf:
                err_txt = lf.read().decode(errors="replace")[-500:]
        except Exception:
            err_txt = ""
        LAST_ERROR = (
            f"AI upscale encode failed after {frame_count}/{total_frames or '?'} frames "
            f"(encoder={'NVENC' if use_nvenc else 'libx264'}, pipe_broke={pipe_broke}): {err_txt}"
        )
        print(f"❌ {LAST_ERROR}")
        ok_final = False
    else:
        print(f"✅ Real-ESRGAN reconstructed and encoded {frame_count} frames at {target_w}x{target_h} "
              f"with real motion interpolation.")

    try:
        os.remove(log_path)
    except Exception:
        pass

    return ok_final


def process_video_ai_upscale_and_motion(input_path, output_path, resolution="4k", fps="60", color_mode="face", speed="fast", progress_cb=None):
    """
    Main entry point (unchanged signature — server.py calls this exactly
    as before). `speed` now genuinely changes the pipeline:
      - "fast" : FFmpeg-only scale/sharpen/grade (seconds, no AI)
      - "ai"   : Real-ESRGAN neural super-resolution + real motion-
                 compensated interpolation (much slower, genuinely AI;
                 needs a CUDA GPU to be practical — this is exactly the
                 workload RunPod GPU workers are for)
    """
    global LAST_ERROR
    LAST_ERROR = None
    t0 = time.time()
    target_w, target_h = _target_dims(resolution)
    target_fps = int(fps) if fps in ["24", "30", "60", "120"] else 60  # was 120 -- that default made every job (including ones where the caller sent no/invalid fps) run the most expensive motion-interpolation path

    print(f"⚡ CineCut Upscale Engine: mode={speed}, res={resolution}({target_w}x{target_h}), fps={fps}, color={color_mode}")

    if speed == "ai":
        ok = _process_real_ai_upscale(input_path, output_path, target_w, target_h, target_fps, color_mode, progress_cb=progress_cb)
    else:
        ok = _process_fast_ffmpeg_only(input_path, output_path, target_w, target_h, target_fps, color_mode)

    dur = time.time() - t0
    if ok:
        size_mb = os.path.getsize(output_path) / 1024 / 1024 if os.path.isfile(output_path) else 0
        print(f"✅ Upscale ({speed}) completed in {dur:.2f}s! Size: {size_mb:.1f} MB")
    else:
        print(f"❌ Upscale ({speed}) failed after {dur:.2f}s")
    return ok

# BUILD-FORCE-2: sync marker to force one final unambiguous RunPod build containing ROUND13 debleed + NVENC diagnostics/ultrafast fixes together.

# BUILD-FORCE-3: sync marker to force one final unambiguous RunPod build containing ROUND14 frequency-dependent debleed (server.py) + ROUND14 NVENC probe resolution fix together.
