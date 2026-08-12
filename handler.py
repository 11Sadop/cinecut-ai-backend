"""
handler.py — RunPod Serverless entry point for CineCut AI Studio.
─────────────────────────────────────────────────────────────────────────
This is what turns the project into an auto-scaling GPU backend: RunPod
Serverless (not a RunPod Pod) spins up GPU workers on demand when jobs are
queued, runs multiple workers in parallel under load, and scales back down
to ZERO when idle — so you only pay for the seconds actual processing
happens. That scaling behavior is native to RunPod's Serverless product;
this file just needs to expose the right `handler(job)` function shape for
RunPod to call, and configuring min/max workers is done in the RunPod
dashboard when you create the Serverless Endpoint (min workers = 0 for
scale-to-zero, max workers = however many concurrent jobs you want to
support at peak).

It deliberately reuses the exact same processing functions already used by
the local FastAPI dev server (server.py) — imported directly, not
duplicated — so behavior never drifts between the two.

FILE TRANSFER — why URLs, not base64:
Both RunPod's job payloads AND Vercel's serverless functions have hard
practical size ceilings (Vercel Functions specifically reject any request
body over 4.5 MB outright), which real video/audio files blow past
immediately. So instead of embedding file bytes in the job JSON, every
input file is uploaded by the browser straight to Vercel Blob (bypassing
Vercel's function body-size limit entirely — see api/blob-upload.js) and
this worker receives just a `file_url` string, downloads it directly, and
uploads its OWN output back to that same Vercel Blob store, returning only
a URL. The frontend never sees base64 for anything.

Required environment variable on the RunPod Endpoint (Settings → Environment
Variables): BLOB_READ_WRITE_TOKEN — same value as the one Vercel generated
for your Blob store (Vercel Project → Storage → your store → copy the
token). This lets this Python worker upload directly to the same store the
browser uploads to.

Request shape (job["input"]):
  {
    "operation": "separate_audio" | "stem_from_url" | "transcribe"
               | "transcribe_url" | "upscale" | "upscale_url"
               | "download_url" | "remove_background_image"
               | "remove_background_video" | "burn_subtitles",
    "file_url": "<https URL of the uploaded input, from /api/blob-upload>",
    "filename": "input.mp4",
    ... operation-specific params (see each branch below) ...
  }
"""

import base64
import os
import tempfile
import time
import traceback
import uuid

import requests
import runpod

# Reuse the exact same processing logic as the local FastAPI server —
# importing server.py runs its module-level setup (FastAPI app object,
# lazy model loaders) but that's harmless/inert until a route or function
# is actually called.
import server
import ai_engine
from ai_engine import process_video_ai_upscale_and_motion
from caption_engine import build_ass, segments_from_plain_text

try:
    from bg_removal_engine import remove_background_image, remove_background_video
    BG_REMOVAL_AVAILABLE = True
except Exception:
    BG_REMOVAL_AVAILABLE = False

try:
    import vercel_blob
    BLOB_AVAILABLE = bool(os.environ.get("BLOB_READ_WRITE_TOKEN"))
except Exception:
    BLOB_AVAILABLE = False


TEMP_DIR = server.TEMP_DIR


def _decode_input_file(job_input, default_ext="mp4"):
    """Fetches the input file. Prefers file_url (Vercel Blob — no size
    limit); falls back to file_base64 for small/legacy callers."""
    filename = job_input.get("filename") or f"input.{default_ext}"
    file_url = job_input.get("file_url")
    if file_url:
        resp = requests.get(file_url, timeout=180)
        resp.raise_for_status()
        return resp.content, filename
    b64 = job_input.get("file_base64")
    if b64:
        return base64.b64decode(b64), filename
    raise ValueError("Missing required field: file_url (or legacy file_base64)")


def _upload_output(local_path, content_type=None):
    """Uploads a finished output file to Vercel Blob and returns its public
    URL, or None if the file is missing or Blob isn't configured (in which
    case the caller should fall back to returning nothing usable — the
    RunPod deployment isn't fully configured yet)."""
    if not local_path or not os.path.isfile(local_path):
        return None
    if not BLOB_AVAILABLE:
        raise RuntimeError(
            "BLOB_READ_WRITE_TOKEN is not set on this RunPod endpoint — "
            "cannot upload results. Copy the token from Vercel → Storage → "
            "your Blob store into the RunPod endpoint's environment variables."
        )
    pathname = f"results/{uuid.uuid4().hex}_{os.path.basename(local_path)}"
    with open(local_path, "rb") as f:
        data = f.read()
    opts = {"addRandomSuffix": "false"}
    if content_type:
        opts["contentType"] = content_type

    # ROOT CAUSE of "تعثرت عملية الترقية" (confirmed via a real FAILED
    # request in RunPod's own request log: error = "Request failed after
    # retries. Please try again.", executionTime ~80s): a 4K/120FPS AI
    # upscale output is a genuinely large video file, and this was doing a
    # single-shot, non-multipart PUT straight to Vercel Blob. Once the SDK's
    # own internal retry budget for that one request was exhausted, it just
    # gave up — no outer retry existed here at all, unlike the frontend's
    # uploadToBlob() which already got a retry wrapper earlier. multipart
    # uploads split large files into resumable chunks (far more reliable
    # over a real network for anything beyond a few MB), and the outer loop
    # below adds a second safety net for purely transient failures.
    file_size_mb = len(data) / (1024 * 1024)
    if file_size_mb > 4:
        opts["multipart"] = True

    last_err = None
    for attempt in range(1, 4):
        try:
            resp = vercel_blob.put(pathname, data, opts)
            return resp.get("url")
        except Exception as e:
            last_err = e
            print(f"⚠️ Blob upload attempt {attempt}/3 failed ({file_size_mb:.1f}MB): {e}")
            if attempt < 3:
                time.sleep(attempt * 2)
    raise RuntimeError(f"فشل رفع الملف الناتج إلى التخزين بعد عدة محاولات ({file_size_mb:.1f}MB): {last_err}")


# ─────────────────────────────────────────────────────────────────────────
#  Operation handlers — each mirrors the matching FastAPI endpoint in
#  server.py, downloading its input from file_url and uploading its
#  output(s) back to Vercel Blob, returning plain URL fields that app.js
#  can use directly (as <video>/<audio> src or for direct download).
# ─────────────────────────────────────────────────────────────────────────
def _op_separate_audio(job_input):
    raw, filename = _decode_input_file(job_input)
    resolution = job_input.get("resolution", "none")
    fps = job_input.get("fps", "none")
    result = server._sync_separate_audio(raw, filename, resolution=resolution, fps=fps)

    session_id = result.get("session_id")
    out = {"status": result.get("status", "success"), "session_id": session_id}

    file_map = {
        "clean_media_url": (os.path.join(TEMP_DIR, f"clean_{session_id}.mp4"), "video/mp4"),
        "vocals_url": (os.path.join(TEMP_DIR, f"vocals_{session_id}.wav"), "audio/wav"),
        "guitar_url": (os.path.join(TEMP_DIR, f"guitar_{session_id}.wav"), "audio/wav"),
        "piano_url": (os.path.join(TEMP_DIR, f"piano_{session_id}.wav"), "audio/wav"),
        "drums_url": (os.path.join(TEMP_DIR, f"drums_{session_id}.wav"), "audio/wav"),
        "bass_url": (os.path.join(TEMP_DIR, f"bass_{session_id}.wav"), "audio/wav"),
        "other_url": (os.path.join(TEMP_DIR, f"other_{session_id}.wav"), "audio/wav"),
    }
    for key, (path, ct) in file_map.items():
        url = _upload_output(path, ct)
        if url:
            out[key] = url
    return out


def _op_transcribe(job_input):
    raw, filename = _decode_input_file(job_input, default_ext="wav")
    language = job_input.get("language", "ar")
    result = server._sync_transcribe(raw, filename, language)
    return result  # already plain JSON (transcript list + language), no files to upload


def _op_upscale(job_input):
    raw, filename = _decode_input_file(job_input)
    resolution = job_input.get("resolution", "4k")
    fps = job_input.get("fps", "120")
    color_mode = job_input.get("color_mode", "face")
    speed = job_input.get("speed", "ai")  # default to the REAL AI path on GPU workers

    session_id = f"rp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    input_path = os.path.join(TEMP_DIR, f"upscale_in_{session_id}.mp4")
    output_path = os.path.join(TEMP_DIR, f"upscale_out_{session_id}.mp4")
    with open(input_path, "wb") as f:
        f.write(raw)

    ok = process_video_ai_upscale_and_motion(
        input_path, output_path, resolution=resolution, fps=fps, color_mode=color_mode, speed=speed
    )
    if not ok:
        return {"status": "error", "error": ai_engine.LAST_ERROR or "Upscale failed"}

    url = _upload_output(output_path, "video/mp4")
    return {"status": "success", "session_id": session_id, "upscale_url": url, "clean_media_url": url}


def _op_upscale_url(job_input):
    url_in = job_input.get("url", "")
    if not url_in:
        return {"status": "error", "error": "Missing 'url'"}
    resolution = job_input.get("resolution", "4k")
    fps = job_input.get("fps", "120")
    color_mode = job_input.get("color_mode", "pure")
    speed = job_input.get("speed", "ai")

    dl_info = server._sync_download_url(url_in, fmt="video")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp4")
    input_path = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(input_path):
        return {"status": "error", "error": "Failed to download source video"}

    output_path = os.path.join(TEMP_DIR, f"upscale_out_{session_id}.mp4")
    ok = process_video_ai_upscale_and_motion(
        input_path, output_path, resolution=resolution, fps=fps, color_mode=color_mode, speed=speed
    )
    if not ok:
        return {"status": "error", "error": ai_engine.LAST_ERROR or "Upscale failed"}
    url = _upload_output(output_path, "video/mp4")
    return {"status": "success", "session_id": session_id, "upscale_url": url, "clean_media_url": url}


def _op_stem_from_url(job_input):
    url_in = job_input.get("url", "")
    if not url_in:
        return {"status": "error", "error": "Missing 'url'"}
    resolution = job_input.get("resolution", "none")
    fps = job_input.get("fps", "none")
    result = server._sync_stem_from_url(url_in, resolution=resolution, fps=fps)
    session_id = result.get("session_id")
    out = {"status": result.get("status", "success"), "session_id": session_id}
    file_map = {
        "clean_media_url": (os.path.join(TEMP_DIR, f"clean_{session_id}.mp4"), "video/mp4"),
        "vocals_url": (os.path.join(TEMP_DIR, f"vocals_{session_id}.wav"), "audio/wav"),
        "guitar_url": (os.path.join(TEMP_DIR, f"guitar_{session_id}.wav"), "audio/wav"),
        "piano_url": (os.path.join(TEMP_DIR, f"piano_{session_id}.wav"), "audio/wav"),
        "drums_url": (os.path.join(TEMP_DIR, f"drums_{session_id}.wav"), "audio/wav"),
        "bass_url": (os.path.join(TEMP_DIR, f"bass_{session_id}.wav"), "audio/wav"),
        "other_url": (os.path.join(TEMP_DIR, f"other_{session_id}.wav"), "audio/wav"),
    }
    for key, (path, ct) in file_map.items():
        url = _upload_output(path, ct)
        if url:
            out[key] = url
    return out


def _op_download_url(job_input):
    url_in = job_input.get("url", "")
    if not url_in:
        return {"status": "error", "error": "Missing 'url'"}
    fmt = job_input.get("fmt", "video")
    result = server._sync_download_url(url_in, fmt=fmt)
    session_id = result.get("session_id")
    ext = result.get("ext", "mp4")
    path = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    content_type = "audio/mpeg" if fmt == "audio" else "video/mp4"
    url = _upload_output(path, content_type)
    return {"status": "success", "session_id": session_id, "file_url": url, "result_url": url}


def _op_transcribe_url(job_input):
    url_in = job_input.get("url", "")
    if not url_in:
        return {"status": "error", "error": "Missing 'url'"}
    language = job_input.get("language", "ar")
    dl_info = server._sync_download_url(url_in, fmt="audio")
    session_id = dl_info["session_id"]
    ext = dl_info.get("ext", "mp3")
    audio_file = os.path.join(TEMP_DIR, f"dl_{session_id}.{ext}")
    if not os.path.isfile(audio_file):
        return {"status": "error", "error": "Failed to download source audio"}
    with open(audio_file, "rb") as f:
        raw_bytes = f.read()
    return server._sync_transcribe(raw_bytes, f"url_audio.{ext}", language)


def _op_remove_background_image(job_input):
    if not BG_REMOVAL_AVAILABLE:
        return {"status": "error", "error": "Background removal engine unavailable on this worker"}
    raw, filename = _decode_input_file(job_input, default_ext="png")
    mode = job_input.get("mode", "transparent")
    color = job_input.get("color", "#00ff00")
    blur_amount = int(job_input.get("blur_amount", 25))

    session_id = f"rp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    ext_in = filename.split(".")[-1].lower() if "." in filename else "png"
    input_path = os.path.join(TEMP_DIR, f"bgimg_in_{session_id}.{ext_in}")
    output_path = os.path.join(TEMP_DIR, f"bgimg_out_{session_id}.png")
    with open(input_path, "wb") as f:
        f.write(raw)

    custom_bg_path = None
    custom_bg_url = job_input.get("custom_bg_url")
    if custom_bg_url:
        custom_bg_path = os.path.join(TEMP_DIR, f"bgimg_custombg_{session_id}.png")
        resp = requests.get(custom_bg_url, timeout=60)
        resp.raise_for_status()
        with open(custom_bg_path, "wb") as f:
            f.write(resp.content)

    ok = remove_background_image(input_path, output_path, mode=mode, color_hex=color,
                                  blur_amount=blur_amount, custom_bg_path=custom_bg_path)
    if not ok:
        return {"status": "error", "error": "Background removal failed"}
    url = _upload_output(output_path, "image/png")
    return {"status": "success", "session_id": session_id, "result_url": url}


def _op_remove_background_video(job_input):
    if not BG_REMOVAL_AVAILABLE:
        return {"status": "error", "error": "Background removal engine unavailable on this worker"}
    raw, filename = _decode_input_file(job_input)
    mode = job_input.get("mode", "color")
    color = job_input.get("color", "#00ff00")
    blur_amount = int(job_input.get("blur_amount", 25))

    session_id = f"rp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    ext_in = filename.split(".")[-1].lower() if "." in filename else "mp4"
    out_ext = "webm" if mode == "transparent" else "mp4"
    input_path = os.path.join(TEMP_DIR, f"bgvid_in_{session_id}.{ext_in}")
    output_path = os.path.join(TEMP_DIR, f"bgvid_out_{session_id}.{out_ext}")
    with open(input_path, "wb") as f:
        f.write(raw)

    custom_bg_path = None
    custom_bg_url = job_input.get("custom_bg_url")
    if custom_bg_url:
        custom_bg_path = os.path.join(TEMP_DIR, f"bgvid_custombg_{session_id}.png")
        resp = requests.get(custom_bg_url, timeout=60)
        resp.raise_for_status()
        with open(custom_bg_path, "wb") as f:
            f.write(resp.content)

    ok = remove_background_video(input_path, output_path, server.FFMPEG_PATH, mode=mode, color_hex=color,
                                  blur_amount=blur_amount, custom_bg_path=custom_bg_path)
    if not ok:
        return {"status": "error", "error": "Background removal failed"}
    content_type = "video/webm" if out_ext == "webm" else "video/mp4"
    url = _upload_output(output_path, content_type)
    return {"status": "success", "session_id": session_id, "output_ext": out_ext, "result_url": url}


def _op_burn_subtitles(job_input):
    raw, filename = _decode_input_file(job_input)
    text = job_input.get("text", "")
    style_mode = job_input.get("style_mode", "credits")
    font_size = int(job_input.get("font_size", 28))
    font_color = job_input.get("font_color", "#ffc800")
    font_name = job_input.get("font_name", "Arial")
    segments_json = job_input.get("segments_json", "")

    result = server._sync_burn_subtitles(raw, filename, text, style_mode, font_size, font_color, font_name, segments_json)
    session_id = result.get("session_id")
    output_path = os.path.join(TEMP_DIR, f"clean_{session_id}.mp4")
    url = _upload_output(output_path, "video/mp4")
    return {"status": "success", "session_id": session_id, "clean_media_url": url}


def _op_tts(job_input):
    """Real neural text-to-speech via edge-tts (Microsoft Edge's free online
    TTS service — no API key needed). server.py already has a working
    /api/tts route that does exactly this, but that FastAPI route only runs
    on a local dev server — it was never reachable in production, because in
    production every request goes through RunPod (this file), not directly
    to server.py. So the frontend's TTS tool had been silently falling back
    to the browser's own built-in speechSynthesis, which ignores the
    specific voice picked from the dropdown entirely (only ~1 Arabic voice
    exists in most browsers) — that's why every "different" voice sounded
    identical (reported bug: "كلها اصوات تتشابه"). This wires the SAME
    edge-tts engine server.py already has into the actual RunPod job path,
    so each of the 8 distinct dropdown voices (already real edge-tts voice
    IDs like ar-SA-HamedNeural, ar-EG-ShakirNeural, etc.) genuinely sounds
    different.
    """
    import asyncio
    import edge_tts

    text = (job_input.get("text") or "").strip()
    if not text:
        return {"status": "error", "error": "النص فارغ"}

    voice = job_input.get("voice") or job_input.get("voice_profile") or "ar-SA-HamedNeural"
    # Accept either a raw edge-tts voice id (what the current dropdown
    # sends) or one of server.py's friendlier VOICE_MAP keys, for safety.
    voice = server.VOICE_MAP.get(voice, voice)
    rate = job_input.get("rate", "+0%")
    pitch = job_input.get("pitch", "+0Hz")

    session_id = f"rp_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    output_path = os.path.join(TEMP_DIR, f"tts_{session_id}.mp3")

    async def _synth():
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)

    try:
        asyncio.run(_synth())
    except Exception as e:
        return {"status": "error", "error": f"فشل توليد الصوت ({voice}): {e}"}

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 100:
        return {"status": "error", "error": "تعذر توليد ملف الصوت"}

    url = _upload_output(output_path, "audio/mpeg")
    return {"status": "success", "session_id": session_id, "result_url": url, "voice": voice}


_LAST_CLEANUP_TS = 0


def _cleanup_old_blobs(prefix, max_age_hours, max_items=2000):
    """Best-effort housekeeping: deletes blobs under `prefix` older than
    `max_age_hours`.

    ROOT CAUSE fix for the recurring upload-storage failures that broke BOTH
    video upscale uploads AND simple image background-removal uploads at the
    same time: every job's output was uploaded to Vercel Blob but NEVER
    deleted afterward, so the store just filled up permanently -- once it
    hit the Hobby plan's 1GB cap, every new upload started failing outright,
    regardless of that upload's own size. This function reclaims space
    automatically as the app is used (called at the start of every job --
    see _maybe_cleanup_old_blobs), so no manual cleanup step is ever
    required again.

    Silent/best-effort: a cleanup hiccup must never break the user's actual
    job, so every exception here is swallowed after a console warning.
    """
    if not BLOB_AVAILABLE:
        return 0
    try:
        import datetime as _dt

        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=max_age_hours)
        cursor = None
        checked = 0
        deleted = 0
        while checked < max_items:
            opts = {"prefix": prefix, "limit": 1000}
            if cursor:
                opts["cursor"] = cursor
            page = vercel_blob.list(opts)
            blobs = page.get("blobs", []) if isinstance(page, dict) else []
            stale_urls = []
            for b in blobs:
                checked += 1
                uploaded_at = b.get("uploadedAt") or b.get("uploaded_at")
                if not uploaded_at:
                    continue
                try:
                    ts = _dt.datetime.fromisoformat(str(uploaded_at).replace("Z", "+00:00"))
                except Exception:
                    continue
                if ts < cutoff:
                    url = b.get("url")
                    if url:
                        stale_urls.append(url)
            if stale_urls:
                try:
                    vercel_blob.delete(stale_urls)
                    deleted += len(stale_urls)
                except Exception as e_del:
                    print(f"blob cleanup: failed deleting a batch under {prefix}: {e_del}")
            has_more = page.get("hasMore") if isinstance(page, dict) else False
            if not has_more:
                break
            cursor = page.get("cursor")
        if deleted:
            print(f"blob cleanup: removed {deleted} stale file(s) under '{prefix}' (older than {max_age_hours}h)")
        return deleted
    except Exception as e:
        print(f"blob cleanup failed for prefix '{prefix}' (non-fatal): {e}")
        return 0


def _maybe_cleanup_old_blobs():
    """Throttled wrapper: runs the actual cleanup at most once every 10
    minutes per warm worker (module-level timestamp resets on cold start),
    so a busy worker handling many jobs back-to-back doesn't re-list the
    whole store on every single request."""
    global _LAST_CLEANUP_TS
    now = time.time()
    if now - _LAST_CLEANUP_TS < 600:
        return
    _LAST_CLEANUP_TS = now
    _cleanup_old_blobs("results/", max_age_hours=6)
    _cleanup_old_blobs("logs/", max_age_hours=24 * 60)


def _log_operation(operation, status, started_at, ended_at, job_input, result, error=None):
    """Persists one small JSON record per job to Vercel Blob under logs/ --
    this is what powers the separate, owner-only ops/cost dashboard.

    Why this exists: the first attempt at a stats dashboard stored
    everything in the BROWSER's localStorage, which only ever sees whatever
    happened in that one specific browser/device -- it silently misses every
    job triggered from any other browser, device, or a future frontend
    entirely. Logging here instead, server-side, in the one place every
    single job (regardless of which frontend/device triggered it) actually
    passes through, makes the record genuinely global and impossible to
    miss. It's also the only place that knows real wall-clock processing
    time, which is what the SAR cost estimate is based on.

    Best-effort and silent on failure: a logging hiccup must never break the
    actual user-facing job, so every exception here is swallowed (after a
    console warning) rather than propagated.
    """
    if not BLOB_AVAILABLE:
        return
    try:
        import json as _json

        clip_duration_sec = job_input.get("clip_duration_sec")
        try:
            clip_duration_sec = float(clip_duration_sec) if clip_duration_sec is not None else None
        except (TypeError, ValueError):
            clip_duration_sec = None

        err_msg = None
        if status == "error":
            err_msg = error or (isinstance(result, dict) and result.get("error")) or "unknown error"

        record = {
            "operation": operation,
            "status": status,  # "done" | "error"
            "started_at": started_at,  # ms epoch
            "ended_at": ended_at,  # ms epoch
            "duration_sec": round(max(0.0, (ended_at - started_at) / 1000.0), 2),
            "clip_duration_sec": clip_duration_sec,
            "error": err_msg,
        }
        pathname = f"logs/{int(started_at)}_{uuid.uuid4().hex[:8]}.json"
        vercel_blob.put(
            pathname,
            _json.dumps(record).encode("utf-8"),
            {"addRandomSuffix": "false", "contentType": "application/json"},
        )
    except Exception as e:
        print(f"warning: ops-log write failed (non-fatal, job itself is unaffected): {e}")


_OPERATIONS = {
    "separate_audio": _op_separate_audio,
    "stem_from_url": _op_stem_from_url,
    "transcribe": _op_transcribe,
    "transcribe_url": _op_transcribe_url,
    "upscale": _op_upscale,
    "upscale_url": _op_upscale_url,
    "download_url": _op_download_url,
    "remove_background_image": _op_remove_background_image,
    "remove_background_video": _op_remove_background_video,
    "burn_subtitles": _op_burn_subtitles,
    "tts": _op_tts,
}


def handler(job):
    job_input = job.get("input", {}) or {}
    operation = job_input.get("operation")

    if operation not in _OPERATIONS:
        return {
            "status": "error",
            "error": f"Unknown or missing 'operation'. Expected one of: {list(_OPERATIONS.keys())}"
        }

    try:
        _maybe_cleanup_old_blobs()
    except Exception as e_cleanup:
        print(f"pre-job blob cleanup skipped (non-fatal): {e_cleanup}")

    started_at = time.time() * 1000
    try:
        result = _OPERATIONS[operation](job_input)
        ended_at = time.time() * 1000
        result_status = (isinstance(result, dict) and result.get("status")) or "success"
        log_status = "error" if result_status == "error" else "done"
        _log_operation(operation, log_status, started_at, ended_at, job_input, result)
        return result
    except Exception as e:
        ended_at = time.time() * 1000
        traceback.print_exc()
        # server.py's _sync_* helpers sometimes raise FastAPI's HTTPException
        # (they were originally written to run inside request handlers) —
        # its str() isn't informative, so surface .detail when present.
        msg = getattr(e, "detail", None) or str(e) or e.__class__.__name__
        _log_operation(operation, "error", started_at, ended_at, job_input, None, error=msg)
        return {"status": "error", "error": msg}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
