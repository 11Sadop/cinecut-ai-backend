"""
bg_removal_engine.py
─────────────────────────────────────────────────────────────────────────
AI background removal engine for CineCut AI Studio — works on both still
images and video, sharing the same neural matting model (rembg / U^2-Net
"isnet-general-use") so quality is consistent across both media types.

Output modes (all four supported for images AND video):
  - "transparent" : keep alpha channel (PNG for images, VP9+alpha WebM for video)
  - "color"        : composite the cut-out subject over a solid color
  - "blur"         : composite the subject over a blurred version of the
                      original background (classic "portrait mode" look)
  - "image"        : composite the subject over a custom uploaded background

Video-specific quality improvement: an exponential-moving-average temporal
smoothing pass is applied to the alpha matte across consecutive frames to
suppress the flicker/edge-jitter that naive per-frame background removal
produces on video.
"""

import os
import io
import subprocess
import numpy as np
import cv2
from PIL import Image, ImageFilter

_rembg_session = None
_rembg_load_failed = False


def _get_session():
    """Lazily creates a single shared rembg inference session (ONNXRuntime).
    Uses CUDA if torch reports a GPU is available, else CPU."""
    global _rembg_session, _rembg_load_failed
    if _rembg_session is not None:
        return _rembg_session
    if _rembg_load_failed:
        return None
    try:
        from rembg import new_session
        providers = None
        try:
            import torch
            if torch.cuda.is_available():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            providers = None
        _rembg_session = new_session("isnet-general-use", providers=providers) if providers else new_session("isnet-general-use")
        return _rembg_session
    except Exception as e:
        print("⚠️ rembg session init failed:", e)
        _rembg_load_failed = True
        return None


def _hex_to_rgb(hex_color: str, default=(0, 255, 0)):
    try:
        h = hex_color.lstrip("#")
        if len(h) == 6:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return default


def _detect_text_mask(pil_img: Image.Image) -> np.ndarray:
    """Heuristic detector for burned-in text/caption overlays (e.g. on-screen
    timestamps, watermarks, subtitles baked into the frame) using MSER blob
    detection + geometric filtering for letter-like shapes, merged into
    word/line blocks. Returns a uint8 mask (255 = likely text, 0 = not) at
    the image's native resolution.

    This exists because rembg's subject-matting model ("isnet-general-use")
    only knows how to keep ONE thing: the main subject it was trained to
    segment (a person, product, etc). Any overlaid text/graphics that aren't
    part of that subject get treated as "background" and erased along with
    everything else — this is the exact complaint that burned-in timestamps
    disappear after background removal. A full OCR/text-detection model
    would be more accurate but needs extra weights shipped in the image;
    this classical CV heuristic (MSER, already available via the
    opencv-python-headless dependency already in requirements.txt) catches
    the common case — small, high-contrast, letter-shaped blobs arranged in
    a line — with no new dependencies or model downloads.
    """
    img_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    mser = cv2.MSER_create()
    mser.setMinArea(10)
    mser.setMaxArea(int(0.02 * w * h))
    regions, _ = mser.detectRegions(gray)

    glyph_mask = np.zeros((h, w), dtype=np.uint8)
    for pts in regions:
        x, y, rw, rh = cv2.boundingRect(pts.reshape(-1, 1, 2))
        if rw == 0 or rh == 0:
            continue
        aspect = rw / float(rh)
        # Letter-like glyphs: roughly upright, not too elongated, small
        # relative to the frame (captions/timestamps are never huge).
        if 0.1 < aspect < 4.0 and 6 <= rh <= max(10, int(h * 0.08)):
            cv2.rectangle(glyph_mask, (x, y), (x + rw, y + rh), 255, -1)

    if not glyph_mask.any():
        return glyph_mask

    # Merge nearby glyphs into word/line blocks (text sits in a row with
    # small gaps between letters — a wide/short dilation kernel bridges
    # those gaps without merging unrelated blobs elsewhere in the frame).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
    merged = cv2.dilate(glyph_mask, kernel, iterations=1)
    merged = cv2.erode(merged, kernel, iterations=1)

    # Only trust blocks that are wide enough to be several merged glyphs —
    # a single isolated blob is more likely texture/noise, not real text.
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(merged)
    for c in contours:
        x, y, rw, rh = cv2.boundingRect(c)
        if rw >= 30 and rh >= 6:
            cv2.rectangle(clean_mask, (x, y), (x + rw, y + rh), 255, -1)
    return clean_mask


def _matte_foreground(pil_img: Image.Image, protect_text: bool = True) -> Image.Image:
    """Runs the neural matting model and returns an RGBA image with a clean
    alpha channel isolating the subject. When protect_text is True (default),
    any region detected as burned-in text/captions (see _detect_text_mask)
    is forced fully opaque, so overlaid timestamps/watermarks survive
    background removal instead of being erased along with the real
    background."""
    from rembg import remove
    session = _get_session()
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=session) if session else remove(buf.getvalue())
    rgba = Image.open(io.BytesIO(out_bytes)).convert("RGBA")

    if protect_text:
        try:
            text_mask = _detect_text_mask(pil_img)
            if text_mask.any():
                r, g, b, a = rgba.split()
                alpha = np.maximum(np.array(a), text_mask)
                rgba = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))
        except Exception as e:
            print("⚠️ text-protect mask failed (non-fatal, continuing without it):", e)

    return rgba


def _composite(rgba_fg: Image.Image, mode: str, color_hex: str, blur_amount: int,
               custom_bg: Image.Image = None, original_rgb: Image.Image = None) -> Image.Image:
    """Composites the matted foreground onto the requested background type.
    Returns an RGBA (transparent mode) or RGB (all other modes) image."""
    w, h = rgba_fg.size

    if mode == "transparent":
        return rgba_fg

    if mode == "color":
        bg = Image.new("RGB", (w, h), _hex_to_rgb(color_hex))
    elif mode == "blur":
        base = original_rgb if original_rgb is not None else rgba_fg.convert("RGB")
        base = base.resize((w, h))
        radius = max(2, min(60, int(blur_amount)))
        bg = base.filter(ImageFilter.GaussianBlur(radius=radius))
    elif mode == "image" and custom_bg is not None:
        bg = custom_bg.convert("RGB").resize((w, h))
    else:
        bg = Image.new("RGB", (w, h), _hex_to_rgb(color_hex))

    bg = bg.convert("RGBA")
    bg.alpha_composite(rgba_fg)
    return bg.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────
#  IMAGE background removal
# ─────────────────────────────────────────────────────────────────────────
def remove_background_image(input_path: str, output_path: str, mode: str = "transparent",
                             color_hex: str = "#00ff00", blur_amount: int = 25,
                             custom_bg_path: str = None) -> bool:
    try:
        original = Image.open(input_path)
        original.load()
        rgba_fg = _matte_foreground(original)
        custom_bg = Image.open(custom_bg_path) if (custom_bg_path and os.path.isfile(custom_bg_path)) else None
        result = _composite(rgba_fg, mode, color_hex, blur_amount, custom_bg, original.convert("RGB"))
        if mode == "transparent":
            result.save(output_path, format="PNG")
        else:
            result.save(output_path, format="PNG" if output_path.lower().endswith(".png") else "JPEG", quality=95)
        return True
    except Exception as e:
        print("❌ remove_background_image error:", e)
        return False


# ─────────────────────────────────────────────────────────────────────────
#  VIDEO background removal (frame-by-frame matting + temporal smoothing)
# ─────────────────────────────────────────────────────────────────────────
def remove_background_video(input_path: str, output_path: str, ffmpeg_path: str,
                             mode: str = "color", color_hex: str = "#00ff00",
                             blur_amount: int = 25, custom_bg_path: str = None,
                             smoothing: float = 0.45, max_side: int = 1280,
                             progress_cb=None) -> bool:
    """Processes every frame with the matting model, applies EMA temporal
    smoothing on the alpha channel to reduce flicker, composites onto the
    requested background, then re-encodes with ffmpeg (re-attaching the
    original audio track). `max_side` downsizes very large frames before
    matting for speed/CPU-RAM safety; the alpha matte is upscaled back to
    the source resolution before compositing so output stays full-res.
    """
    import cv2

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Could not open input video for background removal")
        return False

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    use_alpha_output = (mode == "transparent")
    raw_video_tmp = output_path + ("_rgba.raw" if use_alpha_output else "_rgb.raw")

    custom_bg_img = None
    if mode == "image" and custom_bg_path and os.path.isfile(custom_bg_path):
        custom_bg_img = Image.open(custom_bg_path).convert("RGB")

    prev_alpha_small = None
    frame_idx = 0

    try:
        with open(raw_video_tmp, "wb") as raw_out:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # Downscale for matting speed if needed
                scale = 1.0
                if max(src_w, src_h) > max_side:
                    scale = max_side / float(max(src_w, src_h))
                small_w, small_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
                frame_small = cv2.resize(frame_bgr, (small_w, small_h)) if scale != 1.0 else frame_bgr

                pil_small = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
                rgba_small = _matte_foreground(pil_small)
                alpha_small = np.asarray(rgba_small.split()[-1], dtype=np.float32) / 255.0

                # Temporal EMA smoothing of the alpha matte to reduce flicker
                if prev_alpha_small is not None and prev_alpha_small.shape == alpha_small.shape:
                    alpha_small = smoothing * prev_alpha_small + (1.0 - smoothing) * alpha_small
                prev_alpha_small = alpha_small

                # Upscale matte back to source resolution
                alpha_full = cv2.resize((alpha_small * 255.0).astype(np.uint8), (src_w, src_h), interpolation=cv2.INTER_LINEAR)
                rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgba_full = Image.fromarray(np.dstack([rgb_full, alpha_full]), mode="RGBA")

                original_rgb_full = Image.fromarray(rgb_full)
                composited = _composite(rgba_full, mode, color_hex, blur_amount, custom_bg_img, original_rgb_full)

                if use_alpha_output:
                    raw_out.write(composited.convert("RGBA").tobytes())
                else:
                    raw_out.write(np.array(composited.convert("RGB"))[:, :, ::-1].tobytes())  # RGB->BGR for rawvideo bgr24

                if progress_cb and total_frames:
                    try:
                        progress_cb(frame_idx, total_frames)
                    except Exception:
                        pass
    finally:
        cap.release()

    # ── Encode raw frames + re-attach original audio via ffmpeg ──
    pix_fmt_in = "rgba" if use_alpha_output else "bgr24"
    try:
        if use_alpha_output:
            # Alpha-capable container: WebM (VP9 + yuva420p)
            cmd = [
                ffmpeg_path, "-y",
                "-f", "rawvideo", "-pix_fmt", pix_fmt_in, "-s", f"{src_w}x{src_h}", "-r", str(fps),
                "-i", raw_video_tmp,
                "-i", input_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "28",
                "-c:a", "libopus",
                "-shortest",
                output_path
            ]
        else:
            cmd = [
                ffmpeg_path, "-y",
                "-f", "rawvideo", "-pix_fmt", pix_fmt_in, "-s", f"{src_w}x{src_h}", "-r", str(fps),
                "-i", raw_video_tmp,
                "-i", input_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ]
        res = subprocess.run(cmd, capture_output=True)
        ok_final = (res.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 1000)
        if not ok_final:
            print("ffmpeg bg-removal encode stderr:", res.stderr.decode(errors="ignore")[-800:])
        return ok_final
    except Exception as e:
        print("❌ remove_background_video encode error:", e)
        return False
    finally:
        try:
            if os.path.isfile(raw_video_tmp):
                os.remove(raw_video_tmp)
        except Exception:
            pass
