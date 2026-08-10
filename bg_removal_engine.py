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


def _matte_foreground(pil_img: Image.Image) -> Image.Image:
    """Runs the neural matting model and returns an RGBA image with a clean
    alpha channel isolating the subject."""
    from rembg import remove
    session = _get_session()
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=session) if session else remove(buf.getvalue())
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")


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
