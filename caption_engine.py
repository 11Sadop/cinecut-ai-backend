"""
caption_engine.py
─────────────────────────────────────────────────────────────────────────
Generates .ass (Advanced SubStation Alpha) subtitle content for burning
animated, styled captions onto video with FFmpeg's `ass` filter (libass).

Supports 8 caption styles, matching the live browser preview in app.js:
  - credits    : cinema end-credits style, scrolling upward
  - karaoke    : per-word color sweep synced to speech timing
  - cinematic  : bold text on a dark rounded box (CapCut style)
  - natural    : simple elegant centered/bottom text
  - neon       : glowing neon-tube text (colored blur halo + crisp core)
  - typewriter : characters appear progressively, left to right
  - tiktok_pop : one bold word at a time, bouncy scale-in pop animation
  - glitch     : flickering RGB-split / jitter glitch effect

Input `segments` is the transcript list produced by /api/transcribe:
  [{"start": float, "end": float, "text": str, "words": [{"start","end","word"}, ...]}, ...]

`words` may be empty (e.g. Google STT fallback, or older cached transcripts)
— every style below degrades gracefully to even time-splitting across the
segment's words when per-word timestamps aren't available.
"""

import re

VIDEO_W = 1920
VIDEO_H = 1080


# ─────────────────────────────────────────────────────────────────────────
#  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────
def _esc(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")
    text = text.replace("\r", "").replace("\n", "\\N")
    return text


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _hex_to_ass_color(hex_color: str, default="&H0000C8FF") -> str:
    try:
        h = (hex_color or "").lstrip("#")
        if len(h) == 6:
            r, g, b = h[0:2], h[2:4], h[4:6]
            return f"&H00{b}{g}{r}".upper()
    except Exception:
        pass
    return default


def _words_for_segment(seg: dict) -> list:
    """Returns [{"start","end","word"}] — uses real word timestamps if
    present, else synthesizes evenly-split timing across segment duration."""
    words = seg.get("words") or []
    if words:
        return [w for w in words if (w.get("word") or "").strip()]

    text = (seg.get("text") or "").strip()
    tokens = [t for t in text.split(" ") if t.strip()]
    if not tokens:
        return []
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start + 2.0))
    dur = max(0.3, end - start)
    step = dur / len(tokens)
    out = []
    for i, tok in enumerate(tokens):
        out.append({"start": start + i * step, "end": start + (i + 1) * step, "word": tok})
    return out


def _ass_header(font_name: str, font_size: int, primary_color: str) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_W}
PlayResY: {VIDEO_H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size * 2},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,40,40,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# ─────────────────────────────────────────────────────────────────────────
#  Per-style event builders — each returns a list of raw "Dialogue: ..." lines
# ─────────────────────────────────────────────────────────────────────────
def _build_credits(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        txt = _esc(seg["text"])
        y_start = VIDEO_H + 60
        y_end = -80
        tags = (
            f"{{\\an8\\pos({VIDEO_W // 2},{y_start})"
            f"\\move({VIDEO_W // 2},{y_start},{VIDEO_W // 2},{y_end},0,{int((end - start) * 1000)})"
            f"\\c{color}\\bord2\\shad2\\fad(150,150)}}"
        )
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{tags}{txt}")
    return lines


def _build_karaoke(segments, font_color, font_size):
    highlight = _hex_to_ass_color(font_color)
    dim = "&H00E0E0E0"
    lines = []
    for seg in segments:
        words = _words_for_segment(seg)
        if not words:
            continue
        start = float(words[0]["start"])
        end = float(words[-1]["end"])
        if end <= start:
            continue
        k_parts = []
        for w in words:
            dur_cs = max(1, int(round((float(w["end"]) - float(w["start"])) * 100)))
            k_parts.append(f"{{\\k{dur_cs}}}{_esc(w['word'])}")
        txt = " ".join(k_parts)
        tags = f"{{\\an2\\1c{highlight}\\2c{dim}\\bord2\\shad2\\fad(100,100)}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{tags}{txt}")
    return lines


def _build_cinematic(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        txt = _esc(seg["text"])
        tags = f"{{\\an2\\c{color}\\bord0\\shad0\\3c&H000000&\\4c&H80000000&\\fad(150,150)}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{tags}{txt}")
    return lines


def _build_natural(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        txt = _esc(seg["text"])
        tags = f"{{\\an2\\c{color}\\bord2\\shad3\\fad(200,200)}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{tags}{txt}")
    return lines


def _build_neon(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        txt = _esc(seg["text"])
        # Layer 0: soft blurred glow halo (thick colored outline, heavy blur)
        glow_tags = f"{{\\an2\\c&H00000000&\\3c{color}\\bord10\\blur14\\shad0\\fad(150,150)\\alpha&H30&}}"
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{glow_tags}{txt}")
        # Layer 1: crisp white-hot core on top for the classic neon-tube look
        core_tags = f"{{\\an2\\c&H00FFFFFF&\\3c{color}\\bord2\\blur1\\shad0\\fad(150,150)}}"
        lines.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{core_tags}{txt}")
    return lines


def _build_typewriter(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        words = _words_for_segment(seg)
        if not words:
            continue
        seg_end = float(seg["end"])
        built = ""
        for wi, w in enumerate(words):
            word_text = w["word"]
            w_start = float(w["start"])
            w_end = float(w["end"])
            char_count = max(1, len(word_text))
            char_dur = max(0.02, (w_end - w_start) / char_count)
            for ci in range(1, char_count + 1):
                reveal_at = w_start + (ci - 1) * char_dur
                reveal_until = seg_end if wi == len(words) - 1 and ci == char_count else (w_start + ci * char_dur)
                visible_text = (built + word_text[:ci]).strip()
                if not visible_text:
                    continue
                tags = f"{{\\an2\\c{color}\\bord2\\shad2}}"
                lines.append(
                    f"Dialogue: 0,{_ass_time(reveal_at)},{_ass_time(reveal_until)},Default,,0,0,0,,{tags}{_esc(visible_text)}"
                )
            built += word_text + " "
    return lines


def _build_tiktok_pop(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        words = _words_for_segment(seg)
        for w in words:
            start, end = float(w["start"]), float(w["end"])
            if end <= start:
                continue
            dur_ms = max(60, int((end - start) * 1000))
            pop_in = min(120, dur_ms // 2)
            tags = (
                f"{{\\an5\\c{color}\\bord3\\shad2\\fscx55\\fscy55"
                f"\\t(0,{pop_in},\\fscx118\\fscy118)"
                f"\\t({pop_in},{dur_ms},\\fscx100\\fscy100)}}"
            )
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{tags}{_esc(w['word'])}")
    return lines


def _build_glitch(segments, font_color, font_size):
    color = _hex_to_ass_color(font_color)
    lines = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        txt = _esc(seg["text"])
        dur_ms = max(80, int((end - start) * 1000))
        jitter = min(400, dur_ms)

        # Cyan ghost layer, shifted slightly left
        cyan_tags = (
            f"{{\\an2\\c&H00FFFF00&\\alpha&H70&\\bord0\\shad0\\pos({VIDEO_W // 2 - 4},{VIDEO_H - 90})"
            f"\\t(0,{jitter},\\frz1)\\t({jitter},{jitter * 2},\\frz-1)}}"
        )
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{cyan_tags}{txt}")

        # Magenta ghost layer, shifted slightly right
        magenta_tags = (
            f"{{\\an2\\c&H00FF00FF&\\alpha&H70&\\bord0\\shad0\\pos({VIDEO_W // 2 + 4},{VIDEO_H - 90})"
            f"\\t(0,{jitter},\\frz-1)\\t({jitter},{jitter * 2},\\frz1)}}"
        )
        lines.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{magenta_tags}{txt}")

        # Crisp core layer on top
        core_tags = f"{{\\an2\\c{color}\\bord2\\shad1\\fad(60,60)}}"
        lines.append(f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{core_tags}{txt}")
    return lines


_STYLE_BUILDERS = {
    "credits": _build_credits,
    "karaoke": _build_karaoke,
    "cinematic": _build_cinematic,
    "natural": _build_natural,
    "neon": _build_neon,
    "typewriter": _build_typewriter,
    "tiktok_pop": _build_tiktok_pop,
    "glitch": _build_glitch,
}


def build_ass(segments: list, style_mode: str = "credits", font_name: str = "Arial",
              font_size: int = 28, font_color: str = "#ffc800") -> str:
    """Main entry point used by server.py's /api/burn-subtitles endpoint."""
    builder = _STYLE_BUILDERS.get(style_mode, _build_natural)
    header = _ass_header(font_name, font_size, _hex_to_ass_color(font_color))
    events = builder(segments, font_color, font_size)
    if not events:
        events = [f"Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,{{\\an2}}"]
    return header + "\n".join(events)


def segments_from_plain_text(text: str, total_duration: float = 10.0) -> list:
    """Fallback used when the frontend only sends plain text (no timing
    JSON) — splits lines evenly across the estimated video duration, same
    behaviour as the original engine before per-style ASS generation."""
    lines = [l.strip() for l in (text or "").replace("\r", "").split("\n") if l.strip()]
    if not lines:
        lines = ["تفريغ النص بالذكاء الاصطناعي سينيكات"]
    line_dur = max(2.5, total_duration / max(1, len(lines)))
    segments = []
    t = 0.0
    for l in lines:
        t_end = min(total_duration, t + line_dur) if total_duration > 0 else t + line_dur
        segments.append({"start": t, "end": max(t_end, t + 0.5), "text": l, "words": []})
        t = t_end
    return segments
