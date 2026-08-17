"""
text_correction.py
─────────────────────────────────────────────────────────────────────────
Post-processing / spelling & normalization engine for transcripts produced
by Whisper (faster-whisper) and Google STT inside CineCut AI Studio.

Goals:
  1. Fix common Arabic ASR artifacts (stutter repeats, bracketed noise tags,
     inconsistent spacing/punctuation, known dialect/phonetic mistakes).
  2. Provide a light, dependency-optional English spell-correction pass.
  3. Stay 100% safe: every transform here is either a pure regex cleanup or
     an opt-in dictionary fix — nothing here should change the *meaning* of
     a correctly transcribed sentence.

This module has NO required third-party dependencies. `pyspellchecker` is
used only if installed (optional line in requirements.txt); if missing we
silently skip English spell correction and keep the normalized text.
"""

import re

# ─────────────────────────────────────────────────────────────────────────
#  1) Arabic ASR artifact normalization
# ─────────────────────────────────────────────────────────────────────────
# NOTE: this used to also run an "ARABIC_LYRIC_CORRECTIONS" regex table here
# (e.g. blindly rewriting every occurrence of the real word "شكوى" to
# "شكواي", "لؤاك" to "لقاك", etc). That table was hand-built against ONE
# specific test song's lyrics during an earlier debugging session and then
# left running unconditionally on every single Arabic transcript afterward.
# It directly violated this module's own stated safety rule above ("nothing
# here should change the *meaning* of a correctly transcribed sentence") --
# on any other audio, those patterns silently swapped in real but wrong
# words wherever they happened to match, which is very likely the actual
# cause of repeated "استخراج النص قاعد يخبص" (transcript coming out
# scrambled/wrong) reports. Removed entirely; only the generic,
# content-agnostic cleanup below (noise-tag stripping, punctuation/spacing
# normalization, stutter-repeat collapsing) remains, which can't corrupt
# real words because it never depends on what the words themselves are.

# Bracketed / parenthesized hallucination tags Whisper frequently injects
# for silence, music beds or crowd noise. Safe to strip entirely.
_NOISE_TAG_PATTERN = re.compile(
    r'[\[\(（]\s*(music|applause|laughter|noise|silence|inaudible|'
    r'موسيقى|تصفيق|ضحك|صمت|ضوضاء|غير مسموع|موسيقا)\s*[\]\)）]',
    re.IGNORECASE
)

# Arabic tatweel (kashida) — purely cosmetic elongation character, safe
# to strip since it never changes meaning.
_TATWEEL_PATTERN = re.compile(r'ـ+')

# Arabic diacritics (tashkeel: fatha, damma, kasra, sukun, shadda,
# tanwin, etc). REPORTED BUG: transcript comes out fully diacritized
# (مُشَكَّلة -style marks on every letter) which nobody wants for a normal
# transcript/caption -- diacritics are optional pronunciation guides in
# written Arabic, virtually always omitted outside Quran/school-textbook
# typesetting, and Whisper large-v3 sometimes emits them inconsistently
# (it was trained on some diacritized sources like Quran recitation/MSA
# news). Stripping them is always safe -- removing tashkeel never
# changes which WORDS are on the page, only the (already-omitted-by-
# convention) pronunciation hints on top of them.
_DIACRITICS_PATTERN = re.compile(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED\u08D3-\u08FF]')

# Whisper frequently hallucinates a single bare (unbracketed) word like
# "موسيقى" / "Music" for a whole segment when that stretch of audio is pure
# instrumental with no speech — the noise-tag pattern above only strips the
# BRACKETED form ("[موسيقى]"), so a bare hallucinated word was slipping
# through as if it were real transcribed speech (reported bug: extracting
# text from a music clip returned the literal word "موسيقى" as the
# "transcript"). We only drop a segment when, after trimming punctuation,
# it consists of NOTHING BUT one of these words — never when the word
# appears as part of an actual sentence (e.g. "أحب الموسيقى كثيراً" stays
# untouched), so this can't eat real speech that happens to mention music.
_BARE_HALLUCINATION_WORDS = {
    'موسيقى', 'موسيقا', 'تصفيق', 'ضحك', 'صمت', 'ضوضاء', 'غير مسموع',
    'music', 'applause', 'laughter', 'noise', 'silence', 'inaudible',
}
_TRIM_PUNCT_PATTERN = re.compile(r'^[\s.,!?؟،؛:\-–—…]+|[\s.,!?؟،؛:\-–—…]+$')


def _strip_if_bare_hallucination(text: str) -> str:
    """Returns '' if `text`, once trimmed of surrounding punctuation, is
    NOTHING but one known Whisper hallucination word; otherwise returns
    `text` unchanged."""
    if not text:
        return text
    bare = _TRIM_PUNCT_PATTERN.sub('', text).strip().lower()
    return '' if bare in _BARE_HALLUCINATION_WORDS else text

# Collapse 2+ identical consecutive punctuation marks: "!!!" -> "!", "؟؟" -> "؟"
_REPEAT_PUNCT_PATTERN = re.compile(r'([!؟?.,،؛;:])\1{1,}')

# Collapse runs of whitespace
_MULTI_SPACE_PATTERN = re.compile(r'[ \t]{2,}')

# Fix a space inserted *before* Arabic/Latin punctuation (ASR often does this)
_SPACE_BEFORE_PUNCT_PATTERN = re.compile(r'\s+([!؟?.,،؛;:])')

# Ensure a single space *after* punctuation when followed directly by a letter
_NO_SPACE_AFTER_PUNCT_PATTERN = re.compile(r'([!؟?.,،؛:])([^\s\d])')


def _collapse_stutter_repeats(text: str) -> str:
    """Removes immediate duplicate word repeats caused by ASR stutter
    hallucination, e.g. 'الكلمة الكلمة الجملة' -> 'الكلمة الجملة'.
    Keeps legitimate intentional repetition of length > 2 (common in lyrics
    like 'لا لا لا') by only collapsing runs of the *same* word longer than
    what natural Arabic singing/poetry repetition typically uses (>=4)
    down to 2 repeats, and collapsing simple accidental doubles (word word)
    only when the word has 3+ letters (to avoid eating intentional short
    interjections like 'يا يا').
    """
    words = text.split(' ')
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        run = 1
        while i + run < len(words) and words[i + run] == w:
            run += 1
        if len(w) >= 3 and run == 2:
            # Accidental single duplicate from ASR -> keep one
            out.append(w)
        elif run >= 4:
            # Excessive repeat glitch -> cap at 2 (still reads as emphasis)
            out.extend([w, w])
        else:
            out.extend([w] * run)
        i += run
    return ' '.join(out)


def clean_arabic_lyric(text: str) -> str:
    """Full Arabic transcript cleanup pipeline: dialect corrections +
    generic ASR-artifact normalization. Safe to run on every Arabic
    transcript segment."""
    if not text:
        return text

    text = _NOISE_TAG_PATTERN.sub('', text)
    text = _TATWEEL_PATTERN.sub('', text)
    text = _DIACRITICS_PATTERN.sub('', text)
    text = _REPEAT_PUNCT_PATTERN.sub(r'\1', text)
    text = _SPACE_BEFORE_PUNCT_PATTERN.sub(r'\1', text)
    text = _NO_SPACE_AFTER_PUNCT_PATTERN.sub(r'\1 \2', text)
    text = _MULTI_SPACE_PATTERN.sub(' ', text)
    text = _collapse_stutter_repeats(text)
    text = text.strip()
    return _strip_if_bare_hallucination(text)


# ─────────────────────────────────────────────────────────────────────────
#  2) English normalization + optional spell-check
# ─────────────────────────────────────────────────────────────────────────
_spell_checker_instance = None
_spell_checker_load_failed = False


def _get_spell_checker():
    """Lazily loads pyspellchecker if it's installed. Returns None and
    never raises if the optional dependency is missing."""
    global _spell_checker_instance, _spell_checker_load_failed
    if _spell_checker_instance is not None:
        return _spell_checker_instance
    if _spell_checker_load_failed:
        return None
    try:
        from spellchecker import SpellChecker
        _spell_checker_instance = SpellChecker(distance=1)
        return _spell_checker_instance
    except Exception:
        _spell_checker_load_failed = True
        return None


_WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z']+|[^A-Za-z']+")


def spellcheck_english(text: str) -> str:
    """Best-effort English spell correction. Skips short tokens, tokens
    with digits, ALL-CAPS acronyms and tokens that look like proper nouns
    (capitalized mid-sentence) to avoid mangling names/brands. Falls back
    to the original text untouched if pyspellchecker isn't installed."""
    if not text:
        return text
    checker = _get_spell_checker()
    if checker is None:
        return text

    tokens = _WORD_TOKEN_PATTERN.findall(text)
    out = []
    for idx, tok in enumerate(tokens):
        if not tok.isalpha() or len(tok) <= 2 or not tok.isascii():
            out.append(tok)
            continue
        if tok.isupper():  # acronym like "AI", "USA"
            out.append(tok)
            continue
        if tok[0].isupper() and idx > 0:  # likely a proper noun / name
            out.append(tok)
            continue
        lower = tok.lower()
        if checker.known([lower]):
            out.append(tok)
            continue
        correction = checker.correction(lower)
        if correction and correction != lower:
            out.append(correction)
        else:
            out.append(tok)
    return ''.join(out)


def normalize_english_text(text: str) -> str:
    if not text:
        return text
    text = _REPEAT_PUNCT_PATTERN.sub(r'\1', text)
    text = _SPACE_BEFORE_PUNCT_PATTERN.sub(r'\1', text)
    text = _MULTI_SPACE_PATTERN.sub(' ', text)
    text = text.strip()
    return text


# ─────────────────────────────────────────────────────────────────────────
#  3) Public dispatcher used by server.py
# ─────────────────────────────────────────────────────────────────────────
def postprocess_transcript_text(text: str, language: str = "ar") -> str:
    """Single entry point: routes text through the correct cleanup/spell
    pipeline based on detected/selected language code."""
    if not text:
        return text
    lang = (language or "ar").lower()
    if lang.startswith("ar"):
        return clean_arabic_lyric(text)
    if lang.startswith("en"):
        return _strip_if_bare_hallucination(spellcheck_english(normalize_english_text(text)))
    # Unknown language: run only the safe generic cleanup (punctuation/
    # spacing), skip dictionary-specific corrections.
    text = _REPEAT_PUNCT_PATTERN.sub(r'\1', text)
    text = _MULTI_SPACE_PATTERN.sub(' ', text)
    return _strip_if_bare_hallucination(text.strip())

# BUILD-FORCE: sync marker to force a fresh unambiguous RunPod build containing ROUND12 debleed + NVENC + diacritics-strip + Arabic bg-removal fixes together.
