import os
import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

SERVER = "http://127.0.0.1:5000"

PASS = "✅"
FAIL = "❌"

print("=" * 60)
print("  CineCut AI Server - Full Diagnostic Test")
print("=" * 60)

# ── Test 0: Health Check ──
print("\n[TEST 0] Health Check...")
try:
    r = requests.get(f"{SERVER}/api/health", timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f"  Status     : {data.get('status')}")
        print(f"  Whisper    : {data.get('whisper')}")
        print(f"  Demucs     : {data.get('demucs')}")
        print(f"  ffmpeg     : {data.get('ffmpeg')}")
        print(f"  Result     : [{PASS} PASS]")
    else:
        print(f"  [{FAIL} FAIL] Status code {r.status_code}")
except Exception as e:
    print(f"  [{FAIL} FAIL] Cannot reach server: {e}")

# ── Test 1: Microsoft Neural TTS ──
print("\n[TEST 1] Microsoft Neural TTS (edge-tts)...")
mp3_path = "test_tts_output.mp3"
try:
    r = requests.post(f"{SERVER}/api/tts", data={
        "text": "مرحبا بكم في استوديو المنتاج الاحترافي سينيكا تريبر",
        "voice_profile": "ar-cinematic-male"
    }, timeout=30)
    if r.status_code == 200:
        with open(mp3_path, "wb") as f:
            f.write(r.content)
        size_kb = os.path.getsize(mp3_path) / 1024
        print(f"  [{PASS} PASS] Generated MP3: {size_kb:.1f} KB")
    else:
        print(f"  [{FAIL} FAIL] Status {r.status_code}: {r.text}")
        mp3_path = None
except Exception as e:
    print(f"  [{FAIL} FAIL] TTS error: {e}")
    mp3_path = None

# ── Test 2: OpenAI Whisper Transcription ──
print("\n[TEST 2] OpenAI Whisper Model - Arabic Transcription...")
if mp3_path and os.path.isfile(mp3_path):
    try:
        with open(mp3_path, "rb") as af:
            r = requests.post(f"{SERVER}/api/transcribe",
                              files={"file": ("tts.mp3", af, "audio/mpeg")},
                              timeout=120)
        if r.status_code == 200:
            data = r.json()
            segments = data.get("transcript", [])
            lang = data.get("language", "?")
            print(f"  [{PASS} PASS] Language detected: {lang}")
            print(f"  [{PASS} PASS] Segments extracted: {len(segments)}")
            for s in segments:
                print(f"    [{s.get('start')}s -> {s.get('end')}s] \"{s.get('text')}\"")
        else:
            print(f"  [{FAIL} FAIL] Status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  [{FAIL} FAIL] Whisper error: {e}")

# ── Test 3: Meta Demucs Separation ──
print("\n[TEST 3] Meta Demucs htdemucs - Music Separation...")
if mp3_path and os.path.isfile(mp3_path):
    try:
        with open(mp3_path, "rb") as af:
            r = requests.post(f"{SERVER}/api/separate-audio",
                              files={"file": ("tts.mp3", af, "audio/mpeg")},
                              timeout=120)
        if r.status_code == 200:
            data = r.json()
            print(f"  [{PASS} PASS] Separation successful!")
            print(f"  Vocals URL : {data.get('vocals_url')}")
            print(f"  Music URL  : {data.get('music_url')}")

            # Download and verify vocals file
            rv = requests.get(f"{SERVER}{data['vocals_url']}", timeout=30)
            vm_path = "test_vocals.wav"
            with open(vm_path, "wb") as f:
                f.write(rv.content)
            size_kb = os.path.getsize(vm_path) / 1024
            print(f"  [{PASS} PASS] Vocals WAV downloaded: {size_kb:.1f} KB")

            # Download and verify music file
            rm = requests.get(f"{SERVER}{data['music_url']}", timeout=30)
            mus_path = "test_music.wav"
            with open(mus_path, "wb") as f:
                f.write(rm.content)
            size_kb = os.path.getsize(mus_path) / 1024
            print(f"  [{PASS} PASS] Music WAV downloaded: {size_kb:.1f} KB")
        else:
            print(f"  [{FAIL} FAIL] Separation error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"  [{FAIL} FAIL] Demucs error: {e}")

print("\n" + "=" * 60)
print("  Test Complete")
print("=" * 60)
