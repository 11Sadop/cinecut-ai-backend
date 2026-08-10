import numpy as np
import soundfile as sf
import time
from debleed_engine import apply_spectral_debleed

# Create a test mixture with singing synth (sine 440Hz + 880Hz harmonics) + loud background music (100Hz + noise)
fs = 44100
dur = 4.0
t = np.linspace(0, dur, int(fs * dur), False)

# Vocal part (intermittent singing)
vocal_true = np.zeros_like(t)
vocal_true[int(fs*1):int(fs*3)] = 0.5 * np.sin(2*np.pi*440*t[int(fs*1):int(fs*3)]) + 0.3 * np.sin(2*np.pi*880*t[int(fs*1):int(fs*3)])

# Music part (continuous loud drums/bass/guitar)
music_true = 0.4 * np.sin(2*np.pi*100*t) + 0.2 * np.sin(2*np.pi*200*t) + 0.1 * np.random.randn(len(t))

# Raw separated vocal (with 20% music bleed)
vocal_extracted = vocal_true + 0.25 * music_true
music_extracted = music_true + 0.05 * vocal_true

print("Testing Spectral De-Bleed...")
t0 = time.time()
clean_vocal = apply_spectral_debleed(vocal_extracted, music_extracted, alpha=0.3, threshold_factor=0.1)
print(f"De-bleed finished in {time.time()-t0:.3f}s!")

# Calculate noise residual during vocal silence (0s to 1s)
silence_bleed_before = np.mean(np.abs(vocal_extracted[:int(fs*1)]))
silence_bleed_after = np.mean(np.abs(clean_vocal[:int(fs*1)]))

print(f"Music Bleed in Vocal Silence BEFORE: {silence_bleed_before:.5f}")
print(f"Music Bleed in Vocal Silence AFTER:  {silence_bleed_after:.5f}")
print(f"Bleed Reduction: {(1 - silence_bleed_after/silence_bleed_before)*100:.1f}%")
