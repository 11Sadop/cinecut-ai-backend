from audio_separator.separator import Separator
import os, soundfile as sf, numpy as np

# Test audio with clear sine wave (vocal) and low tone (music)
sample_rate = 44100
duration = 5.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
sine_vocal = 0.8 * np.sin(2 * np.pi * 1000 * t) # High tone representing vocal
sine_music = 0.8 * np.sin(2 * np.pi * 150 * t)  # Low tone representing music
mix = sine_vocal + sine_music

test_in = "test_mix.wav"
sf.write(test_in, np.column_stack((mix, mix)), sample_rate)

sep = Separator(output_dir="./test_out", output_format="WAV", model_file_dir="C:/tmp/audio-separator-models/")
sep.load_model("vocals_mel_band_roformer.ckpt")
outs = sep.separate(test_in)

print("OUTPUT FILES:")
for f in outs:
    full_p = os.path.join("./test_out", f) if not os.path.isabs(f) else f
    d, sr = sf.read(full_p)
    print(f"File: {os.path.basename(f)} | shape={d.shape} | max_val={np.max(np.abs(d)):.4f}")
