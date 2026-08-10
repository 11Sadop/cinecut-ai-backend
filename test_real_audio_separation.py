import os
import soundfile as sf
import numpy as np
import urllib.request
from audio_separator.separator import Separator

# Download a short sample with real human voice + music (e.g. from soundcloud or sample URL)
sample_url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
raw_file = "test_song_sample.mp3"

if not os.path.exists(raw_file):
    print("Downloading sample audio file...")
    urllib.request.urlretrieve(sample_url, raw_file)

print("Running audio-separator MelBand-RoFormer...")
output_dir = "./test_real_out"
os.makedirs(output_dir, exist_ok=True)

sep = Separator(output_dir=output_dir, output_format="WAV", model_file_dir="C:/tmp/audio-separator-models/")
sep.load_model("vocals_mel_band_roformer.ckpt")
output_files = sep.separate(raw_file)

print("\n--- RESULTS FROM AUDIO-SEPARATOR ---")
for f in output_files:
    full_path = os.path.join(output_dir, os.path.basename(f)) if not os.path.isabs(f) else f
    data, sr = sf.read(full_path)
    print(f"Filename: {os.path.basename(f)}")
    print(f"  Path: {full_path}")
    print(f"  Shape: {data.shape}, SampleRate: {sr}")
    print(f"  Peak Amplitude: {np.max(np.abs(data)):.4f}")
    print(f"  Mean Energy: {np.mean(np.abs(data)):.6f}")
