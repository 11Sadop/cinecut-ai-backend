import os
import soundfile as sf
import numpy as np
from audio_separator.separator import Separator

input_file = "test_song_sample.mp3"  # or test audio
output_dir = "./test_bulletproof"
os.makedirs(output_dir, exist_ok=True)

# 1. Extract VOCALS ONLY
print("Step 1: Extracting Vocals...")
sep_v = Separator(output_dir=output_dir, output_format="WAV", model_file_dir="C:/tmp/audio-separator-models/", output_single_stem="Vocals", log_level=40)
sep_v.load_model("vocals_mel_band_roformer.ckpt")
v_files = sep_v.separate(input_file)
print("VOCALS RESULT:", v_files)

# 2. Extract INSTRUMENTAL ONLY
print("Step 2: Extracting Instrumental...")
sep_i = Separator(output_dir=output_dir, output_format="WAV", model_file_dir="C:/tmp/audio-separator-models/", output_single_stem="Instrumental", log_level=40)
sep_i.load_model("vocals_mel_band_roformer.ckpt")
i_files = sep_i.separate(input_file)
print("INSTRUMENTAL RESULT:", i_files)
