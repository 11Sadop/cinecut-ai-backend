from audio_separator.separator import Separator
import os, soundfile as sf, numpy as np

# Test with single stem mode
sep_voc = Separator(
    output_dir="./test_single_out",
    output_format="WAV",
    model_file_dir="C:/tmp/audio-separator-models/",
    output_single_stem="Vocals",
    log_level=40
)
sep_voc.load_model("vocals_mel_band_roformer.ckpt")
outs_voc = sep_voc.separate("test_mix.wav")
print("VOCALS ONLY STEM OUTPUT:", outs_voc)

sep_inst = Separator(
    output_dir="./test_single_out",
    output_format="WAV",
    model_file_dir="C:/tmp/audio-separator-models/",
    output_single_stem="Instrumental",
    log_level=40
)
sep_inst.load_model("vocals_mel_band_roformer.ckpt")
outs_inst = sep_inst.separate("test_mix.wav")
print("INSTRUMENTAL ONLY STEM OUTPUT:", outs_inst)
