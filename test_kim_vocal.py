from audio_separator.separator import Separator
import os, soundfile as sf, numpy as np, time

print("Testing Kim_Vocal_2.onnx...")
t0 = time.time()
sep = Separator(output_dir="./test_kim_out", output_format="WAV", model_file_dir="C:/tmp/audio-separator-models/")
sep.load_model("Kim_Vocal_2.onnx")
outs = sep.separate("test_mix.wav")

print(f"Done in {time.time()-t0:.2f}s!")
print("Kim_Vocal_2 Outputs:")
for f in outs:
    full_p = os.path.join("./test_kim_out", f) if not os.path.isabs(f) else f
    print(" -", os.path.basename(f), "--> full path:", full_p)
