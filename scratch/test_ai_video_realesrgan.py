"""
PyTorch Real-ESRGAN CUDA AI Video Neural Redraw Engine Test
"""
import os
import sys
import io
import time
import cv2
import torch
import subprocess
import json
import numpy as np

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

FFMPEG_PATH = r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"

def test_real_esrgan_video():
    in_video = "sample.mp4"
    if not os.path.exists(in_video):
        in_video = "test_upscale_v80.mp4"

    print(f"🚀 Initializing Real-ESRGAN CUDA AI Neural Redraw for {in_video}...")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(os.path.expanduser('~'), '.cache', 'realesrgan', 'RealESRGAN_x4plus.pth')

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True if device.type == 'cuda' else False,
        device=device
    )

    cap = cv2.VideoCapture(in_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = "test_realesrgan_output_4k.mp4"

    # FFmpeg Pipe for 4K 120FPS NVENC Video Encoding
    ffmpeg_cmd = [
        FFMPEG_PATH, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"3840x2160", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-vf", "fps=fps=120,cas=0.40,eq=contrast=1.10:saturation=1.15",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "16", "-b:v", "30M",
        out_video
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    t0 = time.time()
    count = 0
    print(f"🎬 Starting PyTorch Real-ESRGAN CUDA Neural Redraw on {total_frames} frames ({w}x{h})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        # Downscale for ultra-fast GPU neural tensor inference
        small_frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        output_frame, _ = upsampler.enhance(small_frame, outscale=4)
        output_4k = cv2.resize(output_frame, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)

        proc.stdin.write(output_4k.tobytes())

        if count % 10 == 0 or count == total_frames:
            pct = int((count / total_frames) * 100)
            elapsed = time.time() - t0
            fps_proc = count / elapsed
            print(f"🤖 AI Neural Redraw Pass: frame {count}/{total_frames} ({pct}%) | Speed: {fps_proc:.1f} FPS")

    cap.release()
    proc.stdin.close()
    proc.wait()

    dur = time.time() - t0
    print(f"✅ Real-ESRGAN AI Video Neural Redraw Completed in {dur:.2f}s! Output size: {os.path.getsize(out_video)/1024/1024:.2f} MB")

if __name__ == "__main__":
    test_real_esrgan_video()
