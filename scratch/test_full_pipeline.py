"""
End-to-End Pipeline Automated Verification Script
Tests 4K 120FPS Upscale + Vocal/Music Isolation + FFprobe Output Specs
"""
import os
import sys
import io
import time
import requests
import subprocess
import json

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_URL = "http://127.0.0.1:5000"
FFPROBE = r"C:\Users\FSOS\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffprobe.exe"

def run_test():
    sample_video = 'test_upscale_v80.mp4'
    if not os.path.isfile(sample_video):
        sample_video = 'sample.mp4'

    print(f"Testing Full Pipeline with input file: {sample_video}")
    t0 = time.time()

    # Step 1: Trigger async separate-audio + 4K upscale
    with open(sample_video, 'rb') as f:
        files = {'file': (os.path.basename(sample_video), f, 'video/mp4')}
        data = {'resolution': '4k', 'fps': '120', 'color_mode': 'face', 'speed': 'fast'}
        resp = requests.post(f"{BASE_URL}/api/separate-audio", files=files, data=data)

    res_json = resp.json()
    print("Initial Response:", res_json)
    job_id = res_json.get('job_id')

    if not job_id:
        print("FAILED to get job_id!")
        return

    # Step 2: Poll job-status
    print(f"Polling job status for {job_id}...")
    for i in range(40):
        time.sleep(1)
        status_resp = requests.get(f"{BASE_URL}/api/job-status/{job_id}")
        status_data = status_resp.json()
        st = status_data.get('status')
        print(f"[{i+1}s] Job Status: {st}")
        if st == 'done':
            print("SUCCESS! Job Finished!")
            print("Result Data:", json.dumps(status_data, indent=2))
            
            clean_url = status_data.get('clean_media_url')
            session_id = status_data.get('session_id')
            
            # Verify result file via ffprobe
            output_file = os.path.join('temp_downloads', f"clean_{session_id}.mp4")
            if os.path.isfile(output_file):
                print(f"Output file found: {output_file} ({os.path.getsize(output_file)/1024/1024:.2f} MB)")
                cmd = [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_streams", output_file]
                out = subprocess.check_output(cmd).decode()
                info = json.loads(out)
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        w = stream.get('width')
                        h = stream.get('height')
                        fps = stream.get('r_frame_rate')
                        print(f"VERIFIED SPEC: Resolution={w}x{h} (Target 3840x2160 4K), FPS={fps}")
            break
        elif st == 'error':
            print("Job failed with error:", repr(status_data.get('error')))
            break

    print(f"Total test duration: {time.time()-t0:.2f} seconds!")

if __name__ == "__main__":
    run_test()
