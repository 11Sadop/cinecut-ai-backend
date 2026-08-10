# -*- coding: utf-8 -*-
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from ai_engine import process_video_ai_upscale_and_motion

input_path = r'C:\Users\FSOS\.gemini\antigravity\scratch\capcut-ai-studio\sample.mp4'
output_path = r'C:\Users\FSOS\.gemini\antigravity\scratch\capcut-ai-studio\test_upscale_v80.mp4'

print('=== Testing True Frame-by-Frame Real-ESRGAN V80 ===')
print('Input size:', os.path.getsize(input_path), 'bytes')

result = process_video_ai_upscale_and_motion(
    input_path, output_path,
    resolution='1080', fps='30', color_mode='face'
)

print('Result:', result)
if result and os.path.isfile(output_path):
    sz = os.path.getsize(output_path)/1024/1024
    print(f'Output size: {sz:.2f} MB')
    print('SUCCESS - Real-ESRGAN frame-by-frame upscale works!')
else:
    print('FAILED - check errors above')
