import requests
import json
import numpy as np
import soundfile as sf

url = "https://released-river-entry-canyon.trycloudflare.com/api/separate-audio"
print("Sending test request to:", url)

sample_rate = 44100
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 100 * t)
test_wav = "test_audio_sample.wav"
sf.write(test_wav, np.column_stack((signal, signal)), sample_rate)

files = {"file": open(test_wav, "rb")}
data = {"resolution": "none", "fps": "none"}
headers = {"bypass-tunnel-reminder": "true", "Bypass-Tunnel-Reminder": "true"}

print("Uploading...")
res = requests.post(url, files=files, data=data, headers=headers, timeout=300)
print("HTTP Status Code:", res.status_code)
print("Response JSON:")
print(json.dumps(res.json(), indent=2))
