import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🎬 CineCut AI Pro - Cloud CPU Backend\n",
    "Welcome to the official **CineCut AI Pro** Google Colab server! This notebook provides a **100% Free CPU Server (12.7GB RAM)** to run the heavy AI models (Meta Demucs htdemucs_ft & OpenAI Whisper medium) at studio-grade 4K quality.\n",
    "\n",
    "### 🚀 How to use:\n",
    "1. In the top menu, click **Runtime** -> **Run all** (أو اضغط على **▶ Run all** في شريط الأدوات).\n",
    "2. Wait for the installation to complete. Models are pre-cached for instant processing.\n",
    "3. Scroll down to the bottom cell and wait for the **Localtunnel URL** to appear.\n",
    "4. Copy the URL (e.g., `https://xxxx.loca.lt`) and paste it into the CineCut Studio website!\n",
    "5. Click the link once and click **Click to Continue** on the blue screen to authorize the tunnel."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "#@title 📦 Install dependencies & Pre-cache AI Models (يستغرق دقيقة ونصف)\n",
    "# Install python libraries\n",
    "!pip install fastapi uvicorn python-multipart torch torchaudio soundfile scipy numpy faster-whisper demucs edge-tts librosa SpeechRecognition nest-asyncio\n",
    "\n",
    "# Install localtunnel client\n",
    "!npm install -g localtunnel\n",
    "\n",
    "# Pre-download Meta Demucs htdemucs_ft model weights to prevent timeout later\n",
    "print('📥 Pre-downloading Demucs Neural Model weights...')\n",
    "import subprocess\n",
    "subprocess.run(['python', '-c', 'from demucs.pretrained import get_model; get_model(\"htdemucs_ft\")'])\n",
    "\n",
    "# Pre-download Whisper medium weights\n",
    "print('📥 Pre-downloading Whisper Medium model weights...')\n",
    "subprocess.run(['python', '-c', 'from faster_whisper import WhisperModel; WhisperModel(\"medium\", device=\"cpu\", compute_type=\"int8\")'])\n",
    "\n",
    "print('✅ Installation & Caching Complete!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "#@title 🌐 Start the AI Engine and Generate Public URL\n",
    "import subprocess\n",
    "import time\n",
    "import socket\n",
    "import nest_asyncio\n",
    "import urllib.request\n",
    "\n",
    "# Get Colab server public IP (needed for localtunnel password verification)\n",
    "try:\n",
    "    public_ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()\n",
    "except Exception:\n",
    "    public_ip = 'Unknown'\n",
    "\n",
    "# Download server.py directly from the repo\n",
    "import urllib.request\n",
    "url = 'https://raw.githubusercontent.com/11Sadop/cinecut-ai-backend/master/server.py'\n",
    "urllib.request.urlretrieve(url, 'server.py')\n",
    "\n",
    "# Modify server.py to run on port 5000\n",
    "with open('server.py', 'r') as file:\n",
    "    content = file.read()\n",
    "content = content.replace('IS_CLOUD = os.environ.get(\"RENDER\") is not None', 'IS_CLOUD = False')\n",
    "with open('server.py', 'w') as file:\n",
    "    file.write(content)\n",
    "\n",
    "# Start FastAPI server in the background\n",
    "print('🚀 Starting FastAPI server on port 5000...')\n",
    "server_process = subprocess.Popen(['uvicorn', 'server:app', '--host', '127.0.0.1', '--port', '5000'])\n",
    "time.sleep(5)  # Wait for server to boot\n",
    "\n",
    "# Start Localtunnel tunnel on port 5000\n",
    "print('🌐 Generating your Public Tunnel URL...')\n",
    "lt_process = subprocess.Popen(['lt', '--port', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n",
    "\n",
    "time.sleep(3)\n",
    "tunnel_url = ''\n",
    "for line in lt_process.stdout:\n",
    "    line_str = line.decode('utf-8').strip()\n",
    "    if 'url is:' in line_str:\n",
    "        tunnel_url = line_str.split('url is:')[-1].strip()\n",
    "        break\n",
    "\n",
    "print('\\n==================================================================')\n",
    "print('🎬 CINECUT AI ENGINE IS RUNNING!')\n",
    "print(f'🔗 Your Public Server URL: {tunnel_url}')\n",
    "print(f'🔑 Localtunnel Password (IP): {public_ip}')\n",
    "print('==================================================================\\n')\n",
    "print('💡 Note: When you click the URL, it will ask for a password. Paste the IP shown above.')\n",
    "\n",
    "# Keep running\n",
    "try:\n",
    "    while True:\n",
    "        time.sleep(1)\n",
    "except KeyboardInterrupt:\n",
    "    print('Stopping server...')\n",
    "    server_process.terminate()\n",
    "    lt_process.terminate()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("cinecut_colab_backend.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("CPU-default Jupyter notebook generated successfully!")
