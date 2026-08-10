"""
CineCut AI Server Auto-Restart Daemon
Ensures server.py is 100% online perpetually on port 5000 with CUDA GPU.
"""
import sys
import time
import subprocess
import os

def run_daemon():
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(cwd)
    print("CineCut Auto-Restart Server Daemon active in:", cwd)

    while True:
        try:
            print("Launching CineCut AI GPU Server (server.py)...")
            p = subprocess.Popen([sys.executable, "server.py"], cwd=cwd)
            p.wait()
            print("Server process exited with code", p.returncode, "restarting in 2s...")
        except Exception as e:
            print("Daemon loop exception:", e)
        time.sleep(2)

if __name__ == "__main__":
    run_daemon()
