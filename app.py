import os
import gradio as gr
from server import app

# Simple placeholder UI for Gradio Space interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🎬 CineCut AI Pro Cloud Engine")
    gr.Markdown("This HuggingFace Space hosts the high-performance AI Backend for CineCut Studio (Meta Demucs Neural Vocal Isolation & Whisper STT).")
    gr.HTML("<p style='color: #00f0ff;'>🟢 AI Server status: Active (16GB RAM Neural Compute)</p>")

# Mount FastAPI server onto Gradio so all API endpoints are served securely online 24/7
app = gr.mount_wsgi_app(app, demo, path="/")
