"""
Hugging Face Spaces Entry Point for ZERO-ARCADE
Compatible with the 100% Free Gradio SDK on Hugging Face Spaces.
"""

import os
import uvicorn
from server import app

# Ensure PORT is read from environment (defaults to 7860 on Hugging Face)
PORT = int(os.environ.get("PORT", 7860))

try:
    import gradio as gr
    # Mount a lightweight Gradio interface to satisfy HF Gradio SDK checks
    with gr.Blocks(title="ZERO-ARCADE") as demo:
        gr.Markdown("## 🌌 ZERO-ARCADE is Live!\n[Click here to open the full experience](/)")
    app = gr.mount_gradio_app(app, demo, path="/_gradio")
except Exception as e:
    print(f"Gradio mounting note: {e}")

if __name__ == "__main__":
    print(f"[ZERO-ARCADE] Starting server on 0.0.0.0:{PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
