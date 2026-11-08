"""
ZERO-TEXT: MULTI-MODAL VISUAL GALAXY & SEMANTIC SEARCH SYSTEM
Entry Point for ZERO-TEXT Application.

Usage:
    python main.py                     # Runs the full interactive 2D/3D system app & opens browser
    python main.py --port 8080         # Runs on a custom port
    python main.py --no-browser        # Starts server without automatically opening browser
    python main.py --legacy-gui        # Runs the classic Tkinter desktop explorer
"""

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BANNER = r"""
  ==============================================================
    ______ ___________  _____       _____ _______   _________ 
   |___  /|  ___| ___ \|  _  |     |_   _|  ___\ \ / /_   _|
      / / | |__ | |_/ /| | | |______ | | | |__  \ V /  | |  
     / /  |  __||    / | | | |______|| | |  __| /   \  | |  
   ./ /___| |___| |\ \ \ \_/ /       | | | |___/ /^\ \ | |  
   \_____/\____/\_| \_| \___/        \_/ \____/\/   \/ \_/  
  ==============================================================
   ZERO-TEXT 2.0: INTERACTIVE 2D/3D VISUAL GALAXY & SEARCH SYSTEM
  ==============================================================
"""


def open_browser_after_delay(url: str, delay: float = 1.2):
    def _worker():
        time.sleep(delay)
        print(f"[ZERO-TEXT] Launching interactive Visual Galaxy in your browser: {url}")
        webbrowser.open(url)

    threading.Thread(target=_worker, daemon=True).start()


def run_legacy_gui():
    print("[ZERO-TEXT] Launching classic Tkinter Desktop GUI...")
    import tkinter as tk
    from pipeline import get_engine

    engine = get_engine()
    root = tk.Tk()
    root.title("ZERO-TEXT Classic Explorer")
    root.geometry("1100x700")
    root.configure(bg="#0b1020")

    tk.Label(
        root,
        text="ZERO-TEXT Classic Explorer",
        bg="#0b1020",
        fg="#70e1d1",
        font=("Segoe UI", 18, "bold"),
    ).pack(pady=20)

    tk.Label(
        root,
        text=f"Total Indexed Images: {len(engine.metadata)} | Vectors: {engine.index.ntotal}",
        bg="#0b1020",
        fg="#e7edf7",
        font=("Segoe UI", 12),
    ).pack(pady=5)

    def launch_web():
        root.destroy()
        run_server(port=8000, open_browser=True)

    tk.Button(
        root,
        text="Launch Modern 2D/3D Visual Galaxy App",
        command=launch_web,
        bg="#70e1d1",
        fg="#07121a",
        font=("Segoe UI", 12, "bold"),
        padx=20,
        pady=10,
    ).pack(pady=30)

    root.mainloop()


def run_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    import uvicorn
    from server import app

    url = f"http://{host}:{port}"
    print(BANNER)
    print(f" * Server running on: {url}")
    print(f" * 3D Visual Galaxy: {url}")
    print(f" * 2D Star Map:      {url}")
    print(f" * Sketch Search:    {url}")
    print(f" * Folder Ingestion: {url}")
    print(" Press Ctrl+C in this console to stop the server.\n")

    if open_browser:
        open_browser_after_delay(url)

    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="ZERO-TEXT 2.0 Visual Galaxy & Search System")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--legacy-gui", action="store_true", help="Launch classic Tkinter desktop GUI")

    args = parser.parse_args()

    if args.legacy_gui:
        run_legacy_gui()
    else:
        run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
