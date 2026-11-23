---
title: ZERO-ARCADE
emoji: 🌌
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🌌 ZERO‑ARCADE — Multimodal Visual Galaxy, AI‑Powered Image Search & Kinetic AR Gaming Suite

> **Search your image library by typing text, picking a colour, uploading a photo, or drawing a sketch — powered by OpenCLIP ViT-B/32, FAISS, RapidOCR, and a fine-tuned, CPU‑efficient Sketch‑to‑Photo structural projection architecture — fused with the SHADOWBOX ARCADE 60 FPS zero‑controller motion gaming platform.**

---

## Table of Contents

1. [What is ZERO‑ARCADE?](#what-is-zero-arcade)
2. [Key Features at a Glance](#key-features-at-a-glance)
3. [AI Model Architecture & Pipeline Inventory](#ai-model-architecture--pipeline-inventory)
4. [Prerequisites](#prerequisites)
5. [Installation (Step‑by‑Step)](#installation-step-by-step)
6. [Running the Application](#running-the-application)
7. [How to Use the App](#how-to-use-the-app)
   - [🎓 Interactive Demo Presentation Tour](#1-interactive-demo-presentation-tour)
   - [📊 Performance Benchmarks Dashboard & 100k Scaling](#2-performance-benchmarks-dashboard--100k-scaling)
   - [🛰️ Drag-and-Drop Instant Image Probe & 1-Click Probe](#3-drag-and-drop-instant-image-probe--1-click-probe)
   - [💡 In-Galaxy Smart Vocab Suggestions](#4-in-galaxy-smart-vocab-suggestions)
   - [🔍 Hybrid Multimodal Text Search](#5-hybrid-multimodal-text-search)
   - [✏️ Multi-Color Sketch Studio (Draw to Find)](#6-multi-color-sketch-studio-draw-to-find)
   - [🌈 Colour Search](#7-colour-search)
   - [📁 Upload Photo & Find Similar Images](#8-upload-photo--find-similar-images)
   - [➕ Adding Images & Native Folder Dialog](#9-adding-images--native-folder-dialog)
   - [🗑️ Deleting Images](#10-deleting-images)
   - [🪐 3D Visual Galaxy & Cosmic Volumetrics](#11-3d-visual-galaxy--cosmic-volumetrics)
   - [🗺️ 2D Semantic Star Map](#12-2d-semantic-star-map)
   - [🗂️ Gallery Grid & Persistent Slide-Over Inspector](#13-gallery-grid--persistent-slide-over-inspector)
   - [⚡ Turbo Mode (INT8 Quantized Offline Inference)](#14-turbo-mode-int8-quantized-offline-inference)
   - [👍 Search Result Feedback & Dataset Collection](#15-search-result-feedback--dataset-collection)
   - [⚖️ AI Auto-Weight Decider](#16-ai-auto-weight-decider)
8. [🕹️ Shadowbox Arcade Suite (Secret "SAM" Unlock)](#-shadowbox-arcade-suite-secret-sam-unlock)
   - [Activating the Secret Arcade Portal](#activating-the-secret-arcade-portal)
   - [Included Native Games (Shadow Boxing AR & Fruit Ninja 2.0)](#included-native-games)
   - [Live Sensor & Environmental Diagnostic Mirror](#live-sensor--environmental-diagnostic-mirror)
   - [Hardware Concurrency & Process Lifecycle Supervision](#hardware-concurrency--process-lifecycle-supervision)
   - [Live Telemetry & Hall of Fame Leaderboard](#live-telemetry--hall-of-fame-leaderboard)
9. [Deep Metric Learning & Fine-Tuning Technical Report](#deep-metric-learning--fine-tuning-technical-report)
   - [Cross-Modal Domain Discrepancy](#cross-modal-domain-discrepancy)
   - [Architecture of the `SketchProjection` Module](#architecture-of-the-sketchprojection-module)
   - [Multi-Scale Structural Decomposition Pipeline](#multi-scale-structural-decomposition-pipeline)
   - [Synthetic Pair Generation & InfoNCE Contrastive Optimization](#synthetic-pair-generation--infonce-contrastive-optimization)
   - [Fine-Tuning Benchmark Results (+56.6% Accuracy Gain)](#fine-tuning-benchmark-results-566-accuracy-gain)
   - [Drag & Drop Probe Fine-Tuning (81.63% Accuracy)](#drag--drop-probe-fine-tuning-8163-accuracy)
   - [Sub-Millisecond Barycentric Manifold Coordinate Projection](#sub-millisecond-barycentric-manifold-coordinate-projection)
10. [Quantitative Performance Benchmarks & Scaling](#quantitative-performance-benchmarks--scaling)
    - [FAISS Nearest-Neighbor Scaling (100 to 100,000 Vectors)](#faiss-nearest-neighbor-scaling-100-to-100000-vectors)
    - [INT8 Dynamic Quantization vs FP32 Inference](#int8-dynamic-quantization-vs-fp32-inference)
    - [End-to-End Multimodal Pipeline Latencies](#end-to-end-multimodal-pipeline-latencies)
11. [Academic Research Paper & Presentation Materials](#academic-research-paper--presentation-materials)
12. [Complete REST API Reference](#complete-rest-api-reference)
13. [Project Directory Structure](#project-directory-structure)
14. [Testing, Verification & Offline Utilities](#testing-verification--offline-utilities)
15. [Troubleshooting & FAQ](#troubleshooting--faq)
16. [License](#license)

---

## What is ZERO‑ARCADE?

**ZERO‑ARCADE** is a **fully local, self‑hosted** multimodal intelligence and kinetic entertainment suite. It bridges dense vector search, structural spatial computing, and zero-controller computer vision gaming into a single cyberpunk-styled web application.

The platform provides two deeply integrated experiences:
1. **Multimodal Visual Galaxy**: Explore, cluster, and retrieve images using any combination of:
   - **Natural language** — type "sunset over the ocean" and see matching photos ranked by semantic meaning.
   - **In‑image OCR text** — type "import numpy" and find screenshots or notes containing that code.
   - **Dominant colour** — type "red" or click a colour swatch to find images with that palette.
   - **Hand‑drawn sketch** — draw a rough shape on the canvas with multi-colour brush tools and bucket fill to locate matching photos.
   - **Uploaded photo** — upload reference photos to find visually and semantically related items.
   - **Drag-and-Drop Image Probe** — drop any photo directly onto the celestial galaxy to project an interactive 3D test probe and find nearest neighbors in `<1ms`.
2. **SHADOWBOX ARCADE**: An easter-egg gaming suite unlocked by typing `SAM`, allowing you to play high-framerate motion games using only your standard webcam (no specialized VR headsets or handheld controllers required). Includes **Shadow Boxing AR** (BlazePose 33-point tracking + acoustic Kiai shout detection) and **Fruit Ninja 2.0** (MediaPipe 21-hand dual laser katanas with swept-capsule continuous collision physics).

All indexed images are visualised in an interactive **3D Visual Galaxy** (WebGL) and a **2D Semantic Star Map** where visually similar images cluster together, powered by UMAP dimensionality reduction and HDBSCAN clustering with high-contrast Cyan (`#4cc9f0`) and Pink (`#f72585`) constellation styling.

---

## Key Features at a Glance

| Feature | Category | Description |
|---------|----------|-------------|
| 🎓 **Interactive Demo Tour** | UX / Presentation | Complete 11-step interactive presentation tour with dynamic spotlight polygon cutouts, backdrop blur, canvas visibility mode, and keyboard navigation. |
| 📊 **AI Benchmarks Dashboard** | Performance | In-app modal dashboard visualising FAISS scaling (up to 100,000 vectors), INT8 CPU quantization speedup (1.79x), and pipeline latency metrics with PDF export. |
| 🛰️ **Drag‑and‑Drop Image Probe** | Visual Search / Manifold | Drop any photo directly onto the galaxy. Computes 512‑D CLIP embedding, projects manifold coordinates in `<1ms`, drops a 3D laser probe with rotating gimbal rings & laser beams, and auto-locks nearest neighbor. |
| 🔘 **1-Click Probe Button** | Visual Search / UX | Instant `Probe Image` button in the toolbar allowing one-click demonstration of the 3D manifold probe without dragging external files. |
| 📁 **Native OS Folder Dialog** | Ingestion / UX | Single-click native folder picker (`/api/browse/folder`) via Tkinter topmost dialog to select and index local folders without manual typing. |
| 💡 **In‑Galaxy Smart Vocab** | Visual Search / UX | Real-time autocomplete discovery track under the search bar extracted from dataset OCR text, visual concepts, and dominant colors, featuring dynamic `<mark>` filtering and 1-click execution. |
| 🔍 **Hybrid Text Search** | Visual Search | Combines CLIP semantic understanding + OCR text matching + dominant colour scoring with intelligent auto‑weights. |
| ✏️ **Structural Sketch Search** | Visual Search | Draw strokes or shapes and retrieve matching photos via multi-scale contour + distance-field silhouettes and a fine-tuned projection head (+56.6% accuracy gain). |
| 🎨 **Multi‑Colour Drawing** | Studio | 12 preset colour swatches + custom colour picker for coloured sketches. |
| 🪣 **Bucket Fill (Flood Fill)** | Studio | One‑click flood fill tool with colour tolerance — fill regions of your sketch quickly. |
| ↩️ **Undo / Redo** | Studio | Stroke‑level history (up to 35 steps). Keyboard shortcuts: `Ctrl+Z`, `Ctrl+Y`, `Ctrl+Shift+Z`. |
| 📁 **Upload & Search** | Visual Search | Upload a photo directly onto the sketch canvas and search by visual similarity. |
| 🎯 **Find Similar Images** | Visual Search | Click any image → "Find Visually Similar" to locate related photos in the index. |
| 🌈 **Colour‑Based Search** | Visual Search | Search by hex colour code or colour name — finds images whose dominant palette matches. |
| 📝 **OCR Text Extraction** | Visual Search | RapidOCR automatically reads text from every indexed image (code screenshots, handwritten notes, invoices, etc.). |
| ⚡ **Turbo Mode** | Performance | Toggleable on-the-fly PyTorch INT8 dynamic quantization for offline CPU inference, achieving a 1.79x speedup with zero internet access. |
| 👍 **Feedback & Dataset Curation** | AI Pipeline | Upvote/downvote search results to curate versioned training and evaluation datasets without modifying live model parameters. |
| 🪐 **3D Visual Galaxy & Cosmic Volumetrics** | Visualisation | Interactive WebGL 3D space with procedural breathing nebulae, 1,200+ particle twinkling stardust drift, and orbital constellations in Cyan (`#4cc9f0`) and Pink (`#f72585`). |
| 🗺️ **2D Semantic Star Map** | Visualisation | A 2D UMAP projection showing how all images relate to each other spatially with pan, zoom, and probe support. |
| 🗂️ **Gallery Grid & Persistent Drawer** | Visualisation / UX | Traditional card grid with fixed‑size thumbnails (260 px), cluster chips, and a persistent Slide-Over Inspector Drawer that stays open until dismissed. |
| 🕹️ **Arcade Portal** | Kinetic Gaming | Hidden easter-egg portal unlocked via secret `SAM` keystroke sequence. |
| 🪞 **Live Sensor Diagnostic Mirror** | Kinetic Gaming / UX | Integrated 60 FPS camera mirror with real-time lumen estimation, body framing guides, and audio VU meter for voice calibration. |
| 🥊 **Shadow Boxing AR** | Kinetic Gaming | 33-point BlazePose skeleton tracking, $1\text{€}$ filter stabilization, 6-stage punch state machine, acoustic Kiai battle cries. |
| 🍉 **Fruit Ninja 2.0** | Kinetic Gaming | Dual fist-gripped laser energy swords, swept-capsule continuous collision detection, combo multipliers, bomb hazards. |
| 🏆 **Hall of Fame Leaderboard** | Kinetic Gaming | Live per-player statistics, profile selector, ranking medals, and atomic local persistence. |
| ⚖️ **AI Auto‑Weight Decider** | Intelligence | Analyses your query intent (visual concept vs. text/OCR vs. colour) and auto‑tunes scoring weights. |

---

## AI Model Architecture & Pipeline Inventory

ZERO-ARCADE combines specialized computer vision and metric learning models configured for deterministic, local CPU execution:

```
+───────────────────────────────────────────────────────────────────────────────────+
|                                    INPUT MODES                                    |
|   [ Natural Language Text ]    [ Canvas Drawing ]    [ Photo Upload / Drag-Drop ]  |
+───────────────────────────────────────────────────────────────────────────────────+
         │                               │                               │
         ▼                               ▼                               ▼
+─────────────────+             +─────────────────+             +─────────────────+
| OpenCLIP Text   |             | Multi-Scale     |             | OpenCLIP Visual |
| ViT-B/32        |             | Structural Prep |             | ViT-B/32        |
| (512-D Vectors) |             | (Contours + EDT)|             | (512-D Vectors) |
+─────────────────+             +─────────────────+             +─────────────────+
         │                               │                               │
         │                               ▼                               │
         │                      +─────────────────+                      │
         │                      | SketchProjection|                      │
         │                      | Head (Frozen)   |                      │
         │                      | (Residual MLP)  |                      │
         │                      +─────────────────+                      │
         │                               │                               │
         └───────────────────────┬───────┴───────────────────────────────┘
                                 ▼
                     [ 512-D Cosine Metric Space ]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
+──────────────────────────+               +──────────────────────────+
| FAISS IndexFlatIP Index  |               | Barycentric Manifold     |
| Exact Nearest Neighbors  |               | Projector (< 1ms)        |
+──────────────────────────+               +──────────────────────────+
           │                                           │
           ▼                                           ▼
+──────────────────────────+               +──────────────────────────+
| Hybrid Multi-Modal Score |               | 3D / 2D WebGL Galaxy     |
| CLIP + OCR + Color Blend |               | Interactive Universe     |
+──────────────────────────+               +──────────────────────────+
```

### Complete Model Inventory

| Subsystem | Model / Framework | Configuration / Weights | Purpose |
|-----------|-------------------|-------------------------|---------|
| **Multimodal Vision & Text** | OpenCLIP ViT-B/32 | `laion2b_s34b_b79k` (512-D) | Encodes natural language queries and photographic image inputs into a shared latent metric space. |
| **In-Image Text Recognition** | RapidOCR | ONNX Runtime / PaddleOCR | Automatically extracts embedded text from code screenshots, handwritten notes, and documents. |
| **Sketch Domain Adaptation** | `SketchProjection` | Custom 2-layer residual MLP (`512 -> 768 -> 768 -> 512`) | Maps distance-transformed sketch vectors into the photographic CLIP manifold (InfoNCE fine-tuned, frozen). |
| **Quantized Fast Inference** | PyTorch INT8 Dynamic | `torch.quantization.quantize_dynamic` | Accelerates CPU inference by 1.79x (104.2 ms vs 186.7 ms) with 0.8284 fidelity in Turbo Mode. |
| **Manifold Projection** | Barycentric Weighting | Temperature $\tau = 0.15$, $k=5$ | Projects out-of-sample query embeddings into 3D/2D space in $<1\text{ms}$ without gradient descent. |
| **Dimensionality Reduction** | UMAP | `n_components=2, 3`, `metric="cosine"` | Precomputes non-linear 2D/3D embeddings for the visual galaxy. |
| **Density Clustering** | HDBSCAN | `min_cluster_size=2`, `min_samples=1` | Discovers semantic clusters; mapped to iconic Cyan (`#4cc9f0`) and Pink (`#f72585`) constellation groups. |
| **Kinetic Pose Tracking** | MediaPipe BlazePose | Lite Task Bundle (`33 landmarks`) | 60 FPS skeletal tracking with $1\text{€}$ filter jitter suppression for Shadow Boxing AR. |
| **Kinetic Hand Tracking** | MediaPipe Hand Landmarker | Full Task Bundle (`21 landmarks`) | Dual knuckle/wrist tracking for plasma katana blades in Fruit Ninja 2.0. |
| **Audio Kiai Detection** | `sounddevice` DSP | RMS Decibel Analyzer | Low-latency acoustic microphone capture for martial arts battle cry super moves. |

---

## Prerequisites

Make sure you have the following installed on your system before proceeding:

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **Python** | 3.11+ | Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**. |
| **pip** | (bundled with Python) | Used to install Python dependencies. |
| **Git** | Any recent version | Optional — only needed if cloning the repository. |
| **Web Browser** | Chrome, Edge, or Firefox | Chrome recommended for optimal WebGL 3D performance. |
| **Webcam & Microphone** | Standard USB / Built-in | Required for Shadowbox Arcade motion tracking and audio Kiai detection. |
| **~2.5 GB free disk space** | — | CLIP model (~600 MB) + MediaPipe task models (~13 MB) + indexed dataset assets. |
| **~4 GB RAM** | — | Core search pipeline runs in-memory on CPU. |

> **No GPU required.** The entire system runs locally on CPU with optional INT8 dynamic quantization acceleration.

---

## Installation (Step‑by‑Step)

### 1. Get the project files

```bash
git clone https://github.com/Sudharsan-Parthasarathi/zero-text-aiml.git "Zero text AIML"
cd "Zero text AIML"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install Python dependencies

Install core search and web dependencies:
```bash
pip install -r requirements.txt
```

To enable the native camera, audio, and physics drivers for **Shadowbox Arcade**:
```bash
pip install pygame mediapipe sounddevice
```

#### Complete Package Reference

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch deep learning framework (CPU version). Powers CLIP, INT8 quantization, and SketchProjection. |
| `open-clip-torch` | OpenCLIP library — loads the `ViT‑B‑32` vision‑language foundation model. |
| `faiss-cpu` | Facebook AI Similarity Search — ultra‑fast vector nearest‑neighbour index. |
| `umap-learn` | UMAP dimensionality reduction — projects 512‑D vectors into 2D/3D for galaxy visualisation. |
| `hdbscan` | Hierarchical density‑based clustering — groups similar images into clusters. |
| `fastapi` | Modern async web framework powering the backend API server. |
| `uvicorn` | High-performance ASGI server for FastAPI. |
| `Pillow` | Image loading, resizing, colour extraction. |
| `opencv-python` | Computer vision — Canny edge detection, morphological dilation, distance transforms, webcam video capture. |
| `rapidocr-onnxruntime` | On‑device OCR engine — reads text from screenshots and photos locally. |
| `numpy` | Numerical vector operations and scoring matrices. |
| `pygame` | Native game engine powering Shadowbox Arcade windows, rendering, and input handling. |
| `mediapipe` | Google MediaPipe vision models for 33-point pose landmarking and 21-point hand tracking. |
| `sounddevice` | Low-latency audio DSP for Kiai battle cry microphone detection. |

### 4. First run — model download

The first time you start the app, it will automatically download the **CLIP ViT‑B‑32** model (~600 MB) from Hugging Face and cache it in `model_cache/`. After the first download, the application functions **fully offline**.

---

## Running the Application

### Option A: Quick start (recommended)

**Double‑click** `run_app.bat` in the project root. It activates the virtual environment and launches the server automatically.

### Option B: Command line

```bash
python main.py
```

This boots the FastAPI server on `http://127.0.0.1:8000` and automatically opens your default browser.

```bash
python main.py                     # Default: start server + open browser
python main.py --port 8080         # Use custom port
python main.py --no-browser        # Headless server start
python main.py --legacy-gui        # Classic Tkinter desktop GUI
```

---

## How to Use the App

### 1. Interactive Demo Presentation Tour
Click the **🎓 Demo Tour** button in the header toolbar (`#btn-start-tour`) to launch the interactive presentation walkthrough:
- **Presentation Spotlight**: Dims the screen with a frosted glass blur (`backdrop-filter: blur(8px)`) while carving out a sharp, glowing polygon hole (`clip-path: polygon(...)`) directly over the active target element.
- **Canvas Visibility Mode (`tour-view-canvas`)**: When explaining the **🪐 3D Celestial Galaxy** and **🗺️ 2D Semantic Star Map**, the backdrop remains un-blurred and transparent so the full rotating 3D particle universe remains 100% visible, bright, and interactive during live demonstrations.
- **11 Step Walkthrough**:
  1. *Welcome & System Overview*: Core architecture and dual-engine capabilities.
  2. *3D Visual Galaxy Navigation*: Orbit, pan, and zoom controls.
  3. *Celestial Controls & Constellations*: Volumetric nebulae, stardust, and constellation toggles.
  4. *2D Semantic Star Map*: Continuous planar manifold projection.
  5. *Hybrid Multimodal Text Search*: Natural language, OCR keywords, and color queries.
  6. *In-Galaxy Smart Vocab*: Dynamic autocomplete discovery chips.
  7. *Drag-and-Drop Image Probe*: Manifold projection and 1-click test probe demonstration.
  8. *Multi-Color Sketch Studio*: Freehand drawing, bucket flood fill, and stroke history.
  9. *Gallery Grid & Persistent Drawer*: Card metrics and slide-over inspector.
  10. *AI Performance Benchmarks*: FAISS scaling to 100k, INT8 quantization, and latencies.
  11. *Secret SAM Arcade Suite*: Motion gaming unlock instructions.
- **Keyboard Navigation**: Press `ArrowRight` (Next), `ArrowLeft` (Previous), or `Escape` (Exit Tour) at any time.

### 2. Performance Benchmarks Dashboard & 100k Scaling
Click the **📊 Benchmarks** button in the header (`#btn-tour-benchmark`) to open the **System Performance Benchmarks** modal:
- **FAISS Scaling to 100,000 Vectors**: Live table displaying real empirical benchmarks across 100, 1,000, 10,000, 50,000, and 100,000 vectors with build time, mean latency, P95, P99, QPS, and memory footprint.
- **INT8 Dynamic Quantization vs FP32**: Demonstrates a **1.79x CPU speedup** (104.20 ms INT8 vs 186.72 ms FP32) with 0.8284 cosine fidelity.
- **End-to-End Pipeline Latencies**: Real-world measurements for Hybrid Search (787.3 ms), Sketch Search (341.5 ms), and Image Probe (2,357 ms).
- **Download Official PDF Report**: One-click download button fetching [`BENCHMARK_REPORT.pdf`](file:///e:/Zero%20text%20AIML/BENCHMARK_REPORT.pdf).

### 3. Drag‑and‑Drop Instant Image Probe & 1-Click Probe
Search the 3D universe using any external image file (`JPG`, `PNG`, `WEBP`, `GIF`, `BMP`, `TIFF`, `AVIF`, `ICO`):
1. **Two Ways to Trigger**:
   - **Drag & Drop**: Drag an image file over the browser canvas. An animated full-screen HUD radar targeting reticle (`#galaxy-drop-zone`) appears. Drop the file anywhere.
   - **1-Click Toolbar Button**: Click the **🛰️ Probe Image** button (`#btn-probe-trigger`) in the header toolbar to immediately probe the galaxy with sample telemetry or browse an image file.
2. **Auto-Switch to 3D Space**: Dropping an image automatically shifts focus to the **🪐 3D Galaxy** cluster view.
3. **Instant Manifold Projection (`< 1ms`)**: The system extracts the 512‑D CLIP photographic visual embedding and computes 3D coordinates $(X, Y, Z)$ via exponential distance-weighted barycentric projection, bypassing slow UMAP gradient descent.
4. **3D Glowing Test Probe**:
   - **Central Beacon Core**: Vibrant glowing orb placed at the exact predicted manifold location.
   - **Vertical Light Beam Pillar**: Cyan light column cutting vertically through the coordinate space.
   - **Dual Rotating Gimbal Rings**: Outer pitch and inner yaw orbital rings spinning continuously in real time.
   - **Expanding Sonar Radar Pulses**: Expanding geometric rings broadcasting similarity waves.
   - **Connective Laser Beams**: Cyan dashed energy beams connecting the probe to its closest nearest neighbors.
5. **Telemetry HUD**: Displays probe thumbnail, 3D coordinates, closest match filename, cosine similarity %, and a one-click **✕ Dismiss Probe** button.
6. **Nearest Neighbor Focus & Auto-Select**: Automatically selects the closest match in the library and opens its details in the **Image Inspector Drawer**.
7. **Clean Dismissal**: Clicking **✕ Dismiss Probe** clears all 3D probe visuals, unselects active nodes, closes the inspector drawer, and smoothly restores camera navigation.

### 4. In‑Galaxy Smart Vocab Suggestions
Directly underneath the main search bar is the **In‑Galaxy Smart Vocab** track:
- **Dataset‑Driven Autocomplete**: Tokens are extracted directly from your image index: OCR text snippets (e.g. `numpy`, `login`, `invoice`), visual tags (e.g. `landscape`, `portrait`, `car`), and dominant color keywords.
- **Dynamic Real-Time Filtering**: Typing into the search bar automatically filters suggestions in real time, highlighting matching substrings with `<mark>`.
- **Category Color-Coded Chips**: Chips display their origin type (`OCR`, `Tag`, `Color`) for instant semantic clarity.
- **1-Click Instant Search**: Clicking any chip immediately inputs the term and executes a search across the visual universe.
- **Horizontal Mouse-Wheel Scrolling**: Easily browse hundreds of dataset tokens with smooth horizontal scroll navigation.

### 5. Hybrid Multimodal Text Search
Type any natural language query into the search bar and press Enter:
- `sunset` — finds images with sunset visual content.
- `golden retriever` — finds dog photos.
- `coding` — finds screenshots of code, notes, IDEs.
- `blue ocean waves` — finds ocean/sea images with blue tones.
- `red car` — combines semantic concept ("car") + colour ("red") matching.
- `import numpy` — finds images containing that exact OCR text.

### 6. Multi-Color Sketch Studio (Draw to Find)
1. Click the **✏️ Sketch Search** button in the header.
2. Draw your target on the canvas:
   - **🖊️ Pen** — free-draw strokes with adjustable line width.
   - **🪣 Bucket Fill** — flood-fill bounded regions with pixel-level colour tolerance.
   - **🧹 Eraser** — erase parts of the drawing.
   - **↩️ Undo / ↪️ Redo** — stroke history stack (up to 35 steps, shortcuts: `Ctrl+Z`, `Ctrl+Y`).
3. Pick colours from the 12 preset swatches or custom colour picker.
4. Optionally provide an inline **text hint** (e.g. "car", "cat") to boost Top-1 precision up to 88.6%.
5. Click **🚀 Search Galaxy by Sketch**.

### 7. Colour Search
Search by colour keywords (e.g. `red`, `blue`, `golden`, `teal`, `neon`) or hex codes (`#FF0000`). The engine extracts dominant palettes and applies colour histogram matching.

### 8. Upload Photo & Find Similar Images
- **Upload Reference Photo**: Click **📁 Upload Image** inside the Sketch Studio to load an existing photo and search directly using CLIP photographic embeddings.
- **Find Similar Images**: Click any image card or galaxy node to open the **Image Inspector** drawer, then click **🔍 Find Visually Similar** to retrieve nearest neighbors in the 512-D metric space.

### 9. Adding Images & Native Folder Dialog
Click **➕ Add Images / Folder**:
- **Native OS Folder Picker Dialog**: Click **Browse Folder** to open the native Windows/OS folder picker dialog (via `/api/browse/folder`). Select any local directory on your drive and confirm — no manual typing of long file paths required.
- **Drag & Drop Upload**: Drop individual image files directly into the upload area.
- **Recursive Indexing**: The background pipeline automatically crawls the directory, computes CLIP embeddings, extracts OCR text, extracts dominant color palettes, and updates the FAISS index.

### 10. Deleting Images
Open the Image Inspector drawer for any image and click **🗑️ Delete Image**. The system deletes the image file from disk, removes its vector from the FAISS index, and cleans up metadata files.

### 11. 3D Visual Galaxy & Cosmic Volumetrics
Click the **🪐 3D Galaxy** tab:
- **Click & drag**: Orbit camera in 3D WebGL space.
- **Scroll wheel**: Zoom in and out.
- **Hover point**: Instant tooltip preview.
- **Click point**: Locks selection and opens detailed Image Inspector.
- **Orbit Spin**: Toggles celestial auto-rotation.
- **Constellations**: Toggles cluster connection lines.
- **Procedural Volumetric Nebulae**: Sinusoidal breathing nebula dust clouds with dynamic scale, opacity, and ambient celestial colors.
- **Twinkling Stardust Field**: 1,200+ particle stardust galaxy drift with procedural opacity scintillation.
- **Cyan & Pink Constellations**: Images are grouped into high-contrast semantic clusters styled in vibrant Cyan (`#4cc9f0`) and Pink (`#f72585`).

### 12. 2D Semantic Star Map
Click the **🗺️ 2D Star Map** tab for a continuous 2D planar projection of cluster groupings with pan, zoom, and fit-to-screen controls. Supports Drag-and-Drop Image Probe searches directly onto the canvas.

### 13. Gallery Grid & Persistent Slide-Over Inspector
Click the **🗂️ Gallery Grid** tab for a structured card grid displaying fixed-size thumbnails (260 px), cluster chips, match percentage badges, OCR text snippets, and dominant palette strips.
- **Persistent Slide-Over Inspector**: Opening an image card keeps the inspection drawer open persistently so you can examine high-resolution details, dominant colors, and OCR text without arbitrary auto-dismiss timeouts.

### 14. Turbo Mode (INT8 Quantized Offline Inference)
Toggle Turbo Mode to accelerate CPU inference by **1.79x** (dropping CLIP forward passes from ~186.7 ms to ~104.2 ms):
- **On-Demand Quantization**: Uses PyTorch INT8 dynamic quantization (`torch.quantization.quantize_dynamic`) on local weights.
- **Zero Internet Required**: Operates completely offline with zero external downloads.
- **API Controlled**: Toggleable via `POST /api/turbo` or frontend status settings.

### 15. Search Result Feedback & Dataset Collection
Every search card features positive (👍) and negative (👎) feedback buttons:
- **Positive Feedback**: Confirms query-target pairs and appends validation samples to `data/text_search_dataset.json` and `data/sketch_dataset.json`.
- **Negative Feedback**: Records false positives to `data/feedback_dataset.json` for contrastive evaluation.
- **Zero Runtime Weight Mutation**: User feedback never mutates live model parameters during inference, preventing model drift and ensuring 100% deterministic search.

### 16. AI Auto-Weight Decider
The decider analyzes query intent and automatically balances weights:
- **Visual concept** ("mountain landscape"): 75% CLIP, 15% OCR, 10% Colour.
- **Code / document** ("python code"): 25% CLIP, 65% OCR, 10% Colour.
- **Colour-focused** ("bright red"): 20% CLIP, 5% OCR, 75% Colour.
- **Balanced** ("blue machine learning"): 50% CLIP, 30% OCR, 20% Colour.

---

## 🕹️ Shadowbox Arcade Suite (Secret "SAM" Unlock)

ZERO-ARCADE integrates a complete motion-controlled AR arcade suite directly into the application, executed natively as hardware-accelerated desktop subprocesses.

```
+─────────────────────────────────────────────────────────────────────────+
|                          ZERO-ARCADE BROWSER UI                         |
|   [ 3D Galaxy ]  [ 2D Star Map ]  [ Gallery Grid ]  [ 🕹️ Arcade (SAM) ]  |
+─────────────────────────────────────────────────────────────────────────+
                                    │
                         REST API Calls (/api/arcade/*)
                                    ▼
+─────────────────────────────────────────────────────────────────────────+
|                           FASTAPI BACKEND                               |
|                       arcade_manager.py (Lock)                          |
+─────────────────────────────────────────────────────────────────────────+
         │                              │                             │
         ▼                              ▼                             ▼
+───────────────────+         +───────────────────+         +───────────────────+
|  Arcade Launcher  |         |  Shadow Boxing AR |         |  Fruit Ninja 2.0  |
| 04_arcade_launcher|         |02_shadow_boxer_game|         |03_fruit_ninja_game|
|  Pygame + Audio   |         | MediaPipe 33-Pose |         | MediaPipe 21-Hand |
| Profile Selector  |         | 1€ Filter + Kiai  |         | Laser Blade Physics|
+───────────────────+         +───────────────────+         +───────────────────+
```

### Activating the Secret Arcade Portal
1. Click anywhere in the application outside of active text inputs.
2. Type the secret key sequence: **`S`** $\to$ **`A`** $\to$ **`M`** (case-insensitive, within 2 seconds).
3. The retro-cyberpunk **SHADOWBOX ARCADE UNLOCKED** toast banner slides down, and the **🕹️ Arcade** tab becomes visible in the navigation header.

### Included Native Games

#### 1. Unified Arcade Launcher (`04_arcade_launcher.py`)
- Fullscreen retro-futuristic arcade portal.
- Interactive player profile selector and creation modal.
- Background synthwave audio DSP engine with volume controls.
- Glove skin selector (Classic Red, Electric Blue, Golden Champ, Neon Purple).
- Global high-score leaderboard browser.

#### 2. Shadow Boxing AR (`02_shadow_boxer_game.py`)
- **33-Point BlazePose Skeleton**: Real-time kinematic tracking of body, shoulders, elbows, and wrists.
- **$1\text{€}$ Filter Jitter Suppression**: Adaptive low-pass filter eliminates camera noise while retaining zero-latency response for high-speed strikes.
- **6-Stage Punch State Machine**: Accurately classifies jabs, crosses, hooks, and uppercuts based on velocity thresholds.
- **Acoustic Kiai Microphone Detection**: Low-latency audio DSP captures physical martial arts shouts to trigger screen-clearing super moves.
- **Sonic Shield Claps**: Cross-arm clap detection for defensive shield activation.
- **Mecha Boss Waves**: Increasing difficulty tiers with adaptive AI opponents.

#### 3. Fruit Ninja 2.0 (`03_fruit_ninja_game.py`)
- **Dual Fist Laser Energy Blades**: Tracks left and right knuckles using MediaPipe 21-point hand tracking to generate glowing dual plasma katanas.
- **Swept-Capsule Continuous Collision Detection (CCD)**: Constructs swept cylindrical capsules between consecutive blade frames ($t-1 \to t$), preventing tunneling artifacts on fast-moving fruit slices.
- **Dynamic Combo Multipliers**: Slicing multiple fruits in a single blade stroke triggers cascading score multipliers ($2\times, 3\times, 4\times, 5\times$).
- **Bomb Hazards & Extra-Life Heart Fruits**: Strategic hazards and health recoveries.

### Live Sensor & Environmental Diagnostic Mirror
Built directly into the web Arcade tab, this pre-flight sensor test HUD ensures optimal gaming conditions before launching heavy desktop subprocesses:
- **60 FPS Real-Time Video Mirror**: Low-latency video mirror stream to verify webcam field-of-view and positioning.
- **Photometric Lumen / Lux Estimator**: Samples central frame pixels at 30 Hz and computes perceived relative luminance:
  $$Y = 0.299 R + 0.587 G + 0.114 B$$
  Classifies ambient room lighting into Dark ($< 45$), Dim ($45\text{--}80$), Optimal ($80\text{--}180$), and Overexposed ($> 180$), providing real-time lighting guidance.
- **Body Framing & Stance Guide**: Visual silhouette overlay to help players step into optimal tracking distance (~1.5–2.5 meters from webcam).
- **Acoustic Audio VU Meter**: Real-time Web Audio API analyzer measuring root-mean-square (RMS) decibels to verify microphone responsiveness and sensitivity thresholds for Kiai martial arts shouts.

### Hardware Concurrency & Process Lifecycle Supervision
- Managed by `arcade_manager.py` using thread-safe reentrant locks (`threading.Lock`).
- Ensures only **one** game process owns the webcam and audio hardware at any given moment, preventing device driver deadlocks.
- Web UI provides live process telemetry (`PID`, status dot, active player) and an emergency stop button (**🛑 Emergency Exit**) to terminate background game processes safely.

### Live Telemetry & Hall of Fame Leaderboard
- **Synchronized State**: The web UI automatically reads and synchronizes `player_profiles.json` and high scores.
- **Leaderboard Table**: Displays player ranks, medal badges (`🥇 1st`, `🥈 2nd`, `🥉 3rd`), cumulative high scores, lifetime punch counts, and fruit slice records.
- **Active Player Selector**: Switch the active player directly from the web interface before launching any game.

---

## Deep Metric Learning & Fine-Tuning Technical Report

For detailed mathematical analysis and experiments, see [`FINE_TUNING_REPORT.md`](file:///e:/Zero%20text%20AIML/FINE_TUNING_REPORT.md) and [`FINE_TUNING_REPORT.pdf`](file:///e:/Zero%20text%20AIML/FINE_TUNING_REPORT.pdf).

### Cross-Modal Domain Discrepancy
Raw sketches consist of sparse binary lines and huge white backgrounds. In natural images, each 16x16 patch contains continuous frequency signals, textures, and lighting. Passing raw sketches directly into zero-shot CLIP yields an unacceptable **Top-1 Precision of 21.7%**, with sketches frequently misidentified as typographical characters or source code screenshots (`code.jpg`, `aiml.jpg`).

### Architecture of the `SketchProjection` Module
To solve this without mutating the frozen 86M parameter CLIP backbone, ZERO-ARCADE employs a lightweight parametric residual projection subnetwork: `SketchProjection` (in `pipeline.py`).

```python
class SketchProjection(torch.nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.act1 = torch.nn.GELU()
        
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.act2 = torch.nn.GELU()
        
        self.fc3 = torch.nn.Linear(hidden_dim, dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

        # Identity initialization: zero-out final layer
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc3(self.act2(self.norm2(self.fc2(self.act1(self.norm1(self.fc1(x)))))))
        out = x + self.alpha * delta
        return out / out.norm(dim=-1, keepdim=True)
```

#### Mathematical Formulation
$$\Delta \mathbf{z} = \mathbf{W}_3 \cdot \text{GELU}\Big(\text{LN}\big(\mathbf{W}_2 \cdot \text{GELU}(\text{LN}(\mathbf{W}_1 \mathbf{z}_{\text{in}} + \mathbf{b}_1)) + \mathbf{b}_2\big)\Big) + \mathbf{b}_3$$
$$\mathbf{z}_{\text{res}} = \mathbf{z}_{\text{in}} + \alpha \cdot \Delta \mathbf{z} \quad (\alpha = 0.8)$$
$$\mathbf{z}_{\text{out}} = \frac{\mathbf{z}_{\text{res}}}{\|\mathbf{z}_{\text{res}}\|_2}$$

### Multi-Scale Structural Decomposition Pipeline
Before passing strokes into CLIP, the sketch is decomposed into multi-scale representations:
1. **Foreground Bounding-Box Centering**: Automatically detects stroke boundaries, crops margins, and rescales strokes proportionally onto a standardized $224 \times 224$ white canvas with 12% padding.
2. **Density-Adaptive Stroke Contour (Weight: 1.3)**: Thickens low-density strokes (< 4%) via morphological dilation (`cv2.erode` on inverted canvas), preserving line visibility during Vision Transformer patch tokenization.
3. **Continuous Distance-Field Silhouette (Weight: 0.8)**: Computes exact Euclidean Distance Transforms (EDT):
   $$D(p) = \min_{q \in \text{Strokes}} \|p - q\|_2$$
   Normalized to $[0, 255]$ with exponential falloff ($e^{-D(p)/\sigma}$), capturing 2D spatial mass and volume.
4. **Horizontal Mirror Invariance (Weight: 0.9)**: Evaluates mirrored strokes to eliminate hand-drawing orientation bias.
5. **High-Frequency Code Screenshot Suppression**: Computes horizontal/vertical gradient frequency ratios to suppress false-positive code screenshot matches.

### Synthetic Pair Generation & InfoNCE Contrastive Optimization
Training pairs were generated by extracting multi-scale structural transformations from gallery photos (Canny edge detection at multiple thresholds, morphological skeletonization, and random stroke jitter) aligned with authentic user drawings. The projection head was optimized using symmetric InfoNCE contrastive loss:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{sketch}}, \mathbf{z}_{\text{photo}}^+) / \tau)}{\sum_{j} \exp(\text{sim}(\mathbf{z}_{\text{sketch}}, \mathbf{z}_{\text{photo}}^j) / \tau)}$$

### Fine-Tuning Benchmark Results (+56.6% Accuracy Gain)

| Metric | Raw Zero-Shot CLIP | Fine-Tuned `SketchProjection` | Absolute Gain |
|--------|-------------------|--------------------------------|---------------|
| **Top-1 Precision ($P@1$)** | 21.7% | **78.3%** | **+56.6%** |
| **Top-1 with Text Hint** | 47.8% | **88.6%** | **+40.8%** |
| **Recall@3 ($R@3$)** | 39.1% | **91.3%** | **+52.2%** |
| **Mean Reciprocal Rank (MRR)** | 0.312 | **0.842** | **+0.530** |
| **False-Positive Code Matches** | 43.5% | **0.0%** | **Completely Eliminated** |
| **Forward Pass Latency (CPU)** | 0.00 ms (no head) | **1.15 ms** | Negligible overhead |

### Drag & Drop Probe Fine-Tuning (81.63% Accuracy)
During project evolution, user validation revealed cross-domain confusion on natural photographs (e.g. cat images misclassified as dogs or generic grass). We executed specialized fine-tuning with semantic anchor vectors and hard-negative contrastive mining (`scratch/verify_probe_finetuning.py`, `scratch/test_cat_ginger_grass.py`):
- **Final Top-1 Accuracy**: **81.63%** across diverse validation sets.
- **Cat/Dog Disambiguation**: Semantic anchor weighting ensures exact alignment with target species while maintaining invariant background tolerance.

### Sub-Millisecond Barycentric Manifold Coordinate Projection
When dragging an external image onto the 3D galaxy, standard out-of-sample UMAP projection (`reducer.transform()`) runs numerical gradient descent, taking **60–70 seconds on CPU** and hanging the server event loop.

ZERO-ARCADE eliminates this bottleneck with an **instant ($< 1\text{ ms}$) exponential distance-weighted barycentric coordinate projection**:
$$\mathbf{p}_{\text{probe}} = \frac{\sum_{i=1}^k w_i \cdot \mathbf{y}_i}{\sum_{i=1}^k w_i}, \quad w_i = \exp\left(-\frac{1 - s_i}{\tau}\right)$$

Where:
- $\mathbf{y}_i \in \mathbb{R}^3$ (or $\mathbb{R}^2$): Precomputed manifold coordinates of nearest neighbors already embedded in the galaxy.
- $s_i \in [-1, 1]$: Cosine similarity between query image embedding and nearest neighbor in FAISS.
- $\tau = 0.15$: Temperature constant sharpening focus toward the closest visual neighbors.

---

## Quantitative Performance Benchmarks & Scaling

All benchmarks were executed on a commodity 16-thread CPU workstation. Source data is available in [`BENCHMARK_REPORT.json`](file:///e:/Zero%20text%20AIML/BENCHMARK_REPORT.json) and [`BENCHMARK_REPORT.pdf`](file:///e:/Zero%20text%20AIML/BENCHMARK_REPORT.pdf).

### FAISS Nearest-Neighbor Scaling (100 to 100,000 Vectors)

| Corpus Size ($N$) | Build Time (ms) | Mean Latency (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (QPS) | Memory (MB) |
|---|---|---|---|---|---|---|---|
| **100** | 0.09 ms | **0.032 ms** | 0.026 ms | 0.043 ms | 0.097 ms | **31,352.9** | 0.20 MB |
| **1,000** | 1.04 ms | **0.088 ms** | 0.063 ms | 0.168 ms | 0.393 ms | **11,340.4** | 1.95 MB |
| **10,000** | 5.75 ms | **1.710 ms** | 1.529 ms | 3.842 ms | 4.590 ms | **584.9** | 19.53 MB |
| **50,000** | 35.41 ms | **9.389 ms** | 9.126 ms | 12.968 ms | 14.927 ms | **106.5** | 97.66 MB |
| **100,000** | 72.63 ms | **14.943 ms** | 14.850 ms | 17.665 ms | 18.055 ms | **66.9** | 195.31 MB |

> **Key Takeaway**: Even at **100,000 indexed images**, FAISS CPU searches complete in **~14.9 ms** while consuming less than **200 MB of RAM**, demonstrating seamless scalability from personal photo collections to enterprise galleries.

### INT8 Dynamic Quantization vs FP32 Inference

| Metric | FP32 Standard Inference | INT8 Dynamic Quantization (Turbo Mode) | Speedup / Impact |
|--------|-------------------------|----------------------------------------|------------------|
| **Mean Inference Time** | 186.72 ms | **104.20 ms** | **1.79x faster** |
| **P95 Latency** | 227.18 ms | **148.49 ms** | **1.53x faster** |
| **Cosine Fidelity** | 1.0000 (Reference) | **0.8284** | High structural fidelity |
| **Network Requirement** | None (Local CPU) | None (Local CPU) | Zero cloud dependencies |

### End-to-End Multimodal Pipeline Latencies

| Pipeline Operation | Mean Latency (ms) | P95 Latency (ms) | Primary Computational Phase |
|--------------------|-------------------|------------------|------------------------------|
| **Hybrid Text Search** | 787.3 ms | 964.1 ms | CLIP Text Tokenization + FAISS + RapidOCR Fuzzy Match |
| **Sketch Search** | 341.5 ms | 395.3 ms | Distance Transform + SketchProjection MLP + FAISS |
| **Drag & Drop Image Probe** | 2,357.0 ms | 3,018.8 ms | Full Image Decoding + 512-D CLIP Encoding + Barycentric Manifold Projection |

---

## Academic Research Paper & Presentation Materials

Comprehensive academic, technical, and presentation documentation is included directly in the repository:

- **Academic Research Paper**:
  - Plain Text: [`ZERO_TEXT_RESEARCH_PAPER.txt`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.txt) (1,360 lines)
  - Styled HTML: [`ZERO_TEXT_RESEARCH_PAPER.html`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.html)
  - Formatted Printable PDF: [`ZERO_TEXT_RESEARCH_PAPER.pdf`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.pdf)
- **Master PowerPoint Presentation**:
  - Python Deck Generator: [`generate_master_presentation.py`](file:///e:/Zero%20text%20AIML/generate_master_presentation.py)
  - PowerPoint Deck: [`ZERO_ARCADE_MASTER_FINAL_PRESENTATION.pptx`](file:///e:/Zero%20text%20AIML/ZERO_ARCADE_MASTER_FINAL_PRESENTATION.pptx) (13 formatted slides covering motivation, multi-modal architecture, Shadowbox Arcade, benchmarks, and live demonstrations).
- **Fine-Tuning Report**:
  - Markdown: [`FINE_TUNING_REPORT.md`](file:///e:/Zero%20text%20AIML/FINE_TUNING_REPORT.md)
  - Styled HTML: [`FINE_TUNING_REPORT.html`](file:///e:/Zero%20text%20AIML/FINE_TUNING_REPORT.html)
  - Printable PDF: [`FINE_TUNING_REPORT.pdf`](file:///e:/Zero%20text%20AIML/FINE_TUNING_REPORT.pdf)
- **Benchmarking Report**:
  - JSON Metrics: [`BENCHMARK_REPORT.json`](file:///e:/Zero%20text%20AIML/BENCHMARK_REPORT.json)
  - Printable PDF: [`BENCHMARK_REPORT.pdf`](file:///e:/Zero%20text%20AIML/BENCHMARK_REPORT.pdf)
- **Uncovered Systems & Architectural Extensions**:
  - Text: [`UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.txt`](file:///e:/Zero%20text%20AIML/UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.txt)
  - Printable PDF: [`UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.pdf`](file:///e:/Zero%20text%20AIML/UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.pdf)

---

## Complete REST API Reference

All backend API endpoints are served from `http://127.0.0.1:8000/api/`.

| Endpoint | Method | Category | Description | Parameters / Payload |
|----------|--------|----------|-------------|----------------------|
| `/api/status` | GET | System | System status (image count, cluster count, CLIP readiness, Turbo Mode state). | — |
| `/api/turbo` | POST | System / Performance | Toggle or set Turbo Mode (PyTorch INT8 dynamic quantization). | `enabled` (optional boolean) |
| `/api/browse/folder` | POST | Ingestion / UX | Opens native Windows/OS folder picker dialog via Tkinter topmost window. | — |
| `/api/benchmark` | GET | Performance | Returns latest quantitative benchmark results (FAISS scaling, quantization, latencies). | — |
| `/api/benchmark/pdf` | GET | Performance | Serves and downloads `BENCHMARK_REPORT.pdf`. | — |
| `/api/images` | GET | Gallery | List all indexed images with 2D/3D UMAP coordinates and metadata. | — |
| `/api/image/{filename}` | GET | Gallery | Serves an image file from the library. | `filename` |
| `/api/image/{filename}` | DELETE | Gallery | Permanently deletes an image from disk and drops its embedding from FAISS. | `filename` |
| `/api/search/recommend_weights` | GET | Search | AI-recommended scoring weights based on query intent analysis. | `query` |
| `/api/search/hybrid` | POST | Search | Hybrid multimodal search combining CLIP, OCR text, and dominant colors. | `query`, `clip_weight`, `ocr_weight`, `colour_weight`, `auto_weights` |
| `/api/search/sketch` | POST | Search | Sketch-to-photo retrieval using structural decomposition and frozen projection. | `sketch_data` (base64 PNG), `top_k`, `text_hint` |
| `/api/search/color` | POST | Search | Dominant colour search via hex code or color keyword. | `hex` (e.g. `#FF0000`), `top_k` |
| `/api/search/similar` | POST | Search | Visual similarity search based on an existing photo embedding. | `filename`, `top_k` |
| `/api/search/image_probe` | POST | Search / Manifold | Drag-and-drop probe search: computes 512-D CLIP embedding, projects manifold coordinates in `<1ms`, and retrieves nearest neighbors. | `file` (multipart upload), `top_k` |
| `/api/vocabulary` | GET | Search / Vocab | Dynamic dataset vocabulary tokens for autocomplete discovery (OCR, visual tags, colors). | `limit` (default 250) |
| `/api/feedback` | POST | Feedback | Records user positive/negative search feedback to dataset files. | `search_type`, `feedback`, `filename`, `query` |
| `/api/feedback/stats` | GET | Feedback | Summary counts of collected positive/negative feedback datasets. | — |
| `/api/index/upload` | POST | Indexing | Upload and index new images directly. | `files` (array of base64 data) |
| `/api/index/directory` | POST | Indexing | Recursively index all images in a local directory path. | `directory_path`, `recursive` |
| `/api/index/status` | GET | Indexing | Check background indexing progress and logs. | — |
| `/api/recluster` | POST | Indexing | Re-run 2D/3D UMAP and HDBSCAN clustering over library embeddings. | — |
| `/api/sketches` | GET | Studio | List available sample sketch template files. | — |
| `/api/sketches/{filename}` | GET | Studio | Serves sample sketch template file. | `filename` |
| `/api/arcade/status` | GET | Arcade | Get active arcade subprocess state and verify camera/audio dependencies. | — |
| `/api/arcade/launch` | POST | Arcade | Launch an arcade game (`launcher`, `boxing`, `fruit_ninja`) with hardware lock. | `game`, `player_id` |
| `/api/arcade/stop` | POST | Arcade | Terminate currently running arcade subprocess and release hardware locks. | — |
| `/api/arcade/profiles` | GET | Arcade | Fetch all player profiles, high scores, and Hall of Fame leaderboard. | — |
| `/api/arcade/player/active` | POST | Arcade | Set the active player profile. | `player_id` |

---

## Project Directory Structure

```
Zero text AIML/
│
├── main.py                                  # Primary entry point — launches server & browser
├── server.py                                # FastAPI backend with all search, probe, & arcade routes
├── pipeline.py                              # Core AI engine (CLIP, FAISS, UMAP, HDBSCAN, SketchProjection, INT8)
├── arcade_manager.py                        # Hardware concurrency manager & arcade subprocess lifecycle
├── sketch_search.py                         # Sketch search helper module
├── hybrid_search.py                         # Hybrid search helper module
├── search_images.py                         # Image search CLI utilities
├── cluster_images.py                        # Clustering & manifold projection utilities
├── run_app.bat                              # Windows one‑click launcher
├── requirements.txt                         # Core Python dependencies
├── README.md                                # Comprehensive documentation
│
├── BENCHMARK_REPORT.json                    # Quantitative scaling & quantization benchmarks
├── BENCHMARK_REPORT.pdf                     # Formatted benchmark report PDF
├── FINE_TUNING_REPORT.md                    # Technical report on SketchProjection fine-tuning
├── FINE_TUNING_REPORT.html                  # Styled HTML fine-tuning report
├── FINE_TUNING_REPORT.pdf                   # Formatted printable fine-tuning report PDF
├── UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.txt # Architectural extensions report
├── UNCOVERED_FEATURES_AND_SYSTEMS_REPORT.pdf # Formatted uncovered systems PDF
├── ZERO_TEXT_RESEARCH_PAPER.txt             # Academic research paper documentation
├── ZERO_TEXT_RESEARCH_PAPER.html            # Research paper formatted in HTML
├── ZERO_TEXT_RESEARCH_PAPER.pdf             # Formatted academic research paper PDF
├── generate_master_presentation.py          # Master PowerPoint presentation generator
├── ZERO_ARCADE_MASTER_FINAL_PRESENTATION.pptx # 13-slide master presentation deck
│
├── shadowbox_arcade/                        # Self-contained kinetic AR arcade game engine
│   ├── 02_shadow_boxer_game.py              # Shadow Boxing AR (BlazePose + Kiai voice detection)
│   ├── 03_fruit_ninja_game.py               # Fruit Ninja 2.0 (Dual energy blades + swept capsules)
│   ├── 04_arcade_launcher.py                # Standalone arcade desktop launcher portal
│   ├── player_profiles.py                   # Atomic player profile & score manager
│   ├── player_profiles.json                 # Persistent player records & leaderboard
│   ├── fruit_ninja_high_score.json          # Fruit Ninja high-score persistence
│   ├── pose_landmarker_lite.task            # MediaPipe BlazePose model bundle
│   ├── hand_landmarker.task                 # MediaPipe Hand landmark model bundle
│   └── assets/                              # Sound FX, synthwave music tracks, and glove skins
│
├── scripts/
│   └── train_sketch_projection.py           # Standalone offline training utility for SketchProjection
│
├── scratch/                                 # Verification & experimentation scripts
│   ├── verify_probe_finetuning.py           # Verification of cat/dog probe fine-tuning
│   ├── test_cat_ginger_grass.py             # Cross-domain cat probe disambiguation test
│   └── test_semantic_anchor.py              # Semantic anchor verification suite
│
├── data/
│   ├── sample_images/                       # Gallery images collection
│   ├── sketches/                            # Reference and validated sketches
│   │   └── user_validated/                  # User sketches collected via positive search feedback
│   ├── image_index.faiss                    # FAISS vector index (IndexFlatIP)
│   ├── image_metadata.json                  # Image dimensions, paths, and metadata
│   ├── ocr_results.json                     # Extracted RapidOCR text
│   ├── color_results.json                   # Extracted dominant colour palettes
│   ├── cluster_results.json                 # HDBSCAN cluster assignments + UMAP coords
│   ├── sketch_projection.pt                 # Frozen pretrained SketchProjection weights
│   ├── feedback_dataset.json                # User feedback audit log
│   ├── text_search_dataset.json             # Curated query-to-photo validation pairs
│   └── sketch_dataset.json                  # Curated sketch-to-photo validation pairs
│
├── model_cache/                             # Cached OpenCLIP ViT-B/32 foundation model
│
├── web/                                     # Single-page web application frontend
│   ├── index.html                           # Main frontend HTML with embedded arcade fallback
│   ├── css/
│   │   ├── styles.css                       # OG blue cyberpunk theme & responsive layout
│   │   ├── arcade.css                       # Standalone Shadowbox Arcade stylesheet
│   │   └── tour.css                         # Interactive tour spotlight & presenter styling
│   └── js/
│       ├── app.js                           # Frontend controller, search, tabs, & probe bridge
│       ├── sketch.js                        # Drawing studio canvas, flood fill, undo/redo
│       ├── galaxy3d.js                      # Three.js 3D WebGL galaxy visualization
│       ├── galaxy2d.js                      # Canvas 2D UMAP star map renderer
│       └── tour.js                          # Multi-step presentation tour engine
│
└── tests/
    ├── test_app_endpoints.py                # Integration tests for search & ingestion API
    ├── test_sketch_multicolor.py            # Sketch studio multi-colour & bucket fill tests
    ├── test_feedback_enrichment.py          # Feedback audit & dataset enrichment tests
    ├── test_arcade_manager.py               # Arcade process lifecycle & concurrency tests
    └── test_arcade_endpoints.py             # Arcade FastAPI routes tests
```

---

## Testing, Verification & Offline Utilities

Run these test suites to verify system functionality:

```bash
# 1. Full application endpoint & multi-modal search integration
python tests/test_app_endpoints.py

# 2. Structural & multi-color sketch-to-photo retrieval verification
python tests/test_sketch_multicolor.py

# 3. User feedback logging & dataset curation test suite
python tests/test_feedback_enrichment.py

# 4. Shadowbox Arcade process manager & concurrency locks
python tests/test_arcade_manager.py

# 5. Shadowbox Arcade API endpoints
python tests/test_arcade_endpoints.py

# 6. Verify cat/dog drag & drop probe fine-tuning accuracy
python scratch/verify_probe_finetuning.py

# 7. Standalone offline training utility health check
python scripts/train_sketch_projection.py --help
```

---

## Troubleshooting & FAQ

| Problem | Cause | Solution |
|---------|-------|----------|
| **Arcade missing packages alert** | `pygame`, `mediapipe`, or `sounddevice` not installed | Run `pip install pygame mediapipe sounddevice` in your virtual environment. |
| **Secret arcade tab not opening** | Keystrokes typed while focusing search input | Click anywhere on the background to unfocus inputs, then type **`SAM`** within 2 seconds. |
| **Game fails to access webcam** | Another application is using the camera | Close other apps using the webcam. If a previous game crashed, click **🛑 Emergency Exit** in the status bar to release the hardware lock. |
| **Interactive Tour spotlight misplaced** | Browser window resized during active tour | Press `ArrowRight` or `ArrowLeft` to re-align the spotlight coordinates to the updated viewport. |
| **Image probe failed: Not Found** | Outdated server process running without new probe endpoints | Terminate stale background Python/Uvicorn instances and restart with `python main.py`. |
| **Drag & Drop radar not activating** | Non-image file dragged or dropzone events blocked | Drop any standard image (`JPG`, `PNG`, `WEBP`, `GIF`, `BMP`, `TIFF`, `AVIF`, `ICO`) or click **🛰️ Probe Image** in the header. |
| **Camera Diagnostic Mirror pitch black** | Browser permission blocked or webcam in use | Grant camera and microphone permissions in browser site settings and close background camera apps. |
| **"paging file too small" on Windows** | SafeTensors memory mapping | Handled automatically in code by forcing pread file reading. No manual action needed. |
| **First startup is slow (~30s)** | CLIP model loading into memory | Expected on first boot. Subsequent starts are fast as weights reside in OS cache. |
| **3D Galaxy is blank** | Three.js CDN blocked or offline | Three.js falls back to local simulation; check browser console or network permissions. |
| **Browser caching old CSS/JS** | Aggressive static asset cache | Press **`Ctrl + F5`** (or `Ctrl + Shift + R`) to force refresh. Embedded styles in `index.html` ensure immediate fallback. |

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

**🚀 Welcome to ZERO‑ARCADE!** Explore your visual galaxies, sketch to search, probe high-dimensional manifolds, and step in front of the camera to play.
