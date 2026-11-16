# 🌌 ZERO‑ARCADE — Multimodal Visual Galaxy, AI‑Powered Image Search & Kinetic AR Gaming Suite

> **Search your image library by typing text, picking a colour, uploading a photo, or drawing a sketch — powered by CLIP, FAISS, and a frozen, CPU‑efficient Sketch‑to‑Photo structural projection architecture — fused with the SHADOWBOX ARCADE 60 FPS zero‑controller motion gaming platform.**

---

## Table of Contents

1. [What is ZERO‑ARCADE?](#what-is-zero-arcade)
2. [Key Features at a Glance](#key-features-at-a-glance)
3. [Prerequisites](#prerequisites)
4. [Installation (Step‑by‑Step)](#installation-step-by-step)
5. [Running the Application](#running-the-application)
6. [How to Use the App](#how-to-use-the-app)
   - [Text Search](#1-text-search)
   - [Sketch Search (Draw to Find)](#2-sketch-search-draw-to-find)
   - [Colour Search](#3-colour-search)
   - [Upload / Photo Search](#4-upload--photo-search)
   - [Find Similar Images](#5-find-similar-images)
   - [Adding Images to the Gallery](#6-adding-images-to-the-gallery)
   - [Deleting Images](#7-deleting-images)
   - [Search Result Feedback & Dataset Collection](#8-search-result-feedback--dataset-collection)
   - [3D Visual Galaxy](#9-3d-visual-galaxy)
   - [2D Semantic Star Map](#10-2d-star-map)
   - [Gallery Grid](#11-gallery-grid)
   - [Reclustering](#12-reclustering)
   - [AI Auto‑Weight Decider](#13-ai-auto-weight-decider)
7. [🕹️ Shadowbox Arcade Suite (Secret "SAM" Unlock)](#-shadowbox-arcade-suite-secret-sam-unlock)
   - [Activating the Secret Arcade Portal](#activating-the-secret-arcade-portal)
   - [Included Native Games](#included-native-games)
   - [Hardware Concurrency & Process Lifecycle](#hardware-concurrency--process-lifecycle)
   - [Live Telemetry & Hall of Fame Leaderboard](#live-telemetry--hall-of-fame-leaderboard)
8. [Architecture & Technical Details](#architecture--technical-details)
   - [Structural Sketch‑to‑Photo Pipeline](#structural-sketch-to-photo-pipeline)
   - [Frozen SketchProjection Head](#frozen-sketchprojection-head)
   - [Zero Runtime Training Guarantee](#zero-runtime-training-guarantee)
   - [Offline Training Utility](#offline-training-utility)
   - [Colour‑Aware Sketch Search](#colour-aware-sketch-search)
9. [Academic Research Paper & Master Presentation](#academic-research-paper--master-presentation)
10. [API Endpoints](#api-endpoints)
11. [Project Structure](#project-structure)
12. [Testing & Verification](#testing--verification)
13. [Troubleshooting & Tips](#troubleshooting--tips)
14. [License](#license)

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
2. **SHADOWBOX ARCADE**: An easter-egg gaming suite unlocked by typing `SAM`, allowing you to play high-framerate motion games using only your standard webcam (no specialized VR headsets or handheld controllers required).

All images are visualised in an interactive **3D Visual Galaxy** (WebGL) and a **2D Semantic Star Map** where visually similar images cluster together, powered by UMAP dimensionality reduction and HDBSCAN clustering.

---

## Key Features at a Glance

| Feature | Category | Description |
|---------|----------|-------------|
| 🔍 **Hybrid Text Search** | Visual Search | Combines CLIP semantic understanding + OCR text matching + dominant colour scoring with intelligent auto‑weights. |
| ✏️ **Structural Sketch Search** | Visual Search | Draw strokes or shapes and retrieve matching photos via multi-scale contour + distance-field silhouettes and a frozen projection head (0 startup/runtime training). |
| 🎨 **Multi‑Colour Drawing** | Studio | 12 preset colour swatches + custom colour picker for coloured sketches. |
| 🪣 **Bucket Fill (Flood Fill)** | Studio | One‑click fill tool with colour tolerance — fill regions of your sketch quickly. |
| ↩️ **Undo / Redo** | Studio | Stroke‑level history (up to 35 steps). Keyboard shortcuts: `Ctrl+Z`, `Ctrl+Y`, `Ctrl+Shift+Z`. |
| 📁 **Upload & Search** | Visual Search | Upload a photo directly onto the sketch canvas and search by visual similarity. |
| 🎯 **Find Similar Images** | Visual Search | Click any image → "Find Visually Similar" to locate related photos in the index. |
| 🌈 **Colour‑Based Search** | Visual Search | Search by hex colour code or colour name — finds images whose dominant palette matches. |
| 📝 **OCR Text Extraction** | Visual Search | RapidOCR automatically reads text from every indexed image (code screenshots, handwritten notes, invoices, etc.). |
| 👍 **Feedback & Dataset Curation** | AI Pipeline | Upvote/downvote search results to curate versioned training and evaluation datasets without modifying live model parameters. |
| 🪐 **3D Visual Galaxy** | Visualisation | Interactive WebGL 3D space where each image is a floating point, clustered by visual similarity. Orbit, zoom, spin. |
| 🗺️ **2D Semantic Star Map** | Visualisation | A 2D UMAP projection showing how all images relate to each other spatially. |
| 🗂️ **Gallery Grid** | Visualisation | Traditional card grid with fixed‑size thumbnails (260 px), score badges, cluster labels, and OCR tags. |
| 🕹️ **Arcade Portal** | Kinetic Gaming | Hidden easter-egg portal unlocked via secret `SAM` keystroke sequence. |
| 🥊 **Shadow Boxing AR** | Kinetic Gaming | 33-point BlazePose skeleton tracking, $1\text{€}$ filter stabilization, 6-stage punch state machine, acoustic Kiai battle cries. |
| 🍉 **Fruit Ninja 2.0** | Kinetic Gaming | Dual fist-gripped laser energy swords, swept-capsule anti-tunneling physics, combo multipliers, bomb hazards. |
| 🏆 **Hall of Fame Leaderboard** | Kinetic Gaming | Live per-player statistics, profile selector, ranking medals, and atomic persistence. |
| ⚖️ **AI Auto‑Weight Decider** | Intelligence | Analyses your query intent (visual concept vs. text/OCR vs. colour) and auto‑tunes the scoring weights. |
| 🔄 **Recluster** | Intelligence | One‑click button to re‑run UMAP + HDBSCAN clustering after adding new images. |
| 📂 **Folder Ingestion** | Data Pipeline | Point the app at any local folder to crawl, index, and process every image automatically. |

---

## Prerequisites

Make sure you have the following installed on your system before proceeding:

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **Python** | 3.11+ | Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**. |
| **pip** | (bundled with Python) | Used to install Python dependencies. |
| **Git** | Any recent version | Optional — only needed if cloning the repository. |
| **Web Browser** | Chrome, Edge, or Firefox | Chrome recommended for best WebGL 3D performance. |
| **Webcam & Microphone** | Standard USB / Built-in | Required for Shadowbox Arcade computer vision and audio Kiai detection. |
| **~2.5 GB free disk space** | — | CLIP model (~600 MB) + MediaPipe task models (~13 MB) + assets and images. |
| **~4 GB RAM** | — | Core search pipeline runs in-memory on CPU. |

> **No GPU required.** The entire system runs on CPU. A dedicated graphics card is optional.

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

To enable the native camera and audio drivers for **Shadowbox Arcade**:
```bash
pip install pygame mediapipe sounddevice
```

#### Complete Package Reference

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch deep learning framework (CPU version). Powers CLIP and SketchProjection. |
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

### 1. Text Search
Type any natural language query into the search bar and press Enter:
- `sunset` — finds images with sunset visual content.
- `golden retriever` — finds dog photos.
- `coding` — finds screenshots of code, notes, IDEs.
- `blue ocean waves` — finds ocean/sea images with blue tones.
- `red car` — combines semantic concept ("car") + colour ("red") matching.
- `import numpy` — finds images containing that exact OCR text.

### 2. Sketch Search (Draw to Find)
1. Click the **✏️ Sketch Search** button in the header.
2. Draw your target on the canvas:
   - **🖊️ Pen** — free-draw strokes.
   - **🪣 Bucket Fill** — flood-fill bounded regions.
   - **🧹 Eraser** — erase parts of the drawing.
   - **↩️ Undo / ↪️ Redo** — stroke history (`Ctrl+Z`, `Ctrl+Y`).
3. Pick colours from the 12 swatches or custom colour picker.
4. Optionally provide an inline **text hint** (e.g. "car", "cat") to boost precision.
5. Click **🚀 Search Galaxy by Sketch**.

### 3. Colour Search
Search by colour keywords (e.g. `red`, `blue`, `golden`, `teal`, `neon`). The engine extracts dominant palettes and applies colour histogram matching.

### 4. Upload / Photo Search
Click **📁 Upload Image** inside the Sketch Studio to load an existing image onto the canvas and search directly using CLIP photographic embeddings.

### 5. Find Similar Images
Click any image card or galaxy node to open the **Image Inspector** drawer, then click **🔍 Find Visually Similar**.

### 6. Adding Images to the Gallery
Click **➕ Add Images / Folder**:
- **Upload Files**: Drag & drop image files (JPG, PNG, WEBP) directly into the dropzone.
- **Index Local Directory**: Specify an absolute directory path to recursively ingest folders.

### 7. Deleting Images
Open the Image Inspector and click **🗑️ Delete Image**. The system cleans up the image file from disk, drops its embedding from FAISS, and re-indexes metadata.

### 8. Search Result Feedback & Dataset Collection
Every search card features positive (👍) and negative (👎) feedback buttons:
- **Positive Feedback**: Confirms query-target pairs and saves validation samples to `data/text_search_dataset.json` and `data/sketch_dataset.json`.
- **Negative Feedback**: Records false positives to `data/feedback_dataset.json` for contrastive evaluation.
- **Zero Runtime Weight Mutation**: User feedback never mutates live weights during inference, preventing model drift.

### 9. 3D Visual Galaxy
Click the **🪐 3D Galaxy** tab:
- **Click & drag**: Orbit camera in 3D WebGL space.
- **Scroll wheel**: Zoom in and out.
- **Hover point**: Instant tooltip preview.
- **Click point**: Opens detailed Image Inspector.
- **Orbit Spin**: Toggles celestial auto-rotation.
- **Constellations**: Toggles cluster connection lines.

### 10. 2D Semantic Star Map
Click the **🗺️ 2D Star Map** tab for a continuous 2D planar projection of cluster groupings with pan, zoom, and fit-to-screen controls.

### 11. Gallery Grid
Click the **🗂️ Gallery Grid** tab for a structured card grid displaying fixed-size thumbnails, match percentage badges, cluster chips, OCR text snippets, and dominant palette strips.

### 12. Reclustering
Click **🔄 Recluster** at any time to re-run 2D/3D UMAP projections and HDBSCAN density clustering over newly added images.

### 13. AI Auto‑Weight Decider
The decider analyzes query intent and automatically balances weights:
- **Visual concept** ("mountain landscape"): 75% CLIP, 15% OCR, 10% Colour.
- **Code / document** ("python code"): 25% CLIP, 65% OCR, 10% Colour.
- **Colour-focused** ("bright red"): 20% CLIP, 5% OCR, 75% Colour.
- **Balanced** ("blue machine learning"): 50% CLIP, 30% OCR, 20% Colour.

---

## 🕹️ Shadowbox Arcade Suite (Secret "SAM" Unlock)

ZERO-ARCADE integrates a complete motion-controlled AR arcade suite directly into the application, executed natively as hardware-accelerated desktop subprocesses.

```
+-------------------------------------------------------------------------+
|                          ZERO-ARCADE BROWSER UI                         |
|   [ 3D Galaxy ]  [ 2D Star Map ]  [ Gallery Grid ]  [ 🕹️ Arcade (SAM) ]  |
+-------------------------------------------------------------------------+
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
3. The retro-cyberpunk **SHADOWBOX ARCADE UNLOCKED** toast banner will slide down, and the **🕹️ Arcade** tab will become visible in the navigation header.

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
- **Swept-Capsule Continuous Physics**: Prevents tunneling artifacts on fast-moving fruit slices.
- **Dynamic Combo Multipliers**: Slicing multiple fruits in a single blade stroke triggers cascading score multipliers.
- **Bomb Hazards & Extra-Life Heart Fruits**: Strategic hazards and health recoveries.

### Hardware Concurrency & Process Lifecycle
- Managed by `arcade_manager.py` using thread-safe synchronization locks.
- Ensures only **one** game process owns the webcam and audio hardware at any given moment.
- Web UI provides live process telemetry (`PID`, status dot, active player) and an emergency stop button (🛑 Emergency Exit) to terminate background game processes safely.

### Live Telemetry & Hall of Fame Leaderboard
- **Synchronized State**: The web UI automatically reads and synchronizes `player_profiles.json` and high scores.
- **Leaderboard Table**: Displays player ranks, medal badges (`🥇 1st`, `🥈 2nd`, `🥉 3rd`), cumulative high scores, punch counts, and fruit slice records.
- **Active Player Selector**: Switch the active player directly from the web interface before launching any game.

---

## Architecture & Technical Details

```
                                  [ User Input ]
                   (Text Query, Sketch Drawing, Image Upload)
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [ Text / OCR Query ]                         [ Canvas Sketch / Drawing ]
                │                                             │
      CLIP Text ViT-B/32                            Multi-Scale Structural Views:
                │                                    - Bounding-Box Centering
                │                                    - Normalized Stroke Contour
                │                                    - Distance-Field Silhouette
                │                                    - Horizontal Mirror View
                │                                    - Chromatic Pigment View
                │                                             │
                │                                   CLIP Visual ViT-B/32
                │                                             │
                │                                   [ 512-D Structural Embed ]
                │                                             │
                │                                   Frozen SketchProjection Head
                │                                   (Residual Linear 512->768->512)
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
                       [ Query Vector in 512-D Space ]
                                       │
                      FAISS Inner-Product Cosine Search
                                       │
                       Combined Multi-Modal Scoring:
                    Final = w1*CLIP + w2*OCR + w3*Colour
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
      [ Ranked Gallery Cards ]                     [ 3D / 2D Galaxy Viewports ]
```

### Structural Sketch‑to‑Photo Pipeline

To overcome the domain gap between sparse line drawings and dense natural photographs without fragile online training, ZERO-ARCADE employs a **multi-scale structural decomposition**:

1. **Foreground Bounding-Box Detection & Centering**: Crops empty canvas margins and rescales strokes to canonical $224 \times 224$ dimensions while preserving stroke aspect ratios.
2. **Normalized Stroke Contour**: Inverts lines and applies adaptive morphological dilation (`cv2.dilate`) to normalize variable line widths.
3. **Soft Distance-Field Silhouette**: Computes Euclidean distance transforms (`cv2.distanceTransform`) to encode 2D spatial mass, boundary topology, and volume.
4. **Horizontal Mirror Invariance**: Evaluates mirrored views to guarantee orientation-invariant matching.
5. **Chromatic View**: Extracts stroke pigments and bucket fills via K-Means clustering, blending colour tokens into query prompts.
6. **Code & Screenshot Suppression**: High-frequency text suppression filters out false-positive code screenshots.

### Frozen SketchProjection Head

The structural sketch representation is projected into the photographic embedding space via a frozen pretrained **`SketchProjection`** head:

- **Dimensions**: $512\text{D} \to 512\text{D}$
- **Architecture**:
  $$\text{LayerNorm}(512) \to \text{Linear}(512 \to 768) \to \text{GELU} \to \text{Linear}(768 \to 512)$$
- **Residual Connection**:
  $$\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{in}} + \alpha \cdot \Delta \mathbf{z} \quad (\alpha \approx 0.8)$$
- **L2 Normalization**: Output vectors are unit-normalized for exact cosine ranking in FAISS.
- **Checkpoint**: Loaded directly from static weights file [`data/sketch_projection.pt`](file:///e:/Zero%20text%20AIML/data/sketch_projection.pt).

### Zero Runtime Training Guarantee
- **Instant Startup**: Cold boots load model weights in **~1.5 seconds**.
- **Deterministic Search**: Zero runtime gradient updates; live model weights never mutate during application execution.
- **Decoupled User Feedback**: Feedback is recorded purely as training datasets (`data/feedback_dataset.json`, `data/sketch_dataset.json`).

### Offline Training Utility
To refine projection weights independently of the production app:
```bash
python scripts/train_sketch_projection.py --epochs 60 --lr 0.0003 --temperature 0.07
```
- **InfoNCE Contrastive Loss**: Learns alignment between synthetic multi-view sketch transforms and photo gallery embeddings.
- **Evaluation Metrics**: Logs Precision@1 ($P@1$), Recall@3 ($R@3$), and Mean Reciprocal Rank ($\text{MRR}$).

---

## Academic Research Paper & Master Presentation

Comprehensive academic and technical materials are included in the repository:

- **Research Paper**:
  - Plain Text: [`ZERO_TEXT_RESEARCH_PAPER.txt`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.txt) (1,360 lines)
  - Styled HTML: [`ZERO_TEXT_RESEARCH_PAPER.html`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.html)
  - Printable PDF: [`ZERO_TEXT_RESEARCH_PAPER.pdf`](file:///e:/Zero%20text%20AIML/ZERO_TEXT_RESEARCH_PAPER.pdf)
- **Master Presentation**:
  - Python Deck Generator: [`generate_master_presentation.py`](file:///e:/Zero%20text%20AIML/generate_master_presentation.py)
  - PowerPoint Deck: [`ZERO_ARCADE_MASTER_FINAL_PRESENTATION.pptx`](file:///e:/Zero%20text%20AIML/ZERO_ARCADE_MASTER_FINAL_PRESENTATION.pptx) (13 formatted slides covering motivation, multi-modal architecture, Shadowbox Arcade, benchmarks, and live demonstrations).

---

## API Endpoints

All endpoints are served from `http://127.0.0.1:8000/api/`.

| Endpoint | Method | Category | Description | Key Parameters |
|----------|--------|----------|-------------|----------------|
| `/api/status` | GET | System | System status (images, clusters, CLIP readiness). | — |
| `/api/images` | GET | Gallery | List all indexed images with 2D/3D coordinates & metadata. | — |
| `/api/image/{filename}` | GET | Gallery | Serve an image file. | `filename` |
| `/api/image/{filename}` | DELETE | Gallery | Permanently delete image from disk and FAISS. | `filename` |
| `/api/search/hybrid` | POST | Search | Hybrid text + OCR + colour search. | `query`, `clip_weight`, `ocr_weight`, `colour_weight`, `auto_weights` |
| `/api/search/sketch` | POST | Search | Sketch-to-photo search with structural matching. | `sketch_data` (base64), `top_k`, `text_hint` |
| `/api/search/color` | POST | Search | Dominant colour search. | `hex` (e.g. "#FF0000"), `top_k` |
| `/api/search/similar` | POST | Search | Visual similarity search from photo embedding. | `filename`, `top_k` |
| `/api/search/recommend_weights` | GET | Search | AI-recommended scoring weights for query. | `query` |
| `/api/feedback` | POST | Feedback | Record user feedback (positive/negative). | `search_type`, `feedback`, `filename`, `query` |
| `/api/feedback/stats` | GET | Feedback | Summary statistics of collected feedback datasets. | — |
| `/api/index/upload` | POST | Indexing | Upload and index new images. | `files` (array of base64 data) |
| `/api/index/directory` | POST | Indexing | Recursively index all images in a local directory. | `directory_path`, `recursive` |
| `/api/index/status` | GET | Indexing | Check background indexing progress. | — |
| `/api/recluster` | POST | Indexing | Re-run UMAP (2D/3D) and HDBSCAN clustering. | — |
| `/api/sketches` | GET | Studio | List available sample sketch files. | — |
| `/api/sketches/{filename}` | GET | Studio | Serve sample sketch file. | `filename` |
| `/api/arcade/status` | GET | Arcade | Get active arcade game status and missing dependencies. | — |
| `/api/arcade/launch` | POST | Arcade | Launch an arcade game (`launcher`, `boxing`, `fruit_ninja`). | `game`, `player_id` |
| `/api/arcade/stop` | POST | Arcade | Terminate currently running arcade subprocess. | — |
| `/api/arcade/profiles` | GET | Arcade | Fetch all player profiles and Hall of Fame leaderboard. | — |
| `/api/arcade/player/active` | POST | Arcade | Set the active player profile. | `player_id` |

---

## Project Structure

```
Zero text AIML/
│
├── main.py                                  # Entry point — starts server, opens browser
├── server.py                                # FastAPI backend with all search & arcade routes
├── pipeline.py                              # Core AI engine (CLIP, FAISS, UMAP, HDBSCAN, SketchProjection, OCR)
├── arcade_manager.py                        # Process lifecycle & hardware concurrency manager
├── sketch_search.py                         # Sketch search helper module
├── hybrid_search.py                         # Hybrid search helper module
├── search_images.py                         # Image search CLI utilities
├── cluster_images.py                        # Clustering utilities
├── run_app.bat                              # Windows one‑click launcher
├── requirements.txt                         # Core Python dependencies
├── README.md                                # This documentation file
├── ZERO_TEXT_RESEARCH_PAPER.txt             # Academic research paper documentation
├── ZERO_TEXT_RESEARCH_PAPER.html            # Research paper formatted in HTML
├── ZERO_TEXT_RESEARCH_PAPER.pdf             # Printable PDF research paper
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
├── data/
│   ├── sample_images/                       # Gallery images collection
│   ├── sketches/                            # Reference and validated sketches
│   │   └── user_validated/                  # User sketches collected via positive search feedback
│   ├── image_index.faiss                    # FAISS vector index
│   ├── image_metadata.json                  # Image dimensions, paths, and metadata
│   ├── ocr_results.json                     # Extracted OCR text
│   ├── color_results.json                   # Dominant colour palettes
│   ├── cluster_results.json                 # Cluster assignments + UMAP coords
│   ├── sketch_projection.pt                 # Frozen pretrained SketchProjection weights
│   ├── feedback_dataset.json                # User feedback audit log
│   ├── text_search_dataset.json             # Curated query-to-photo validation pairs
│   └── sketch_dataset.json                  # Curated sketch-to-photo validation pairs
│
├── model_cache/                             # Cached CLIP ViT-B/32 foundation model
│
├── web/                                     # Single-page web application
│   ├── index.html                           # Main frontend HTML with embedded arcade fallback
│   ├── css/
│   │   ├── styles.css                       # OG blue cyberpunk theme & responsive layout
│   │   └── arcade.css                       # Standalone Shadowbox Arcade stylesheet
│   └── js/
│       ├── app.js                           # Frontend controller, search, tabs, & arcade bridge
│       ├── sketch.js                        # Drawing studio canvas, flood fill, undo/redo
│       ├── galaxy3d.js                      # Three.js 3D WebGL galaxy visualization
│       └── galaxy2d.js                      # Canvas 2D UMAP star map renderer
│
└── tests/
    ├── test_app_endpoints.py                # Integration tests for search & ingestion API
    ├── test_sketch_multicolor.py            # Sketch studio multi-colour & bucket fill tests
    ├── test_feedback_enrichment.py          # Feedback audit & dataset enrichment tests
    ├── test_arcade_manager.py               # Arcade process lifecycle & concurrency tests
    └── test_arcade_endpoints.py             # Arcade FastAPI routes tests
```

---

## Testing & Verification

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

# 6. Standalone offline training utility health check
python scripts/train_sketch_projection.py --help
```

---

## Troubleshooting & Tips

| Problem | Cause | Solution |
|---------|-------|----------|
| **Arcade missing packages alert** | `pygame`, `mediapipe`, or `sounddevice` not installed | Run `pip install pygame mediapipe sounddevice` in your virtual environment. |
| **Secret arcade tab not opening** | Keystrokes typed while focusing search input | Click anywhere on the background to unfocus inputs, then type **`SAM`** within 2 seconds. |
| **Game fails to access webcam** | Another application is using the camera | Close other apps using the webcam. If a previous game crashed, click **🛑 Emergency Exit** in the status bar to release the hardware lock. |
| **"paging file too small" on Windows** | SafeTensors memory mapping | Handled automatically in code by forcing pread file reading. No manual action needed. |
| **First startup is slow (~30s)** | CLIP model loading into memory | Expected on first boot. Subsequent starts are fast as weights reside in OS cache. |
| **3D Galaxy is blank** | Three.js CDN blocked or offline | Three.js falls back to local simulation; check browser console or network permissions. |
| **Arcade CSS styling not updating** | Browser aggressively caching static CSS | Press **`Ctrl + F5`** (or `Ctrl + Shift + R`) to force refresh. Embedded styles in `index.html` ensure immediate fallback. |

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

**🚀 Welcome to ZERO‑ARCADE!** Explore your visual galaxies, sketch to search, and step in front of the camera to play.
