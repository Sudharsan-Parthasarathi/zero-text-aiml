# 🌌 ZERO‑TEXT — Multimodal Visual Galaxy & AI‑Powered Image Search

> **Search your image library by typing text, picking a colour, uploading a photo, or drawing a sketch — all powered by CLIP, FAISS, and a frozen, CPU‑efficient Sketch‑to‑Photo structural projection architecture.**

---

## Table of Contents

1. [What is ZERO‑TEXT?](#what-is-zero-text)
2. [Key Features at a Glance](#key-features-at-a-glance)
3. [Prerequisites](#prerequisites)
4. [Installation (Step‑by‑Step)](#installation-step-by-step)
5. [Running the Application](#running-the-application)
6. [How to Use the App](#how-to-use-the-app)
   - [Text Search](#1-text-search)
   - [Sketch Search](#2-sketch-search-draw-to-find)
   - [Colour Search](#3-colour-search)
   - [Upload / Photo Search](#4-upload--photo-search)
   - [Find Similar Images](#5-find-similar-images)
   - [Adding Images to the Gallery](#6-adding-images-to-the-gallery)
   - [Deleting Images](#7-deleting-images)
   - [Search Result Feedback & Dataset Collection](#8-search-result-feedback--dataset-collection)
   - [3D Visual Galaxy](#9-3d-visual-galaxy)
   - [2D Star Map](#10-2d-star-map)
   - [Gallery Grid](#11-gallery-grid)
   - [Reclustering](#12-reclustering)
   - [AI Auto‑Weight Decider](#13-ai-auto-weight-decider)
7. [Architecture & Technical Details](#architecture--technical-details)
   - [Structural Sketch‑to‑Photo Pipeline](#structural-sketch-to-photo-pipeline)
   - [Frozen SketchProjection Head](#frozen-sketchprojection-head)
   - [Zero Runtime Training Guarantee](#zero-runtime-training-guarantee)
   - [Offline Training Utility](#offline-training-utility)
   - [Colour‑Aware Sketch Search](#colour-aware-sketch-search)
8. [API Endpoints](#api-endpoints)
9. [Project Structure](#project-structure)
10. [Testing & Verification](#testing--verification)
11. [Troubleshooting & Tips](#troubleshooting--tips)
12. [License](#license)

---

## What is ZERO‑TEXT?

ZERO‑TEXT is a **fully local, self‑hosted** multimodal image search engine. It lets you find images using any combination of:

- **Natural language** — type "sunset over the ocean" and see matching photos ranked by semantic meaning.
- **In‑image OCR text** — type "import numpy" and find screenshots or notes containing that code.
- **Dominant colour** — type "red" or click a colour swatch to find images with that palette.
- **Hand‑drawn sketch** — open the sketch canvas, draw a rough shape (with multiple colours, bucket fill, undo/redo), and the AI finds photos that match your drawing.
- **Uploaded photo** — upload a reference image and find visually similar ones.

All images are organised into an interactive **3D Visual Galaxy** (WebGL) and a **2D Semantic Star Map** where visually similar images cluster together, powered by UMAP dimensionality reduction and HDBSCAN clustering.

---

## Key Features at a Glance

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Text Search** | Combines CLIP semantic understanding + OCR text matching + dominant colour scoring with intelligent auto‑weights. |
| ✏️ **Structural Sketch Search** | Draw strokes or shapes and retrieve matching photos via multi-scale contour + distance-field silhouettes and a frozen projection head (0 startup/runtime training). |
| 🎨 **Multi‑Colour Drawing** | 12 preset colour swatches + a custom colour picker for coloured sketches. |
| 🪣 **Bucket Fill (Flood Fill)** | One‑click fill tool with colour tolerance — fill regions of your sketch quickly. |
| ↩️ **Undo / Redo** | Stroke‑level history (up to 35 steps). Keyboard shortcuts: `Ctrl+Z`, `Ctrl+Y`, `Ctrl+Shift+Z`. |
| 📁 **Upload & Search** | Upload a photo directly onto the sketch canvas and search by visual similarity. |
| 🎯 **Find Similar Images** | Click any image → "Find Visually Similar" to locate related photos in the index. |
| 🌈 **Colour‑Based Search** | Search by hex colour code or colour name — finds images whose dominant palette matches. |
| 📝 **OCR Text Extraction** | RapidOCR automatically reads text from every indexed image (code screenshots, handwritten notes, invoices, etc.). |
| 👍 **Feedback & Dataset Curation** | Upvote/downvote search results to curate versioned training and evaluation datasets without modifying live model parameters. |
| 🪐 **3D Visual Galaxy** | Interactive WebGL 3D space where each image is a floating point, clustered by visual similarity. Orbit, zoom, spin. |
| 🗺️ **2D Semantic Star Map** | A 2D UMAP projection showing how all images relate to each other spatially. |
| 🗂️ **Gallery Grid** | Traditional card grid with fixed‑size thumbnails (260 px), score badges, cluster labels, and OCR tags. |
| 🗑️ **Safe Image Deletion** | Delete images with a confirmation modal — removes from FAISS index, disk, and all metadata. |
| ⚖️ **AI Auto‑Weight Decider** | Analyses your query intent (visual concept vs. text/OCR vs. colour) and auto‑tunes the scoring weights. |
| 🔄 **Recluster** | One‑click button to re‑run UMAP + HDBSCAN clustering after adding new images. |
| 📂 **Folder Ingestion** | Point the app at any local folder and it will crawl, index, and process every image automatically. |

---

## Prerequisites

Make sure you have the following installed on your system before proceeding:

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| **Python** | 3.11+ | Download from [python.org](https://www.python.org/downloads/). During install, check **"Add Python to PATH"**. |
| **pip** | (bundled with Python) | Comes with Python. Used to install packages. |
| **Git** | Any recent version | Optional — only needed if cloning the repo. |
| **Web Browser** | Chrome, Edge, or Firefox | For the interactive UI. Chrome recommended for best WebGL 3D performance. |
| **~2 GB free disk space** | — | The CLIP model file is ~600 MB. Rest is for images and data. |
| **~4 GB RAM** | — | CLIP + FAISS + UMAP run in‑memory on CPU. |

> **No GPU required.** The entire system runs on CPU. A GPU is not needed and not used.

---

## Installation (Step‑by‑Step)

### 1. Get the project files

If you have Git:
```bash
git clone <repository-url>
cd "Zero text AIML"
```
Or simply download and extract the ZIP into a folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
- **macOS / Linux:** `source .venv/bin/activate`

### 3. Install all Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything the app needs:

| Package | What it does |
|---------|-------------|
| `torch` | PyTorch deep learning framework (CPU version). Powers CLIP and the SketchAdapter. |
| `open-clip-torch` | OpenCLIP library — loads the `ViT‑B‑32` vision‑language model. |
| `faiss-cpu` | Facebook AI Similarity Search — ultra‑fast vector nearest‑neighbour index. |
| `umap-learn` | UMAP dimensionality reduction — projects 512‑D vectors into 2D/3D for visualisation. |
| `hdbscan` | Hierarchical density‑based clustering — groups similar images into clusters. |
| `fastapi` | Modern async web framework powering the API server. |
| `uvicorn` | ASGI server to run FastAPI. |
| `Pillow` | Image loading, resizing, colour extraction. |
| `opencv-python` | Computer vision — Canny edge detection, thresholding, morphology for sketch preprocessing. |
| `rapidocr-onnxruntime` | On‑device OCR engine — reads text from images (no cloud API needed). |
| `numpy` | Numerical operations for embeddings and scoring. |
| `scikit-learn` | Used internally by UMAP/HDBSCAN. |

### 4. First run — model download

The first time you start the app, it will automatically download the **CLIP ViT‑B‑32** model (~600 MB) from Hugging Face and cache it in the `model_cache/` folder. After the first download, the app works **fully offline**.

---

## Running the Application

### Option A: Quick start (recommended)

**Double‑click** `run_app.bat` in the project folder. It activates the virtual environment and launches the server automatically.

### Option B: Command line

```bash
python main.py
```

This starts the FastAPI server on `http://127.0.0.1:8000` and opens your browser automatically.

### Command‑line options

```bash
python main.py                     # Default: start server + open browser
python main.py --port 8080         # Use a different port
python main.py --no-browser        # Start server without auto‑opening browser
python main.py --legacy-gui        # Open classic Tkinter desktop GUI
```

### Option C: Direct uvicorn (for development)

```bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

---

## How to Use the App

### 1. Text Search

Type any natural language query into the search bar and press Enter or click 🔍.

**Examples:**
- `sunset` — finds images with sunset‑like visual content.
- `golden retriever` — finds dog photos.
- `coding` — finds screenshots of code, programming‑related images.
- `blue ocean waves` — finds ocean/sea images with blue tones.
- `red car` — combines semantic ("car") + colour ("red") matching.
- `import numpy` — finds images containing that exact OCR text.

The **AI Auto‑Weight Decider** automatically detects whether your query is a visual concept, a text/OCR query, or a colour query, and tunes the scoring accordingly.

### 2. Sketch Search (Draw to Find)

1. Click the **✏️ Sketch Search** button in the header.
2. A drawing canvas opens. Use the tools:
   - **🖊️ Pen** — free‑draw strokes.
   - **🪣 Bucket Fill** — tap to flood‑fill a region with the selected colour.
   - **🧹 Eraser** — erase parts of your drawing.
   - **↩️ Undo / ↪️ Redo** — undo/redo strokes (or press `Ctrl+Z` / `Ctrl+Y`).
   - **🗑️ Clear** — wipe the canvas.
3. Pick a **colour** from the 12 swatches or click 🎨 for a custom colour.
4. Choose a **brush size** (Fine, Medium, Bold).
5. Optionally type a **text hint** (e.g., "car", "cat") for better accuracy.
6. Click **🚀 Search Galaxy by Sketch**.

The AI processes your sketch through:
- **Foreground Bounding-Box Detection & Centering**: Crops whitespace and centers the drawing with aspect-ratio preservation.
- **Normalized Stroke Contour**: Inverts lines and applies adaptive morphological dilation to capture delicate line work.
- **Soft Distance-Field Silhouette**: Computes Euclidean distance transforms to capture 2D spatial mass, boundary topology, and volume.
- **Horizontal Mirroring**: Evaluates mirrored views for left/right orientation invariance.
- **Chromatic View**: Extracts bucket fills and colored strokes when non-grayscale colors are detected.
- **Frozen CLIP ViT-B/32 Visual Encoder**: Fast, zero-training CPU inference.
- **Frozen `SketchProjection` Head**: Lightweight projection head loaded directly from checkpoint (`data/sketch_projection.pt`).
- **Code & Screenshot Suppression**: Suppresses high-frequency text/code documents from false-positive sketch matches.
- **FAISS Cosine Similarity Search**: Fast nearest-neighbour retrieval.

**Tips for better sketch results:**
- Draw the overall **shape/silhouette** of the object.
- Use **colours** that match the target (red for a red car, yellow for a sunflower).
- Use **bucket fill** to colour large regions.
- Add an optional **text hint** for extra accuracy.

### 3. Colour Search

Type a colour name in the search bar (e.g., `red`, `blue`, `golden`, `teal`). The system detects the colour keyword and automatically boosts colour‑based matching.

Supported colour names: red, crimson, scarlet, green, emerald, forest, blue, navy, cyan, teal, yellow, amber, gold, golden, orange, purple, violet, magenta, pink, black, dark, white, light, grey/gray, silver, brown, tan, lime, coral, indigo, neon.

You can also combine colour + concept: `red sports car`, `blue ocean`, `golden sunflower`.

### 4. Upload / Photo Search

1. Open the **Sketch Studio** (✏️ button).
2. Click the **📁 Upload Image** button on the toolbar.
3. Select a photo from your computer.
4. The photo loads onto the canvas. If you don't draw over it, the system searches using the **original photo's full CLIP accuracy** (no sketch processing applied).
5. Click **🚀 Search Galaxy by Sketch** to find similar images.

### 5. Find Similar Images

1. Click on any image in the Gallery Grid, 3D Galaxy, or 2D Star Map.
2. The **Image Inspector** drawer slides open on the right.
3. Click **🔍 Find Visually Similar**.
4. The system finds and ranks the most visually similar images using CLIP embedding cosine similarity.

### 6. Adding Images to the Gallery

Click the **➕ Add Images / Folder** button. Two options:

**Upload Files:**
- Drag & drop image files (JPG, PNG, WEBP) onto the dropzone, or click to browse.
- Images are copied to `data/sample_images/` and automatically indexed (CLIP embedding, OCR, colour extraction, clustering).

**Index Local Directory:**
- Enter an absolute folder path (e.g., `C:\Users\YourName\Pictures`).
- The system recursively scans the folder and indexes every image found.
- A live progress bar shows indexing status.

### 7. Deleting Images

1. Open the Image Inspector by clicking any image.
2. Scroll down and click **🗑️ Delete Image**.
3. A confirmation modal appears with the image thumbnail and filename.
4. Click **Confirm Delete** to permanently remove the image from disk, FAISS index, and all metadata.
5. Click **Cancel** to abort.

### 8. Search Result Feedback & Dataset Collection

Every search result card and inspector panel includes quick feedback buttons (👍 / 👎):
- **Positive Feedback (👍)**: Confirms that the retrieved photo correctly matches the search query or sketch. The query-target pair is saved to `data/text_search_dataset.json` or `data/sketch_dataset.json` (along with user sketch canvas PNGs saved in `data/sketches/user_validated/`).
- **Negative Feedback (👎)**: Logs false-positive retrievals for auditing and negative-pair contrastive training.
- **Audit Log**: Every feedback event is logged with timestamps, scores, clusters, and session IDs to `data/feedback_dataset.json`.
- **Zero Runtime Weight Modification**: Unlike fragile online feedback learning loops, feedback **never** mutates live weights or FAISS embeddings during execution. Data collection is purely additive and feeds into the offline training script [`scripts/train_sketch_projection.py`](file:///e:/Zero%20text%20AIML/scripts/train_sketch_projection.py).

### 9. 3D Visual Galaxy

Click the **🪐 3D Galaxy** tab. Every indexed image appears as a glowing point in 3D space. Images that are visually similar cluster together.

Controls:
- **Click & drag** — orbit the camera.
- **Scroll wheel** — zoom in/out.
- **Hover over a point** — see the image thumbnail tooltip.
- **Click a point** — open the Image Inspector.
- **🎯 Reset** — reset the camera position.
- **💫 Orbit Spin** — toggle auto‑rotation.
- **🕸️ Constellations** — toggle cluster connection lines.

When you search, matching images are highlighted and a **Match Navigator HUD** lets you fly between results.

### 10. 2D Star Map

Click the **🗺️ 2D Star Map** tab. A 2D UMAP projection of all images. Similar images appear close together.

Controls:
- **Click & drag** — pan the map.
- **Scroll wheel** — zoom.
- **+/−** buttons — zoom in/out.
- **⛶** — fit all points to screen.
- **Hover** — preview tooltip.
- **Click** — open Inspector.

### 11. Gallery Grid

Click the **🗂️ Gallery Grid** tab. A traditional card layout showing:
- Image thumbnail (fixed 260 px height).
- Match score badge (percentage).
- Cluster label (colour‑coded).
- OCR tag (if text was detected in the image).
- Filename and extracted text preview.
- Colour palette bar at the bottom of each card.

### 12. Reclustering

After adding or removing images, click the **🔄 Recluster** button. This re‑runs:
- **UMAP 2D** projection (n_neighbours=30, min_dist=0.1)
- **UMAP 3D** projection
- **HDBSCAN** density‑based clustering

The 3D Galaxy and 2D Star Map update with the new cluster assignments.

### 13. AI Auto‑Weight Decider

The search system blends three scoring signals:

| Signal | What it measures |
|--------|-----------------|
| **CLIP Semantic** | Visual meaning similarity (computed by the CLIP ViT‑B‑32 model). |
| **OCR Text** | Fuzzy text matching against OCR‑extracted text in each image. |
| **Dominant Colour** | How closely the image's palette matches colour keywords in your query. |

The **AI Auto‑Weight Decider** analyses your query and automatically decides the blend:

| Query Type | Example | Weights (CLIP / OCR / Colour) |
|-----------|---------|-------------------------------|
| Visual concept | "mountain landscape" | 75% / 15% / 10% |
| Code / document | "python code" | 25% / 65% / 10% |
| Colour‑focused | "bright red" | 20% / 5% / 75% |
| Balanced | "blue machine learning" | 50% / 30% / 20% |

You can also manually adjust weights by clicking **⚖️ Weights** or selecting a preset pill (Semantic, Text/OCR, Colour Focus, Balanced).

---

## Architecture & Technical Details

### Structural Sketch‑to‑Photo Pipeline

Rather than relying on fragile online training loops or unstable pixel-space heuristics, ZERO‑TEXT uses a **multi-scale structural feature representation** that bridges the sketch→photo domain gap directly on CPU:

1. **Foreground Bounding-Box Detection & Centering**: Automatically detects drawing stroke boundaries, removes empty canvas margins, and rescales the drawing to a canonical $224 \times 224$ canvas while preserving stroke aspect ratios.
2. **Normalized Stroke Contour**: Inverts dark strokes on bright canvases and applies adaptive morphological dilation (`cv2.dilate`) to compensate for ultra-thin stylus and pencil strokes.
3. **Soft Distance-Field Silhouette**: Computes Euclidean distance transforms (`cv2.distanceTransform`) to encode continuous 2D mass, boundary topology, and volume.
4. **Horizontal Flip Invariance**: Synthesizes mirrored views to ensure robust retrieval regardless of drawing orientation (e.g. car facing left vs. right).
5. **Color-Enhanced View**: Preserves stroke pigments and bucket fills when chromatic content is detected, extracting dominant colors via K-Means and blending text-prompt color tokens.
6. **Code & Screenshot Suppression**: Suppresses high-frequency textual screenshots from cluttering natural sketch results.

### Frozen SketchProjection Head

The extracted structural representations are mapped into the shared 512-D photographic embedding space using a frozen, pretrained **`SketchProjection`** head:

- **Input / Output**: $512 \text{D} \to 512 \text{D}$
- **Architecture**:
  $$\text{LayerNorm}(512) \to \text{Linear}(512 \to 768) \to \text{GELU} \to \text{Linear}(768 \to 512)$$
- **Residual Connection**:
  $$\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{in}} + \alpha \cdot \Delta \mathbf{z} \quad (\alpha \approx 0.8)$$
- **L2 Normalization**: Output is L2-normalized for exact cosine compatibility with the FAISS photographic gallery index.
- **Weights**: Loaded directly from static checkpoint [`data/sketch_projection.pt`](file:///e:/Zero%20text%20AIML/data/sketch_projection.pt).

### Zero Runtime Training Guarantee

- **No Startup Training**: Cold boots load model weights in **~1.5 seconds**. The previous 180-epoch training loop has been completely eliminated.
- **Deterministic Search**: Every search query executes through frozen inference paths; model weights never mutate during runtime.
- **Decoupled User Feedback**: User upvotes and downvotes are stored purely as structured audit records and training pairs (`data/feedback_dataset.json`, `data/sketch_dataset.json`), completely isolated from model parameters.

### Offline Training Utility

To train or refine the projection weights without affecting the running application, ZERO-TEXT provides an offline training CLI:

```bash
python scripts/train_sketch_projection.py --epochs 60 --lr 0.0003 --temperature 0.07
```

Key features of the training script:
- **InfoNCE Contrastive Loss**: Learns invariant representations between synthetic multi-view sketches and photo gallery embeddings.
- **Evaluation Metrics**: Logs Precision@1 ($P@1$), Recall@3 ($R@3$), and Mean Reciprocal Rank ($\text{MRR}$).
- **Versioned Export**: Saves frozen weights directly to `data/sketch_projection.pt`.

### Colour‑Aware Sketch Search

When you draw with multiple colours or use bucket fill, the system:
- Extracts dominant colours from the sketch using K‑Means clustering.
- Generates colour text prompts (e.g., "red and blue coloured object").
- Blends sketch‑adapter embedding (55%), photo CLIP embedding (15%), and colour‑text embedding (30%) for the final query vector.

---

## API Endpoints

All endpoints are served from `http://127.0.0.1:8000/api/`.

| Endpoint | Method | Description | Key Parameters |
|----------|--------|-------------|----------------|
| `/api/status` | GET | System status (image count, cluster count, model readiness). | — |
| `/api/images` | GET | List all indexed images with full 2D/3D coordinates & metadata. | — |
| `/api/image/{filename}` | GET | Serve an image file by filename. | `filename` |
| `/api/image/{filename}` | DELETE | Delete an image permanently from disk and FAISS. | `filename` |
| `/api/search/hybrid` | POST | Hybrid text + OCR + colour search. | `query`, `clip_weight`, `ocr_weight`, `colour_weight`, `auto_weights`, `cluster_filter`, `top_k` |
| `/api/search/sketch` | POST | Sketch‑to‑image search with multi-scale structural matching. | `sketch_data` (base64), `top_k`, `text_hint`, `is_canvas_sketch` |
| `/api/search/color` | POST | Colour‑based search. | `hex` (e.g., "#FF0000"), `top_k` |
| `/api/search/similar` | POST | Find visually similar images. | `filename`, `top_k` |
| `/api/search/recommend_weights` | GET | Get AI‑recommended scoring weights for a query. | `query` |
| `/api/feedback` | POST | Record user feedback (positive/negative) for text or sketch queries. | `search_type`, `feedback`, `filename`, `query`, `sketch_data`, `text_hint` |
| `/api/feedback/stats` | GET | Get summary statistics of collected user feedback and datasets. | — |
| `/api/index/upload` | POST | Upload and index new images. | `files` (array of base64 data) |
| `/api/index/directory` | POST | Index all images in a local folder. | `directory_path`, `recursive` |
| `/api/index/status` | GET | Check background indexing progress. | — |
| `/api/recluster` | POST | Re‑run UMAP + HDBSCAN clustering. | — |
| `/api/sketches` | GET | List available sample sketch files. | — |
| `/api/sketches/{filename}` | GET | Serve a sample sketch file. | `filename` |

All POST endpoints accept JSON bodies and return JSON responses.

---

## Project Structure

```
Zero text AIML/
│
├── main.py                          # Entry point — starts server, opens browser
├── server.py                        # FastAPI backend with all API routes & feedback handlers
├── pipeline.py                      # Core AI engine (CLIP, FAISS, UMAP, HDBSCAN, SketchProjection, OCR)
├── sketch_search.py                 # Sketch search helper module
├── hybrid_search.py                 # Hybrid search helper module
├── search_images.py                 # Image search utilities
├── cluster_images.py                # Clustering utilities
├── run_app.bat                      # Windows one‑click launcher
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
│
├── scripts/
│   └── train_sketch_projection.py   # Standalone offline training utility for SketchProjection
│
├── data/
│   ├── sample_images/               # Your indexed images live here
│   ├── sketches/                    # Sample reference sketches (cat.jpg, mountain.jpg, rocket.jpg)
│   │   └── user_validated/          # Curated user sketches from verified search feedback
│   ├── image_index.faiss            # FAISS vector index (auto‑generated)
│   ├── image_metadata.json          # Image metadata (auto‑generated)
│   ├── ocr_results.json             # OCR text extraction results (auto‑generated)
│   ├── color_results.json           # Dominant colour palettes (auto‑generated)
│   ├── cluster_results.json         # Cluster assignments + UMAP coords (auto‑generated)
│   ├── sketch_projection.pt         # Frozen pretrained SketchProjection weights
│   ├── feedback_dataset.json        # User feedback audit log
│   ├── text_search_dataset.json     # Curated query-to-photo validation pairs
│   └── sketch_dataset.json          # Curated sketch-to-photo validation pairs
│
├── model_cache/                     # CLIP model files (auto‑downloaded on first run)
│   └── hub/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/
│
├── web/                             # Frontend (served as static files by FastAPI)
│   ├── index.html                   # Single‑page app HTML
│   ├── css/
│   │   └── styles.css               # All styling (dark theme, galaxy aesthetic)
│   └── js/
│       ├── app.js                   # Main application logic (search, gallery, inspector, modals)
│       ├── sketch.js                # Sketch Studio (canvas drawing, bucket fill, undo/redo, upload)
│       ├── galaxy3d.js              # 3D WebGL galaxy renderer (Three.js)
│       └── galaxy2d.js              # 2D UMAP star map renderer (Canvas 2D)
│
└── tests/
    ├── test_app_endpoints.py        # End-to-end integration & endpoint tests
    ├── test_sketch_multicolor.py    # Multi‑colour sketch, bucket fill, B&W, and upload tests
    ├── test_feedback_enrichment.py  # User feedback & dataset enrichment tests
    ├── verify_endpoints.py          # API route health checks
    ├── verify_search.py             # Search quality verification
    ├── test_color_extraction.py     # Colour extraction tests
    ├── test_image_embedding.py      # CLIP embedding tests
    ├── test_ocr.py                  # OCR extraction tests
    └── test.py                      # General unit tests
```

---

## Testing & Verification

Run these scripts to verify complete system health:

```bash
# 1. Full application endpoint & multi-modal search integration
python test_app_endpoints.py

# 2. Structural & multi-color sketch-to-photo retrieval verification
python test_sketch_multicolor.py

# 3. User feedback logging & dataset curation test suite
python tests/test_feedback_enrichment.py

# 4. Standalone offline training utility health check
python scripts/train_sketch_projection.py --help
```

All tests exit with **code 0** and print detailed diagnostic summaries.

---

## Troubleshooting & Tips

| Problem | Solution |
|---------|----------|
| **"paging file too small" error on Windows** | Already handled automatically — the code patches `safetensors` to use `pread` backend. No action needed. |
| **First startup is slow (~30–60 seconds)** | The CLIP model (~600 MB) is loading into RAM. Subsequent starts are faster because the model is cached on disk. |
| **Sketch search returns unexpected results** | Add a **text hint** in the sketch studio (e.g., type "car" while drawing a car). Use colours that match the target. Draw the overall silhouette/shape. |
| **3D Galaxy is blank** | Make sure Three.js CDN loaded (requires internet on first load). If offline, the 3D view gracefully degrades. |
| **Images not appearing after upload** | Wait for the background indexing to complete (check the progress bar). Then click **🔄 Recluster** to update the galaxy views. |
| **Search returns low scores** | Try the **Weights** button to manually boost CLIP, OCR, or Colour weights depending on your query type. |
| **Port 8000 is in use** | Run with a different port: `python main.py --port 8080` |

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

**🚀 Enjoy using ZERO‑TEXT!** Explore your images in the Visual Galaxy, sketch to search, and discover connections you never knew existed.
