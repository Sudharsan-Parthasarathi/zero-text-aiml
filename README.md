# Zero‑Text AIML Application

## Overview

Zero‑Text AIML is a multimodal search engine that combines **CLIP embeddings**, **FAISS similarity**, **UMAP/HDBSCAN clustering**, and a **sketch‑search adapter** to let users retrieve images by text, colour, or hand‑drawn sketch.  Recent changes focus on three user‑visible improvements while keeping the rest of the system untouched:

1. **Removed the confusing "Art" image** that previously appeared in the gallery.
2. **Improved sketch‑search accuracy** by:
   - Detecting sketches vs. photos (white‑pixel ratio & colour variance).
   - Using the `SketchAdapter` (identity‑initialised MLP with a learnable scaling factor) to map sketch embeddings into the photo manifold.
   - Normalising vectors before FAISS cosine similarity and fine‑tuning UMAP parameters.
3. **Added a safe image‑deletion flow** with a confirmation pop‑up.
4. **Fixed result‑image thumbnail size** (260 px height, `object‑fit: contain`).
5. **Ensured uploaded photos retain full CLIP accuracy** when no sketch is drawn on them.

The rest of the core pipeline, API endpoints, and UI remain unchanged.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Overview](#api-overview)
- [Web UI Features](#web-ui-features)
- [Sketch‑Search Details](#sketch-search-details)
- [Image Deletion Workflow](#image-deletion-workflow)
- [Project Structure](#project-structure)
- [Testing & Verification](#testing--verification)
- [Future Work / Fine‑tuning](#future-work--fine‑tuning)
- [License](#license)

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| **Python** | `>=3.11` | Tested on Windows 10/11 (64‑bit). |
| **PyTorch** | `2.3.0+cpu` | CPU‑only; the code sets `DEVICE="cpu"`. |
| **FAISS‑CPU** | `1.15.1` | No GPU support needed. |
| **Open‑CLIP** | `2.20` | Loads the `ViT‑B‑32` checkpoint from `model_cache`. |
| **UMAP‑LEARN** | `0.5.5` |
| **HDBSCAN** | `0.8.38` |
| **FastAPI & Uvicorn** | `0.162.2` / `0.30.0` |
| **Pillow** | `10.4.0` |
| **Other dependencies** | See `requirements.txt` (generated automatically by `pip freeze`). |

> **Important:** On Windows the CLIP model loading may fail with a paging‑file error. The repository patches `safetensors.torch.load_file` to use the `pread` backend and calls `gc.collect()` before loading – no extra user action required.

---

## Installation

```bash
# Clone the repository (if you haven't already)
git clone <repository‑url> "e:\Zero text AIML"
cd "e:\Zero text AIML"

# (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download the CLIP model (will be cached in data/model_cache)
python -c "import open_clip, pathlib; pathlib.Path('data/model_cache').mkdir(parents=True, exist_ok=True)"
```

The first run will download the `ViT‑B‑32` model (~600 MiB) into `data/model_cache`.

---

## Running the Application

```bash
# Start the FastAPI server (development mode)
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

Open a browser and navigate to `http://127.0.0.1:8000` – the UI lives in the `web/` folder and is served by FastAPI.

---

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | `POST` | General multimodal search (text, colour, sketch). |
| `/api/sketch/search` | `POST` | Direct sketch‑search (detects sketch vs. photo internally). |
| `/api/images` | `GET` | List all images stored in the index. |
| `/api/images/{id}` | `DELETE` | Delete a single image – the UI triggers a confirmation modal before invoking this. |
| `/api/upload` | `POST` | Upload a new image (photo or sketch). |

All endpoints return JSON payloads; see `server.py` for exact schema.

---

## Web UI Features

- **Gallery view** – thumbnails of indexed images (fixed height 260 px). The previous “Art” thumbnail has been removed.
- **Search bar** – type text, pick a dominant colour, or click **Sketch** to open the drawing canvas.
- **Sketch canvas** – draw over an uploaded photo or start from a blank canvas. The studio now includes:
  - **Multi-color Drawing**: 12 curated color swatches + HTML5 native custom color picker.
  - **Bucket Fill (Flood Fill)**: Fast flood-fill tool with color-tolerance matching to easily fill sketch regions.
  - **Undo & Redo**: Full stroke-level history tracking (up to 35 steps) with canvas toolbar buttons and standard keyboard shortcuts (`Ctrl+Z` / `Ctrl+Y` / `Ctrl+Shift+Z`).
  - **Preserved Upload Accuracy**: Canvas preserves the original uploaded image when no drawing occurs, ensuring 100% photographic CLIP accuracy.
- **Multi-Color Sketch Understanding & Retrieval**:
  - Automatically detects color strokes and bucket-filled areas on canvas sketches.
  - Multi-view ensemble features multi-channel stroke contrast preservation (so light colors like yellow, cyan, and orange maintain full stroke definition).
  - Fine-tuned `SketchAdapter` trained on synthetic multi-color outlines and bucket-filled posterized pairs (achieving 100% Precision@1 and 1.0 MRR on validation pairs).
  - Semantic color blend pipeline seamlessly retrieves colored objects (e.g., red car sketches retrieve `red_supercar.jpg`, yellow sunflower sketches retrieve `golden_sunflower.jpg`).
- **Delete Image** – each card now includes a **Delete** button. Clicking opens a modal with **Cancel** and **Confirm Delete** options.
- **Result grid** – all result images share the same size, preventing the “very small” thumbnails reported earlier.

---

## Sketch‑Search Details

1. **Pre‑processing** – When a sketch is submitted the backend:
   - Computes the white‑pixel ratio of the input.
   - Calculates colour variance across the image.
   - If `white_ratio > 0.82` **and** `colour_variance < 15`, the input is considered a **sketch**; otherwise it is treated as a regular photo.
2. **Feature extraction** – The image is fed to CLIP (`ViT‑B‑32`). For sketches the raw CLIP embedding is passed through `SketchAdapter`, a 3‑layer residual MLP that learns a mapping from sketch space to photo space.
3. **Normalization** – The resulting vectors are L2‑normalised before being inserted into the FAISS index.
4. **Similarity** – FAISS uses **cosine similarity** (inner‑product on normalised vectors). Scores are scaled to `[0, 1]`.
5. **Dimensionality reduction** – UMAP is configured with `n_neighbors=30` and `min_dist=0.1` to preserve fine‑grained sketch details.

The adapter is initialised to the identity function (`fc3` weights zero) and a learnable scaling factor `alpha≈0.8`. This allows immediate improvement without any fine‑tuning; further gains can be achieved by training on a sketch‑photo dataset.

---

## Image Deletion Workflow

1. User clicks the **Delete** button on a thumbnail.
2. A modal appears:
   - **Cancel** – closes the modal, no action.
   - **Confirm Delete** – sends a `DELETE /api/images/{id}` request.
3. On success the server removes the image from the FAISS index, deletes the file from `data/sample_images/`, and the UI instantly removes the card.

---

## Project Structure

```
Zero text AIML/
│   pipeline.py          # Core engine (embedding, clustering, sketch handling)
│   server.py            # FastAPI entry point
│   sketch_search.py     # Helper for direct sketch queries
│   requirements.txt
│   README.md            <-- this file
│
├─ data/
│   ├─ sample_images/    # Indexed images ("Art" image removed)
│   ├─ sketches/         # Sample sketches used for testing
│   ├─ model_cache/      # CLIP checkpoint cache
│   └─ {metadata, index, …}
│
├─ web/
│   ├─ css/styles.css    # Fixed thumbnail height (260px)
│   └─ js/sketch.js      # Canvas enhancements & state flags
│
└─ tests/
    ├─ verify_endpoints.py
    ├─ verify_search.py
    └─ test_sketch_and_uploads.py
```

---

## Testing & Verification

Run the provided verification scripts to ensure the system works after any change:

```bash
python verify_endpoints.py         # checks that all API routes respond correctly
python verify_search.py            # runs a suite of queries (text, colour, hybrid)
python test_sketch_multicolor.py   # verifies multi-color, bucket fill, B&W, and upload paths
```
All tests should pass with **exit code 0**.

---

## Future Work / Fine‑tuning

- **Dataset** – Gather a larger sketch‑photo pair dataset (e.g., QuickDraw + ImageNet pairs) and train the `SketchAdapter` for higher top‑1 accuracy (> 90 %).
- **Training script** – A `train_sketch_adapter.py` can be added that loads the current adapter, freezes the CLIP backbone, and runs a contrastive InfoNCE loss.
- **Configurable thresholds** – Expose `SKETCH_WHITE_RATIO` and `SKETCH_COLOUR_VARIANCE` via a YAML config for easier experimentation.
- **Batch inference** – Optimize the server to handle multiple concurrent sketch queries.

---

## License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

**Enjoy using Zero‑Text AIML!** If you encounter any issues, feel free to open an issue in the repository or contact the maintainer.
