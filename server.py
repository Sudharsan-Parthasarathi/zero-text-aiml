"""
ZERO-TEXT API SERVER
FastAPI backend powering the Interactive 2D/3D Visual Galaxy and Multi-modal Search System.
"""

import base64
import io
import mimetypes
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from pipeline import BASE_DIR, DATA_DIR, SAMPLE_DIR, SKETCHES_DIR, get_engine

app = FastAPI(
    title="ZERO-TEXT API",
    description="Multimodal Visual Galaxy Explorer & Semantic Search System",
    version="2.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_engine()

# Request schemas
class HybridSearchRequest(BaseModel):
    query: str
    clip_weight: float = 0.55
    ocr_weight: float = 0.30
    colour_weight: float = 0.15
    cluster_filter: Optional[int] = None
    top_k: int = 40

class SketchSearchRequest(BaseModel):
    sketch_data: str  # Base64 data URL e.g. "data:image/png;base64,..."
    top_k: int = 20

class ColorSearchRequest(BaseModel):
    hex: str
    top_k: int = 20

class SimilarSearchRequest(BaseModel):
    filename: str
    top_k: int = 10

class DirectoryIndexRequest(BaseModel):
    directory_path: str
    recursive: bool = True

class FileUploadItem(BaseModel):
    filename: str
    data: str  # Base64 data URL

class UploadPayload(BaseModel):
    files: List[FileUploadItem]


# -----------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return engine.get_system_status()


@app.get("/api/images")
def get_images():
    return engine.get_all_images()


@app.get("/api/image/{filename}")
def get_image(filename: str):
    """
    Safely retrieves the image file from disk.
    """
    path_str = engine.get_image_path(filename)
    path = Path(path_str)
    if not path.exists():
        alt_path = SAMPLE_DIR / filename
        if alt_path.exists():
            path = alt_path
        else:
            alt_sketch = SKETCHES_DIR / filename
            if alt_sketch.exists():
                path = alt_sketch
            else:
                raise HTTPException(status_code=404, detail=f"Image {filename} not found.")

    mime_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(path, media_type=mime_type or "image/jpeg")


@app.post("/api/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    results = engine.hybrid_search(
        query=req.query,
        clip_weight=req.clip_weight,
        ocr_weight=req.ocr_weight,
        colour_weight=req.colour_weight,
        cluster_filter=req.cluster_filter,
        top_k=req.top_k,
    )
    words, colours = engine.analyse_query(req.query)
    return {
        "query": req.query,
        "detected_words": words,
        "detected_colours": colours,
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/search/sketch")
def search_sketch(req: SketchSearchRequest):
    try:
        data_str = req.sketch_data
        if "," in data_str:
            data_str = data_str.split(",", 1)[1]
        img_bytes = base64.b64decode(data_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid sketch data: {e}")

    results = engine.sketch_search(img, top_k=req.top_k)
    return {
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/search/color")
def search_color(req: ColorSearchRequest):
    results = engine.color_search(req.hex, top_k=req.top_k)
    return {
        "color": req.hex,
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/search/similar")
def search_similar(req: SimilarSearchRequest):
    results = engine.similar_images(req.filename, top_k=req.top_k)
    return {
        "target": req.filename,
        "total_results": len(results),
        "results": results,
    }


@app.post("/api/index/upload")
def upload_and_index(payload: UploadPayload):
    """
    Accepts one or multiple image file uploads via Base64, saves them into
    data/sample_images, and triggers the background indexing pipeline.
    """
    saved_paths = []
    upload_dir = DATA_DIR / "sample_images"
    upload_dir.mkdir(parents=True, exist_ok=True)

    for item in payload.files:
        fn = Path(item.filename).name
        if not fn:
            continue
        try:
            data_str = item.data
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]
            img_bytes = base64.b64decode(data_str)
            dest = upload_dir / fn
            with open(dest, "wb") as f:
                f.write(img_bytes)
            saved_paths.append(dest)
        except Exception as e:
            print(f"Failed to save upload {fn}: {e}")

    if saved_paths:
        engine.index_files_background(saved_paths)

    return {
        "message": f"Received {len(saved_paths)} images. Background indexing started.",
        "files": [p.name for p in saved_paths],
    }


@app.post("/api/index/directory")
def index_directory(req: DirectoryIndexRequest):
    """
    Indexes all images found in a specified local directory.
    """
    try:
        found_count = engine.index_directory(req.directory_path, recursive=req.recursive)
        return {
            "message": f"Found {found_count} images in {req.directory_path}. Indexing started in background.",
            "directory": req.directory_path,
            "images_found": found_count,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/status")
def get_indexing_status():
    with engine._lock:
        return dict(engine.indexing_state)


@app.post("/api/recluster")
def recluster_images():
    clusters = engine.run_clustering()
    engine._save_data()
    return {
        "message": "Reclustering completed successfully.",
        "clusters": clusters,
    }


@app.get("/api/sketches")
def get_sample_sketches():
    sketches = []
    if SKETCHES_DIR.exists():
        for p in sorted(SKETCHES_DIR.glob("*.*")):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                sketches.append({
                    "filename": p.name,
                    "url": f"/api/sketches/{p.name}",
                })
    return sketches


@app.get("/api/sketches/{filename}")
def get_sample_sketch_file(filename: str):
    p = SKETCHES_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sketch not found.")
    mime_type, _ = mimetypes.guess_type(str(p))
    return FileResponse(p, media_type=mime_type or "image/jpeg")


# -----------------------------------------------------------------
# FRONTEND STATIC FILES & SPA SERVING
# -----------------------------------------------------------------

WEB_DIR = BASE_DIR / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>ZERO-TEXT Frontend Initializing...</h1>"
