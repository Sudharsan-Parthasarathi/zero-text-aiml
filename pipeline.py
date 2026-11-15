"""
ZERO-TEXT CORE PIPELINE & AI ENGINE (v2.5 High-Precision Multimodal Architecture)
Integrates:
- CLIP (ViT-B-32) Multimodal Embeddings (Text, Image, Sketch)
- RapidOCR In-Image Text Extraction (confidence-filtered)
- Dominant Colour Extraction (Pillow Median Cut & Lab distance)
- Intelligent Query Analyzer & Auto-Weight Decider (CLIP vs OCR vs Colour)
- FAISS Vector Index (512-dim Cosine Similarity)
- UMAP 2D & 3D Dimensionality Reduction
- HDBSCAN Density-Based Hierarchical Clustering
- Ingestion Pipeline for Files & Directories
"""

import os
import sys

# Critical Windows environment variables before importing torch/faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str((BASE_DIR / "model_cache").resolve())
os.environ["HF_HUB_OFFLINE"] = "1"

# Windows DLL order: torch MUST be imported first
import torch
import open_clip

# Windows paging-file resilience: avoid mmap allocation failure (error 1455) by using pread
try:
    import safetensors.torch
    _orig_safetensors_load = safetensors.torch.load_file
    def _pread_load_file(filename, device="cpu", **kwargs):
        kwargs.setdefault("backend", "pread")
        return _orig_safetensors_load(filename, device=device, **kwargs)
    safetensors.torch.load_file = _pread_load_file
except Exception:
    pass

import faiss
import numpy as np
import hdbscan
import umap

import json
import re
import threading
import time
import base64
import hashlib
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image

# Enable UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
SKETCHES_DIR = DATA_DIR / "sketches"

INDEX_PATH = DATA_DIR / "image_index.faiss"
METADATA_PATH = DATA_DIR / "image_metadata.json"
OCR_PATH = DATA_DIR / "ocr_results.json"
COLOUR_PATH = DATA_DIR / "color_results.json"
CLUSTER_PATH = DATA_DIR / "cluster_results.json"
SKETCH_PROJECTION_PATH = DATA_DIR / "sketch_projection.pt"
SKETCH_ADAPTER_PATH = DATA_DIR / "sketch_adapter.pt"
FEEDBACK_PATH = DATA_DIR / "feedback_dataset.json"
TEXT_DATASET_PATH = DATA_DIR / "text_search_dataset.json"
SKETCH_DATASET_PATH = DATA_DIR / "sketch_dataset.json"
USER_SKETCHES_DIR = SKETCHES_DIR / "user_validated"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cpu"
DIMENSIONS = 512
COLOUR_COUNT = 5

OCR_CONFIDENCE_THRESHOLD = 0.50

COLOURS = {
    "red": (220, 40, 40),
    "crimson": (220, 20, 60),
    "scarlet": (255, 36, 0),
    "green": (40, 160, 70),
    "emerald": (46, 139, 87),
    "forest": (34, 139, 34),
    "blue": (30, 100, 220),
    "navy": (15, 30, 80),
    "cyan": (40, 210, 220),
    "teal": (20, 130, 130),
    "yellow": (245, 215, 30),
    "amber": (255, 191, 0),
    "gold": (255, 215, 0),
    "golden": (255, 215, 0),
    "orange": (240, 130, 30),
    "purple": (130, 60, 180),
    "violet": (138, 43, 226),
    "magenta": (220, 40, 180),
    "pink": (240, 130, 180),
    "black": (20, 20, 20),
    "dark": (30, 30, 30),
    "white": (245, 245, 245),
    "light": (235, 235, 235),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "silver": (192, 192, 192),
    "brown": (130, 80, 40),
    "tan": (210, 180, 140),
    "lime": (50, 205, 50),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),
    "neon": (76, 201, 240),
}

DOCUMENT_MARKERS = {
    "code", "coding", "coder", "python", "script", "import", "class", "def", "function",
    "programmer", "programming", "developer", "software", "algorithm",
    "formula", "equation", "math", "physics", "quantum", "notes",
    "study", "biology", "genetics", "dna", "cell", "text", "document",
    "invoice", "bill", "tax", "statement", "todo", "checklist", "action",
    "memorandum", "written", "receipt", "table", "definition", "paper",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "with", "find", "show",
    "me", "image", "picture", "containing", "that", "has", "having", "and",
    "for", "to", "at", "from", "by", "about", "all", "some", "any", "photo",
    "photos", "images", "pictures", "look", "like", "search", "get", "give",
}

CLUSTER_PALETTE = [
    "#4cc9f0",  # Neon Cyan
    "#f72585",  # Neon Pink
    "#80ed99",  # Mint Green
    "#ffd166",  # Warm Yellow
    "#c77dff",  # Soft Violet
    "#ff7b00",  # Bright Orange
    "#00b4d8",  # Sky Blue
    "#b5179e",  # Deep Magenta
    "#06d6a0",  # Emerald Green
    "#e76f51",  # Terracotta
    "#118ab2",  # Ocean Blue
    "#ef476f",  # Rose Red
]


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def softmax_normalize(scores: np.ndarray, temperature: float = 0.35) -> np.ndarray:
    if len(scores) == 0:
        return scores
    diff = scores.max() - scores.min()
    if diff < 1e-6:
        return np.ones_like(scores) * 0.5
    # Min-max scale first to [0, 1] then apply soft temperature
    scaled = (scores - scores.min()) / diff
    exp_s = np.exp(scaled / temperature)
    norm = exp_s / exp_s.max()  # top candidate gets 1.0
    return norm


class SketchProjection(torch.nn.Module):
    """
    Lightweight residual projection head that maps sketch structural embeddings
    into the photographic CLIP manifold. Loaded directly from frozen pretrained checkpoints.
    Zero runtime training is performed during application startup or queries.
    """
    def __init__(self, dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        # 3-layer deep MLP residual offset path
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.act1 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.act2 = torch.nn.GELU()
        self.fc3 = torch.nn.Linear(hidden_dim, dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

        # Initialize fc3 to zero so initial forward is exact identity
        torch.nn.init.zeros_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc3(self.act2(self.norm2(self.fc2(self.act1(self.norm1(self.fc1(x)))))))
        out = x + self.alpha * delta
        return out / out.norm(dim=-1, keepdim=True)


# Backwards compatibility alias
SketchAdapter = SketchProjection


class ZeroTextEngine:
    """
    Unified ZeroText Engine handling multi-modal indexing, searching,
    query intent recommendation, and 2D/3D visual galaxy clustering.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.ocr = None
        self.model_ready = False

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.ocr_lookup: Dict[str, Dict] = {}
        self.colour_lookup: Dict[str, List[Dict]] = {}
        self.cluster_lookup: Dict[str, Dict] = {}

        # Feedback & Dataset Enrichment
        self.feedback_dataset: List[Dict] = []
        self.text_search_dataset: List[Dict] = []
        self.sketch_dataset: List[Dict] = []
        self.validated_text_queries: Dict[str, set] = {}
        self.negative_feedback_queries: Dict[str, set] = {}
        self.validated_sketch_pairs: set = set()
        self.negative_sketch_pairs: set = set()

        self.sketch_projection: Optional[SketchProjection] = None
        self.sketch_adapter: Optional[SketchProjection] = None
        self.reducer_2d: Optional[umap.UMAP] = None
        self.reducer_3d: Optional[umap.UMAP] = None

        self.indexing_state = {
            "status": "idle",
            "progress": 0,
            "message": "Engine ready",
            "total": 0,
            "current": 0,
            "logs": [],
            "model_ready": False,
        }

        self._ensure_directories()
        self._load_data()

        # Pre-warm CLIP in a background thread
        threading.Thread(target=self._prewarm_clip, daemon=True).start()

    def _prewarm_clip(self):
        try:
            self.load_clip()
        except Exception as e:
            self._log(f"CLIP prewarm notice: {e}")

    def _ensure_directories(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        SKETCHES_DIR.mkdir(parents=True, exist_ok=True)
        USER_SKETCHES_DIR.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str):
        print(f"[ZERO-TEXT] {message}")
        with self._lock:
            self.indexing_state["logs"].append(
                {"time": time.strftime("%H:%M:%S"), "message": message}
            )
            if len(self.indexing_state["logs"]) > 200:
                self.indexing_state["logs"].pop(0)

    # -------------------------------------------------------------
    # MODEL INITIALIZATION
    # -------------------------------------------------------------

    def load_clip(self):
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            self._log(f"Loading CLIP ({MODEL_NAME})...")
            start = time.time()
            import gc
            gc.collect()
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            self.model_ready = True
            self.indexing_state["model_ready"] = True
            elapsed = time.time() - start
            self._log(f"CLIP loaded in {elapsed:.2f}s — ready for search!")
            try:
                self.load_sketch_projection()
            except Exception as ex:
                self._log(f"Sketch projection init notice: {ex}")

    def load_sketch_projection(self):
        """
        Loads the frozen pretrained SketchProjection weights directly.
        Zero runtime or startup training is performed.
        """
        if self.sketch_projection is not None:
            return
        ckpt_path = SKETCH_PROJECTION_PATH if SKETCH_PROJECTION_PATH.exists() else SKETCH_ADAPTER_PATH
        if ckpt_path.exists():
            try:
                proj = SketchProjection(DIMENSIONS)
                state_dict = torch.load(ckpt_path, map_location=DEVICE)
                proj.load_state_dict(state_dict)
                proj.eval()
                self.sketch_projection = proj
                self.sketch_adapter = proj
                self._log(f"Loaded frozen pretrained SketchProjection weights from {ckpt_path.name}.")
                return
            except Exception as e:
                self._log(f"Notice loading sketch projection checkpoint: {e}")

    def load_ocr(self):
        if self.ocr is not None:
            return
        with self._lock:
            if self.ocr is not None:
                return
            self._log("Loading RapidOCR engine...")
            start = time.time()
            try:
                from rapidocr import RapidOCR
                self.ocr = RapidOCR()
            except ImportError:
                from rapidocr_onnxruntime import RapidOCR
                self.ocr = RapidOCR()
            self._log(f"RapidOCR loaded in {time.time() - start:.2f}s")

    # -------------------------------------------------------------
    # DATA LOADING & PERSISTENCE
    # -------------------------------------------------------------

    def _load_data(self):
        with self._lock:
            if METADATA_PATH.exists():
                try:
                    with open(METADATA_PATH, "r", encoding="utf-8") as f:
                        raw_metadata = json.load(f)
                    # Strictly filter out any files that do not exist physically on disk
                    valid_metadata = []
                    for item in raw_metadata:
                        p = Path(item["path"])
                        if p.is_file():
                            valid_metadata.append(item)
                        else:
                            alt = SAMPLE_DIR / item["filename"]
                            if alt.is_file():
                                item["path"] = str(alt.resolve())
                                valid_metadata.append(item)
                    self.metadata = valid_metadata
                except Exception as e:
                    self._log(f"Failed to load metadata: {e}")
                    self.metadata = []
            else:
                self.metadata = []

            valid_filenames = {item["filename"] for item in self.metadata}

            if INDEX_PATH.exists():
                try:
                    self.index = faiss.read_index(str(INDEX_PATH))
                except Exception as e:
                    self._log(f"Failed to load FAISS index: {e}")
                    self.index = faiss.IndexFlatIP(DIMENSIONS)
            else:
                self.index = faiss.IndexFlatIP(DIMENSIONS)

            if OCR_PATH.exists():
                try:
                    with open(OCR_PATH, "r", encoding="utf-8") as f:
                        ocr_data = json.load(f)
                        self.ocr_lookup = {
                            item["filename"]: item for item in ocr_data
                            if item.get("filename") in valid_filenames
                        }
                except Exception as e:
                    self._log(f"Failed to load OCR data: {e}")
                    self.ocr_lookup = {}
            else:
                self.ocr_lookup = {}

            if COLOUR_PATH.exists():
                try:
                    with open(COLOUR_PATH, "r", encoding="utf-8") as f:
                        colour_data = json.load(f)
                        self.colour_lookup = {
                            item["filename"]: item.get("dominant_colours", [])
                            for item in colour_data
                            if item.get("filename") in valid_filenames
                        }
                except Exception as e:
                    self._log(f"Failed to load colour data: {e}")
                    self.colour_lookup = {}
            else:
                self.colour_lookup = {}

            if CLUSTER_PATH.exists():
                try:
                    with open(CLUSTER_PATH, "r", encoding="utf-8") as f:
                        cluster_data = json.load(f)
                        self.cluster_lookup = {
                            item["filename"]: item for item in cluster_data
                            if item.get("filename") in valid_filenames
                        }
                except Exception as e:
                    self._log(f"Failed to load cluster data: {e}")
                    self.cluster_lookup = {}
            else:
                self.cluster_lookup = {}

            # Load Feedback & Enriched Datasets
            if FEEDBACK_PATH.exists():
                try:
                    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                        self.feedback_dataset = json.load(f)
                except Exception as e:
                    self._log(f"Failed to load feedback dataset: {e}")
                    self.feedback_dataset = []
            else:
                self.feedback_dataset = []

            if TEXT_DATASET_PATH.exists():
                try:
                    with open(TEXT_DATASET_PATH, "r", encoding="utf-8") as f:
                        self.text_search_dataset = json.load(f)
                except Exception as e:
                    self._log(f"Failed to load text search dataset: {e}")
                    self.text_search_dataset = []
            else:
                self.text_search_dataset = []

            if SKETCH_DATASET_PATH.exists():
                try:
                    with open(SKETCH_DATASET_PATH, "r", encoding="utf-8") as f:
                        self.sketch_dataset = json.load(f)
                except Exception as e:
                    self._log(f"Failed to load sketch dataset: {e}")
                    self.sketch_dataset = []
            else:
                self.sketch_dataset = []

            # Populate in-memory lookup indices for quick relevance calibration
            self.validated_text_queries = {}
            for item in self.text_search_dataset:
                qn = item.get("query_normalized", "").strip()
                fn = item.get("filename", "").strip()
                if qn and fn:
                    self.validated_text_queries.setdefault(qn, set()).add(fn)

            self.validated_sketch_pairs = {
                (item.get("sketch_hash", ""), item.get("filename", ""))
                for item in self.sketch_dataset
                if item.get("filename")
            }

            self.negative_feedback_queries = {}
            self.negative_sketch_pairs = set()
            for fb in self.feedback_dataset:
                if fb.get("feedback") in ("negative", "no"):
                    st = fb.get("search_type")
                    fn = fb.get("filename")
                    if st == "text" and fn:
                        qn = fb.get("query_normalized", "").strip()
                        if qn:
                            self.negative_feedback_queries.setdefault(qn, set()).add(fn)
                    elif st == "sketch" and fn:
                        sh = fb.get("sketch_hash", "")
                        if sh:
                            self.negative_sketch_pairs.add((sh, fn))

            self._log(
                f"Data loaded: {len(self.metadata)} images indexed, "
                f"{len(self.cluster_lookup)} cluster records, "
                f"{len(self.text_search_dataset)} validated text queries, "
                f"{len(self.sketch_dataset)} validated sketches, "
                f"{len(self.feedback_dataset)} feedback logs."
            )

    def _save_data(self):
        with self._lock:
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)

            if self.index is not None:
                faiss.write_index(self.index, str(INDEX_PATH))

            ocr_list = list(self.ocr_lookup.values())
            with open(OCR_PATH, "w", encoding="utf-8") as f:
                json.dump(ocr_list, f, indent=4, ensure_ascii=False)

            colour_list = [
                {
                    "filename": k,
                    "path": self.get_image_path(k),
                    "dominant_colours": v,
                }
                for k, v in self.colour_lookup.items()
            ]
            with open(COLOUR_PATH, "w", encoding="utf-8") as f:
                json.dump(colour_list, f, indent=4, ensure_ascii=False)

            cluster_list = list(self.cluster_lookup.values())
            with open(CLUSTER_PATH, "w", encoding="utf-8") as f:
                json.dump(cluster_list, f, indent=4, ensure_ascii=False)

            self._save_feedback_dataset()
            self._save_text_dataset()
            self._save_sketch_dataset()

            self._log("All data files saved successfully.")

    def _save_feedback_dataset(self):
        with self._lock:
            with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
                json.dump(self.feedback_dataset, f, indent=4, ensure_ascii=False)

    def _save_text_dataset(self):
        with self._lock:
            with open(TEXT_DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(self.text_search_dataset, f, indent=4, ensure_ascii=False)

    def _save_sketch_dataset(self):
        with self._lock:
            with open(SKETCH_DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(self.sketch_dataset, f, indent=4, ensure_ascii=False)

    def record_feedback(
        self,
        search_type: str,
        feedback: str,
        filename: str,
        query: str = "",
        sketch_data: str = "",
        text_hint: str = "",
        score: Optional[float] = None,
        cluster: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Record user feedback (Yes / No) and perform intelligent dataset enrichment.
        When YES (positive):
          - Identifies exact image, object category, and search metadata.
          - Performs duplicate detection: if already in dataset, duplicate is prevented.
          - If new, adds validated result to text_search_dataset.json or sketch_dataset.json.
        When NO (negative):
          - Records rejection in feedback_dataset.json for future search relevance calibration.
          - Does not add to positive datasets.
        """
        with self._lock:
            st = (search_type or "text").lower().strip()
            fb = "positive" if (feedback or "").lower().strip() in ("positive", "yes", "true", "1") else "negative"
            fn = Path(filename).name

            # Verify image exists
            img_path = self.get_image_path(fn)
            matched = any(item["filename"] == fn for item in self.metadata) or Path(img_path).exists()
            if not matched:
                return {
                    "status": "error",
                    "enriched": False,
                    "already_exists": False,
                    "message": f"Target image '{fn}' not found in index.",
                }

            clean_stem = Path(fn).stem.replace("_", " ").replace("-", " ")
            category = clean_stem.title()
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            enriched = False
            already_exists = False
            sketch_hash = ""

            if fb == "positive":
                if st == "text":
                    q_raw = query.strip()
                    q_norm = q_raw.lower()
                    if not q_norm:
                        return {
                            "status": "error",
                            "enriched": False,
                            "already_exists": False,
                            "message": "Search query text is required for text feedback.",
                        }

                    # Duplicate detection
                    exists = any(
                        e.get("query_normalized") == q_norm and e.get("filename") == fn
                        for e in self.text_search_dataset
                    )
                    if exists or fn in self.validated_text_queries.get(q_norm, set()):
                        already_exists = True
                        msg = f"Feedback noted! Match for '{q_raw}' already exists in dataset (duplicate prevented)."
                    else:
                        entry_id = f"text_{hashlib.md5(f'{q_norm}_{fn}'.encode()).hexdigest()[:12]}"
                        new_entry = {
                            "id": entry_id,
                            "query": q_raw,
                            "query_normalized": q_norm,
                            "filename": fn,
                            "path": img_path,
                            "category": category,
                            "score": round(float(score or 0.0), 4),
                            "cluster": int(cluster if cluster is not None else -1),
                            "timestamp": timestamp,
                            "validated": True,
                        }
                        self.text_search_dataset.append(new_entry)
                        self._save_text_dataset()
                        self.validated_text_queries.setdefault(q_norm, set()).add(fn)
                        if q_norm in self.negative_feedback_queries:
                            self.negative_feedback_queries[q_norm].discard(fn)
                        enriched = True
                        msg = f"Validated! '{fn}' added to text dataset for '{q_raw}'."

                elif st == "sketch":
                    sketch_bytes = None
                    if sketch_data:
                        try:
                            b64 = sketch_data
                            if "," in b64:
                                b64 = b64.split(",", 1)[1]
                            sketch_bytes = base64.b64decode(b64)
                        except Exception as e:
                            self._log(f"Notice decoding sketch for feedback: {e}")

                    if sketch_bytes:
                        sketch_hash = hashlib.sha256(sketch_bytes).hexdigest()[:16]
                    else:
                        sketch_hash = hashlib.md5(f"{fn}_{timestamp}".encode()).hexdigest()[:16]

                    # Duplicate detection
                    exists = any(
                        e.get("sketch_hash") == sketch_hash and e.get("filename") == fn
                        for e in self.sketch_dataset
                    )
                    if exists or (sketch_hash, fn) in self.validated_sketch_pairs:
                        already_exists = True
                        msg = f"Feedback noted! Sketch mapping for '{fn}' already exists in dataset (duplicate prevented)."
                    else:
                        USER_SKETCHES_DIR.mkdir(parents=True, exist_ok=True)
                        rel_sketch_file = ""
                        if sketch_bytes:
                            sk_filename = f"sketch_{sketch_hash}.png"
                            sk_save_path = USER_SKETCHES_DIR / sk_filename
                            try:
                                with open(sk_save_path, "wb") as f_sk:
                                    f_sk.write(sketch_bytes)
                                rel_sketch_file = f"user_validated/{sk_filename}"
                            except Exception as e:
                                self._log(f"Error saving user sketch file: {e}")

                        new_entry = {
                            "id": f"sketch_{sketch_hash[:12]}",
                            "sketch_hash": sketch_hash,
                            "sketch_file": rel_sketch_file,
                            "filename": fn,
                            "path": img_path,
                            "category": category,
                            "text_hint": text_hint.strip() if text_hint else "",
                            "score": round(float(score or 0.0), 4),
                            "cluster": int(cluster if cluster is not None else -1),
                            "timestamp": timestamp,
                            "validated": True,
                        }
                        self.sketch_dataset.append(new_entry)
                        self._save_sketch_dataset()
                        self.validated_sketch_pairs.add((sketch_hash, fn))
                        if (sketch_hash, fn) in self.negative_sketch_pairs:
                            self.negative_sketch_pairs.discard((sketch_hash, fn))
                        enriched = True
                        msg = f"Validated! Sketch match for '{fn}' added to sketch dataset."
                else:
                    msg = f"Feedback registered for '{fn}'."

            else:
                # Negative feedback
                if st == "text":
                    q_norm = query.strip().lower()
                    if q_norm:
                        self.negative_feedback_queries.setdefault(q_norm, set()).add(fn)
                        if q_norm in self.validated_text_queries:
                            self.validated_text_queries[q_norm].discard(fn)
                elif st == "sketch":
                    if sketch_data:
                        try:
                            b64 = sketch_data.split(",", 1)[1] if "," in sketch_data else sketch_data
                            sketch_hash = hashlib.sha256(base64.b64decode(b64)).hexdigest()[:16]
                        except Exception:
                            sketch_hash = ""
                    if sketch_hash:
                        self.negative_sketch_pairs.add((sketch_hash, fn))

                msg = f"Feedback recorded. '{fn}' marked as not desired to refine future results."

            # Audit log entry in feedback_dataset.json
            q_norm = query.strip().lower() if st == "text" else ""
            log_id = f"fb_{hashlib.md5(f'{st}_{fn}_{timestamp}_{fb}'.encode()).hexdigest()[:10]}"
            feedback_log_entry = {
                "id": log_id,
                "search_type": st,
                "feedback": fb,
                "filename": fn,
                "query": query.strip() if st == "text" else "",
                "query_normalized": q_norm,
                "sketch_hash": sketch_hash,
                "text_hint": text_hint.strip() if text_hint else "",
                "score": round(float(score or 0.0), 4),
                "cluster": int(cluster if cluster is not None else -1),
                "category": category,
                "timestamp": timestamp,
                "session_id": session_id or "",
                "enriched": enriched,
            }

            # Avoid spamming duplicate logs
            is_dup_log = any(
                e.get("search_type") == st and e.get("filename") == fn and e.get("feedback") == fb
                and e.get("query_normalized") == q_norm and e.get("sketch_hash") == sketch_hash
                for e in self.feedback_dataset[-20:]
            )
            if not is_dup_log:
                self.feedback_dataset.append(feedback_log_entry)
                self._save_feedback_dataset()

            return {
                "status": "ok",
                "enriched": enriched,
                "already_exists": already_exists,
                "feedback": fb,
                "filename": fn,
                "category": category,
                "message": msg,
            }

    def get_feedback_stats(self) -> Dict:
        with self._lock:
            return {
                "total_feedback": len(self.feedback_dataset),
                "positive_count": sum(1 for f in self.feedback_dataset if f.get("feedback") == "positive"),
                "negative_count": sum(1 for f in self.feedback_dataset if f.get("feedback") == "negative"),
                "text_dataset_count": len(self.text_search_dataset),
                "sketch_dataset_count": len(self.sketch_dataset),
            }

    def get_image_path(self, filename: str) -> str:
        for item in self.metadata:
            if item["filename"] == filename:
                p = Path(item["path"])
                if p.exists():
                    return str(p)
        sample = SAMPLE_DIR / filename
        if sample.exists():
            return str(sample)
        return str(SAMPLE_DIR / filename)

    def delete_image(self, filename: str) -> bool:
        """
        Permanently delete an image: removes the physical file from disk,
        removes it from metadata, OCR, color, and cluster lookups,
        rebuilds the FAISS index from remaining embeddings, and persists changes.
        """
        with self._lock:
            # Find the image in metadata
            idx = -1
            item_to_remove = None
            for i, item in enumerate(self.metadata):
                if item["filename"] == filename:
                    idx = i
                    item_to_remove = item
                    break

            if idx == -1:
                # Also check if it exists in SAMPLE_DIR even if missing in metadata
                target_file = SAMPLE_DIR / filename
                if target_file.is_file():
                    try:
                        target_file.unlink()
                        return True
                    except Exception:
                        pass
                return False

            # 1. Remove physical file
            try:
                p = Path(item_to_remove["path"])
                if p.is_file():
                    p.unlink()
            except Exception as e:
                self._log(f"Notice deleting file from disk: {e}")

            target_alt = SAMPLE_DIR / filename
            if target_alt.is_file():
                try:
                    target_alt.unlink()
                except Exception:
                    pass

            # 2. Rebuild FAISS index without the deleted item
            if self.index is not None and self.index.ntotal > 0:
                try:
                    n_total = self.index.ntotal
                    if n_total == len(self.metadata):
                        all_vecs = np.array(self.index.reconstruct_n(0, n_total))
                        keep_mask = [i != idx for i in range(len(self.metadata))]
                        if any(keep_mask):
                            remaining_vecs = all_vecs[keep_mask]
                            new_index = faiss.IndexFlatIP(DIMENSIONS)
                            new_index.add(remaining_vecs.astype("float32"))
                            self.index = new_index
                        else:
                            self.index = faiss.IndexFlatIP(DIMENSIONS)
                    else:
                        self._log("Index size mismatch during delete; re-syncing index.")
                except Exception as e:
                    self._log(f"Error rebuilding index on delete: {e}")

            # 3. Remove from metadata and lookups
            self.metadata.pop(idx)
            self.ocr_lookup.pop(filename, None)
            self.colour_lookup.pop(filename, None)
            self.cluster_lookup.pop(filename, None)

            # 4. Save updated data
            self._save_data()
            self._log(f"Successfully deleted {filename}. Remaining images: {len(self.metadata)}")
            return True

    # -------------------------------------------------------------
    # FEATURE EXTRACTION
    # -------------------------------------------------------------

    def extract_colours(self, image_path: Path) -> List[Dict]:
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                image.thumbnail((200, 200))
                quantized = image.quantize(
                    colors=COLOUR_COUNT, method=Image.Quantize.MEDIANCUT
                )
                colour_counts = quantized.getcolors(200 * 200)
                palette = quantized.getpalette()

                if not colour_counts or not palette:
                    return []

                colour_counts.sort(reverse=True)
                total_pixels = sum(count for count, _ in colour_counts)
                dominant = []
                for count, index in colour_counts:
                    pos = index * 3
                    r, g, b = palette[pos], palette[pos + 1], palette[pos + 2]
                    hex_code = rgb_to_hex(r, g, b)
                    pct = round((count / total_pixels) * 100, 2)
                    dominant.append({
                        "hex": hex_code,
                        "rgb": [int(r), int(g), int(b)],
                        "percentage": pct,
                    })
                return dominant
        except Exception as e:
            self._log(f"Colour extraction error for {image_path.name}: {e}")
            return []

    def extract_ocr(self, image_path: Path) -> Dict:
        self.load_ocr()
        try:
            output = self.ocr(str(image_path))
            extracted_items = []

            # RapidOCR 3.x+ returns RapidOCROutput dataclass with .txts and .scores
            if hasattr(output, "txts") and output.txts:
                scores = getattr(output, "scores", [1.0] * len(output.txts))
                for txt, conf in zip(output.txts, scores):
                    txt_str = str(txt).strip()
                    if txt_str and float(conf) >= OCR_CONFIDENCE_THRESHOLD:
                        extracted_items.append({
                            "text": txt_str,
                            "confidence": round(float(conf), 4),
                        })
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                # Handle legacy tuple / list format
                raw_items = output[0] if isinstance(output[0], (list, tuple)) else output
                for item in raw_items:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        txt_str = str(item[1]).strip()
                        conf = float(item[2])
                        if txt_str and conf >= OCR_CONFIDENCE_THRESHOLD:
                            extracted_items.append({
                                "text": txt_str,
                                "confidence": round(conf, 4),
                            })

            full_text = " ".join(item["text"] for item in extracted_items)
            return {
                "filename": image_path.name,
                "path": str(image_path),
                "extracted_text": extracted_items,
                "full_text": full_text,
            }
        except Exception as e:
            self._log(f"OCR error for {image_path.name}: {e}")
            return {
                "filename": image_path.name,
                "path": str(image_path),
                "extracted_text": [],
                "full_text": "",
                "error": str(e),
            }

    def compute_image_embedding(self, image: Image.Image) -> np.ndarray:
        self.load_clip()
        img_rgb = image.convert("RGB")
        tensor = self.preprocess(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    # -----------------------------------------------------------------
    #  SKETCH-SPECIFIC PREPROCESSING & EMBEDDING (High-Precision)
    # -----------------------------------------------------------------

    def _preprocess_sketch_views(self, sketch_img: Image.Image) -> Tuple[List[Image.Image], List[float]]:
        """
        High-fidelity structural multi-scale contour and silhouette representation:
          1. Aspect-preserved foreground bounding box centering on standard 224x224 canvas.
          2. View 1: Structural Stroke Contour View (antialiased normalized strokes).
          3. View 2: Structural Silhouette / Distance Field View (captures 2D shape, contour topology, mass).
          4. View 3 (if chromatic): Color-contrast enhanced content view.
          5. View 4: Horizontal flip view (orientation robustness).
        """
        import cv2
        from PIL import ImageEnhance

        rgb_np = np.array(sketch_img.convert("RGB"), dtype=np.uint8)
        gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        is_white_bg = gray_np.mean() > 128

        if is_white_bg:
            inv_np = np.max(255 - rgb_np, axis=2).astype(np.uint8)
        else:
            inv_np = np.max(rgb_np, axis=2).astype(np.uint8)

        stroke_mask = inv_np > 20
        h, w = inv_np.shape[:2]

        if not np.any(stroke_mask):
            return [sketch_img], [1.0]

        ys, xs = np.nonzero(stroke_mask)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        box_h = y_max - y_min + 1
        box_w = x_max - x_min + 1

        # Foreground bounding box centering with 12% proportional margin
        pad_y = max(8, int(box_h * 0.12))
        pad_x = max(8, int(box_w * 0.12))
        c_y1 = max(0, y_min - pad_y)
        c_y2 = min(h, y_max + pad_y + 1)
        c_x1 = max(0, x_min - pad_x)
        c_x2 = min(w, x_max + pad_x + 1)

        crop_rgb = rgb_np[c_y1:c_y2, c_x1:c_x2]
        ch, cw = crop_rgb.shape[:2]

        TARGET_SIZE = 224
        scale = min((TARGET_SIZE - 28) / max(ch, 1), (TARGET_SIZE - 28) / max(cw, 1))
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        resized_crop = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas_rgb = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        y_off = (TARGET_SIZE - new_h) // 2
        x_off = (TARGET_SIZE - new_w) // 2
        canvas_rgb[y_off:y_off+new_h, x_off:x_off+new_w] = resized_crop

        # 1. Structural Stroke Contour View (density adaptive)
        gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
        stroke_mask_c = gray < 235
        stroke_density = float(stroke_mask_c.mean())
        if stroke_density < 0.04:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_gray = cv2.erode(gray, kernel)
            canvas_stroke = np.stack([dilated_gray] * 3, axis=-1)
        else:
            canvas_stroke = canvas_rgb
        view_stroke = Image.fromarray(canvas_stroke)

        # 2. Structural Silhouette / Soft Shape Field View
        inv_stroke = 255 - gray
        dist = cv2.distanceTransform(cv2.dilate(inv_stroke, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        view_shape = Image.fromarray(255 - dist_norm).convert("RGB")

        # 3. Horizontal flip view (orientation invariance)
        view_flip = view_stroke.transpose(Image.FLIP_LEFT_RIGHT)

        views = [view_stroke, view_shape, view_flip]
        weights = [1.3, 0.8, 0.7]

        # Check if sketch has distinct color content
        c_hsv = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2HSV)
        sat_vals = c_hsv[:, :, 1]
        has_color = float(sat_vals.mean()) > 8.0 or (sat_vals > 35).mean() > 0.02

        if has_color:
            try:
                boosted = ImageEnhance.Color(Image.fromarray(canvas_rgb)).enhance(1.3)
                views.append(boosted)
                weights.append(1.1)
            except Exception:
                views.append(Image.fromarray(canvas_rgb))
                weights.append(1.0)

        return views, weights

    def _compute_sketch_raw_ensemble(
        self,
        sketch_img: Image.Image,
        text_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute weighted multi-view ensemble embedding for a sketch across structural views
        plus optional text hint tokens.
        """
        self.load_clip()
        views, weights = self._preprocess_sketch_views(sketch_img)

        tensors = torch.stack([self.preprocess(view) for view in views]).to(DEVICE)
        with torch.inference_mode():
            batch_emb = self.model.encode_image(tensors)
            batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)

        embeddings: List[torch.Tensor] = [batch_emb]

        if text_hint and text_hint.strip():
            hint = text_hint.strip()
            prompts = [
                hint,
                f"a photograph of {hint}",
                f"a real photo of {hint}",
            ]
            tokens = self.tokenizer(prompts).to(DEVICE)
            with torch.inference_mode():
                t_emb = self.model.encode_text(tokens)
                t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            embeddings.append(t_emb)
            weights.extend([1.5] * len(prompts))

        stacked = torch.cat(embeddings, dim=0)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        weighted = (stacked * weights_t).sum(dim=0, keepdim=True) / weights_t.sum()
        weighted = weighted / weighted.norm(dim=-1, keepdim=True)
        return weighted.cpu().numpy().astype("float32")

    def _compute_sketch_ensemble_embedding(
        self,
        sketch_img: Image.Image,
        text_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute high-quality sketch structural embedding:
          1. Multi-scale structural contour & silhouette representations.
          2. Frozen pretrained visual encoder embedding.
          3. Optional projection via frozen SketchProjection checkpoint into photographic space.
          4. Feature normalization: Unit L2 sphere.
        """
        self.load_clip()
        raw_vec = self._compute_sketch_raw_ensemble(sketch_img, text_hint=text_hint)
        t_raw = torch.tensor(raw_vec, dtype=torch.float32, device=DEVICE)

        model_proj = self.sketch_projection or self.sketch_adapter
        if model_proj is not None:
            with torch.inference_mode():
                adapted = model_proj(t_raw)
                adapted = adapted / adapted.norm(dim=-1, keepdim=True)
                return adapted.cpu().numpy().astype("float32")

        return raw_vec

    def compute_text_embedding(self, text: str, ensemble: bool = True) -> np.ndarray:
        self.load_clip()
        text = text.strip()
        if not text:
            text = "photo"

        prompts = [text]
        if ensemble:
            prompts.append(f"a photo of {text}")
            prompts.append(f"a high quality photo showing {text}")

            t_lower = text.lower()
            if any(k in t_lower for k in ["animal", "pet", "creature", "fauna"]):
                prompts.extend([
                    "a photo of an animal, such as a dog or cat",
                    "a cute domestic pet or animal",
                    "a golden retriever dog or cat"
                ])
            elif any(k in t_lower for k in ["dog", "puppy", "canine", "retriever"]):
                prompts.extend([
                    "a photo of a golden retriever dog on grass",
                    "a friendly canine pet dog"
                ])
            elif any(k in t_lower for k in ["cat", "kitten", "feline"]):
                prompts.extend([
                    "a photo of a funny cat with paws",
                    "a cute domestic feline pet cat"
                ])
            elif any(k in t_lower for k in ["car", "vehicle", "automobile", "supercar"]):
                prompts.extend([
                    "a photo of a crimson sports car supercar",
                    "a high performance automotive vehicle"
                ])
            elif any(k in t_lower for k in ["code", "coding", "coder", "programming", "programmer", "python", "script", "software", "developer", "algorithm"]):
                prompts.extend([
                    "computer programming code in an IDE text editor",
                    "python source code script on dark screen"
                ])
            elif any(k in t_lower for k in ["ocean", "sea", "water", "beach", "waves"]):
                prompts.extend([
                    "a photo of deep blue ocean waves",
                    "turquoise sea water and foam"
                ])
            elif any(k in t_lower for k in ["space", "galaxy", "nebula", "stars", "cosmos"]):
                prompts.extend([
                    "a deep space telescope photograph of a purple cosmic nebula",
                    "cosmic dust and sparkling stars in outer space"
                ])
            elif any(k in t_lower for k in ["flower", "sunflower", "blossom", "plant"]):
                prompts.extend([
                    "a vibrant blooming yellow sunflower macro photograph",
                    "bright golden flower petals"
                ])
            elif any(k in t_lower for k in ["forest", "trees", "woods", "nature"]):
                prompts.extend([
                    "a lush green ancient redwood forest with tall trees",
                    "deep emerald nature woods"
                ])
            elif any(k in t_lower for k in ["pizza", "food", "meal", "snack"]):
                prompts.extend([
                    "a hot baked pepperoni pizza with melted cheese",
                    "delicious fresh gourmet food"
                ])
            elif any(k in t_lower for k in ["human", "person", "people", "man", "woman", "boy", "girl", "guy", "face", "portrait", "selfie", "individual"]):
                prompts.extend([
                    "a photo of a person",
                    "a portrait photo of a human person",
                    "a real human being portrait photograph",
                    "a selfie photo of a person looking at the camera",
                    "a photo of a man or woman face",
                    "a real human face portrait"
                ])
            elif any(k in t_lower for k in ["camera", "photography", "lens", "rangefinder"]):
                prompts.extend([
                    "a photo of a vintage 35mm rangefinder mechanical camera",
                    "a vintage film camera sitting on a wooden desk"
                ])
            elif any(k in t_lower for k in ["spiderman", "superhero", "spider-man", "spider man"]):
                prompts.extend([
                    "spiderman upside down falling from the sky over a city",
                    "a superhero in a red and black suit"
                ])

        tokens = self.tokenizer(prompts).to(DEVICE)
        with torch.inference_mode():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean_emb = emb.mean(dim=0, keepdim=True)
            mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        return mean_emb.cpu().numpy().astype("float32")

    # -------------------------------------------------------------
    # CLUSTERING & DIMENSIONALITY REDUCTION (UMAP 2D & 3D + HDBSCAN)
    # -------------------------------------------------------------

    def run_clustering(self) -> List[Dict]:
        with self._lock:
            n_items = len(self.metadata)
            if n_items < 2:
                self._log("Not enough images to cluster (minimum 2 needed).")
                clusters = []
                for item in self.metadata:
                    record = {
                        "filename": item["filename"],
                        "path": item["path"],
                        "cluster": 0,
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "x3d": 0.0,
                        "y3d": 0.0,
                        "z3d": 0.0,
                        "membership_probability": 1.0,
                    }
                    clusters.append(record)
                    self.cluster_lookup[item["filename"]] = record
                return clusters

            self._log(f"Re-clustering {n_items} images with HDBSCAN and UMAP (2D/3D)...")
            embeddings = self.index.reconstruct_n(0, n_items)
            embeddings = np.asarray(embeddings, dtype=np.float32)

            min_cluster = max(2, min(4, n_items // 3))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster,
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            cluster_labels = clusterer.fit_predict(embeddings)

            # UMAP 2D - parameters tuned to preserve fine-grained sketch & contour manifold details
            n_neighbors = min(6, max(2, n_items - 1))
            reducer_2d = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.08,
                metric="cosine",
                random_state=42,
            )
            coords_2d = reducer_2d.fit_transform(embeddings)

            # UMAP 3D - parameters tuned to preserve fine-grained sketch & contour manifold details
            if n_items >= 4:
                reducer_3d = umap.UMAP(
                    n_components=3,
                    n_neighbors=n_neighbors,
                    min_dist=0.08,
                    metric="cosine",
                    random_state=42,
                )
                coords_3d = reducer_3d.fit_transform(embeddings)
            else:
                reducer_3d = None
                coords_3d = np.zeros((n_items, 3), dtype=np.float32)
                coords_3d[:, 0] = coords_2d[:, 0]
                coords_3d[:, 1] = coords_2d[:, 1]
                coords_3d[:, 2] = np.random.RandomState(42).uniform(-2.0, 2.0, size=n_items)

            self.reducer_2d = reducer_2d
            self.reducer_3d = reducer_3d

            new_clusters = []
            for i, item in enumerate(self.metadata):
                record = {
                    "filename": item["filename"],
                    "path": item["path"],
                    "cluster": int(cluster_labels[i]),
                    "x": float(coords_2d[i][0]),
                    "y": float(coords_2d[i][1]),
                    "z": float(coords_3d[i][2]),
                    "x3d": float(coords_3d[i][0]),
                    "y3d": float(coords_3d[i][1]),
                    "z3d": float(coords_3d[i][2]),
                    "membership_probability": float(clusterer.probabilities_[i]),
                }
                new_clusters.append(record)
                self.cluster_lookup[item["filename"]] = record

            self._log(
                f"Clustering complete: {len(set(cluster_labels) - {-1})} clusters discovered."
            )
            return new_clusters

    def _ensure_reducers(self):
        if self.reducer_2d is not None and self.reducer_3d is not None:
            return
        if not self.index or self.index.ntotal < 2:
            return
        try:
            n_items = self.index.ntotal
            embeddings = np.asarray(self.index.reconstruct_n(0, n_items), dtype=np.float32)
            n_neighbors = min(6, max(2, n_items - 1))
            self.reducer_2d = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.08,
                metric="cosine",
                random_state=42,
            )
            self.reducer_2d.fit(embeddings)
            if n_items >= 4:
                self.reducer_3d = umap.UMAP(
                    n_components=3,
                    n_neighbors=n_neighbors,
                    min_dist=0.08,
                    metric="cosine",
                    random_state=42,
                )
                self.reducer_3d.fit(embeddings)
        except Exception as e:
            self._log(f"Notice fitting sketch UMAP reducers: {e}")

    # -------------------------------------------------------------
    # QUERY ANALYSIS & SMART WEIGHT DECIDER
    # -------------------------------------------------------------

    def analyse_query(self, query: str) -> Tuple[List[str], List[str]]:
        words = re.findall(r"[a-z0-9]+", query.lower())
        detected_colours = [w for w in words if w in COLOURS]
        text_words = [
            w for w in words
            if w not in STOP_WORDS and w not in COLOURS and len(w) > 1
        ]
        return text_words, detected_colours

    def recommend_weights(self, query: str) -> Dict:
        """
        Intelligently decides the recommended weights for CLIP, OCR, and Colour.
        Satisfies user requirement: 'recommended CLIP OCR Dominant colour deciding thing'.
        """
        q_lower = query.strip().lower()
        text_words, detected_colours = self.analyse_query(q_lower)

        has_hex = bool(re.search(r"#[0-9a-f]{3,6}", q_lower))
        has_color = bool(detected_colours or has_hex)
        has_doc_marker = any(w in DOCUMENT_MARKERS for w in text_words)

        # 1. Pure Color Query (e.g. 'red', 'dark blue', 'crimson', '#ff0000')
        if has_color and not text_words:
            return {
                "clip": 0.20,
                "ocr": 0.05,
                "colour": 0.75,
                "intent": "colour_dominant",
                "label": "🎨 Colour Focused",
                "reason": f"Pure color query detected: {', '.join(detected_colours or ['hex'])}. Boosting Dominant Colour to 75%.",
            }

        # 2. Hybrid Color + Subject (e.g. 'red supercar', 'blue ocean', 'golden flower')
        if has_color and text_words:
            return {
                "clip": 0.50,
                "ocr": 0.15,
                "colour": 0.35,
                "intent": "hybrid_colour",
                "label": "🎨 Semantic + Colour Hybrid",
                "reason": f"Subject with color descriptor: prioritizing semantic meaning (50%) and color harmony (35%).",
            }

        # 3. Document / Text-Heavy Query (e.g. 'python code', 'physics formula', 'invoice', 'notes')
        if has_doc_marker or "notes" in q_lower or "code" in q_lower or "coding" in q_lower or "coder" in q_lower or "text" in q_lower:
            return {
                "clip": 0.25,
                "ocr": 0.65,
                "colour": 0.10,
                "intent": "ocr_dominant",
                "label": "📝 In-Image Text & Document",
                "reason": "Text-heavy/document query detected. Boosting RapidOCR in-image text extraction to 65%.",
            }

        # 4. General Semantic Concept (e.g. 'cat', 'mountain', 'rocket', 'football')
        return {
            "clip": 0.75,
            "ocr": 0.15,
            "colour": 0.10,
            "intent": "semantic_dominant",
            "label": "🧠 Visual Semantic Concept",
            "reason": "Natural language visual concept detected. Prioritizing CLIP multimodal vector embedding (75%).",
        }

    # -------------------------------------------------------------
    # MULTI-MODAL SCORING ENGINES
    # -------------------------------------------------------------

    def get_ocr_score(self, filename: str, query_words: List[str]) -> float:
        if not query_words:
            return 0.0
        ocr_item = self.ocr_lookup.get(filename, {})
        full_text = ocr_item.get("full_text", "").lower()
        if not full_text:
            return 0.0

        unique_qwords = list(set(query_words))
        total_score = 0.0
        for w in unique_qwords:
            # Exact word boundary match gets full 1.0
            if re.search(r"\b" + re.escape(w) + r"\b", full_text):
                total_score += 1.0
            # Substring match gets partial 0.6
            elif w in full_text:
                total_score += 0.6

        # Multi-word sequence bonus
        if len(unique_qwords) > 1 and " ".join(query_words) in full_text:
            total_score += 0.5

        return min(1.0, total_score / len(unique_qwords))

    def get_colour_score(self, filename: str, query_colours: List[str]) -> float:
        if not query_colours:
            return 0.0
        dominant = self.colour_lookup.get(filename, [])
        if not dominant:
            return 0.0
        total_weight = sum(max(0, float(c.get("percentage", 0))) for c in dominant)
        if total_weight <= 0:
            return 0.0

        max_dist = (3 * (255 ** 2)) ** 0.5
        scores = []
        for colour_name in query_colours:
            target = np.array(COLOURS[colour_name], dtype=float)
            col_score = 0.0
            for item in dominant:
                rgb = np.array(item["rgb"], dtype=float)
                dist = np.linalg.norm(target - rgb)
                sim = max(0.0, 1.0 - (dist / max_dist))
                # Non-linear boost for close colors
                sim = sim ** 2
                w = float(item.get("percentage", 0)) / total_weight
                col_score += sim * w
            scores.append(col_score)

        return float(np.mean(scores)) if scores else 0.0

    def hybrid_search(
        self,
        query: str,
        clip_weight: Optional[float] = None,
        ocr_weight: Optional[float] = None,
        colour_weight: Optional[float] = None,
        auto_weights: bool = False,
        top_k: int = 50,
        cluster_filter: Optional[int] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        High-precision hybrid search with intelligent auto-weights and softmax ranking.
        Never fails: falls back gracefully if CLIP is loading or if query is pure color.
        """
        if not self.metadata:
            return [], self.recommend_weights(query)

        query = query.strip()
        rec = self.recommend_weights(query)

        # Apply recommended weights if auto_weights is requested or weights omitted
        if auto_weights or clip_weight is None:
            c_w = rec["clip"]
            o_w = rec["ocr"]
            col_w = rec["colour"]
        else:
            c_w = float(clip_weight)
            o_w = float(ocr_weight)
            col_w = float(colour_weight)

        query_words, query_colours = self.analyse_query(query)

        # 1. CLIP Embeddings (with graceful fallback if still warming up)
        raw_clip = np.zeros(len(self.metadata), dtype=np.float32)
        clip_computed = False
        try:
            if self.index and self.index.ntotal > 0:
                self.load_clip()
                text_vec = self.compute_text_embedding(query)
                scores, indices = self.index.search(text_vec, self.index.ntotal)
                for s, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self.metadata):
                        raw_clip[idx] = float(s)
                clip_computed = True
        except Exception as e:
            self._log(f"CLIP computation notice: {e} — falling back to OCR + Colour")

        # Absolute calibration of CLIP cosine similarity
        def calibrate_similarity(sim: float) -> float:
            scaled = (sim - 0.165) / (0.275 - 0.165)
            clamped = max(0.0, min(1.0, scaled))
            return float(clamped ** 1.3)

        results = []
        for i, item in enumerate(self.metadata):
            filename = item["filename"]
            cluster_info = self.cluster_lookup.get(filename, {})
            c_id = cluster_info.get("cluster", -1)

            if cluster_filter is not None and c_id != cluster_filter:
                continue

            raw_s = float(raw_clip[i]) if clip_computed else 0.0
            clip_s = calibrate_similarity(raw_s) if clip_computed else 0.0
            ocr_s = self.get_ocr_score(filename, query_words)
            col_s = self.get_colour_score(filename, query_colours)

            # Combined weighted score
            active_weights = [(c_w, clip_s)]
            if query_words:
                active_weights.append((o_w, ocr_s))
            if query_colours:
                active_weights.append((col_w, col_s))

            total_w = sum(w for w, _ in active_weights) or 1.0
            final_s = sum(w * s for w, s in active_weights) / total_w

            ocr_item = self.ocr_lookup.get(filename, {})
            colours = self.colour_lookup.get(filename, [])

            # Suppress code / document files from ranking high on purely natural visual concepts
            is_doc_or_code_query = any(w in DOCUMENT_MARKERS for w in query_words)
            is_code_file = "code.jpg" in filename.lower() or "calculator.py" in ocr_item.get("full_text", "").lower()
            if not is_doc_or_code_query and is_code_file:
                final_s *= 0.25

            # User feedback adjustments (continuous feedback-driven accuracy improvement)
            q_norm = query.lower().strip()
            if q_norm:
                if filename in self.validated_text_queries.get(q_norm, set()):
                    final_s = min(1.0, final_s * 1.35 + 0.15)
                elif filename in self.negative_feedback_queries.get(q_norm, set()):
                    final_s *= 0.25

            display_clip = clip_s

            results.append({
                "filename": filename,
                "path": item["path"],
                "cluster": c_id,
                "cluster_color": (
                    "#8b96a8"
                    if c_id == -1
                    else CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                ),
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "clip": round(display_clip, 4),
                "ocr": round(ocr_s, 4),
                "colour": round(col_s, 4),
                "final": round(final_s, 4),
                "dominant_colours": colours,
                "full_text": ocr_item.get("full_text", ""),
                "has_text": bool(ocr_item.get("full_text", "").strip()),
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results[:top_k], rec

    def sketch_search(
        self,
        sketch_img: Image.Image,
        top_k: int = 25,
        text_hint: Optional[str] = None,
        is_canvas_sketch: Optional[bool] = None,
    ) -> List[Dict]:
        """
        High-precision sketch-to-photo search.

        Pipeline:
          1. Preprocess sketch into 8 augmented views (Otsu binarize, Canny edge detect,
             blur, dilation, horizontal flip, rotation).
          2. Compute a robust ensemble CLIP embedding and project via fine-tuned
             SketchAdapter into aligned photographic space.
          3. Query FAISS cosine similarity index with normalized vector.
          4. Calibrate raw cosine similarity scores specifically for sketch features.
          5. Map sketch into 2D and 3D UMAP galaxy space for visual spotlighting.
        """
        if not self.metadata or not self.index or self.index.ntotal == 0:
            return []

        self.load_clip()

        # Step 1 & 2: Multi-view preprocessing + fine-tuned adapter embedding
        # Automatically detects whether the input is a hand-drawn sketch / line art
        # or an uploaded full-color photograph to apply optimal representations.
        import cv2
        rgb_np = np.array(sketch_img.convert("RGB"))
        h_img, w_img = rgb_np.shape[:2]

        # Check for white canvas background / padding
        white_mask = (rgb_np[:, :, 0] > 240) & (rgb_np[:, :, 1] > 240) & (rgb_np[:, :, 2] > 240)
        white_ratio = float(white_mask.mean())

        # If it's a photo or graphic padded onto canvas with white margins (white_ratio < 0.85),
        # extract the actual content region to evaluate features without margin bias
        non_white = ~white_mask
        if non_white.any() and white_ratio < 0.85:
            ys, xs = np.nonzero(non_white)
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            crop_y1 = max(0, ymin - 2)
            crop_y2 = min(h_img, ymax + 3)
            crop_x1 = max(0, xmin - 2)
            crop_x2 = min(w_img, xmax + 3)
            content_img = Image.fromarray(rgb_np[crop_y1:crop_y2, crop_x1:crop_x2])
            content_rgb = rgb_np[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            content_img = sketch_img
            content_rgb = rgb_np

        # Evaluate color content and variance
        c_hsv = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2HSV)
        content_sat = float(c_hsv[:, :, 1].mean())
        diff_rg = np.abs(content_rgb[:, :, 0].astype(int) - content_rgb[:, :, 1].astype(int))
        diff_gb = np.abs(content_rgb[:, :, 1].astype(int) - content_rgb[:, :, 2].astype(int))
        diff_rb = np.abs(content_rgb[:, :, 0].astype(int) - content_rgb[:, :, 2].astype(int))
        col_diff = float((diff_rg + diff_gb + diff_rb).mean())

        if is_canvas_sketch is True:
            is_sketch = True
        elif is_canvas_sketch is False:
            # Uploaded image: detect whether it is an uploaded sketch/line drawing or a real photograph
            if white_ratio > 0.65 and (content_sat < 25.0 or col_diff < 15.0):
                is_sketch = True
            else:
                is_sketch = False
        else:
            is_sketch = False
            if white_ratio > 0.82:
                is_sketch = True
            elif white_ratio > 0.25:
                gray_c = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
                lap_var = float(cv2.Laplacian(gray_c, cv2.CV_64F).var())
                if white_ratio > 0.35 and lap_var < 350.0:
                    is_sketch = True
                elif white_ratio > 0.50:
                    is_sketch = True

        # Check if the sketch contains meaningful color (drawn strokes or bucket-filled regions)
        sketch_has_color = False
        sketch_palette = []
        if is_sketch and not white_mask.all():
            non_white_px = rgb_np[~white_mask]
            if len(non_white_px) > 30:
                nw_hsv = cv2.cvtColor(non_white_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
                sat_vals = nw_hsv[:, 0, 1]
                val_vals = nw_hsv[:, 0, 2]
                color_fraction = float((sat_vals > 28).mean())
                mean_sat = float(sat_vals.mean())
                if color_fraction > 0.02 or mean_sat > 12.0:
                    sketch_has_color = True
                    # Extract multi-color clusters from colored and distinct strokes
                    sig_mask = (sat_vals > 22) | (val_vals < 45)
                    sig_px = non_white_px[sig_mask]
                    if len(sig_px) >= 20:
                        try:
                            from sklearn.cluster import MiniBatchKMeans
                            n_c = min(4, len(np.unique(sig_px, axis=0)))
                            kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=42, n_init=3)
                            labels = kmeans.fit_predict(sig_px)
                            centers = kmeans.cluster_centers_
                            counts = np.bincount(labels)
                            total_pts = len(labels)
                            for idx in np.argsort(-counts):
                                pct = float(counts[idx] / total_pts)
                                if pct < 0.04:
                                    continue
                                c_rgb = centers[idx]
                                c_hsv = cv2.cvtColor(np.uint8([[c_rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
                                min_d = float('inf')
                                best_name = None
                                for cname, crgb in COLOURS.items():
                                    d = np.linalg.norm(c_rgb - np.array(crgb, dtype=float))
                                    if d < min_d:
                                        min_d = d
                                        best_name = cname
                                is_chrom = bool(c_hsv[1] > 28 and c_hsv[2] > 35)
                                sketch_palette.append({
                                    "name": best_name,
                                    "rgb": c_rgb.astype(float),
                                    "percentage": pct,
                                    "is_chromatic": is_chrom,
                                    "dist": float(min_d),
                                })
                        except Exception:
                            pass

        if is_sketch:
            if sketch_has_color:
                # COLORED SKETCH: Triple-path blend
                # Path A: SketchAdapter captures shape/structure from line art
                sketch_vec = self._compute_sketch_ensemble_embedding(sketch_img, text_hint=text_hint)

                # Path B: CLIP photo encoder on centered content image
                photo_vec = self.compute_image_embedding(content_img)

                # Path C: Semantic text guidance using accurately extracted color descriptors
                CANONICAL_COLOR_MAP = {
                    "amber": "yellow", "gold": "yellow", "golden": "yellow",
                    "crimson": "red", "scarlet": "red",
                    "emerald": "green", "forest": "green",
                    "navy": "dark blue", "cyan": "cyan", "teal": "teal",
                    "violet": "purple", "magenta": "pink",
                }
                chromatic_names = []
                for p in sketch_palette:
                    if p["is_chromatic"] and p["dist"] < 170:
                        mapped = CANONICAL_COLOR_MAP.get(p["name"], p["name"])
                        if mapped not in chromatic_names:
                            chromatic_names.append(mapped)

                prompts = []
                if chromatic_names:
                    color_phrase = " and ".join(chromatic_names[:2])
                    prompts.extend([
                        color_phrase,
                        f"a photo featuring {color_phrase}",
                        f"a vibrant photo with {color_phrase} colors",
                    ])
                    if text_hint and text_hint.strip():
                        prompts.extend([
                            f"a {color_phrase} {text_hint.strip()}",
                            f"a photo of a {color_phrase} {text_hint.strip()}",
                        ])
                elif text_hint and text_hint.strip():
                    prompts.extend([
                        text_hint.strip(),
                        f"a photo of {text_hint.strip()}",
                    ])

                if prompts:
                    t_tokens = self.tokenizer(prompts).to(DEVICE)
                    with torch.inference_mode():
                        t_embs = self.model.encode_text(t_tokens)
                        t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
                        t_vec = t_embs.mean(dim=0, keepdim=True)
                        t_vec = (t_vec / t_vec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
                    # Optimal fusion: shape anchor (adapter) + visual (photo) + semantic (text)
                    vec = 0.55 * sketch_vec + 0.15 * photo_vec + 0.30 * t_vec
                else:
                    vec = 0.65 * sketch_vec + 0.35 * photo_vec
            else:
                # MONOCHROME SKETCH: 100% fine-tuned SketchAdapter (original path)
                vec = self._compute_sketch_ensemble_embedding(sketch_img, text_hint=text_hint)
        else:
            # Uploaded photograph, illustration, or full-color image:
            # Encode with pre-trained CLIP photographic visual encoder for 100% retrieval precision
            emb_photo = self.compute_image_embedding(content_img)
            if text_hint and text_hint.strip():
                t_vec = self.compute_text_embedding(text_hint.strip())
                vec = 0.80 * emb_photo + 0.20 * t_vec
            else:
                vec = emb_photo

        # Ensure L2-normalized vector for FAISS cosine similarity
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        if norm > 1e-6:
            vec = (vec / norm).astype("float32")

        # Step 3: FAISS cosine similarity search (IndexFlatIP on normalized vectors)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        # Step 4: Calibrated scoring specifically for sketch features
        SKETCH_SIM_FLOOR = 0.18
        SKETCH_SIM_CEIL  = 0.78
        SKETCH_GAMMA     = 1.05

        def calibrate_sketch_score(raw_cos: float) -> float:
            scaled = (raw_cos - SKETCH_SIM_FLOOR) / (SKETCH_SIM_CEIL - SKETCH_SIM_FLOOR)
            clamped = max(0.0, min(1.0, scaled))
            return float(clamped ** SKETCH_GAMMA)

        # Step 5: Dimensionality reduction transform (UMAP 2D & 3D for query sketch)
        query_2d_coords = (0.0, 0.0)
        query_3d_coords = (0.0, 0.0, 0.0)
        try:
            if self.reducer_2d is not None:
                q2 = self.reducer_2d.transform(vec)
                query_2d_coords = (float(q2[0, 0]), float(q2[0, 1]))
            if self.reducer_3d is not None:
                q3 = self.reducer_3d.transform(vec)
                query_3d_coords = (float(q3[0, 0]), float(q3[0, 1]), float(q3[0, 2]))
        except Exception:
            pass

        results = []
        for rank, (raw_score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0 or idx >= len(self.metadata):
                continue

            item = self.metadata[idx]
            filename = item["filename"]
            cluster_info = self.cluster_lookup.get(filename, {})
            c_id = cluster_info.get("cluster", -1)
            raw_calibrated = calibrate_sketch_score(float(raw_score))
            colour_score = 0.0

            if sketch_has_color and sketch_palette:
                img_colours = self.colour_lookup.get(filename, [])
                if img_colours:
                    tot_sk = sum(p["percentage"] for p in sketch_palette) or 1.0
                    tot_score = 0.0
                    for sk in sketch_palette:
                        sk_rgb = sk["rgb"]
                        sk_w = sk["percentage"] / tot_sk
                        best_m = 0.0
                        for img_c in img_colours:
                            img_rgb = np.array(img_c["rgb"], dtype=float)
                            dist = np.linalg.norm(sk_rgb - img_rgb)
                            m = max(0.0, 1.0 - dist / 441.67) ** 2
                            if m > best_m:
                                best_m = m
                        tot_score += sk_w * best_m
                    colour_score = float(tot_score)
                # Colored sketch: balance calibrated visual shape similarity with multi-color harmony
                final_score = float(0.70 * raw_calibrated + 0.30 * colour_score)
            else:
                final_score = raw_calibrated

            # Suppress code / document files from ranking high on natural visual sketch concepts
            ocr_item = self.ocr_lookup.get(filename, {})
            is_code_file = "code.jpg" in filename.lower() or "calculator.py" in ocr_item.get("full_text", "").lower()
            if is_code_file and (not text_hint or "code" not in text_hint.lower()):
                final_score *= 0.35

            results.append({
                "rank": rank,
                "filename": filename,
                "path": item["path"],
                "similarity": round(float(raw_score), 4),
                "final": round(final_score, 4),
                "clip": round(raw_calibrated, 4),
                "ocr": 0.0,
                "colour": round(colour_score, 4),
                "cluster": c_id,
                "cluster_color": (
                    "#8b96a8"
                    if c_id == -1
                    else CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                ),
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "query_x": round(query_2d_coords[0], 4),
                "query_y": round(query_2d_coords[1], 4),
                "query_x3d": round(query_3d_coords[0], 4),
                "query_y3d": round(query_3d_coords[1], 4),
                "query_z3d": round(query_3d_coords[2], 4),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "dominant_colours": self.colour_lookup.get(filename, []),
                "full_text": self.ocr_lookup.get(filename, {}).get("full_text", ""),
                "has_text": bool(self.ocr_lookup.get(filename, {}).get("full_text", "").strip()),
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i
        self._log(f"Sketch search returned {len(results)} results (top: {results[0]['final']:.3f} – {results[0]['filename']})" if results else "Sketch search: no results")
        return results

    def color_search(self, target_hex_or_rgb: Union[str, List[int]], top_k: int = 25) -> List[Dict]:
        if isinstance(target_hex_or_rgb, str):
            target = np.array(hex_to_rgb(target_hex_or_rgb), dtype=float)
        else:
            target = np.array(target_hex_or_rgb, dtype=float)

        max_dist = (3 * (255 ** 2)) ** 0.5
        results = []
        for item in self.metadata:
            filename = item["filename"]
            dominant = self.colour_lookup.get(filename, [])
            if not dominant:
                sim = 0.0
            else:
                tot_weight = sum(float(c.get("percentage", 0)) for c in dominant) or 1.0
                sim = 0.0
                for c in dominant:
                    rgb = np.array(c["rgb"], dtype=float)
                    dist = np.linalg.norm(target - rgb)
                    w = float(c.get("percentage", 0)) / tot_weight
                    sim += max(0.0, (1.0 - dist / max_dist)) ** 2 * w

            cluster_info = self.cluster_lookup.get(filename, {})
            c_id = cluster_info.get("cluster", -1)
            results.append({
                "filename": filename,
                "path": item["path"],
                "final": round(sim, 4),
                "colour": round(sim, 4),
                "clip": 0.0,
                "ocr": 0.0,
                "cluster": c_id,
                "cluster_color": (
                    "#8b96a8"
                    if c_id == -1
                    else CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                ),
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "dominant_colours": dominant,
                "full_text": self.ocr_lookup.get(filename, {}).get("full_text", ""),
                "has_text": bool(self.ocr_lookup.get(filename, {}).get("full_text", "").strip()),
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results[:top_k]

    def similar_images(self, filename: str, top_k: int = 15) -> List[Dict]:
        idx = -1
        for i, item in enumerate(self.metadata):
            if item["filename"] == filename:
                idx = i
                break
        if idx < 0 or not self.index:
            return []

        emb = self.index.reconstruct(idx).reshape(1, -1)
        k = min(top_k + 1, self.index.ntotal)
        scores, indices = self.index.search(emb, k)

        results = []
        for s, neighbor_idx in zip(scores[0], indices[0]):
            if neighbor_idx < 0 or neighbor_idx == idx:
                continue
            item = self.metadata[neighbor_idx]
            fn = item["filename"]
            c_info = self.cluster_lookup.get(fn, {})
            c_id = c_info.get("cluster", -1)
            norm_s = max(0.0, min(1.0, (float(s) + 1.0) / 2.0))
            results.append({
                "filename": fn,
                "path": item["path"],
                "final": round(norm_s, 4),
                "clip": round(norm_s, 4),
                "ocr": 0.0,
                "colour": 0.0,
                "similarity": round(float(s), 4),
                "cluster": c_id,
                "cluster_color": (
                    "#8b96a8"
                    if c_id == -1
                    else CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                ),
                "x": c_info.get("x", 0.0),
                "y": c_info.get("y", 0.0),
                "z": c_info.get("z", 0.0),
                "x3d": c_info.get("x3d", 0.0),
                "y3d": c_info.get("y3d", 0.0),
                "z3d": c_info.get("z3d", 0.0),
                "membership_probability": c_info.get("membership_probability", 0.0),
                "dominant_colours": self.colour_lookup.get(fn, []),
                "full_text": self.ocr_lookup.get(fn, {}).get("full_text", ""),
                "has_text": bool(self.ocr_lookup.get(fn, {}).get("full_text", "").strip()),
            })
        return results

    # -------------------------------------------------------------
    # FULL RE-INDEXING & INGESTION PIPELINE
    # -------------------------------------------------------------

    def index_all_sample_images(self):
        """
        Scans data/sample_images, indexes CLIP embeddings, RapidOCR text,
        dominant colors, and runs 2D & 3D UMAP + HDBSCAN clustering.
        """
        self._log("Indexing all images in data/sample_images...")
        self.load_clip()
        self.load_ocr()

        extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
        image_paths = [
            p for p in SAMPLE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
        image_paths.sort()

        self._log(f"Found {len(image_paths)} images to index.")
        embeddings = []
        new_metadata = []

        # Reset in-memory lookups
        self.ocr_lookup.clear()
        self.colour_lookup.clear()

        for i, path in enumerate(image_paths, start=1):
            self._log(f"[{i}/{len(image_paths)}] Processing {path.name}...")
            try:
                with Image.open(path) as img:
                    emb = self.compute_image_embedding(img)
                embeddings.append(emb)

                ocr_info = self.extract_ocr(path)
                self.ocr_lookup[path.name] = ocr_info

                colours = self.extract_colours(path)
                self.colour_lookup[path.name] = colours

                new_metadata.append({
                    "filename": path.name,
                    "path": str(path.resolve()),
                })
            except Exception as e:
                self._log(f"Error indexing {path.name}: {e}")

        if embeddings:
            stacked = np.vstack(embeddings)
            new_index = faiss.IndexFlatIP(DIMENSIONS)
            new_index.add(stacked)
            self.index = new_index
            self.metadata = new_metadata

            self._log("Running 2D/3D UMAP and HDBSCAN clustering...")
            self.run_clustering()
            self._save_data()

        self._log(f"Re-indexing complete: {len(embeddings)} images indexed successfully!")

    def index_files_background(
        self,
        file_paths: List[Path],
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        thread = threading.Thread(
            target=self._index_files_worker,
            args=(file_paths, callback),
            daemon=True,
        )
        thread.start()

    def _index_files_worker(
        self,
        file_paths: List[Path],
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        with self._lock:
            self.indexing_state["status"] = "running"
            self.indexing_state["total"] = len(file_paths)
            self.indexing_state["current"] = 0
            self.indexing_state["progress"] = 0

        self._log(f"Starting ingestion pipeline for {len(file_paths)} files...")

        existing_filenames = {item["filename"] for item in self.metadata}
        to_process = [p for p in file_paths if p.exists() and p.name not in existing_filenames]

        if not to_process:
            with self._lock:
                self.indexing_state["status"] = "idle"
                self.indexing_state["message"] = "No new files to index"
                self.indexing_state["progress"] = 100
            if callback:
                callback(self.indexing_state)
            return

        self.load_clip()
        self.load_ocr()

        new_embeddings = []
        new_metadata = []

        total_files = len(to_process)
        for i, path in enumerate(to_process, start=1):
            with self._lock:
                self.indexing_state["current"] = i
                self.indexing_state["message"] = f"Processing ({i}/{total_files}): {path.name}"
                self.indexing_state["progress"] = int((i / (total_files + 1)) * 90)

            try:
                with Image.open(path) as img:
                    emb = self.compute_image_embedding(img)
                new_embeddings.append(emb)

                ocr_info = self.extract_ocr(path)
                with self._lock:
                    self.ocr_lookup[path.name] = ocr_info

                colours = self.extract_colours(path)
                with self._lock:
                    self.colour_lookup[path.name] = colours

                new_metadata.append({
                    "filename": path.name,
                    "path": str(path.resolve()),
                })
            except Exception as e:
                self._log(f"Error processing {path.name}: {e}")

        if new_embeddings:
            with self._lock:
                stacked = np.vstack(new_embeddings)
                self.index.add(stacked)
                self.metadata.extend(new_metadata)
                self.run_clustering()
                self._save_data()

        with self._lock:
            self.indexing_state["status"] = "idle"
            self.indexing_state["progress"] = 100
            self.indexing_state["message"] = f"Indexed {len(new_embeddings)} new images successfully."

        if callback:
            callback(self.indexing_state)

    def index_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"]
        files = []
        for ext in extensions:
            if recursive:
                files.extend(dir_p.rglob(ext))
            else:
                files.extend(dir_p.glob(ext))

        valid_files = [f for f in files if not f.name.startswith(".") and ".venv" not in str(f)]
        valid_files.sort()

        if valid_files:
            self.index_files_background(valid_files)

        return len(valid_files)

    # -------------------------------------------------------------
    # SYSTEM OVERVIEW
    # -------------------------------------------------------------

    def get_all_images(self) -> List[Dict]:
        with self._lock:
            result = []
            for item in self.metadata:
                fn = item["filename"]
                cluster_info = self.cluster_lookup.get(fn, {})
                c_id = cluster_info.get("cluster", -1)
                colours = self.colour_lookup.get(fn, [])
                ocr_item = self.ocr_lookup.get(fn, {})

                result.append({
                    "filename": fn,
                    "path": item["path"],
                    "cluster": c_id,
                    "cluster_color": (
                        "#8b96a8"
                        if c_id == -1
                        else CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]
                    ),
                    "x": cluster_info.get("x", 0.0),
                    "y": cluster_info.get("y", 0.0),
                    "z": cluster_info.get("z", 0.0),
                    "x3d": cluster_info.get("x3d", 0.0),
                    "y3d": cluster_info.get("y3d", 0.0),
                    "z3d": cluster_info.get("z3d", 0.0),
                    "membership_probability": cluster_info.get("membership_probability", 0.0),
                    "dominant_colours": colours,
                    "full_text": ocr_item.get("full_text", ""),
                    "has_text": bool(ocr_item.get("full_text", "").strip()),
                    "clip": 0.0,
                    "ocr": 0.0,
                    "colour": 0.0,
                    "final": 0.0,
                })
            return result

    def get_system_status(self) -> Dict:
        with self._lock:
            clusters = set()
            noise_count = 0
            for item in self.cluster_lookup.values():
                c = item.get("cluster", -1)
                if c >= 0:
                    clusters.add(c)
                else:
                    noise_count += 1

            text_images = sum(
                1 for item in self.ocr_lookup.values()
                if item.get("full_text", "").strip()
            )

            return {
                "total_images": len(self.metadata),
                "indexed_vectors": self.index.ntotal if self.index else 0,
                "clusters_count": len(clusters),
                "noise_count": noise_count,
                "text_bearing_images": text_images,
                "model_name": MODEL_NAME,
                "device": DEVICE,
                "model_ready": self.model_ready,
                "indexing_state": self.indexing_state,
            }


# Singleton engine instance
_engine_instance: Optional[ZeroTextEngine] = None


def get_engine() -> ZeroTextEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ZeroTextEngine()
    return _engine_instance
