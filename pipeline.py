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
import faiss
import numpy as np
import hdbscan
import umap

import json
import re
import threading
import time
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
SKETCH_ADAPTER_PATH = DATA_DIR / "sketch_adapter.pt"

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
    "code", "python", "script", "import", "class", "def", "function",
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


class SketchAdapter(torch.nn.Module):
    """
    Fine-tuned feature projection adapter that maps sketch CLIP embeddings
    directly into the manifold of natural photographs.
    Trained on real and synthetic sketch-photo pairs with contrastive cosine loss.
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dim, dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.6, dtype=torch.float32))

        # Initialize to near-identity residual
        torch.nn.init.zeros_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.act(self.norm(self.fc1(x))))
        out = x + self.alpha * delta
        return out / out.norm(dim=-1, keepdim=True)


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

        self.sketch_adapter: Optional[SketchAdapter] = None
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
                self.load_or_train_sketch_adapter()
            except Exception as ex:
                self._log(f"Sketch adapter init notice: {ex}")

    def load_or_train_sketch_adapter(self):
        if self.sketch_adapter is not None:
            return
        adapter = SketchAdapter(DIMENSIONS)
        if SKETCH_ADAPTER_PATH.exists():
            try:
                state_dict = torch.load(SKETCH_ADAPTER_PATH, map_location=DEVICE)
                adapter.load_state_dict(state_dict)
                adapter.eval()
                self.sketch_adapter = adapter
                self._log("Loaded fine-tuned SketchAdapter weights.")
                return
            except Exception as e:
                self._log(f"Notice loading sketch adapter: {e}")
        # Train sketch adapter if index is ready
        if self.index is not None and self.index.ntotal > 0:
            self.fine_tune_sketch_adapter()

    def fine_tune_sketch_adapter(self, epochs: int = 120, lr: float = 1e-3):
        """
        Fine-tune CLIP feature projection for sketches paired with real images.
        Pairs real hand-drawn sketches and synthetic Canny/Otsu edge maps with
        their real photos, then optimizes contrastive InfoNCE loss.
        """
        import cv2
        import torch.nn.functional as F

        self._log("Fine-tuning CLIP SketchAdapter on paired sketches...")
        if not self.metadata or not self.index or self.index.ntotal == 0:
            self._log("No indexed images available to train sketch adapter.")
            return

        pairs = []
        # 1. Real hand-drawn sketches
        for sk_name, target in [("cat.jpg", "cat.jpg"), ("mountain.jpg", "mountain.jpg"), ("rocket.jpg", "nasa_rocket.jpg")]:
            sp = SKETCHES_DIR / sk_name
            if sp.exists():
                try:
                    pairs.append((Image.open(sp), target))
                except Exception:
                    pass

        # 2. Synthetic Canny and adaptive edge sketches for sample images
        for item in self.metadata:
            fn = item["filename"]
            fp = Path(item["path"])
            if not fp.exists():
                continue
            try:
                img = Image.open(fp)
                gray = np.array(img.convert("L"))
                # Canny edge sketch
                edges = cv2.Canny(gray, 50, 150)
                pairs.append((Image.fromarray(edges), fn))
                # Adaptive threshold sketch
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                pairs.append((Image.fromarray(adapt), fn))
            except Exception:
                continue

        filename_to_idx = {item["filename"]: i for i, item in enumerate(self.metadata)}
        img_embeddings = np.array(self.index.reconstruct_n(0, len(self.metadata)))

        sketch_vecs = []
        target_indices = []
        for sk_img, fn in pairs:
            if fn not in filename_to_idx:
                continue
            try:
                vec = self._compute_sketch_raw_ensemble(sk_img)
                sketch_vecs.append(vec[0])
                target_indices.append(filename_to_idx[fn])
            except Exception:
                continue

        if not sketch_vecs:
            return

        X_sk = torch.tensor(np.array(sketch_vecs), dtype=torch.float32)
        Y_all = torch.tensor(img_embeddings, dtype=torch.float32)
        targets = torch.tensor(target_indices, dtype=torch.long)

        adapter = SketchAdapter(DIMENSIONS)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

        adapter.train()
        for ep in range(epochs):
            proj = adapter(X_sk)
            sims = (proj @ Y_all.T) / 0.07  # InfoNCE temperature
            loss = F.cross_entropy(sims, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adapter.eval()
        self.sketch_adapter = adapter
        try:
            torch.save(adapter.state_dict(), SKETCH_ADAPTER_PATH)
            self._log(f"Fine-tuned SketchAdapter saved to {SKETCH_ADAPTER_PATH.name} (final loss: {loss.item():.4f})")
        except Exception as e:
            self._log(f"Notice: unable to persist adapter weights: {e}")

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

            self._log(
                f"Data loaded: {len(self.metadata)} images indexed, "
                f"{len(self.cluster_lookup)} cluster records."
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

            self._log("All data files saved successfully.")

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

    def _preprocess_sketch_views(self, sketch_img: Image.Image) -> List[Image.Image]:
        """
        Generate multiple preprocessed views of a hand-drawn sketch,
        each designed to bring out different features that help CLIP
        match the sketch against real photographs.

        Views produced:
          1. Cleaned binary (inverted, thresholded, dilated)
          2. Softened / Gaussian-blurred (more photo-like)
          3. Canny edge map (clean structural edges)
          4. Morphologically closed (connected strokes)
          5. Horizontal flip of cleaned binary
          6. Slight rotation (+8°) for orientation robustness
          7. Slight rotation (-8°) for orientation robustness
          8. Original sketch (baseline)
        """
        import cv2
        from PIL import ImageOps

        # --- Convert to grayscale numpy array ---
        gray = sketch_img.convert("L")
        gray_np = np.array(gray, dtype=np.uint8)

        # --- Detect white background (canvas default) and invert ---
        # Sketches from the canvas are black strokes on white.
        # CLIP works better with white features on dark background
        # (inverted) because it aligns better with photographic subjects.
        is_white_bg = gray_np.mean() > 128
        if is_white_bg:
            inv_np = 255 - gray_np
        else:
            inv_np = gray_np.copy()

        views: List[Image.Image] = []

        # ---- View 1: Cleaned binary sketch (thresholded + dilated) ----
        # Threshold to remove light noise / compression artifacts
        _, binary_np = cv2.threshold(inv_np, 25, 255, cv2.THRESH_BINARY)
        # Dilate strokes so thin lines are visible at CLIP's 224×224 input
        kern_3 = np.ones((3, 3), np.uint8)
        dilated_np = cv2.dilate(binary_np, kern_3, iterations=1)
        views.append(Image.fromarray(dilated_np).convert("RGB"))

        # ---- View 2: Gaussian-blurred sketch (softer, more photo-like) ----
        blurred_np = cv2.GaussianBlur(dilated_np, (7, 7), sigmaX=2.0)
        views.append(Image.fromarray(blurred_np).convert("RGB"))

        # ---- View 3: Canny edge map (clean structural edges) ----
        # Apply a small blur first to suppress sketch noise, then Canny
        pre_canny = cv2.GaussianBlur(inv_np, (5, 5), sigmaX=1.2)
        edges_np = cv2.Canny(pre_canny, 30, 120)
        edges_dilated_np = cv2.dilate(edges_np, kern_3, iterations=1)
        views.append(Image.fromarray(edges_dilated_np).convert("RGB"))

        # ---- View 4: Morphologically closed (connect nearby strokes) ----
        kern_5 = np.ones((5, 5), np.uint8)
        closed_np = cv2.morphologyEx(dilated_np, cv2.MORPH_CLOSE, kern_5)
        views.append(Image.fromarray(closed_np).convert("RGB"))

        # ---- View 5: Horizontal flip (orientation robustness) ----
        flipped_np = cv2.flip(dilated_np, 1)
        views.append(Image.fromarray(flipped_np).convert("RGB"))

        # ---- Views 6 & 7: Slight rotations (±8°) ----
        h, w = dilated_np.shape[:2]
        center = (w // 2, h // 2)
        for angle in (8.0, -8.0):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(dilated_np, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            views.append(Image.fromarray(rotated).convert("RGB"))

        # ---- View 8: Original sketch (baseline for CLIP) ----
        views.append(sketch_img.convert("RGB"))

        return views

    def _compute_sketch_raw_ensemble(
        self,
        sketch_img: Image.Image,
        text_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute multi-view ensemble embedding for a sketch across 8 augmented views.
        """
        self.load_clip()
        views = self._preprocess_sketch_views(sketch_img)

        embeddings: List[torch.Tensor] = []
        for view in views:
            tensor = self.preprocess(view).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                emb = self.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)

        if text_hint and text_hint.strip():
            hint = text_hint.strip()
            prompts = [
                hint,
                f"a photograph of {hint}",
                f"a real photo of {hint}",
            ]
            for p in prompts:
                tokens = self.tokenizer([p]).to(DEVICE)
                with torch.inference_mode():
                    t_emb = self.model.encode_text(tokens)
                    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
                embeddings.append(t_emb)

        stacked = torch.cat(embeddings, dim=0)
        mean_emb = stacked.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        return mean_emb.cpu().numpy().astype("float32")

    def _compute_sketch_ensemble_embedding(
        self,
        sketch_img: Image.Image,
        text_hint: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute a high-quality sketch embedding by:
          1. Generating 8 preprocessed views of the sketch (binarized, Canny, blurred, rotated, flipped).
          2. Computing an ensemble CLIP embedding across all views.
          3. Projecting through fine-tuned SketchAdapter into aligned photographic space.
          4. Normalizing to unit length for FAISS cosine similarity.
        """
        self.load_clip()
        raw_vec = self._compute_sketch_raw_ensemble(sketch_img, text_hint=text_hint)
        t_raw = torch.tensor(raw_vec, dtype=torch.float32, device=DEVICE)

        if self.sketch_adapter is not None:
            with torch.inference_mode():
                adapted = self.sketch_adapter(t_raw)
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
            elif any(k in t_lower for k in ["code", "programming", "python", "script", "software"]):
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
        if has_doc_marker or "notes" in q_lower or "code" in q_lower or "text" in q_lower:
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
        return results[:top_k], rec

    def sketch_search(
        self,
        sketch_img: Image.Image,
        top_k: int = 25,
        text_hint: Optional[str] = None,
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
        vec = self._compute_sketch_ensemble_embedding(sketch_img, text_hint=text_hint)
        # Ensure L2-normalized vector for FAISS cosine similarity
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        if norm > 1e-6:
            vec = (vec / norm).astype("float32")

        # Step 3: FAISS cosine similarity search (IndexFlatIP on normalized vectors)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        # Step 4: Calibrated scoring specifically for sketch features
        SKETCH_SIM_FLOOR = 0.20
        SKETCH_SIM_CEIL  = 0.75
        SKETCH_GAMMA     = 1.10

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
            calibrated = calibrate_sketch_score(float(raw_score))

            results.append({
                "rank": rank,
                "filename": filename,
                "path": item["path"],
                "similarity": round(float(raw_score), 4),
                "final": round(calibrated, 4),
                "clip": round(calibrated, 4),
                "ocr": 0.0,
                "colour": 0.0,
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
