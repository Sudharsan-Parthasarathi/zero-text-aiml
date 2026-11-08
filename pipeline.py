"""
ZERO-TEXT CORE PIPELINE & AI ENGINE
Integrates:
- CLIP (ViT-B-32) Multimodal Embeddings (Text, Image, Sketch)
- RapidOCR In-Image Text Extraction
- Dominant Colour Extraction (Pillow Median Cut)
- FAISS Vector Index (512-dim Cosine Similarity)
- UMAP 2D & 3D Dimensionality Reduction
- HDBSCAN Density-Based Hierarchical Clustering
- Multi-Directory and File Ingestion Pipeline
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import faiss
import hdbscan
import numpy as np
import open_clip
from PIL import Image
import torch
import umap

# Enable UTF-8 terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(BASE_DIR / "model_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample_images"
SKETCHES_DIR = DATA_DIR / "sketches"

INDEX_PATH = DATA_DIR / "image_index.faiss"
METADATA_PATH = DATA_DIR / "image_metadata.json"
OCR_PATH = DATA_DIR / "ocr_results.json"
COLOUR_PATH = DATA_DIR / "color_results.json"
CLUSTER_PATH = DATA_DIR / "cluster_results.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DEVICE = "cpu"
DIMENSIONS = 512
COLOUR_COUNT = 5

COLOURS = {
    "red": (220, 40, 40),
    "green": (40, 160, 70),
    "blue": (40, 90, 210),
    "yellow": (240, 210, 40),
    "orange": (240, 130, 30),
    "purple": (130, 60, 180),
    "pink": (240, 130, 180),
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "brown": (130, 80, 40),
    "cyan": (40, 210, 220),
    "magenta": (220, 40, 180),
    "navy": (15, 30, 80),
    "teal": (20, 130, 130),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "violet": (138, 43, 226),
    "gold": (255, 215, 0),
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "with", "find", "show",
    "me", "image", "picture", "containing", "that", "has", "having", "and",
    "for", "to", "at", "from", "by", "about", "all", "some", "any", "photo",
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
]


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


class ZeroTextEngine:
    """
    Unified ZeroText Engine handling multi-modal indexing, searching,
    and 2D/3D visual clustering.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.ocr = None

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.ocr_lookup: Dict[str, Dict] = {}
        self.colour_lookup: Dict[str, List[Dict]] = {}
        self.cluster_lookup: Dict[str, Dict] = {}

        self.indexing_state = {
            "status": "idle",
            "progress": 0,
            "message": "Engine ready",
            "total": 0,
            "current": 0,
            "logs": [],
        }

        self._ensure_directories()
        self._load_data()

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
    # MODEL INITIALIZATION (LAZY LOADED)
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
            self._log(f"CLIP loaded in {time.time() - start:.2f}s")

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
            # Metadata
            if METADATA_PATH.exists():
                try:
                    with open(METADATA_PATH, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    self._log(f"Failed to load metadata: {e}")
                    self.metadata = []
            else:
                self.metadata = []

            # FAISS Index
            if INDEX_PATH.exists():
                try:
                    self.index = faiss.read_index(str(INDEX_PATH))
                except Exception as e:
                    self._log(f"Failed to load FAISS index: {e}")
                    self.index = faiss.IndexFlatIP(DIMENSIONS)
            else:
                self.index = faiss.IndexFlatIP(DIMENSIONS)

            # OCR Data
            if OCR_PATH.exists():
                try:
                    with open(OCR_PATH, "r", encoding="utf-8") as f:
                        ocr_data = json.load(f)
                        self.ocr_lookup = {item["filename"]: item for item in ocr_data}
                except Exception as e:
                    self._log(f"Failed to load OCR data: {e}")
                    self.ocr_lookup = {}
            else:
                self.ocr_lookup = {}

            # Colour Data
            if COLOUR_PATH.exists():
                try:
                    with open(COLOUR_PATH, "r", encoding="utf-8") as f:
                        colour_data = json.load(f)
                        self.colour_lookup = {
                            item["filename"]: item.get("dominant_colours", [])
                            for item in colour_data
                        }
                except Exception as e:
                    self._log(f"Failed to load colour data: {e}")
                    self.colour_lookup = {}
            else:
                self.colour_lookup = {}

            # Cluster Data
            if CLUSTER_PATH.exists():
                try:
                    with open(CLUSTER_PATH, "r", encoding="utf-8") as f:
                        cluster_data = json.load(f)
                        self.cluster_lookup = {
                            item["filename"]: item for item in cluster_data
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
    # FEATURE EXTRACTION (OCR, COLOURS, EMBEDDING)
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
            result, elapsed = self.ocr(str(image_path))
            extracted_items = []
            if result:
                for item in result:
                    text = str(item[1]).strip()
                    conf = float(item[2])
                    if text:
                        extracted_items.append({"text": text, "confidence": round(conf, 4)})

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

    def compute_text_embedding(self, text: str) -> np.ndarray:
        self.load_clip()
        tokens = self.tokenizer([text]).to(DEVICE)
        with torch.inference_mode():
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype("float32")

    # -------------------------------------------------------------
    # CLUSTERING & DIMENSIONALITY REDUCTION (UMAP 2D & 3D + HDBSCAN)
    # -------------------------------------------------------------

    def run_clustering(self) -> List[Dict]:
        """
        Runs HDBSCAN and UMAP (both 2D and 3D) on all embeddings currently
        in the FAISS index.
        """
        with self._lock:
            n_items = len(self.metadata)
            if n_items < 2:
                self._log("Not enough images to cluster (minimum 2 needed).")
                # Default coordinates for 1 image
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

            # HDBSCAN
            min_cluster = 2 if n_items >= 3 else 2
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster,
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            cluster_labels = clusterer.fit_predict(embeddings)

            # UMAP 2D
            n_neighbors = min(5, max(2, n_items - 1))
            reducer_2d = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            coords_2d = reducer_2d.fit_transform(embeddings)

            # UMAP 3D (if enough items, else project)
            if n_items >= 4:
                reducer_3d = umap.UMAP(
                    n_components=3,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=42,
                )
                coords_3d = reducer_3d.fit_transform(embeddings)
            else:
                # Synthetic z coordinate based on embedding norm / cluster
                coords_3d = np.zeros((n_items, 3), dtype=np.float32)
                coords_3d[:, 0] = coords_2d[:, 0]
                coords_3d[:, 1] = coords_2d[:, 1]
                coords_3d[:, 2] = np.random.RandomState(42).uniform(-2.0, 2.0, size=n_items)

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

    # -------------------------------------------------------------
    # MULTI-MODAL SEARCH ENGINES
    # -------------------------------------------------------------

    def analyse_query(self, query: str) -> Tuple[List[str], List[str]]:
        words = re.findall(r"[a-z0-9]+", query.lower())
        detected_colours = [w for w in words if w in COLOURS]
        text_words = [
            w for w in words
            if w not in STOP_WORDS and w not in COLOURS and len(w) > 2
        ]
        return text_words, detected_colours

    def get_ocr_score(self, filename: str, query_words: List[str]) -> float:
        if not query_words:
            return 0.0
        ocr_item = self.ocr_lookup.get(filename, {})
        full_text = ocr_item.get("full_text", "").lower()
        if not full_text:
            return 0.0
        image_words = set(re.findall(r"[a-z0-9]+", full_text))
        matched = sum(1 for w in set(query_words) if w in image_words)
        return matched / len(set(query_words))

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
                sim = 1.0 - (dist / max_dist)
                w = float(item.get("percentage", 0)) / total_weight
                col_score += sim * w
            scores.append(col_score)
        return float(np.mean(scores)) if scores else 0.0

    def hybrid_search(
        self,
        query: str,
        clip_weight: float = 0.55,
        ocr_weight: float = 0.30,
        colour_weight: float = 0.15,
        top_k: int = 50,
        cluster_filter: Optional[int] = None,
    ) -> List[Dict]:
        """
        Performs hybrid multi-modal search combining CLIP semantic vector similarity,
        OCR extracted word matching, and dominant color similarity.
        """
        if not self.metadata or self.index.ntotal == 0:
            return []

        query = query.strip()
        query_words, query_colours = self.analyse_query(query)

        # 1. CLIP Search
        text_vec = self.compute_text_embedding(query)
        scores, indices = self.index.search(text_vec, self.index.ntotal)
        clip_scores = np.zeros(self.index.ntotal, dtype=np.float32)
        for s, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                clip_scores[idx] = max(0.0, min(1.0, (float(s) + 1.0) / 2.0))

        results = []
        for i, item in enumerate(self.metadata):
            filename = item["filename"]
            cluster_info = self.cluster_lookup.get(filename, {})
            c_id = cluster_info.get("cluster", -1)

            if cluster_filter is not None and c_id != cluster_filter:
                continue

            clip_s = float(clip_scores[i])
            ocr_s = self.get_ocr_score(filename, query_words)
            col_s = self.get_colour_score(filename, query_colours)

            # Active components weighting
            active = [(clip_weight, clip_s)]
            if query_words:
                active.append((ocr_weight, ocr_s))
            if query_colours:
                active.append((colour_weight, col_s))

            tot_w = sum(w for w, _ in active) or 1.0
            final_s = sum(w * s for w, s in active) / tot_w

            ocr_item = self.ocr_lookup.get(filename, {})
            colours = self.colour_lookup.get(filename, [])

            results.append({
                "filename": filename,
                "path": item["path"],
                "cluster": c_id,
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "clip": round(clip_s, 4),
                "ocr": round(ocr_s, 4),
                "colour": round(col_s, 4),
                "final": round(final_s, 4),
                "dominant_colours": colours,
                "full_text": ocr_item.get("full_text", ""),
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        return results[:top_k]

    def sketch_search(self, sketch_img: Image.Image, top_k: int = 20) -> List[Dict]:
        """
        Uses CLIP to embed a user sketch and perform nearest neighbor search
        against indexed photographic images.
        """
        if not self.metadata or self.index.ntotal == 0:
            return []

        vec = self.compute_image_embedding(sketch_img)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[idx]
            filename = item["filename"]
            cluster_info = self.cluster_lookup.get(filename, {})
            norm_score = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))

            results.append({
                "rank": rank,
                "filename": filename,
                "path": item["path"],
                "similarity": round(float(score), 4),
                "final": round(norm_score, 4),
                "cluster": cluster_info.get("cluster", -1),
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "dominant_colours": self.colour_lookup.get(filename, []),
                "full_text": self.ocr_lookup.get(filename, {}).get("full_text", ""),
            })

        return results

    def color_search(self, target_hex_or_rgb: Union[str, List[int]], top_k: int = 20) -> List[Dict]:
        """
        Searches images closest to a specific target color.
        """
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
                    sim += (1.0 - dist / max_dist) * w

            cluster_info = self.cluster_lookup.get(filename, {})
            results.append({
                "filename": filename,
                "path": item["path"],
                "final": round(sim, 4),
                "colour": round(sim, 4),
                "clip": 0.0,
                "ocr": 0.0,
                "cluster": cluster_info.get("cluster", -1),
                "x": cluster_info.get("x", 0.0),
                "y": cluster_info.get("y", 0.0),
                "z": cluster_info.get("z", 0.0),
                "x3d": cluster_info.get("x3d", 0.0),
                "y3d": cluster_info.get("y3d", 0.0),
                "z3d": cluster_info.get("z3d", 0.0),
                "membership_probability": cluster_info.get("membership_probability", 0.0),
                "dominant_colours": dominant,
                "full_text": self.ocr_lookup.get(filename, {}).get("full_text", ""),
            })

        results.sort(key=lambda x: x["final"], reverse=True)
        return results[:top_k]

    def similar_images(self, filename: str, top_k: int = 10) -> List[Dict]:
        """
        Finds images closest in visual semantic space to a given indexed image.
        """
        # Find index
        idx = -1
        for i, item in enumerate(self.metadata):
            if item["filename"] == filename:
                idx = i
                break
        if idx < 0:
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
            norm_s = max(0.0, min(1.0, (float(s) + 1.0) / 2.0))
            results.append({
                "filename": fn,
                "path": item["path"],
                "final": round(norm_s, 4),
                "similarity": round(float(s), 4),
                "cluster": c_info.get("cluster", -1),
                "x": c_info.get("x", 0.0),
                "y": c_info.get("y", 0.0),
                "z": c_info.get("z", 0.0),
                "dominant_colours": self.colour_lookup.get(fn, []),
                "full_text": self.ocr_lookup.get(fn, {}).get("full_text", ""),
            })
        return results

    # -------------------------------------------------------------
    # INGESTION & INDEXING PIPELINE (FILES & DIRECTORIES)
    # -------------------------------------------------------------

    def index_files_background(
        self,
        file_paths: List[Path],
        callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Spawns background worker to index files without blocking the main server.
        """
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

        # Filter out existing by filename or exact path
        existing_filenames = {item["filename"] for item in self.metadata}
        to_process = []
        for p in file_paths:
            if not p.exists():
                self._log(f"File not found: {p}")
                continue
            if p.name in existing_filenames:
                self._log(f"Skipping already indexed file: {p.name}")
                continue
            to_process.append(p)

        if not to_process:
            self._log("No new images to index.")
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

            self._log(f"[{i}/{total_files}] Processing {path.name}...")

            try:
                # 1. CLIP Embedding
                with Image.open(path) as img:
                    emb = self.compute_image_embedding(img)
                new_embeddings.append(emb)

                # 2. OCR Text Extraction
                ocr_info = self.extract_ocr(path)
                with self._lock:
                    self.ocr_lookup[path.name] = ocr_info

                # 3. Dominant Colours
                colours = self.extract_colours(path)
                with self._lock:
                    self.colour_lookup[path.name] = colours

                # 4. Metadata
                new_meta = {
                    "filename": path.name,
                    "path": str(path.resolve()),
                }
                new_metadata.append(new_meta)

            except Exception as e:
                self._log(f"Error processing {path.name}: {e}")

        if new_embeddings:
            with self._lock:
                stacked_emb = np.vstack(new_embeddings)
                self.index.add(stacked_emb)
                self.metadata.extend(new_metadata)

            # Re-run clustering with all data
            with self._lock:
                self.indexing_state["message"] = "Running UMAP 2D/3D & HDBSCAN clustering..."
            self._log("Updating 2D/3D UMAP galaxy coordinates and HDBSCAN clusters...")
            self.run_clustering()

            # Save state
            self._save_data()

        with self._lock:
            self.indexing_state["status"] = "idle"
            self.indexing_state["progress"] = 100
            self.indexing_state["message"] = f"Indexed {len(new_embeddings)} new images successfully."

        self._log(f"Ingestion complete: {len(new_embeddings)} new images added.")
        if callback:
            callback(self.indexing_state)

    def index_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        """
        Discovers image files in directory and triggers background indexing.
        """
        dir_p = Path(directory_path)
        if not dir_p.exists() or not dir_p.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []
        for ext in extensions:
            if recursive:
                files.extend(dir_p.rglob(ext))
            else:
                files.extend(dir_p.glob(ext))

        # Filter out hidden or cache files
        valid_files = [f for f in files if not f.name.startswith(".") and not ".venv" in str(f)]
        valid_files.sort()

        if valid_files:
            self.index_files_background(valid_files)

        return len(valid_files)

    # -------------------------------------------------------------
    # SYSTEM OVERVIEW
    # -------------------------------------------------------------

    def get_all_images(self) -> List[Dict]:
        """
        Returns rich metadata for every indexed image.
        """
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
                "indexing_state": self.indexing_state,
            }


# Singleton engine instance
_engine_instance: Optional[ZeroTextEngine] = None


def get_engine() -> ZeroTextEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ZeroTextEngine()
    return _engine_instance
