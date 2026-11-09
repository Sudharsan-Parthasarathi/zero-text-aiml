
import torch
import open_clip
import faiss
import numpy as np

from PIL import Image
from pathlib import Path
import sys

# ==========================================
# ZERO-TEXT: SKETCH-TO-IMAGE SEARCH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "data" / "image_index.faiss"
METADATA_PATH = BASE_DIR / "data" / "image_metadata.json"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DIMENSIONS = 512

TOP_K = 5

print("=" * 55)
print("       ZERO-TEXT SKETCH SEARCH ENGINE")
print("=" * 55)

# ==========================================
# 1. LOAD CLIP MODEL
# ==========================================

print("\nLoading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE
)

model.eval()

print("CLIP model loaded successfully!")

# ==========================================
# 2. LOAD EXISTING FAISS INDEX
# ==========================================

print("\nLoading existing FAISS index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError("FAISS index not found!")

if not METADATA_PATH.exists():
    raise FileNotFoundError("Image metadata not found!")

index = faiss.read_index(str(INDEX_PATH))

import json

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if index.d != DIMENSIONS:
    raise ValueError("Embedding dimension mismatch!")

if index.ntotal != len(metadata):
    raise ValueError("Index and metadata mismatch!")

print("FAISS index loaded!")
print("Indexed images:", index.ntotal)

# ==========================================
# 3. GENERATE SKETCH EMBEDDING
# ==========================================

def generate_sketch_embedding(sketch_path, augment=False):
    """Generate a CLIP embedding for a sketch.
    If *augment* is True, also compute a horizontally flipped version and
    average the two normalized embeddings – this often improves robustness
    for hand‑drawn sketches where left/right orientation can vary.
    """
    sketch_path = Path(sketch_path)
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch not found: {sketch_path}")
    print("\nProcessing sketch:", sketch_path.name)
    image = Image.open(sketch_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        embedding = model.encode_image(image_tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    # Augmentation: horizontal flip
    if augment:
        flipped = Image.open(sketch_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGB")
        flipped_tensor = preprocess(flipped).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            flipped_emb = model.encode_image(flipped_tensor)
        flipped_emb = flipped_emb / flipped_emb.norm(dim=-1, keepdim=True)
        embedding = (embedding + flipped_emb) / 2.0
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    vector = embedding.cpu().numpy().astype("float32")
    print("Sketch embedding generated!")
    print("Embedding shape:", vector.shape)
    return vector

    sketch_path = Path(sketch_path)

    if not sketch_path.exists():
        raise FileNotFoundError(
            f"Sketch not found: {sketch_path}"
        )

    print("\nProcessing sketch:", sketch_path.name)

    image = Image.open(sketch_path).convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():

        embedding = model.encode_image(image_tensor)

    embedding = embedding / embedding.norm(
        dim=-1,
        keepdim=True
    )

    vector = embedding.cpu().numpy().astype("float32")

    print("Sketch embedding generated!")
    print("Embedding shape:", vector.shape)

    return vector

# ==========================================
# 4. SEARCH SIMILAR IMAGES
# ==========================================

def search_sketch(sketch_path, top_k=TOP_K, augment=False):
    """Search for images similar to a sketch.
    *augment* – if True, the embedding is averaged with a horizontally
    flipped version of the sketch (see ``generate_sketch_embedding``).
    """
    query_vector = generate_sketch_embedding(sketch_path, augment=augment)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, k)
    print("\n" + "=" * 55)
    print("          SKETCH SEARCH RESULTS")
    print("=" * 55)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        result = metadata[idx]
        print(f"\nRank: {rank}")
        print("Image:", result["filename"])
        print("Similarity:", f"{score * 100:.2f}%")
        print("Path:", result["path"])
    print("\nSketch search completed!")

# Duplicate block removed

# ==========================================
# 5. MAIN PROGRAM
# ==========================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print(
        'python sketch_search.py "path/to/sketch.png"'
    )

    raise SystemExit

sketch_path = sys.argv[1]

search_sketch(sketch_path)

print("\nSUCCESS! ZERO-TEXT SKETCH SEARCH FINISHED.")