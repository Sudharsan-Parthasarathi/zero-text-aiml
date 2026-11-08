
import torch
import open_clip
import faiss
import numpy as np
import json
import re
import sys

from pathlib import Path

# ==========================================
# ZERO-TEXT HYBRID SEARCH ENGINE
# ==========================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INDEX_PATH = DATA_DIR / "image_index.faiss"
METADATA_PATH = DATA_DIR / "image_metadata.json"
OCR_PATH = DATA_DIR / "ocr_results.json"
COLOUR_PATH = DATA_DIR / "color_results.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

DEVICE = "cpu"
TOP_K = 5

# Initial experimental weights
CLIP_WEIGHT = 0.55
OCR_WEIGHT = 0.30
COLOUR_WEIGHT = 0.15

# ==========================================
# COLOUR DATABASE
# ==========================================

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
    "brown": (130, 80, 40)
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "of",
    "in", "on", "with", "find", "show",
    "me", "image", "picture", "containing",
    "that", "has", "having", "and", "for"
}

# ==========================================
# LOAD JSON FILES
# ==========================================

def load_json(path):

    if not path.exists():
        print("WARNING: Missing file:", path.name)
        return []

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


print("=" * 60)
print("          ZERO-TEXT HYBRID SEARCH ENGINE")
print("=" * 60)

# ==========================================
# LOAD FAISS INDEX
# ==========================================

print("\nLoading FAISS index...")

if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise SystemExit("FAISS index or metadata missing!")

index = faiss.read_index(str(INDEX_PATH))

metadata = load_json(METADATA_PATH)
ocr_data = load_json(OCR_PATH)
colour_data = load_json(COLOUR_PATH)

if index.ntotal != len(metadata):
    raise SystemExit("FAISS and metadata mismatch!")

if index.ntotal == 0:
    raise SystemExit("No indexed images!")

print("Indexed images:", index.ntotal)

# Create filename lookup dictionaries

ocr_lookup = {
    item["filename"]: item.get("full_text", "")
    for item in ocr_data
}

colour_lookup = {
    item["filename"]: item.get("dominant_colours", [])
    for item in colour_data
}

# ==========================================
# LOAD CLIP MODEL
# ==========================================

print("\nLoading CLIP model...")

model, _, _ = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=PRETRAINED,
    device=DEVICE
)

model.eval()

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("CLIP loaded successfully!")

# ==========================================
# QUERY ANALYSIS
# ==========================================

def analyse_query(query):

    words = re.findall(r"[a-z0-9]+", query.lower())

    detected_colours = [
        word for word in words
        if word in COLOURS
    ]

    text_words = [
        word for word in words
        if word not in STOP_WORDS
        and word not in COLOURS
        and len(word) > 2
    ]

    return text_words, detected_colours


# ==========================================
# CLIP SEMANTIC SEARCH
# ==========================================

def get_clip_scores(query):

    tokens = tokenizer([query]).to(DEVICE)

    with torch.inference_mode():
        features = model.encode_text(tokens)

    features = features / features.norm(
        dim=-1,
        keepdim=True
    )

    vector = features.cpu().numpy().astype("float32")

    scores, indices = index.search(
        vector,
        index.ntotal
    )

    clip_scores = np.zeros(index.ntotal)

    for score, idx in zip(scores[0], indices[0]):

        if idx >= 0:
            # Convert cosine similarity to 0-1 range
            clip_scores[idx] = (float(score) + 1) / 2

    return clip_scores


# ==========================================
# OCR TEXT MATCHING
# ==========================================

def get_ocr_score(filename, query_words):

    if not query_words:
        return 0.0

    extracted_text = ocr_lookup.get(
        filename, ""
    ).lower()

    image_words = set(
        re.findall(r"[a-z0-9]+", extracted_text)
    )

    matched = sum(
        1 for word in set(query_words)
        if word in image_words
    )

    return matched / len(set(query_words))


# ==========================================
# COLOUR SIMILARITY
# ==========================================

def get_colour_score(filename, query_colours):

    if not query_colours:
        return 0.0

    image_colours = colour_lookup.get(
        filename, []
    )

    if not image_colours:
        return 0.0

    total_weight = sum(
        max(0, float(item.get("percentage", 0)))
        for item in image_colours
    )

    if total_weight == 0:
        return 0.0

    results = []

    for colour_name in query_colours:

        target = np.array(
            COLOURS[colour_name],
            dtype=float
        )

        similarity = 0.0

        for item in image_colours:

            rgb = np.array(
                item["rgb"],
                dtype=float
            )

            distance = np.linalg.norm(
                target - rgb
            )

            max_distance = np.sqrt(3 * 255 ** 2)

            current_similarity = (
                1 - distance / max_distance
            )

            weight = float(
                item.get("percentage", 0)
            ) / total_weight

            similarity += current_similarity * weight

        results.append(similarity)

    return float(np.mean(results))


# ==========================================
# HYBRID SCORING
# ==========================================

def hybrid_search(query):

    print("\n" + "=" * 60)
    print("SEARCH QUERY:", query)
    print("=" * 60)

    query_words, query_colours = analyse_query(query)

    print("\nDetected text keywords:", query_words)
    print("Detected colours:", query_colours)

    clip_scores = get_clip_scores(query)

    results = []

    for i, item in enumerate(metadata):

        filename = item["filename"]

        clip_score = float(clip_scores[i])

        ocr_score = get_ocr_score(
            filename,
            query_words
        )

        colour_score = get_colour_score(
            filename,
            query_colours
        )

        # Activate only relevant search components
        active_weights = [
            (CLIP_WEIGHT, clip_score),
            (OCR_WEIGHT, ocr_score)
            if query_words else None,
            (COLOUR_WEIGHT, colour_score)
            if query_colours else None
        ]

        active_weights = [
            item for item in active_weights
            if item is not None
        ]

        total_weight = sum(
            weight for weight, score in active_weights
        )

        final_score = sum(
            weight * score
            for weight, score in active_weights
        ) / total_weight

        results.append({
            "filename": filename,
            "path": item["path"],
            "clip": clip_score,
            "ocr": ocr_score,
            "colour": colour_score,
            "final": final_score
        })

    results.sort(
        key=lambda item: item["final"],
        reverse=True
    )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\nHYBRID SEARCH RESULTS")
    print("-" * 60)

    for rank, result in enumerate(
        results[:TOP_K], start=1
    ):

        print(f"\nRANK: {rank}")
        print("Image:", result["filename"])

        print(
            "Final score:",
            round(result["final"] * 100, 2), "%"
        )

        print(
            "CLIP:",
            round(result["clip"] * 100, 2), "%"
        )

        print(
            "OCR:",
            round(result["ocr"] * 100, 2), "%"
        )

        print(
            "Colour:",
            round(result["colour"] * 100, 2), "%"
        )

        print("Path:", result["path"])

    print("\nSearch completed!")


# ==========================================
# INTERACTIVE SEARCH
# ==========================================

print("\n" + "=" * 60)
print("ZERO-TEXT HYBRID SEARCH IS READY!")
print("=" * 60)

print("Type 'exit' to close.")

while True:

    query = input("\nSearch: ").strip()

    if query.lower() == "exit":
        print("Closing ZERO-TEXT.")
        break

    if not query:
        continue

    try:
        hybrid_search(query)

    except Exception as error:
        print("Search error:", error)