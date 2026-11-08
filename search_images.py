
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
from pathlib import Path
import json
import time

# ==========================================
# ZERO-TEXT: FAISS SEMANTIC IMAGE SEARCH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "data" / "sample_images"
INDEX_PATH = BASE_DIR / "data" / "image_index.faiss"
METADATA_PATH = BASE_DIR / "data" / "image_metadata.json"

DEVICE = "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
DIMENSIONS = 512

print("=" * 55)
print("       ZERO-TEXT SEMANTIC IMAGE SEARCH")
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

tokenizer = open_clip.get_tokenizer(MODEL_NAME)

print("CLIP model loaded!")

# ==========================================
# 2. BUILD FAISS INDEX
# ==========================================

def build_index():

    print("\nScanning image directory...")

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

    image_paths = []

    for ext in extensions:
        image_paths.extend(IMAGE_DIR.glob(ext))

    image_paths = sorted(image_paths)

    if not image_paths:
        print("No images found!")
        return False

    print("Found", len(image_paths), "images.")

    image_tensors = []
    metadata = []

    for path in image_paths:

        try:
            image = Image.open(path).convert("RGB")

            image_tensor = preprocess(image)

            image_tensors.append(image_tensor)

            metadata.append({
                "filename": path.name,
                "path": str(path)
            })

            print("Processed:", path.name)

        except Exception as error:
            print("Skipping:", path.name, error)

    if not image_tensors:
        print("No valid images!")
        return False

    print("\nGenerating image embeddings...")

    batch = torch.stack(image_tensors).to(DEVICE)

    with torch.inference_mode():
        embeddings = model.encode_image(batch)

    embeddings = embeddings / embeddings.norm(
        dim=-1,
        keepdim=True
    )

    # FAISS expects float32 NumPy arrays
    vectors = embeddings.cpu().numpy().astype("float32")

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(DIMENSIONS)

    index.add(vectors)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))

    # Save corresponding image information
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nIndex successfully created!")
    print("Total vectors:", index.ntotal)
    print("Dimensions:", DIMENSIONS)

    return True


# ==========================================
# 3. LOAD SAVED INDEX
# ==========================================

def load_index():

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        print("Index and metadata mismatch!")
        return None, None

    return index, metadata


# ==========================================
# 4. TEXT-TO-IMAGE SEARCH
# ==========================================

def search_images(query, index, metadata, top_k=5):

    print("\nSearching for:", query)

    # Convert text query into CLIP embedding
    tokens = tokenizer([query]).to(DEVICE)

    with torch.inference_mode():
        text_features = model.encode_text(tokens)

    text_features = text_features / text_features.norm(
        dim=-1,
        keepdim=True
    )

    query_vector = text_features.cpu().numpy().astype("float32")

    # Search FAISS
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    print("\n" + "=" * 55)
    print("              SEARCH RESULTS")
    print("=" * 55)

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):

        if idx < 0:
            continue

        result = metadata[idx]

        print(f"\nRank: {rank}")
        print("Image:", result["filename"])
        print("Similarity:", f"{score * 100:.2f}%")
        print("Path:", result["path"])

    print("\nSearch completed!")


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

index, metadata = load_index()

if index is None:

    print("\nNo saved index found.")
    print("Building a new image index...")

    success = build_index()

    if not success:
        raise SystemExit

    index, metadata = load_index()

else:

    print("\nExisting FAISS index loaded!")
    print("Stored images:", index.ntotal)

# ==========================================
# 6. INTERACTIVE SEARCH LOOP
# ==========================================

print("\nZERO-TEXT SEARCH IS READY!")
print("Type 'exit' to quit.")

while True:

    query = input("\nSearch: ").strip()

    if query.lower() == "exit":
        print("Closing ZERO-TEXT.")
        break

    if not query:
        continue

    search_images(query, index, metadata, top_k=5)
