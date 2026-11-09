import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import open_clip
from PIL import Image
from pathlib import Path
import time

print("=" * 50)
print("       ZERO-TEXT IMAGE EMBEDDING ENGINE")
print("=" * 50)

device = "cpu"

# Load pretrained CLIP model
print("\nLoading CLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device
)

model.eval()

# Locate sample images
IMAGE_DIR = Path("data/sample_images")

image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    image_paths.extend(IMAGE_DIR.glob(ext))

if not image_paths:
    print("No images found! Add images to data/sample_images.")
    raise SystemExit

print(f"\nFound {len(image_paths)} images.")

# Process images
image_tensors = []
valid_paths = []

for path in image_paths:
    try:
        image = Image.open(path).convert("RGB")
        image_tensor = preprocess(image)

        image_tensors.append(image_tensor)
        valid_paths.append(path)

        print("Processed:", path.name)

    except Exception as e:
        print("Skipping:", path.name, "|", e)

if not image_tensors:
    print("No valid images could be processed.")
    raise SystemExit

# Generate image embeddings
print("\nGenerating image embeddings...")

start = time.time()

batch = torch.stack(image_tensors).to(device)

with torch.inference_mode():
    image_features = model.encode_image(batch)

# Normalize embeddings
image_features = image_features / image_features.norm(
    dim=-1,
    keepdim=True
)

print("\n" + "=" * 50)
print("IMAGE EMBEDDING RESULTS")
print("=" * 50)

print("Images processed:", len(valid_paths))
print("Embedding shape:", image_features.shape)
print("Embedding dimensions:", image_features.shape[1])
print("Processing time:", round(time.time() - start, 2), "seconds")

print("\nSUCCESS! IMAGE EMBEDDINGS GENERATED.")
