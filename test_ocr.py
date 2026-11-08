from pathlib import Path
from importlib import import_module
import json
import time
import sys

# Enable UTF-8 output in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

try:
    RapidOCR = import_module("rapidocr_onnxruntime").RapidOCR
except ImportError as error:
    raise SystemExit(
        "RapidOCR is not installed. Install it with: "
        "python -m pip install rapidocr-onnxruntime"
    ) from error

# ==========================================
# ZERO-TEXT OCR ENGINE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "data" / "sample_images"
OUTPUT_FILE = BASE_DIR / "data" / "ocr_results.json"

print("=" * 55)
print("          ZERO-TEXT OCR ENGINE")
print("=" * 55)

# Initialize OCR
print("\nLoading RapidOCR...")

ocr = RapidOCR()

print("RapidOCR loaded successfully!")

# Find images
extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

image_paths = []

for ext in extensions:
    image_paths.extend(IMAGE_DIR.glob(ext))

image_paths = sorted(image_paths)

if not image_paths:
    print("No images found!")
    raise SystemExit

print(f"\nFound {len(image_paths)} images.")

# Process images
all_results = []

start = time.time()

for image_path in image_paths:

    print("\nProcessing:", image_path.name)

    try:
        result, elapsed = ocr(str(image_path))

        extracted_text = []

        if result:
            for item in result:
                text = item[1]
                confidence = float(item[2])

                extracted_text.append({
                    "text": text,
                    "confidence": confidence
                })

        combined_text = " ".join(
            item["text"] for item in extracted_text
        )

        print("Extracted text:", combined_text or "[No text detected]")

        all_results.append({
            "filename": image_path.name,
            "path": str(image_path),
            "extracted_text": extracted_text,
            "full_text": combined_text
        })

    except Exception as error:

        print("OCR error:", error)

        all_results.append({
            "filename": image_path.name,
            "path": str(image_path),
            "extracted_text": [],
            "full_text": "",
            "error": str(error)
        })

# Save results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(all_results, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 55)
print("OCR PROCESSING COMPLETED")
print("=" * 55)

print("Images processed:", len(all_results))
print("Images containing detected text:",
      sum(bool(x["full_text"]) for x in all_results))

print("Output saved to:", OUTPUT_FILE)
print("Total time:", round(time.time() - start, 2), "seconds")

print("\nSUCCESS! ZERO-TEXT OCR ENGINE IS WORKING.")
