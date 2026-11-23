import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipeline import get_engine
from PIL import Image

eng = get_engine()
eng.load_clip()
eng.load_visual_probe_projection()

print("\n--- TEST 1: CODE SCREENSHOT PROBE ---")
code_img = Image.open("data/sample_images/code.jpg")
res_code = eng.image_probe_search(code_img, filename="screenshot.png", top_k=3)
for r in res_code["results"][:3]:
    print(f"  Rank {r['rank']}: {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | OCR={r['ocr']:.4f}")

print("\n--- TEST 2: HANDWRITTEN MATH NOTES PROBE ---")
math_img = Image.open("data/sample_images/math_notes.jpg")
res_math = eng.image_probe_search(math_img, filename="notes.png", top_k=3)
for r in res_math["results"][:3]:
    print(f"  Rank {r['rank']}: {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | OCR={r['ocr']:.4f}")

print("\n--- TEST 3: GOLDEN RETRIEVER PROBE ---")
dog_img = Image.open("data/sample_images/golden_retriever.jpg")
res_dog = eng.image_probe_search(dog_img, filename="unknown_pet.png", top_k=3)
for r in res_dog["results"][:3]:
    print(f"  Rank {r['rank']}: {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | Col={r['colour']:.4f}")
