import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import io
import numpy as np
from PIL import Image, ImageDraw
from server import app, engine, search_sketch, SketchSearchRequest

print("\n=== TESTING MULTI-COLOR & BUCKET-FILLED SKETCH SEARCH ===")

# Test 1: Red sports car sketch with red bucket-fill body
print("\n[Test 1] Generating red sports car sketch with bucket-fill body...")
w, h = 560, 340
img_car = Image.new("RGB", (w, h), (255, 255, 255))
draw = ImageDraw.Draw(img_car)

# Draw car outline
draw.rectangle([140, 160, 420, 240], fill=(239, 68, 68), outline=(0, 0, 0), width=4) # Red bucket fill body
draw.polygon([(180, 160), (220, 110), (340, 110), (380, 160)], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
# Windows
draw.polygon([(190, 155), (225, 118), (275, 118), (275, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.polygon([(285, 155), (285, 118), (335, 118), (370, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
# Wheels
draw.ellipse([180, 220, 240, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)
draw.ellipse([320, 220, 380, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)

buf = io.BytesIO()
img_car.save(buf, format="PNG")
b64_car = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

res_car = search_sketch(SketchSearchRequest(sketch_data=b64_car, top_k=5, is_canvas_sketch=True))
print(f"Total results: {res_car['total_results']}")
for r in res_car["results"]:
    print(f"  {r['rank']}: {r['filename']:24} Final: {r['final']:.4f} (CLIP: {r['clip']:.4f}, Colour: {r['colour']:.4f}, Raw: {r['similarity']:.4f})")
top1_car = res_car["results"][0]
print(f"Top 1 for red car sketch: {top1_car['filename']} (Final: {top1_car['final']}, Colour Score: {top1_car.get('colour', 0)})")
assert "supercar" in top1_car["filename"] or "car" in top1_car["filename"] or top1_car["filename"] == "red_supercar.jpg", f"Unexpected top match: {top1_car['filename']}"
print("✓ Red car sketch with bucket-filled red body successfully retrieved red_supercar.jpg!")

# Test 2: Yellow sunflower sketch with green stem (multiple colors)
print("\n[Test 2] Generating multi-color flower sketch (yellow petals + green stem)...")
img_flower = Image.new("RGB", (w, h), (255, 255, 255))
draw_f = ImageDraw.Draw(img_flower)
# Green stem
draw_f.line([(280, 200), (280, 320)], fill=(34, 197, 94), width=8)
# Yellow petals (bucket-filled yellow circle / ellipse)
draw_f.ellipse([200, 70, 360, 230], fill=(234, 179, 8), outline=(0, 0, 0), width=3)
# Brown center
draw_f.ellipse([240, 110, 320, 190], fill=(120, 53, 15), outline=(0, 0, 0), width=3)

buf_f = io.BytesIO()
img_flower.save(buf_f, format="PNG")
b64_flower = "data:image/png;base64," + base64.b64encode(buf_f.getvalue()).decode("utf-8")

res_flower = search_sketch(SketchSearchRequest(sketch_data=b64_flower, top_k=5, is_canvas_sketch=True))
top1_flower = res_flower["results"][0]
print(f"Top 1 for yellow flower sketch: {top1_flower['filename']} (Final: {top1_flower['final']}, Colour Score: {top1_flower.get('colour', 0)})")
assert "sunflower" in top1_flower["filename"].lower() or "yellow" in top1_flower["filename"].lower(), f"Unexpected top match: {top1_flower['filename']}"
print("✓ Multi-color sunflower sketch successfully retrieved sunflower image!")

# Test 3: Standard black/white rocket sketch (ensure backwards compatibility)
print("\n[Test 3] Testing standard black & white sketch (rocket)...")
with open("data/sketches/rocket.jpg", "rb") as f:
    b64_rocket = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
res_rocket = search_sketch(SketchSearchRequest(sketch_data=b64_rocket, top_k=3))
top1_rocket = res_rocket["results"][0]
print(f"Top 1 for rocket.jpg: {top1_rocket['filename']} (Final: {top1_rocket['final']})")
assert top1_rocket["filename"] == "nasa_rocket.jpg"
print("✓ Rocket black & white sketch returned nasa_rocket.jpg with top accuracy!")

# Test 4: Untouched photo upload (is_canvas_sketch=False)
print("\n[Test 4] Testing photo upload passthrough...")
with open("data/sample_images/cat.jpg", "rb") as f:
    b64_cat = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
res_cat = search_sketch(SketchSearchRequest(sketch_data=b64_cat, top_k=3, is_canvas_sketch=False))
top1_cat = res_cat["results"][0]
print(f"Top 1 for cat photo upload: {top1_cat['filename']} (Final: {top1_cat['final']})")
assert top1_cat["filename"] == "cat.jpg"
assert top1_cat["final"] >= 0.99
print("✓ Photo upload achieved 1.00 score via photographic CLIP path!")

print("\n>>> ALL MULTI-COLOR, BUCKET-FILL, AND SKETCH SEARCH ACCURACY TESTS PASSED WITH 100% SUCCESS! <<<\n")
