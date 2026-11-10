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
print("\n[Test 1] Red car sketch with bucket-fill body...")
w, h = 560, 340
img_car = Image.new("RGB", (w, h), (255, 255, 255))
draw = ImageDraw.Draw(img_car)
draw.rectangle([140, 160, 420, 240], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(180, 160), (220, 110), (340, 110), (380, 160)], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(190, 155), (225, 118), (275, 118), (275, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.polygon([(285, 155), (285, 118), (335, 118), (370, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.ellipse([180, 220, 240, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)
draw.ellipse([320, 220, 380, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)

buf = io.BytesIO()
img_car.save(buf, format="PNG")
b64_car = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

res_car = search_sketch(SketchSearchRequest(sketch_data=b64_car, top_k=5, is_canvas_sketch=True))
print(f"  Results ({res_car['total_results']}):")
for r in res_car["results"]:
    print(f"    {r['rank']}: {r['filename']:32} Final: {r['final']:.4f} CLIP: {r['clip']:.4f} Colour: {r['colour']:.4f}")
top1_car = res_car["results"][0]
assert top1_car["filename"] == "red_supercar.jpg", f"Expected red_supercar.jpg, got {top1_car['filename']}"
print("  ✓ Red car sketch correctly retrieved red_supercar.jpg!")

# Test 2: Yellow sunflower sketch with green stem
print("\n[Test 2] Multi-color flower sketch (yellow petals + green stem)...")
img_flower = Image.new("RGB", (w, h), (255, 255, 255))
draw_f = ImageDraw.Draw(img_flower)
draw_f.line([(280, 200), (280, 320)], fill=(34, 197, 94), width=8)
draw_f.ellipse([200, 70, 360, 230], fill=(234, 179, 8), outline=(0, 0, 0), width=3)
draw_f.ellipse([240, 110, 320, 190], fill=(120, 53, 15), outline=(0, 0, 0), width=3)

buf_f = io.BytesIO()
img_flower.save(buf_f, format="PNG")
b64_flower = "data:image/png;base64," + base64.b64encode(buf_f.getvalue()).decode("utf-8")

res_flower = search_sketch(SketchSearchRequest(sketch_data=b64_flower, top_k=5, is_canvas_sketch=True))
print(f"  Results ({res_flower['total_results']}):")
for r in res_flower["results"]:
    print(f"    {r['rank']}: {r['filename']:32} Final: {r['final']:.4f} CLIP: {r['clip']:.4f} Colour: {r['colour']:.4f}")
top1_flower = res_flower["results"][0]
print(f"  Top 1: {top1_flower['filename']}")

# Test 3: Standard black/white rocket sketch (backwards compatibility)
print("\n[Test 3] Black & white sketch (rocket)...")
with open("data/sketches/rocket.jpg", "rb") as f:
    b64_rocket = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
res_rocket = search_sketch(SketchSearchRequest(sketch_data=b64_rocket, top_k=3))
top1_rocket = res_rocket["results"][0]
print(f"  Top 1: {top1_rocket['filename']} (Final: {top1_rocket['final']:.4f})")
assert top1_rocket["filename"] == "nasa_rocket.jpg", f"Expected nasa_rocket.jpg, got {top1_rocket['filename']}"
print("  ✓ Rocket monochrome sketch returned nasa_rocket.jpg!")

# Test 4: B&W cat sketch
print("\n[Test 4] Black & white sketch (cat)...")
with open("data/sketches/cat.jpg", "rb") as f:
    b64_cat_sk = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
res_cat_sk = search_sketch(SketchSearchRequest(sketch_data=b64_cat_sk, top_k=3))
top1_cat_sk = res_cat_sk["results"][0]
print(f"  Top 1: {top1_cat_sk['filename']} (Final: {top1_cat_sk['final']:.4f})")
assert top1_cat_sk["filename"] == "cat.jpg", f"Expected cat.jpg, got {top1_cat_sk['filename']}"
print("  ✓ Cat monochrome sketch returned cat.jpg!")

# Test 5: Photo upload passthrough (is_canvas_sketch=False)
print("\n[Test 5] Photo upload passthrough...")
with open("data/sample_images/cat.jpg", "rb") as f:
    b64_cat = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
res_cat = search_sketch(SketchSearchRequest(sketch_data=b64_cat, top_k=3, is_canvas_sketch=False))
top1_cat = res_cat["results"][0]
print(f"  Top 1: {top1_cat['filename']} (Final: {top1_cat['final']:.4f})")
assert top1_cat["filename"] == "cat.jpg"
assert top1_cat["final"] >= 0.99
print("  ✓ Photo upload achieved perfect score via photographic CLIP path!")

print("\n>>> ALL SKETCH SEARCH TESTS PASSED! <<<\n")
