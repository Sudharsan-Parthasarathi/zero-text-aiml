"""
ZERO-TEXT INTEGRATION & ENDPOINT VERIFICATION TEST
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import sys
from pathlib import Path

from server import (
    serve_index,
    get_status,
    get_images,
    search_hybrid,
    search_sketch,
    search_color,
    search_similar,
    HybridSearchRequest,
    SketchSearchRequest,
    ColorSearchRequest,
    SimilarSearchRequest,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("     ZERO-TEXT 2.0 INTEGRATION TEST SUITE")
print("=" * 60)

# 1. Test Static / HTML Route
print("\n[1] Testing GET / (HTML Frontend)...")
html_text = serve_index()
assert "ZERO-TEXT" in html_text
print("✓ Frontend HTML served successfully!")

# 2. Test GET /api/status
print("\n[2] Testing GET /api/status...")
status = get_status()
print("✓ Status:", status)
assert status["total_images"] >= 10
assert status["clusters_count"] >= 1

# 3. Test GET /api/images
print("\n[3] Testing GET /api/images...")
images = get_images()
print(f"✓ Retrieved {len(images)} images.")
sample = images[0]
print("  Sample image record:", {
    "filename": sample["filename"],
    "cluster": sample["cluster"],
    "x2d": sample.get("x"),
    "y2d": sample.get("y"),
    "z3d": sample.get("z3d"),
})
assert "x" in sample and "y" in sample and "z3d" in sample

# 4. Test Hybrid Search
print("\n[4] Testing POST /api/search/hybrid...")
search_req = HybridSearchRequest(
    query="blue artificial intelligence and neural network",
    clip_weight=0.55,
    ocr_weight=0.30,
    colour_weight=0.15,
)
search_data = search_hybrid(search_req)
print(f"✓ Hybrid search returned {search_data['total_results']} results.")
print("  Detected text keywords:", search_data["detected_words"])
print("  Detected colors:", search_data["detected_colours"])
top_result = search_data["results"][0]
print("  Top result:", top_result["filename"], "Final Score:", top_result["final"])

# 5. Test Sketch Search
print("\n[5] Testing POST /api/search/sketch...")
sketch_path = Path("data/sketches/rocket.jpg")
if sketch_path.exists():
    with open(sketch_path, "rb") as f:
        sketch_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    sketch_req = SketchSearchRequest(sketch_data=sketch_b64, top_k=5)
    sketch_data = search_sketch(sketch_req)
    print(f"✓ Sketch search returned {sketch_data['total_results']} results.")
    print("  Top result for rocket sketch:", sketch_data["results"][0]["filename"])
else:
    print("  (Skipped sketch test: rocket.jpg not found)")

# 6. Test Color Search
print("\n[6] Testing POST /api/search/color...")
color_req = ColorSearchRequest(hex="#0D2C48", top_k=5)
color_data = search_color(color_req)
print(f"✓ Color search returned {color_data['total_results']} results.")
print("  Top result for navy color:", color_data["results"][0]["filename"])

# 7. Test Similar Images
print("\n[7] Testing POST /api/search/similar...")
similar_req = SimilarSearchRequest(filename="cat.jpg", top_k=3)
similar_data = search_similar(similar_req)
print(f"✓ Similar search for cat.jpg returned {similar_data['total_results']} neighbors.")

print("\n" + "=" * 60)
print("ALL ZERO-TEXT INTEGRATION TESTS PASSED PERFECTLY!")
print("=" * 60)
