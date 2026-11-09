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
from starlette.testclient import TestClient

from server import app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 60)
print("     ZERO-TEXT 2.0 INTEGRATION TEST SUITE")
print("=" * 60)

client = TestClient(app)

# 1. Test Static / HTML Route
print("\n[1] Testing GET / (HTML Frontend)...")
res = client.get("/")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
assert "ZERO-TEXT" in res.text
print("✓ Frontend HTML served successfully!")

# 2. Test GET /api/status
print("\n[2] Testing GET /api/status...")
res = client.get("/api/status")
assert res.status_code == 200
status = res.json()
print("✓ Status:", status)
assert status["total_images"] >= 10
assert status["clusters_count"] >= 1

# 3. Test GET /api/images
print("\n[3] Testing GET /api/images...")
res = client.get("/api/images")
assert res.status_code == 200
images = res.json()
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
search_payload = {
    "query": "blue artificial intelligence and neural network",
    "clip_weight": 0.55,
    "ocr_weight": 0.30,
    "colour_weight": 0.15,
}
res = client.post("/api/search/hybrid", json=search_payload)
assert res.status_code == 200
search_data = res.json()
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
    res = client.post("/api/search/sketch", json={"sketch_data": sketch_b64, "top_k": 5})
    assert res.status_code == 200
    sketch_data = res.json()
    print(f"✓ Sketch search returned {sketch_data['total_results']} results.")
    print("  Top result for rocket sketch:", sketch_data["results"][0]["filename"])
else:
    print("  (Skipped sketch test: rocket.jpg not found)")

# 6. Test Color Search
print("\n[6] Testing POST /api/search/color...")
res = client.post("/api/search/color", json={"hex": "#0D2C48", "top_k": 5})
assert res.status_code == 200
color_data = res.json()
print(f"✓ Color search returned {color_data['total_results']} results.")
print("  Top result for navy color:", color_data["results"][0]["filename"])

# 7. Test Similar Images
print("\n[7] Testing POST /api/search/similar...")
res = client.post("/api/search/similar", json={"filename": "cat.jpg", "top_k": 3})
assert res.status_code == 200
similar_data = res.json()
print(f"✓ Similar search for cat.jpg returned {similar_data['total_results']} neighbors.")

print("\n" + "=" * 60)
print("ALL ZERO-TEXT INTEGRATION TESTS PASSED PERFECTLY!")
print("=" * 60)
