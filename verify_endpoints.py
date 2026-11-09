import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from server import (
    app,
    engine,
    get_status,
    get_images,
    search_hybrid,
    search_color,
    search_similar,
    get_recommended_weights,
    HybridSearchRequest,
    ColorSearchRequest,
    SimilarSearchRequest
)

print("\n--- ZERO-TEXT DIRECT ENDPOINT VERIFICATION ---")

# 1. Status
status = get_status()
print(f"[1] get_status(): total_images={status['total_images']}, clusters={status['clusters_count']}")

# 2. All images
images = get_images()
print(f"[2] get_images(): retrieved {len(images)} images.")
assert len(images) == 20, f"Expected 20 images, got {len(images)}"

# 3. Hybrid search (text + colour + clip)
res_ocean = search_hybrid(HybridSearchRequest(query="blue ocean", top_k=3))
top_ocean = res_ocean["results"][0]
print(f"[3] search_hybrid('blue ocean'): Top 1 = {top_ocean['filename']} (Final: {top_ocean['final']})")
assert top_ocean["filename"] == "blue_ocean.jpg", f"Expected blue_ocean.jpg, got {top_ocean['filename']}"

res_car = search_hybrid(HybridSearchRequest(query="red supercar", top_k=3))
top_car = res_car["results"][0]
print(f"[4] search_hybrid('red supercar'): Top 1 = {top_car['filename']} (Final: {top_car['final']})")
assert top_car["filename"] == "red_supercar.jpg", f"Expected red_supercar.jpg, got {top_car['filename']}"

# 4. Color search
res_color = search_color(ColorSearchRequest(hex="#D90429", top_k=3))
top_col = res_color["results"][0]
print(f"[5] search_color('#D90429'): Top 1 = {top_col['filename']} (Colour Score: {top_col['colour']})")

# 5. Similar search
res_sim = search_similar(SimilarSearchRequest(filename="cat.jpg", top_k=3))
print(f"[6] search_similar('cat.jpg'): Returned {len(res_sim['results'])} neighbors.")

# 6. Weight Recommendation
rec = get_recommended_weights("crimson red sportscar")
print(f"[7] get_recommended_weights('crimson red sportscar'): intent={rec['intent']}, label={rec['label']}, weights=CLIP {rec['clip']}/OCR {rec['ocr']}/Colour {rec['colour']}")

print("\n>>> ALL API ENDPOINTS FUNCTION PERFECTLY WITHOUT ERRORS! <<<\n")
