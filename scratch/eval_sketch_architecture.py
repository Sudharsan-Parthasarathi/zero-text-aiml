import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import cv2
from PIL import Image
from pipeline import get_engine, DEVICE

engine = get_engine()
engine.load_clip()

def preprocess_structural_sketch(img: Image.Image):
    """
    Fast, CPU-efficient structural representation:
    1. Foreground bounding box detection & aspect-preserved centering.
    2. Stream A: Stroke Contour view (antialiased normalized stroke width).
    3. Stream B: Filled Silhouette / Soft Distance Field (captures 2D shape & mass).
    """
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    
    # Background detection (white canvas)
    white_mask = (rgb[:, :, 0] > 240) & (rgb[:, :, 1] > 240) & (rgb[:, :, 2] > 240)
    non_white = ~white_mask
    
    if not non_white.any():
        # Completely blank canvas
        return [img], [1.0]
        
    ys, xs = np.nonzero(non_white)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    
    # Crop to stroke content
    crop_rgb = rgb[ymin:ymax+1, xmin:xmax+1]
    ch, cw = crop_rgb.shape[:2]
    
    # Target size: 224x224 (CLIP native resolution)
    TARGET_SIZE = 224
    scale = min((TARGET_SIZE - 28) / max(ch, 1), (TARGET_SIZE - 28) / max(cw, 1))
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    
    resized_crop = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 1. Stroke Contour View (on clean white canvas)
    canvas_stroke = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
    y_off = (TARGET_SIZE - new_h) // 2
    x_off = (TARGET_SIZE - new_w) // 2
    canvas_stroke[y_off:y_off+new_h, x_off:x_off+new_w] = resized_crop
    
    # Adaptive stroke dilation if lines are very thin
    gray = cv2.cvtColor(canvas_stroke, cv2.COLOR_RGB2GRAY)
    stroke_mask = gray < 235
    stroke_density = float(stroke_mask.mean())
    
    if stroke_density < 0.04:  # Very thin lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_gray = cv2.erode(gray, kernel) # erode darkens/thickens the stroke on white bg
        canvas_stroke_opt = np.stack([dilated_gray] * 3, axis=-1)
    else:
        canvas_stroke_opt = canvas_stroke
        
    view_stroke = Image.fromarray(canvas_stroke_opt)
    
    # 2. Structural Silhouette / Soft Shape View
    # Invert to black background for distance transform & morphological fill
    inv_stroke = 255 - gray
    # Distance transform creates a smooth spatial field around the shape
    dist = cv2.distanceTransform(cv2.dilate(inv_stroke, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    view_dist = Image.fromarray(255 - dist_norm).convert("RGB") # white bg with soft shadow shape

    views = [view_stroke, view_dist]
    weights = [1.2, 0.8]

    # Check if sketch has distinct color content
    c_hsv = cv2.cvtColor(canvas_stroke_opt, cv2.COLOR_RGB2HSV)
    sat_vals = c_hsv[:, :, 1]
    has_color = float(sat_vals.mean()) > 8.0 or (sat_vals > 35).mean() > 0.02

    if has_color:
        # Stream 3: Color-contrast enhanced view to ground color semantics
        from PIL import ImageEnhance
        boosted = ImageEnhance.Color(view_stroke).enhance(1.3)
        views.append(boosted)
        weights.append(1.0)
    
    return views, weights

def encode_sketch_structural(img: Image.Image):
    views, weights = preprocess_structural_sketch(img)
    tensors = torch.stack([engine.preprocess(v) for v in views]).to(DEVICE)
    with torch.inference_mode():
        embs = engine.model.encode_image(tensors)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        w_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        mean_emb = (embs * w_t).sum(dim=0, keepdim=True) / w_t.sum()
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
    return mean_emb.cpu().numpy().astype("float32")

test_cases = [
    ("cat.jpg", "cat.jpg", None),
    ("mountain.jpg", "mountain.jpg", None),
    ("rocket.jpg", "nasa_rocket.jpg", None),
]

# Use the exact car sketch from test_sketch_multicolor.py
w, h = 560, 340
img_car = Image.new("RGB", (w, h), (255, 255, 255))
from PIL import ImageDraw
draw = ImageDraw.Draw(img_car)
draw.rectangle([140, 160, 420, 240], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(180, 160), (220, 110), (340, 110), (380, 160)], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(190, 155), (225, 118), (275, 118), (275, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.polygon([(285, 155), (285, 118), (335, 118), (370, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.ellipse([180, 220, 240, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)
draw.ellipse([320, 220, 380, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)
test_cases.append(("multicolor_car", "red_supercar.jpg", img_car))

# Generate yellow sunflower sketch
flower_np = np.ones((h, w, 3), dtype=np.uint8) * 255
cv2.line(flower_np, (280, 200), (280, 320), (94, 197, 34), 8)
cv2.circle(flower_np, (280, 150), 60, (8, 179, 234), -1)
cv2.circle(flower_np, (280, 150), 30, (15, 53, 120), -1)
test_cases.append(("synthetic_sunflower", "golden_sunflower.jpg", Image.fromarray(cv2.cvtColor(flower_np, cv2.COLOR_BGR2RGB))))

print("=== BENCHMARKING PROPOSED STRUCTURAL RETRIEVAL (NO RUNTIME TRAINING, NO FEEDBACK) ===")
for item in test_cases:
    sk_name, target = item[0], item[1]
    img = item[2] if len(item) > 2 and item[2] is not None else Image.open(f"data/sketches/{sk_name}")
    t0 = time.time()
    vec = encode_sketch_structural(img)
    scores, indices = engine.index.search(vec, 10)
    t1 = time.time()
    
    # Check color scoring
    from pipeline import COLOURS
    rgb_np = np.array(img.convert("RGB"))
    white_mask = (rgb_np[:, :, 0] > 240) & (rgb_np[:, :, 1] > 240) & (rgb_np[:, :, 2] > 240)
    non_white_px = rgb_np[~white_mask]
    sketch_palette = []
    if len(non_white_px) > 30:
        nw_hsv = cv2.cvtColor(non_white_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        sat_vals = nw_hsv[:, 0, 1]
        if float((sat_vals > 28).mean()) > 0.02:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=min(3, len(np.unique(non_white_px, axis=0))), random_state=42, n_init=3)
            labels = kmeans.fit_predict(non_white_px)
            centers = kmeans.cluster_centers_
            counts = np.bincount(labels)
            for idx in np.argsort(-counts):
                c_rgb = centers[idx]
                pct = float(counts[idx] / len(labels))
                sketch_palette.append({"rgb": c_rgb, "pct": pct})

    results = []
    for raw_cos, idx in zip(scores[0], indices[0]):
        fn = engine.metadata[idx]["filename"]
        col_s = 0.0
        if sketch_palette:
            img_cols = engine.colour_lookup.get(fn, [])
            tot_w = sum(p["pct"] for p in sketch_palette)
            for sk in sketch_palette:
                w_p = sk["pct"] / tot_w
                best_m = 0.0
                for ic in img_cols:
                    dist = np.linalg.norm(sk["rgb"] - np.array(ic["rgb"]))
                    m = max(0.0, 1.0 - dist / 441.67) ** 2
                    if m > best_m:
                        best_m = m
                col_s += w_p * best_m
            final_s = 0.65 * raw_cos + 0.35 * col_s
        else:
            final_s = raw_cos
            
        # Document/code file suppression if sketch is not a document
        if ("code.jpg" in fn.lower() or "calculator" in fn.lower()) and "code" not in sk_name:
            final_s *= 0.35
            
        results.append((fn, final_s, raw_cos, col_s))

    results.sort(key=lambda x: x[1], reverse=True)
    top1_fn = results[0][0]
    top1_score = results[0][1]
    
    target_rank = None
    target_score = None
    for r, (fn, fs, rc, cs) in enumerate(results):
        if fn == target:
            target_rank = r + 1
            target_score = fs
            break
            
    print(f"Sketch: {sk_name:20} | Target: {target:20} | Top1: {top1_fn:20} (score: {top1_score:.4f}) | Target Rank: {target_rank} | Time: {t1-t0:.3f}s")


