import sys
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageEnhance
from pipeline import get_engine, DEVICE, COLOURS

engine = get_engine()
engine.load_clip()

def preprocess_sketch_structural_clean(sketch_img: Image.Image):
    """
    Fast, CPU-efficient, structurally robust sketch preprocessing.
    Generates 2 to 3 high-signal views (instead of 12 redundant views):
      1. Structural Stroke Contour View (foreground-centered, stroke-density normalized)
      2. Shape Silhouette / Distance Field View (captures 2D geometry, boundary mass, aspect ratio)
      3. (If colored) Color-content enhanced view
    """
    rgb_np = np.array(sketch_img.convert("RGB"), dtype=np.uint8)
    gray_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    is_white_bg = gray_np.mean() > 128
    
    if is_white_bg:
        inv_np = np.max(255 - rgb_np, axis=2).astype(np.uint8)
    else:
        inv_np = np.max(rgb_np, axis=2).astype(np.uint8)
        
    stroke_mask = inv_np > 20
    h, w = inv_np.shape[:2]
    
    if not np.any(stroke_mask):
        return [sketch_img], [1.0]

    ys, xs = np.nonzero(stroke_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1

    # Foreground bounding box centering with 12% proportional margin
    pad_y = max(8, int(box_h * 0.12))
    pad_x = max(8, int(box_w * 0.12))
    c_y1 = max(0, y_min - pad_y)
    c_y2 = min(h, y_max + pad_y + 1)
    c_x1 = max(0, x_min - pad_x)
    c_x2 = min(w, x_max + pad_x + 1)
    
    cropped_inv = inv_np[c_y1:c_y2, c_x1:c_x2]
    cropped_rgb = rgb_np[c_y1:c_y2, c_x1:c_x2]
    ch, cw = cropped_inv.shape[:2]
    sq = max(ch, cw)
    
    # Place on square canvas
    centered_inv = np.zeros((sq, sq), dtype=np.uint8)
    centered_rgb = np.ones((sq, sq, 3), dtype=np.uint8) * (255 if is_white_bg else 0)
    oy = (sq - ch) // 2
    ox = (sq - cw) // 2
    centered_inv[oy:oy+ch, ox:ox+cw] = cropped_inv
    centered_rgb[oy:oy+ch, ox:ox+cw] = cropped_rgb
    
    TARGET_SIZE = 224
    norm_inv = cv2.resize(centered_inv, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    norm_rgb = cv2.resize(centered_rgb, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    
    # 1. Structural Stroke Contour View (density-normalized)
    _, binary_np = cv2.threshold(norm_inv, 25, 255, cv2.THRESH_BINARY)
    stroke_fraction = float((binary_np > 0).mean())
    if stroke_fraction < 0.05:
        # Thin strokes: dilate with 3x3 kernel so strokes activate ViT patch projection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_inv = cv2.dilate(binary_np, kernel, iterations=1)
    else:
        dilated_inv = binary_np
        
    # Standard white background with dark strokes for vision encoder
    view_stroke = Image.fromarray(255 - dilated_inv).convert("RGB")
    
    # 2. Structural Silhouette / Soft Shape Field View (captures 2D silhouette & boundary mass)
    dist = cv2.distanceTransform(cv2.dilate(dilated_inv, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # White background with soft distance shadow
    view_shape = Image.fromarray(255 - dist_norm).convert("RGB")
    
    views = [view_stroke, view_shape]
    weights = [1.2, 0.8]
    
    # Check if sketch has distinct color content
    c_hsv = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2HSV)
    sat_vals = c_hsv[:, :, 1]
    has_color = float(sat_vals.mean()) > 8.0 or (sat_vals > 35).mean() > 0.02
    
    if has_color:
        # View 3: Color-contrast enhanced content view
        img_color = Image.fromarray(norm_rgb).convert("RGB")
        try:
            boosted = ImageEnhance.Color(img_color).enhance(1.3)
            views.append(boosted)
            weights.append(1.1)
        except Exception:
            views.append(img_color)
            weights.append(1.0)
            
    return views, weights

def compute_sketch_embedding_clean(sketch_img: Image.Image, text_hint: str = ""):
    views, weights = preprocess_sketch_structural_clean(sketch_img)
    tensors = torch.stack([engine.preprocess(v) for v in views]).to(DEVICE)
    with torch.inference_mode():
        embs = engine.model.encode_image(tensors)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        w_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        mean_emb = (embs * w_t).sum(dim=0, keepdim=True) / w_t.sum()
        
    if text_hint and text_hint.strip():
        hint = text_hint.strip()
        prompts = [hint, f"a photograph of {hint}", f"a photo of {hint}"]
        tokens = engine.tokenizer(prompts).to(DEVICE)
        with torch.inference_mode():
            t_embs = engine.model.encode_text(tokens)
            t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
            t_mean = t_embs.mean(dim=0, keepdim=True)
        mean_emb = 0.80 * mean_emb + 0.20 * t_mean

    norm = mean_emb.norm(dim=-1, keepdim=True)
    if norm > 1e-6:
        mean_emb = mean_emb / norm
    return mean_emb.cpu().numpy().astype("float32")

print("=== TESTING CLEAN STRUCTURAL EMBEDDING (ZERO TRAINING, NO RUNTIME ADAPTER) ===")
test_sketches = [
    ("cat.jpg", "cat.jpg"),
    ("mountain.jpg", "mountain.jpg"),
    ("rocket.jpg", "nasa_rocket.jpg"),
]

for sk_name, target in test_sketches:
    img = Image.open(f"data/sketches/{sk_name}")
    t0 = time.time()
    vec = compute_sketch_embedding_clean(img)
    scores, indices = engine.index.search(vec, 5)
    t1 = time.time()
    top1 = engine.metadata[indices[0][0]]["filename"]
    top1_sim = scores[0][0]
    print(f"Sketch: {sk_name:15} -> Target: {target:18} Top1: {top1:18} (cos: {top1_sim:.4f}) Time: {t1-t0:.3f}s")
    assert top1 == target, f"Expected {target}, got {top1}"

print("All monochrome test sketches retrieved target with Rank 1!")
