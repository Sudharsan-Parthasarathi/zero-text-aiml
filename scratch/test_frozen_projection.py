import sys
import time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageEnhance
from pipeline import get_engine, DEVICE, COLOURS, SKETCH_ADAPTER_PATH

engine = get_engine()
engine.load_clip()

# Load frozen checkpoint into model
class SketchProjection(torch.nn.Module):
    def __init__(self, dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.act1 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = torch.nn.LayerNorm(hidden_dim)
        self.act2 = torch.nn.GELU()
        self.fc3 = torch.nn.Linear(hidden_dim, dim)
        self.alpha = torch.nn.Parameter(torch.tensor(0.8, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc3(self.act2(self.norm2(self.fc2(self.act1(self.norm1(self.fc1(x)))))))
        out = x + self.alpha * delta
        return out / out.norm(dim=-1, keepdim=True)

proj = SketchProjection()
if SKETCH_ADAPTER_PATH.exists():
    proj.load_state_dict(torch.load(SKETCH_ADAPTER_PATH, map_location=DEVICE))
    proj.eval()
    print("Loaded frozen projection checkpoint.")

def preprocess_structural(sketch_img: Image.Image):
    rgb_np = np.array(sketch_img.convert("RGB"), dtype=np.uint8)
    h_img, w_img = rgb_np.shape[:2]
    white_mask = (rgb_np[:, :, 0] > 240) & (rgb_np[:, :, 1] > 240) & (rgb_np[:, :, 2] > 240)
    non_white = ~white_mask
    
    if not non_white.any():
        return [sketch_img], [1.0], False, [], sketch_img
        
    ys, xs = np.nonzero(non_white)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    ch = ymax - ymin + 1
    cw = xmax - xmin + 1
    
    # Check for color content
    non_white_px = rgb_np[non_white]
    has_color = False
    sketch_palette = []
    if len(non_white_px) > 30:
        nw_hsv = cv2.cvtColor(non_white_px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        sat_vals = nw_hsv[:, 0, 1]
        val_vals = nw_hsv[:, 0, 2]
        color_fraction = float((sat_vals > 28).mean())
        mean_sat = float(sat_vals.mean())
        if color_fraction > 0.02 or mean_sat > 12.0:
            has_color = True
            sig_mask = (sat_vals > 22) | (val_vals < 45)
            sig_px = non_white_px[sig_mask]
            if len(sig_px) >= 20:
                try:
                    from sklearn.cluster import MiniBatchKMeans
                    n_c = min(4, len(np.unique(sig_px, axis=0)))
                    kmeans = MiniBatchKMeans(n_clusters=n_c, random_state=42, n_init=3)
                    labels = kmeans.fit_predict(sig_px)
                    centers = kmeans.cluster_centers_
                    counts = np.bincount(labels)
                    total_pts = len(labels)
                    for idx in np.argsort(-counts):
                        pct = float(counts[idx] / total_pts)
                        if pct < 0.04:
                            continue
                        c_rgb = centers[idx]
                        c_hsv = cv2.cvtColor(np.uint8([[c_rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
                        min_d = float('inf')
                        best_name = None
                        for cname, crgb in COLOURS.items():
                            d = np.linalg.norm(c_rgb - np.array(crgb, dtype=float))
                            if d < min_d:
                                min_d = d
                                best_name = cname
                        is_chrom = bool(c_hsv[1] > 28 and c_hsv[2] > 35)
                        sketch_palette.append({
                            "name": best_name,
                            "rgb": c_rgb.astype(float),
                            "percentage": pct,
                            "is_chromatic": is_chrom,
                            "dist": float(min_d),
                        })
                except Exception:
                    pass

    TARGET_SIZE = 224
    scale = min((TARGET_SIZE - 28) / max(ch, 1), (TARGET_SIZE - 28) / max(cw, 1))
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))
    
    crop_rgb = rgb_np[ymin:ymax+1, xmin:xmax+1]
    resized_crop = cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas_rgb = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
    y_off = (TARGET_SIZE - new_h) // 2
    x_off = (TARGET_SIZE - new_w) // 2
    canvas_rgb[y_off:y_off+new_h, x_off:x_off+new_w] = resized_crop
    
    gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
    stroke_mask_c = gray < 235
    stroke_density = float(stroke_mask_c.mean())
    if stroke_density < 0.04:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_gray = cv2.erode(gray, kernel)
        canvas_stroke = np.stack([dilated_gray] * 3, axis=-1)
    else:
        canvas_stroke = canvas_rgb
    view_stroke = Image.fromarray(canvas_stroke)
    
    inv_stroke = 255 - gray
    dist = cv2.distanceTransform(cv2.dilate(inv_stroke, np.ones((3, 3), np.uint8)), cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    view_shape = Image.fromarray(255 - dist_norm).convert("RGB")
    
    views = [view_stroke, view_shape]
    weights = [1.2, 0.8]
    
    if has_color:
        try:
            boosted = ImageEnhance.Color(Image.fromarray(canvas_rgb)).enhance(1.3)
            views.append(boosted)
            weights.append(1.0)
        except Exception:
            views.append(Image.fromarray(canvas_rgb))
            weights.append(1.0)
            
    return views, weights, has_color, sketch_palette, Image.fromarray(resized_crop)

def search_test(sketch_img: Image.Image, is_canvas_sketch=True):
    views, weights, has_color, sketch_palette, content_crop = preprocess_structural(sketch_img)
    tensors = torch.stack([engine.preprocess(v) for v in views]).to(DEVICE)
    with torch.inference_mode():
        embs = engine.model.encode_image(tensors)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        w_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        raw_vec = (embs * w_t).sum(dim=0, keepdim=True) / w_t.sum()
        raw_vec = raw_vec / raw_vec.norm(dim=-1, keepdim=True)
        
        # Apply frozen shared projection
        proj_vec = proj(raw_vec)
        sketch_vec = (proj_vec / proj_vec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

    if has_color and sketch_palette:
        photo_vec = engine.compute_image_embedding(content_crop)
        CANONICAL_COLOR_MAP = {
            "amber": "yellow", "gold": "yellow", "golden": "yellow",
            "crimson": "red", "scarlet": "red",
            "emerald": "green", "forest": "green",
            "navy": "dark blue", "cyan": "cyan", "teal": "teal",
            "violet": "purple", "magenta": "pink",
        }
        chromatic_names = []
        for p in sketch_palette:
            if p["is_chromatic"] and p["dist"] < 170:
                mapped = CANONICAL_COLOR_MAP.get(p["name"], p["name"])
                if mapped not in chromatic_names:
                    chromatic_names.append(mapped)
        prompts = []
        if chromatic_names:
            color_phrase = " and ".join(chromatic_names[:2])
            prompts.extend([
                color_phrase,
                f"a photo featuring {color_phrase}",
                f"a vibrant photo with {color_phrase} colors",
            ])
            t_tokens = engine.tokenizer(prompts).to(DEVICE)
            with torch.inference_mode():
                t_embs = engine.model.encode_text(t_tokens)
                t_embs = t_embs / t_embs.norm(dim=-1, keepdim=True)
                t_vec = t_embs.mean(dim=0, keepdim=True)
                t_vec = (t_vec / t_vec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
            vec = 0.55 * sketch_vec + 0.15 * photo_vec + 0.30 * t_vec
        else:
            vec = 0.65 * sketch_vec + 0.35 * photo_vec
    else:
        vec = sketch_vec

    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    if norm > 1e-6:
        vec = (vec / norm).astype("float32")
        
    scores, indices = engine.index.search(vec, 5)
    
    results = []
    for raw_s, idx in zip(scores[0], indices[0]):
        fn = engine.metadata[idx]["filename"]
        col_s = 0.0
        if has_color and sketch_palette:
            img_colours = engine.colour_lookup.get(fn, [])
            tot_w = sum(p["percentage"] for p in sketch_palette) or 1.0
            for sk in sketch_palette:
                w_p = sk["percentage"] / tot_w
                best_m = 0.0
                for ic in img_colours:
                    dist = np.linalg.norm(sk["rgb"] - np.array(ic["rgb"]))
                    m = max(0.0, 1.0 - dist / 441.67) ** 2
                    if m > best_m:
                        best_m = m
                col_s += w_p * best_m
            final_s = 0.65 * raw_s + 0.35 * col_s
        else:
            final_s = raw_s
        results.append((fn, final_s, raw_s, col_s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

print("\n--- Testing Car Sketch ---")
w, h = 560, 340
img_car = Image.new("RGB", (w, h), (255, 255, 255))
draw = ImageDraw.Draw(img_car)
draw.rectangle([140, 160, 420, 240], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(180, 160), (220, 110), (340, 110), (380, 160)], fill=(239, 68, 68), outline=(0, 0, 0), width=4)
draw.polygon([(190, 155), (225, 118), (275, 118), (275, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.polygon([(285, 155), (285, 118), (335, 118), (370, 155)], fill=(200, 230, 255), outline=(0, 0, 0), width=2)
draw.ellipse([180, 220, 240, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)
draw.ellipse([320, 220, 380, 280], fill=(20, 20, 20), outline=(0, 0, 0), width=3)

res_car = search_test(img_car)
for r in res_car:
    print(" ", r[0], "final:", round(r[1], 4), "raw:", round(r[2], 4), "col:", round(r[3], 4))
assert res_car[0][0] == "red_supercar.jpg", f"Expected red_supercar.jpg, got {res_car[0][0]}"
print("-> Car sketch passed!")

print("\n--- Testing Cat Sketch ---")
res_cat = search_test(Image.open("data/sketches/cat.jpg"))
for r in res_cat:
    print(" ", r[0], "final:", round(r[1], 4))
assert res_cat[0][0] == "cat.jpg", f"Expected cat.jpg, got {res_cat[0][0]}"
print("-> Cat sketch passed!")

print("\n--- Testing Mountain Sketch ---")
res_m = search_test(Image.open("data/sketches/mountain.jpg"))
for r in res_m:
    print(" ", r[0], "final:", round(r[1], 4))
assert res_m[0][0] == "mountain.jpg", f"Expected mountain.jpg, got {res_m[0][0]}"
print("-> Mountain sketch passed!")

print("\n--- Testing Rocket Sketch ---")
res_r = search_test(Image.open("data/sketches/rocket.jpg"))
for r in res_r:
    print(" ", r[0], "final:", round(r[1], 4))
assert res_r[0][0] == "nasa_rocket.jpg", f"Expected nasa_rocket.jpg, got {res_r[0][0]}"
print("-> Rocket sketch passed!")

print("\nALL TESTS PASSED WITH CLEAN FROZEN PROJECTION & STRUCTURAL VIEWS!")
