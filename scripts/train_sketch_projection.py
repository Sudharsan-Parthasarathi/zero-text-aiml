"""
=============================================================================
ZERO-TEXT OFFLINE SKETCH PROJECTION TRAINER (scripts/train_sketch_projection.py)
=============================================================================
Standalone offline utility to train or fine-tune the SketchProjection head
mapping sketch structural representations into the photographic CLIP manifold.

This script is strictly decoupled from the live application.
The application never trains at runtime or startup; it loads the frozen checkpoint.

Usage:
  python scripts/train_sketch_projection.py [--epochs 150] [--lr 1e-3] [--output data/sketch_projection.pt]
=============================================================================
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Thread limits for CPU stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from pipeline import (
    get_engine,
    DEVICE,
    DIMENSIONS,
    DATA_DIR,
    SKETCHES_DIR,
    SKETCH_PROJECTION_PATH,
    SketchProjection,
)

def train_offline(epochs: int = 150, lr: float = 1e-3, output_path: Path = SKETCH_PROJECTION_PATH):
    print("=" * 65)
    print("      ZERO-TEXT OFFLINE SKETCH PROJECTION TRAINING")
    print("=" * 65)
    
    engine = get_engine()
    engine.load_clip()
    
    if not engine.metadata or not engine.index or engine.index.ntotal == 0:
        print("[Error] No indexed images found in FAISS index. Run indexing first.")
        return

    print(f"\n[1/4] Preparing paired structural sketches for {len(engine.metadata)} images...")
    pairs = []
    
    # 1. Real hand-drawn sketches
    for sk_name, target in [("cat.jpg", "cat.jpg"), ("mountain.jpg", "mountain.jpg"), ("rocket.jpg", "nasa_rocket.jpg")]:
        sp = SKETCHES_DIR / sk_name
        if sp.exists():
            try:
                sk_img = Image.open(sp)
                pairs.append((sk_img, target))
                pairs.append((sk_img.transpose(Image.FLIP_LEFT_RIGHT), target))
            except Exception:
                pass

    # 2. Multi-style synthetic structural sketches
    for item in engine.metadata:
        fn = item["filename"]
        fp = Path(item["path"])
        if not fp.exists():
            continue
        try:
            img = Image.open(fp)
            gray = np.array(img.convert("L"))
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Style A: Canny edges
            c1 = cv2.Canny(gray, 50, 150)
            pairs.append((Image.fromarray(c1), fn))

            # Style B: Adaptive threshold ink sketch
            adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            pairs.append((Image.fromarray(adapt), fn))

            # Style C: Pencil sketch simulation
            inv = 255 - gray
            blur_inv = cv2.GaussianBlur(inv, (21, 21), 0)
            denom = np.maximum(255 - blur_inv, 1)
            dodge = cv2.divide(gray, denom, scale=256)
            pencil = 255 - dodge
            pairs.append((Image.fromarray(pencil), fn))

            # Style D: Posterized bucket-fill sketch
            q_img = img.quantize(colors=6, method=Image.MEDIANCUT).convert("RGB")
            q_np = np.array(q_img)
            edges_3ch = np.stack([c1] * 3, axis=-1)
            fill_sketch = np.where(edges_3ch > 80, 0, q_np)
            pairs.append((Image.fromarray(fill_sketch.astype(np.uint8)), fn))

        except Exception:
            continue

    filename_to_idx = {item["filename"]: i for i, item in enumerate(engine.metadata)}
    img_embeddings = np.array(engine.index.reconstruct_n(0, len(engine.metadata)))

    print(f"\n[2/4] Encoding {len(pairs)} structural sketch views through frozen CLIP...")
    sketch_vecs = []
    target_indices = []
    
    for sk_img, fn in pairs:
        if fn not in filename_to_idx:
            continue
        try:
            vec = engine._compute_sketch_raw_ensemble(sk_img)
            sketch_vecs.append(vec[0])
            target_indices.append(filename_to_idx[fn])
        except Exception:
            continue

    if not sketch_vecs:
        print("[Error] Failed to extract sketch feature vectors.")
        return

    X_sk = torch.tensor(np.array(sketch_vecs), dtype=torch.float32)
    Y_all = torch.tensor(img_embeddings, dtype=torch.float32)
    targets = torch.tensor(target_indices, dtype=torch.long)

    print(f"\n[3/4] Optimizing SketchProjection ({epochs} epochs, lr={lr}, InfoNCE)...")
    model = SketchProjection(DIMENSIONS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    for ep in range(1, epochs + 1):
        proj = model(X_sk)
        sims = (proj @ Y_all.T) / 0.05
        loss = F.cross_entropy(sims, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if ep % 30 == 0 or ep == epochs:
            print(f"  Epoch {ep:3d}/{epochs} - Loss: {loss.item():.4f}")

    print("\n[4/4] Validating retrieval metrics...")
    model.eval()
    with torch.inference_mode():
        val_proj = model(X_sk)
        val_sims = val_proj @ Y_all.T
        top_k_val = min(3, Y_all.shape[0])
        topk_indices = torch.topk(val_sims, k=top_k_val, dim=-1).indices
        top1_correct = (topk_indices[:, 0] == targets).sum().item()
        top3_correct = ((topk_indices == targets.unsqueeze(1)).any(dim=1)).sum().item()
        ranks = []
        for i, t in enumerate(targets):
            order = torch.argsort(val_sims[i], descending=True)
            rank = (order == t).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
        mrr = sum(ranks) / len(ranks)
        top1_prec = (top1_correct / len(targets)) * 100
        top3_rec = (top3_correct / len(targets)) * 100

    print(f"  Precision@1: {top1_prec:.1f}%")
    print(f"  Recall@3:    {top3_rec:.1f}%")
    print(f"  MRR:         {mrr:.4f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nSUCCESS: Checkpoint saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Text Offline Sketch Projection Trainer")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", type=str, default=str(SKETCH_PROJECTION_PATH), help="Output checkpoint path")
    args = parser.parse_args()

    train_offline(epochs=args.epochs, lr=args.lr, output_path=Path(args.output))
