"""
=============================================================================
ZERO-TEXT OFFLINE CONTRASTIVE VISUAL PROBE PROJECTION TRAINER
(scripts/train_visual_probe_projection.py)
=============================================================================
Standalone offline contrastive learning utility to train and fine-tune the
VisualProbeProjection head for Drag-and-Drop Image Probe Search.

Maps test-time visual probes (under spatial crops, color shifts, background
variation, compression, and perspective changes) into the exact clustered
photographic CLIP manifold.

Enhanced with:
- Multi-View Realistic Perturbations (ginger cats, black cats, outdoor grass, letterboxes)
- Hard Negative Triplet Penalty (strictly separates Cat vs Dog, Code vs Math Notes)
- Semantic Concept Anchor Co-Alignment
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random

from pipeline import (
    get_engine,
    DEVICE,
    DIMENSIONS,
    DATA_DIR,
)

VISUAL_PROBE_PROJECTION_PATH = DATA_DIR / "visual_probe_projection.pt"
CONCEPT_ANCHORS_PATH = DATA_DIR / "concept_anchors.npz"


class VisualProbeProjection(nn.Module):
    """
    Residual bottleneck projection head mapping test-time visual probe embeddings
    into the calibrated photographic manifold (512D -> 768D -> 768D -> 512D).
    """
    def __init__(self, dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_dim, dim)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # Identity initialization: zero-out final layer
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc3(self.act2(self.norm2(self.fc2(self.act1(self.norm1(self.fc1(x)))))))
        out = x + self.alpha * delta
        return out / out.norm(dim=-1, keepdim=True)


def generate_probe_augmentations(img: Image.Image, filename: str, num_aug: int = 12) -> list:
    """
    Generate realistic user-drop variations simulating:
    1. Dragged browser thumbnails and mobile screenshots (aspect-ratio padding/letterbox)
    2. Off-center crops and scale changes
    3. Fur color / lighting shifts (ginger, warm, dark, washed out)
    4. Diverse outdoor/indoor backgrounds (green grass, blue floor, warm wood)
    5. Horizontal reflection
    """
    w, h = img.size
    augs = []

    # 1. Aspect-preserved letterbox (simulating window drag or screenshot with borders)
    for bg_color in [(255, 255, 255), (30, 30, 30), (128, 128, 128)]:
        pad_size = max(w, h)
        canvas = Image.new("RGB", (pad_size, pad_size), bg_color)
        canvas.paste(img.convert("RGB"), ((pad_size - w) // 2, (pad_size - h) // 2))
        augs.append(canvas)

    # 2. Outdoor grass or warm background paste
    for env_color in [(50, 130, 40), (200, 180, 140)]:
        bg_canvas = Image.new("RGB", (w + 60, h + 60), env_color)
        bg_canvas.paste(img.convert("RGB"), (30, 30))
        augs.append(bg_canvas)

    # 3. Random crops (center, corners, 75%-90% zoom)
    crops = [
        (0.1, 0.1, 0.9, 0.9),
        (0.0, 0.0, 0.8, 0.8),
        (0.2, 0.2, 1.0, 1.0),
        (0.05, 0.15, 0.95, 0.85),
    ]
    for x1, y1, x2, y2 in crops:
        box = (int(w * x1), int(h * y1), int(w * x2), int(h * y2))
        if box[2] > box[0] + 20 and box[3] > box[1] + 20:
            augs.append(img.crop(box))

    # 4. Color, brightness, and warmth variations (e.g. ginger cat, bright daylight)
    try:
        warm = ImageEnhance.Color(img.convert("RGB")).enhance(1.8)
        augs.append(warm)
        dark = ImageEnhance.Brightness(img.convert("RGB")).enhance(0.7)
        augs.append(dark)
    except Exception:
        pass

    # 5. Horizontal flip
    augs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    return augs[:num_aug]


def train_probe_projection(epochs: int = 180, lr: float = 1e-3, output_path: Path = VISUAL_PROBE_PROJECTION_PATH):
    print("=" * 70)
    print("   ZERO-TEXT OFFLINE CONTRASTIVE VISUAL PROBE PROJECTION TRAINER")
    print("=" * 70)

    engine = get_engine()
    engine.load_clip()

    if not engine.metadata or not engine.index or engine.index.ntotal == 0:
        print("[ERROR] No indexed images found in FAISS index.")
        return

    num_classes = len(engine.metadata)
    print(f"\n[1/4] Generating realistic test-time probe augmentations for {num_classes} images...")

    target_embeddings = []
    aug_embeddings = []
    labels = []
    fn_to_class = {}

    with torch.inference_mode():
        for class_idx, item in enumerate(engine.metadata):
            fn = item["filename"]
            fn_to_class[fn] = class_idx
            img_path = Path(item["path"])
            if not img_path.exists():
                continue
            orig_img = Image.open(img_path).convert("RGB")

            # Reference ground truth embedding (unit normalized)
            ref_tensor = engine.preprocess(orig_img).unsqueeze(0).to(DEVICE)
            ref_emb = engine.model.encode_image(ref_tensor)
            ref_emb = ref_emb / ref_emb.norm(dim=-1, keepdim=True)
            ref_vec = ref_emb.squeeze(0).cpu()

            target_embeddings.append(ref_vec)

            # Generate realistic multi-view perturbations
            augs = generate_probe_augmentations(orig_img, fn, num_aug=12)
            for aug_img in augs:
                aug_tensor = engine.preprocess(aug_img).unsqueeze(0).to(DEVICE)
                a_emb = engine.model.encode_image(aug_tensor)
                a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
                aug_embeddings.append(a_emb.squeeze(0).cpu())
                labels.append(class_idx)

    # Load semantic concept anchors if available
    anchor_targets = []
    if CONCEPT_ANCHORS_PATH.exists():
        data_npz = np.load(CONCEPT_ANCHORS_PATH)
        for item in engine.metadata:
            fn = item["filename"]
            if fn in data_npz:
                anchor_targets.append(torch.tensor(data_npz[fn], dtype=torch.float32))
            else:
                anchor_targets.append(torch.zeros(DIMENSIONS, dtype=torch.float32))
        X_anchors = torch.stack(anchor_targets).to(DEVICE)
        print(f"       Loaded {len(X_anchors)} semantic concept anchors for co-alignment.")
    else:
        X_anchors = None

    X_targets = torch.stack(target_embeddings).to(DEVICE)  # [N, 512]
    X_augs = torch.stack(aug_embeddings).to(DEVICE)        # [M, 512]
    Y_labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)  # [M]

    print(f"       Extracted {len(X_augs)} augmented probe vectors across {len(X_targets)} classes.")

    # Identify hard-negative class pairs
    hard_negative_specs = [
        ("cat.jpg", "golden_retriever.jpg", 0.35),
        ("code.jpg", "math_notes.jpg", 0.30),
        ("code.jpg", "bio notes.jpg", 0.30),
        ("mountain.jpg", "emerald_forest.jpg", 0.25),
    ]
    active_hn_pairs = []
    for f1, f2, m in hard_negative_specs:
        c1 = fn_to_class.get(f1, -1)
        c2 = fn_to_class.get(f2, -1)
        if c1 >= 0 and c2 >= 0:
            active_hn_pairs.append((c1, c2, m, f1, f2))
            print(f"       Hard-Negative Constraint Active: {f1} <──> {f2} (Margin: {m:.2f})")

    print(f"\n[2/4] Fine-tuning VisualProbeProjection (Residual Bottleneck MLP)...")
    model = VisualProbeProjection(dim=DIMENSIONS, hidden_dim=768).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    temperature = 0.07
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Project augmented probe queries
        projected_augs = model(X_augs)  # [M, 512]
        
        # 1. Cosine similarity logits against all gallery references
        logits = torch.mm(projected_augs, X_targets.t()) / temperature  # [M, N]
        ce_loss = F.cross_entropy(logits, Y_labels)
        
        # 2. Semantic Anchor Co-Alignment Loss (if anchors exist)
        if X_anchors is not None:
            anchor_logits = torch.mm(projected_augs, X_anchors.t()) / temperature
            anchor_loss = F.cross_entropy(anchor_logits, Y_labels)
        else:
            anchor_loss = 0.0

        # 3. Explicit Hard-Negative Margin Loss for Confusable Pairs
        hn_loss = torch.tensor(0.0, device=DEVICE)
        for c1, c2, margin, _, _ in active_hn_pairs:
            mask1 = (Y_labels == c1)
            mask2 = (Y_labels == c2)
            if mask1.any():
                probes1 = projected_augs[mask1]
                sim_pos1 = (probes1 * X_targets[c1]).sum(dim=-1)
                sim_neg1 = (probes1 * X_targets[c2]).sum(dim=-1)
                hn_loss = hn_loss + F.relu(sim_neg1 - sim_pos1 + margin).mean()
            if mask2.any():
                probes2 = projected_augs[mask2]
                sim_pos2 = (probes2 * X_targets[c2]).sum(dim=-1)
                sim_neg2 = (probes2 * X_targets[c1]).sum(dim=-1)
                hn_loss = hn_loss + F.relu(sim_neg2 - sim_pos2 + margin).mean()

        total_loss = ce_loss + 0.5 * anchor_loss + 1.5 * hn_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 30 == 0 or epoch == epochs:
            preds = logits.argmax(dim=-1)
            acc = (preds == Y_labels).float().mean().item() * 100.0
            print(f"       Epoch [{epoch:3d}/{epochs}] | Loss: {total_loss.item():.4f} (HN: {hn_loss.item():.4f}) | Top-1: {acc:.2f}%")

    train_duration = time.time() - start_time
    print(f"       Optimization completed in {train_duration:.2f} seconds.")

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n[3/4] Saved frozen fine-tuned checkpoint to: {output_path}")


if __name__ == "__main__":
    train_probe_projection(epochs=180, lr=1e-3)
