import sys, os
sys.path.insert(0, '.')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from server import engine

engine.load_clip()

prompts = [
    "a photo of a person",
    "a portrait of a human being",
    "a photo of a man or person",
    "a real human face portrait photograph",
    "a picture of a person looking at the camera",
    "a portrait photo of a person"
]

tokens = engine.tokenizer(prompts).to("cpu")
with torch.inference_mode():
    emb = engine.model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    mean_emb = emb.mean(dim=0, keepdim=True)
    mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
    vec = mean_emb.cpu().numpy().astype("float32")

scores, indices = engine.index.search(vec, 5)
print("=== Results ===")
for s, idx in zip(scores[0], indices[0]):
    print(f"  {engine.metadata[idx]['filename']:50} Sim: {s:.4f}")
