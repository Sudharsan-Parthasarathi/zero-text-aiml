import sys, os
sys.path.insert(0, '.')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from server import engine

engine.load_clip()

human_prompts = [
    "a photo of a person",
    "a portrait photo of a human person",
    "a real human being portrait photograph",
    "a selfie photo of a person looking at the camera",
    "a photo of a man or woman",
    "a real human face portrait",
]

tokens = engine.tokenizer(human_prompts).to("cpu")
with torch.inference_mode():
    emb = engine.model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    mean_emb = emb.mean(dim=0, keepdim=True)
    mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
    vec = mean_emb.cpu().numpy().astype("float32")

scores, indices = engine.index.search(vec, len(engine.metadata))

def calibrate_similarity(sim: float) -> float:
    scaled = (sim - 0.165) / (0.275 - 0.165)
    clamped = max(0.0, min(1.0, scaled))
    return float(clamped ** 1.3)

print("Ranked results for human expansion:")
for rank, (s, idx) in enumerate(zip(scores[0], indices[0]), start=1):
    fn = engine.metadata[idx]['filename']
    cal = calibrate_similarity(s)
    final = 0.75 * cal / 0.90
    print(f"  {rank:2d}. {fn:50} Raw: {s:.4f}  Calibrated: {cal:.4f}  Final: {final:.4f} ({int(round(final*100))}%)")
