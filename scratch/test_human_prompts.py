import sys, os
sys.path.insert(0, '.')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from server import engine

engine.load_clip()

test_prompts_sets = [
    ("Default (no expansion)", ["human", "a photo of human", "a high quality photo showing human"]),
    ("Set A (portrait person)", ["a photo of a person", "a photo of a human being", "a portrait of a person", "a human face", "a photo of a man or person"]),
    ("Set B (human person individual)", ["a photo of a human person", "a portrait photo of a person", "a human being", "a person looking at the camera", "a selfie or portrait of a person"]),
    ("Set C (focused human)", ["a photo of a human person", "a portrait of a person", "a real human being", "a man or woman portrait photo", "a person face"]),
]

for name, p_list in test_prompts_sets:
    tokens = engine.tokenizer(p_list).to("cpu")
    with torch.inference_mode():
        emb = engine.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        mean_emb = emb.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        vec = mean_emb.cpu().numpy().astype("float32")
    scores, indices = engine.index.search(vec, 5)
    print(f"\n--- {name} ---")
    for s, idx in zip(scores[0], indices[0]):
        fn = engine.metadata[idx]['filename']
        print(f"  {fn:50}: {s:.4f}")
