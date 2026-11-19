import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
import numpy as np

engine = ZeroTextEngine()
engine.load_clip()

cat_text_queries = [
    "a photo of a cat",
    "a cute cat sitting",
    "a tabby cat with whiskers",
    "a domestic feline pet cat",
    "close up of cat face and eyes",
    "a fluffy white kitten cat",
    "an orange cat looking at camera"
]

cat_idx = [i for i, m in enumerate(engine.metadata) if m["filename"] == "cat.jpg"][0]
dog_idx = [i for i, m in enumerate(engine.metadata) if m["filename"] == "golden_retriever.jpg"][0]

cat_img_vec = engine.index.reconstruct(cat_idx)
dog_img_vec = engine.index.reconstruct(dog_idx)

print("=" * 60)
print("COMPARING EXTERNAL CAT CONCEPT VECTORS AGAINST GALLERY IMAGES")
print("=" * 60)

for q in cat_text_queries:
    t_vec = engine.compute_text_embedding(q, ensemble=True)
    t_vec = t_vec / np.linalg.norm(t_vec)
    
    sim_to_cat = float(np.dot(t_vec.flatten(), cat_img_vec.flatten()))
    sim_to_dog = float(np.dot(t_vec.flatten(), dog_img_vec.flatten()))
    
    # Also search entire index
    scores, indices = engine.index.search(t_vec.reshape(1, -1).astype("float32"), 5)
    top1_fn = engine.metadata[indices[0][0]]["filename"]
    
    print(f"Query: '{q}'")
    print(f"  -> Sim to cat.jpg: {sim_to_cat:.4f} | Sim to golden_retriever.jpg: {sim_to_dog:.4f}")
    print(f"  -> Top-1 in index: {top1_fn} ({scores[0][0]:.4f}) | Win: {'CAT' if sim_to_cat > sim_to_dog else 'DOG'}")
