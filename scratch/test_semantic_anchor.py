import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
import numpy as np
import torch
from PIL import Image

engine = ZeroTextEngine()
engine.load_clip()

# Category semantic definitions for the dataset:
CONCEPT_PROMPTS = {
    "cat.jpg": ["a photo of a cat", "a cute feline kitten cat", "domestic cat with pointed ears and whiskers"],
    "golden_retriever.jpg": ["a photo of a golden retriever dog", "a canine puppy dog", "friendly dog on green grass"],
    "red_supercar.jpg": ["a red supercar sports car", "a high performance exotic automotive car", "crimson racing vehicle"],
    "blue_ocean.jpg": ["deep blue ocean sea waves", "clear turquoise ocean water", "beach ocean coast"],
    "emerald_forest.jpg": ["emerald green forest trees", "dense woodland nature foliage", "green jungle canopy"],
    "coffee.jpg": ["a cup of hot coffee espresso", "latte with coffee beans", "steaming coffee mug"],
    "football.jpg": ["a soccer ball football on stadium field", "classic white and black football", "soccer sports ball"],
    "nasa_rocket.jpg": ["space rocket launch liftoff", "spacecraft booster launchpad", "space shuttle exploration rocket"],
    "code.jpg": ["computer programming code screen", "python developer source code", "software script syntax in IDE"],
    "math_notes.jpg": ["handwritten mathematics calculus formulas", "math equations notebook paper", "algebra calculus study notes"],
    "bio notes.jpg": ["biology study notes diagrams", "biological science notebook", "botany anatomy handwritten notes"],
    "pepperoni_pizza.jpg": ["baked pepperoni pizza with melted cheese", "fresh round slice of pizza pie", "italian pizza"],
    "purple_nebula.jpg": ["deep space purple cosmic nebula", "astronomical telescope star dust cosmos", "outer space nebula galaxy"],
    "golden_sunflower.jpg": ["blooming yellow sunflower in field", "bright golden sunflower petals", "sunflower blossom flora"],
    "mountain.jpg": ["snow covered mountain peaks", "majestic rocky mountain summit", "alpine alpine landscape"],
    "neon_cyberpunk.jpg": ["neon illuminated cyberpunk city night", "futuristic glowing retro synthwave streets", "neon night cityscape"],
    "vintage_camera.jpg": ["vintage classic analog film camera", "retro mechanical camera lens", "antique photo camera"],
    "Iron-Man-featured-image.avif": ["iron man marvel superhero armor suit", "tony stark red and gold robotic suit", "avengers iron man"],
    "spiderman-upside-down_3840x2160_xtrafondos.com (2).jpg": ["spiderman hanging upside down", "peter parker superhero red and blue suit", "spiderman web slinger"],
    "triangle-with-planet_3840x2160_xtrafondos.com.jpg": ["geometric triangle framing outer space planet", "retro sci-fi cosmic planetary artwork", "space synthwave art"],
    "among-us-vicious-smile-512x512.png": ["among us crewmate imposter character", "red among us astronaut with vicious smile", "among us video game"],
    "aiml.jpg": ["artificial intelligence machine learning neural network diagram", "ai technology futuristic network concept", "deep learning circuit visual"],
    "WIN_20261016_21_52_16_Pro.jpg": ["indoor personal webcam portrait snapshot", "room ambient photo webcam capture", "user webcam frame"]
}

print("Computing semantic anchor embeddings...")
concept_anchors = {}
with torch.no_grad():
    for fn, prompts in CONCEPT_PROMPTS.items():
        tokens = engine.tokenizer(prompts).to("cpu")
        embs = engine.model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        concept_anchors[fn] = mean_emb.squeeze(0).numpy()

print(f"Generated {len(concept_anchors)} semantic concept anchors.")

# Test with external cat test queries (simulating external cat photos):
test_cats = [
    "a photo of an orange tabby cat sitting on floor",
    "a fluffy white angora cat with blue eyes",
    "a black cat looking directly at camera",
    "a playful kitten playing with string",
    "a domestic short hair cat resting"
]

print("\n--- TESTING EXTERNAL CAT SIMULATION WITH SEMANTIC ANCHOR FUSION ---")
for t in test_cats:
    t_tok = engine.tokenizer([t]).to("cpu")
    with torch.no_grad():
        t_vec = engine.model.encode_text(t_tok)
        t_vec = (t_vec / t_vec.norm(dim=-1, keepdim=True)).squeeze(0).numpy()
    
    # Check visual dot product with indexed cat.jpg vs golden_retriever.jpg
    cat_idx = [i for i, m in enumerate(engine.metadata) if m["filename"] == "cat.jpg"][0]
    dog_idx = [i for i, m in enumerate(engine.metadata) if m["filename"] == "golden_retriever.jpg"][0]
    
    cat_gallery_vec = engine.index.reconstruct(cat_idx).flatten()
    dog_gallery_vec = engine.index.reconstruct(dog_idx).flatten()
    
    vis_sim_cat = float(np.dot(t_vec, cat_gallery_vec))
    vis_sim_dog = float(np.dot(t_vec, dog_gallery_vec))
    
    sem_sim_cat = float(np.dot(t_vec, concept_anchors["cat.jpg"]))
    sem_sim_dog = float(np.dot(t_vec, concept_anchors["golden_retriever.jpg"]))
    
    # Fused score: 0.65 * visual + 0.35 * semantic anchor
    fused_cat = 0.65 * vis_sim_cat + 0.35 * sem_sim_cat
    fused_dog = 0.65 * vis_sim_dog + 0.35 * sem_sim_dog
    
    print(f"\nQuery: '{t}'")
    print(f"  Raw Visual:  Cat={vis_sim_cat:.3f} | Dog={vis_sim_dog:.3f}")
    print(f"  Concept:     Cat={sem_sim_cat:.3f} | Dog={sem_sim_dog:.3f}")
    print(f"  Fused Score: Cat={fused_cat:.3f} | Dog={fused_dog:.3f}  ==> Winner: {'CAT ✅' if fused_cat > fused_dog else 'DOG ❌'}")
