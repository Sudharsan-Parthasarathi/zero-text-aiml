import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
import numpy as np
import torch
from PIL import Image, ImageEnhance

engine = ZeroTextEngine()
engine.load_clip()

CONCEPT_PROMPTS = {
    "cat.jpg": ["a photo of a cat", "a cute feline kitten cat", "domestic cat with pointed ears whiskers", "feline pet cat"],
    "golden_retriever.jpg": ["a photo of a golden retriever dog", "a canine puppy dog", "friendly pet dog", "dog canine"],
    "red_supercar.jpg": ["a red supercar sports car", "a high performance exotic automotive vehicle", "crimson sports coupe car"],
    "blue_ocean.jpg": ["deep blue ocean sea waves", "clear turquoise ocean water", "beach ocean coast sea"],
    "emerald_forest.jpg": ["emerald green forest trees", "dense woodland nature foliage", "green jungle trees plants"],
    "coffee.jpg": ["a cup of hot coffee espresso", "latte mug with roasted coffee beans", "coffee drink cafe"],
    "football.jpg": ["a soccer ball football on field", "classic white and black football", "soccer sports ball match"],
    "nasa_rocket.jpg": ["space rocket launch liftoff", "spacecraft booster launchpad", "space rocket cosmos exploration"],
    "code.jpg": ["computer programming code screen", "python developer source code", "software script syntax in IDE text editor"],
    "math_notes.jpg": ["handwritten mathematics calculus formulas", "math equations notebook paper", "algebra calculus study notes"],
    "bio notes.jpg": ["biology study notes diagrams", "biological science notebook", "botany anatomy handwritten notes text"],
    "pepperoni_pizza.jpg": ["baked pepperoni pizza with melted cheese", "fresh round slice of pizza pie", "italian pizza crust"],
    "purple_nebula.jpg": ["deep space purple cosmic nebula", "astronomical telescope star dust cosmos", "outer space nebula galaxy stars"],
    "golden_sunflower.jpg": ["blooming yellow sunflower in field", "bright golden sunflower petals", "sunflower blossom flora plant"],
    "mountain.jpg": ["snow covered mountain peaks", "majestic rocky mountain summit", "alpine mountain landscape peak"],
    "neon_cyberpunk.jpg": ["neon illuminated cyberpunk city night", "futuristic glowing retro synthwave streets", "neon night cityscape skyline"],
    "vintage_camera.jpg": ["vintage classic analog film camera", "retro mechanical camera lens", "antique photo camera equipment"],
    "Iron-Man-featured-image.avif": ["iron man marvel superhero armor suit", "tony stark red and gold robotic suit", "avengers iron man"],
    "spiderman-upside-down_3840x2160_xtrafondos.com (2).jpg": ["spiderman hanging upside down", "peter parker superhero red and blue suit", "spiderman web slinger"],
    "triangle-with-planet_3840x2160_xtrafondos.com.jpg": ["geometric triangle framing outer space planet", "retro sci-fi cosmic planetary artwork", "space synthwave art"],
    "among-us-vicious-smile-512x512.png": ["among us crewmate imposter character", "red among us astronaut with vicious smile", "among us video game"],
    "aiml.jpg": ["artificial intelligence machine learning neural network diagram", "ai technology futuristic network concept", "deep learning circuit visual"],
    "WIN_20261016_21_52_16_Pro.jpg": ["indoor personal webcam portrait snapshot", "room ambient photo webcam capture", "user webcam frame"]
}

concept_anchors = {}
with torch.no_grad():
    for fn, prompts in CONCEPT_PROMPTS.items():
        tokens = engine.tokenizer(prompts).to("cpu")
        embs = engine.model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        concept_anchors[fn] = mean_emb.squeeze(0).numpy()

# Test 1: Real cat.jpg
cat_img = Image.open("data/sample_images/cat.jpg")

# Test 2: Cat with golden/yellow color jitter (simulating ginger/orange cat on grass)
ginger_cat = ImageEnhance.Color(cat_img).enhance(2.0)
w, h = ginger_cat.size
grass_cat = Image.new("RGB", (w+80, h+80), (70, 140, 40)) # Green grass
grass_cat.paste(ginger_cat, (40, 40))

test_cases = [
    ("Original cat.jpg", cat_img),
    ("Ginger Cat on Grass", grass_cat),
]

for label, img in test_cases:
    # Compute multi-view probe ensemble
    emb = engine._compute_probe_image_ensemble(img, filename="unknown_pet.png")
    emb_norm = emb.flatten() / np.linalg.norm(emb)
    
    # Compare against all 23 database items
    scores = []
    for item in engine.metadata:
        fn = item["filename"]
        idx = [i for i, m in enumerate(engine.metadata) if m["filename"] == fn][0]
        vis_vec = engine.index.reconstruct(idx).flatten()
        vis_sim = float(np.dot(emb_norm, vis_vec))
        
        sem_sim = float(np.dot(emb_norm, concept_anchors[fn]))
        
        # Color match
        col_score = 0.0
        if fn in engine.colour_lookup:
            p_query = engine.extract_colours_from_image(img)
            p_target = engine.colour_lookup[fn]
            max_dist = (3 * (255 ** 2)) ** 0.5
            total_w = sum(float(c.get("percentage", 10)) for c in p_query) or 100.0
            sim_sum = 0.0
            for c1 in p_query:
                rgb1 = np.array(c1["rgb"], dtype=float)
                w1 = float(c1.get("percentage", 10)) / total_w
                best_match = 0.0
                for c2 in p_target:
                    rgb2 = np.array(c2["rgb"], dtype=float)
                    dist = np.linalg.norm(rgb1 - rgb2)
                    match = max(0.0, 1.0 - (dist / max_dist)) ** 2
                    if match > best_match: best_match = match
                sim_sum += w1 * best_match
            col_score = sim_sum
        
        # Multimodal fusion
        fused = 0.55 * vis_sim + 0.35 * sem_sim + 0.10 * col_score
        scores.append((fn, fused, vis_sim, sem_sim, col_score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\nResults for [{label}]:")
    for rank, (fn, fused, vis, sem, col) in enumerate(scores[:3], 1):
        print(f"  {rank}. {fn:25s} | Fused={fused:.4f} (Vis={vis:.3f}, Sem={sem:.3f}, Col={col:.3f})")
