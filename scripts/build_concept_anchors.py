import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine, DATA_DIR
import numpy as np
import torch
import json

engine = ZeroTextEngine()
engine.load_clip()

CONCEPT_PROMPTS = {
    "cat.jpg": [
        "a photo of a cat",
        "a cute feline kitten cat",
        "domestic cat with pointed ears whiskers",
        "feline pet cat kitty",
        "a close up photo of a cat face and eyes"
    ],
    "golden_retriever.jpg": [
        "a photo of a golden retriever dog",
        "a canine puppy dog",
        "friendly golden pet dog on grass",
        "canine dog with long muzzle snout",
        "a domestic pet dog puppy"
    ],
    "red_supercar.jpg": [
        "a red supercar sports car",
        "a high performance exotic automotive vehicle",
        "crimson sports coupe car automobile",
        "luxury sports car supercar"
    ],
    "blue_ocean.jpg": [
        "deep blue ocean sea waves",
        "clear turquoise ocean water",
        "beach ocean coast marine waves",
        "tropical sea ocean aerial"
    ],
    "emerald_forest.jpg": [
        "emerald green forest trees",
        "dense woodland nature foliage landscape",
        "green jungle trees plants canopy",
        "rainforest forest vegetation"
    ],
    "coffee.jpg": [
        "a cup of hot coffee espresso",
        "latte mug with roasted coffee beans",
        "steaming coffee drink cafe cup",
        "coffee beverage in ceramic mug"
    ],
    "football.jpg": [
        "a soccer ball football on stadium field",
        "classic white and black football",
        "soccer sports ball match",
        "fifa football soccer ball on grass"
    ],
    "nasa_rocket.jpg": [
        "space rocket launch liftoff",
        "spacecraft booster launchpad",
        "space rocket cosmos exploration",
        "apollo nasa rocket ascending"
    ],
    "code.jpg": [
        "computer programming code screen",
        "python developer source code",
        "software script syntax in IDE text editor",
        "programming software script coding"
    ],
    "math_notes.jpg": [
        "handwritten mathematics calculus formulas",
        "math equations notebook paper",
        "algebra calculus study notes",
        "mathematical derivation handwriting"
    ],
    "bio notes.jpg": [
        "biology study notes diagrams",
        "biological science notebook paper",
        "botany anatomy handwritten notes text",
        "biology science lecture notes"
    ],
    "pepperoni_pizza.jpg": [
        "baked pepperoni pizza with melted cheese",
        "fresh round slice of pizza pie",
        "italian pizza crust tomato cheese",
        "delicious pepperoni pizza"
    ],
    "purple_nebula.jpg": [
        "deep space purple cosmic nebula",
        "astronomical telescope star dust cosmos",
        "outer space nebula galaxy stars",
        "hubble purple cosmos space nebula"
    ],
    "golden_sunflower.jpg": [
        "blooming yellow sunflower in field",
        "bright golden sunflower petals",
        "sunflower blossom flora plant",
        "yellow agricultural sunflower"
    ],
    "mountain.jpg": [
        "snow covered mountain peaks",
        "majestic rocky mountain summit",
        "alpine mountain landscape peak",
        "himalayas mountain range snow"
    ],
    "neon_cyberpunk.jpg": [
        "neon illuminated cyberpunk city night",
        "futuristic glowing retro synthwave streets",
        "neon night cityscape skyline",
        "blade runner cyberpunk neon city"
    ],
    "vintage_camera.jpg": [
        "vintage classic analog film camera",
        "retro mechanical camera lens",
        "antique photo camera equipment",
        "old fashioned 35mm film camera"
    ],
    "Iron-Man-featured-image.avif": [
        "iron man marvel superhero armor suit",
        "tony stark red and gold robotic suit",
        "avengers iron man mechanical armor",
        "marvel superhero iron man flying"
    ],
    "spiderman-upside-down_3840x2160_xtrafondos.com (2).jpg": [
        "spiderman hanging upside down",
        "peter parker superhero red and blue suit",
        "spiderman web slinger superhero",
        "marvel comic spiderman inverted"
    ],
    "triangle-with-planet_3840x2160_xtrafondos.com.jpg": [
        "geometric triangle framing outer space planet",
        "retro sci-fi cosmic planetary artwork",
        "space synthwave digital fantasy art",
        "astronomical triangle planet graphic"
    ],
    "among-us-vicious-smile-512x512.png": [
        "among us crewmate imposter character",
        "red among us astronaut with vicious smile",
        "among us video game character",
        "imposter among us cartoon mascot"
    ],
    "aiml.jpg": [
        "artificial intelligence machine learning neural network diagram",
        "ai technology futuristic network concept",
        "deep learning circuit visual infographic",
        "data science ai machine learning chart"
    ],
    "WIN_20261016_21_52_16_Pro.jpg": [
        "indoor personal webcam portrait snapshot",
        "room ambient photo webcam capture",
        "user webcam frame portrait indoor",
        "selfie webcam face capture"
    ]
}

print("Computing semantic concept anchors...")
anchors_dict = {}
with torch.no_grad():
    for fn, prompts in CONCEPT_PROMPTS.items():
        tokens = engine.tokenizer(prompts).to("cpu")
        embs = engine.model.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        mean_emb = embs.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        anchors_dict[fn] = mean_emb.squeeze(0).cpu().numpy().astype(np.float32)

anchors_file = DATA_DIR / "concept_anchors.npz"
np.savez_compressed(anchors_file, **anchors_dict)

prompts_file = DATA_DIR / "concept_prompts.json"
with open(prompts_file, "w", encoding="utf-8") as f:
    json.dump(CONCEPT_PROMPTS, f, indent=2)

print(f"Saved {len(anchors_dict)} concept anchors to {anchors_file}")
