import torch
import open_clip
import time

print("=" * 45)
print("       ZERO-TEXT AI ENGINE TEST")
print("=" * 45)

device = "cpu"

print("\nDevice:", device)
print("Loading CLIP model...")

start = time.time()

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k",
    device=device
)

model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

print("Model loaded in:", round(time.time() - start, 2), "seconds")

# Generate text embedding
prompts = [
    "A blue neural network architecture diagram",
    "A funny cat meme",
    "A handwritten mathematics notebook"
]

tokens = tokenizer(prompts)

with torch.no_grad():
    text_features = model.encode_text(tokens)

# Normalize embeddings
text_features = text_features / text_features.norm(
    dim=-1, keepdim=True
)

print("\nEmbedding shape:", text_features.shape)
print("Number of prompts:", len(prompts))
print("Embedding dimensions:", text_features.shape[1])

print("\nSUCCESS! ZERO-TEXT CLIP ENGINE IS WORKING.")
