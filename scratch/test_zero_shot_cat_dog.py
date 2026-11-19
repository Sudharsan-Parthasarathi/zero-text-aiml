import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
import torch
import torch.nn.functional as F
from PIL import Image

engine = ZeroTextEngine()
engine.load_clip()

cat_prompts = [
    "a photo of a cat, kitten, feline, with pointed ears and whiskers",
    "a photo of a cat",
    "a pet cat"
]
dog_prompts = [
    "a photo of a dog, puppy, canine, golden retriever, with a snout",
    "a photo of a dog",
    "a pet dog"
]

cat_tokens = engine.tokenizer(cat_prompts).to("cpu")
dog_tokens = engine.tokenizer(dog_prompts).to("cpu")

with torch.no_grad():
    cat_text_vec = engine.model.encode_text(cat_tokens)
    cat_text_vec = (cat_text_vec / cat_text_vec.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
    cat_text_vec = cat_text_vec / cat_text_vec.norm(dim=-1, keepdim=True)

    dog_text_vec = engine.model.encode_text(dog_tokens)
    dog_text_vec = (dog_text_vec / dog_text_vec.norm(dim=-1, keepdim=True)).mean(dim=0, keepdim=True)
    dog_text_vec = dog_text_vec / dog_text_vec.norm(dim=-1, keepdim=True)

text_stack = torch.cat([cat_text_vec, dog_text_vec], dim=0) # [2, 512]

# Test on cat image:
cat_img = Image.open("data/sample_images/cat.jpg")
cat_t = engine.preprocess(cat_img).unsqueeze(0).to("cpu")
with torch.no_grad():
    cat_feat = engine.model.encode_image(cat_t)
    cat_feat = cat_feat / cat_feat.norm(dim=-1, keepdim=True)
    logits = (cat_feat @ text_stack.t()) * 100.0
    probs = F.softmax(logits, dim=-1)

print("Zero-Shot Classification of cat.jpg:")
print(f"  P(Cat): {probs[0,0].item()*100:.2f}% | P(Dog): {probs[0,1].item()*100:.2f}%")

dog_img = Image.open("data/sample_images/golden_retriever.jpg")
dog_t = engine.preprocess(dog_img).unsqueeze(0).to("cpu")
with torch.no_grad():
    dog_feat = engine.model.encode_image(dog_t)
    dog_feat = dog_feat / dog_feat.norm(dim=-1, keepdim=True)
    logits = (dog_feat @ text_stack.t()) * 100.0
    probs = F.softmax(logits, dim=-1)

print("\nZero-Shot Classification of golden_retriever.jpg:")
print(f"  P(Cat): {probs[0,0].item()*100:.2f}% | P(Dog): {probs[0,1].item()*100:.2f}%")
