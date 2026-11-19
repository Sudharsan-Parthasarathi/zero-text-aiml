import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
from PIL import Image
import numpy as np

engine = ZeroTextEngine()
engine.load_clip()

text_cat = engine.compute_text_embedding("a photo of a cat")
text_dog = engine.compute_text_embedding("a photo of a dog")

cat_img = Image.open("data/sample_images/cat.jpg")
dog_img = Image.open("data/sample_images/golden_retriever.jpg")

cat_vec = engine.compute_image_embedding(cat_img)
dog_vec = engine.compute_image_embedding(dog_img)

print("cat_img dot text_cat:", float((cat_vec @ text_cat.T)[0,0]))
print("cat_img dot text_dog:", float((cat_vec @ text_dog.T)[0,0]))
print("dog_img dot text_cat:", float((dog_vec @ text_cat.T)[0,0]))
print("dog_img dot text_dog:", float((dog_vec @ text_dog.T)[0,0]))

print("cat_vec dot dog_vec:", float((cat_vec @ dog_vec.T)[0,0]))

# Let's inspect cat.jpg image properties
print("cat_img mode:", cat_img.mode, "size:", cat_img.size)
