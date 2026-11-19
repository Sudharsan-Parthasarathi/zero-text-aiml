import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ZeroTextEngine
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

engine = ZeroTextEngine()
engine.load_clip()

cat_img = Image.open("data/sample_images/cat.jpg")

# What if the user drags a cat on grass? (Green background)
# Or a cat with different colors?
# Or a flipped/rotated cat?
# Or what if filename is 'cat.png', 'cat (1).jpg', 'cat_image.jpg', 'kitten.jpg'?

# Let's test with a simulated external cat:
# cat.jpg with yellow/warm lighting (like sunlight/grass outdoors):
warm_cat = ImageEnhance.Color(cat_img).enhance(1.5)
res_warm = engine.image_probe_search(warm_cat, filename="cat.jpg", top_k=5)
print("Warm cat filename='cat.jpg':")
for r in res_warm["results"][:3]:
    print(f"  {r['rank']}. {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | Col={r['colour']:.4f}")

# What if filename is generic 'image.png' or 'download.jpg'?
res_warm_gen = engine.image_probe_search(warm_cat, filename="download.jpg", top_k=5)
print("\nWarm cat filename='download.jpg':")
for r in res_warm_gen["results"][:3]:
    print(f"  {r['rank']}. {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | Col={r['colour']:.4f}")

# What if cat is pasted on grass background?
w, h = cat_img.size
grass_bg = Image.new("RGB", (w + 100, h + 100), (34, 139, 34)) # forest green
grass_bg.paste(cat_img, (50, 50))
res_grass = engine.image_probe_search(grass_bg, filename="download.jpg", top_k=5)
print("\nCat on green background filename='download.jpg':")
for r in res_grass["results"][:3]:
    print(f"  {r['rank']}. {r['filename']} | Final={r['final']:.4f} | CLIP={r['clip']:.4f} | Col={r['colour']:.4f}")
