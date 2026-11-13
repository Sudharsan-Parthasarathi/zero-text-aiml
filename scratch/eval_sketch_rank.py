"""One-off sketch ranking diagnostic. Not part of the app."""
import json
from pathlib import Path

from PIL import Image

from pipeline import SKETCHES_DIR, get_engine

eng = get_engine()
eng.load_clip()
print("INDEX", eng.index.ntotal, "PROJ", eng.sketch_projection is not None)

ds = json.loads(Path("data/sketch_dataset.json").read_text(encoding="utf-8"))
target = {}
for item in ds:
    target.setdefault(item["sketch_file"], set()).add(item["filename"])

cases = [
    ("cat.jpg", {"cat.jpg"}),
    ("rocket.jpg", {"nasa_rocket.jpg"}),
    ("mountain.jpg", {"mountain.jpg"}),
]
for rel, names in target.items():
    cases.append((rel, names))

for sf, names in cases:
    img = Image.open(SKETCHES_DIR / sf)
    res = eng.sketch_search(img, top_k=5, is_canvas_sketch=True)
    hit = next((r["rank"] for r in res if r["filename"] in names), None)
    print(f"\n{sf} expect {names} rank={hit}")
    for r in res:
        mark = " *" if r["filename"] in names else ""
        print(
            f"  {r['rank']} {r['filename'][:42]:42} "
            f"{r['final']:.3f} clip {r['clip']:.3f} col {r['colour']:.3f}{mark}"
        )
