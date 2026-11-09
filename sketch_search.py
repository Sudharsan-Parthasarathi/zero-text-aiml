import os
import sys
from pathlib import Path
from PIL import Image

# Critical Windows environment variables before importing torch/faiss
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pipeline import get_engine

TOP_K = 5

print("=" * 55)
print("       ZERO-TEXT HIGH-PRECISION SKETCH SEARCH")
print("=" * 55)

engine = get_engine()
engine.load_clip()


def search_sketch(sketch_path: str, top_k: int = TOP_K, text_hint: str = ""):
    sketch_path = Path(sketch_path)
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch not found: {sketch_path}")

    print(f"\nProcessing sketch: {sketch_path.name}")
    img = Image.open(sketch_path)
    results = engine.sketch_search(img, top_k=top_k, text_hint=text_hint)

    print("\n" + "=" * 55)
    print("          SKETCH SEARCH RESULTS")
    print("=" * 55)

    for r in results:
        print(f"\nRank: {r['rank']}")
        print(f"Image: {r['filename']}")
        print(f"Similarity (Raw Cosine): {r['similarity'] * 100:.2f}%")
        print(f"Calibrated Match Score:  {r['final'] * 100:.2f}%")
        print(f"Cluster: {r['cluster']}")
        print(f"Path: {r['path']}")

    print("\nSketch search completed successfully!")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print('  python sketch_search.py "path/to/sketch.png" [optional_text_hint]')
        raise SystemExit

    sketch_path = sys.argv[1]
    hint = sys.argv[2] if len(sys.argv) > 2 else ""
    search_sketch(sketch_path, text_hint=hint)
    print("\nSUCCESS! ZERO-TEXT SKETCH SEARCH FINISHED.")
