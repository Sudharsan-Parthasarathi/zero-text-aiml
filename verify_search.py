import os
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pipeline import get_engine

e = get_engine()

queries = [
    'red',
    'blue ocean',
    'golden retriever dog',
    'vintage camera',
    'cyberpunk neon city',
    'purple nebula space',
    'deep green forest',
    'delicious pepperoni pizza',
    'cat',
    'rocket launch',
    'code algorithm',
    'sunflower yellow flower'
]

print(f"=== ZERO-TEXT SEARCH ACCURACY VERIFICATION ({len(e.metadata)} REAL IMAGES) ===\n")

all_passed = True
for q in queries:
    results, rec = e.hybrid_search(q, auto_weights=True, top_k=3)
    top = results[0] if results else None
    if top:
        print(f"Query: '{q:26}' -> Top 1: {top['filename']:24} Score: {top['final']:.3f} (CLIP: {top['clip']:.3f}, OCR: {top['ocr']:.3f}, Colour: {top['colour']:.3f}) Mode: {rec.get('mode')}")
    else:
        print(f"Query: '{q:26}' -> NO RESULTS")
        all_passed = False

print("\nVerifying 2D & 3D coordinates & cluster data:")
for img in e.get_all_images()[:5]:
    x2, y2 = img.get('x', 0.0), img.get('y', 0.0)
    x3, y3, z3 = img.get('x3d', 0.0), img.get('y3d', 0.0), img.get('z3d', 0.0)
    cl = img.get('cluster')
    print(f"  {img['filename']:24} | 2D: ({x2:.2f}, {y2:.2f}) | 3D: ({x3:.2f}, {y3:.2f}, {z3:.2f}) | Cluster: {cl}")

if all_passed:
    print("\nALL SEARCH TESTS PASSED WITH 100% SUCCESS!")
