import sys, os
sys.path.insert(0, '.')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from server import engine

engine.load_clip()
for q in ['human', 'person', 'a photo of a human', 'a photo of a person', 'man']:
    print(f'=== Query: {q} ===')
    results, rec = engine.hybrid_search(q, top_k=5)
    for r in results:
        print(f"  {r['filename']:50} Final: {r['final']:.4f} CLIP: {r['clip']:.4f}")
