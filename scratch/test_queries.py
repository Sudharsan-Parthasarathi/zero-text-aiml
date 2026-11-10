"""Test both 'human' and 'coding' queries after fixes."""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from pipeline import ZeroTextEngine

engine = ZeroTextEngine()
engine.load_clip()

print("=" * 60)
print("TEST 1: Query 'human'")
print("=" * 60)
results, rec = engine.hybrid_search("human", top_k=5)
print(f"AI Decider: {rec.get('label', 'N/A')} | Intent: {rec.get('intent', 'N/A')}")
print(f"Weights: CLIP={rec.get('clip')}, OCR={rec.get('ocr')}, Colour={rec.get('colour')}")
for i, r in enumerate(results):
    fn = r.get("filename", "?")
    score = r.get("final", r.get("similarity", 0))
    clip_s = r.get("clip", 0)
    ocr_s = r.get("ocr", 0)
    print(f"  #{i+1}: {fn}  final={score:.4f}  clip={clip_s:.4f}  ocr={ocr_s:.4f}")

expected_human = "WIN_20261016_21_52_16_Pro.jpg"
if results and expected_human in results[0].get("filename", ""):
    print(f"\n✅ PASS: '{expected_human}' is #1 for 'human'")
else:
    top_fn = results[0].get("filename", "?") if results else "NO RESULTS"
    print(f"\n❌ FAIL: Expected '{expected_human}' at #1, got '{top_fn}'")

print("\n" + "=" * 60)
print("TEST 2: Query 'coding'")
print("=" * 60)
results2, rec2 = engine.hybrid_search("coding", top_k=5)
print(f"AI Decider: {rec2.get('label', 'N/A')} | Intent: {rec2.get('intent', 'N/A')}")
print(f"Weights: CLIP={rec2.get('clip')}, OCR={rec2.get('ocr')}, Colour={rec2.get('colour')}")
for i, r in enumerate(results2):
    fn = r.get("filename", "?")
    score = r.get("final", r.get("similarity", 0))
    clip_s = r.get("clip", 0)
    ocr_s = r.get("ocr", 0)
    print(f"  #{i+1}: {fn}  final={score:.4f}  clip={clip_s:.4f}  ocr={ocr_s:.4f}")

expected_coding = "code.jpg"
if results2 and expected_coding in results2[0].get("filename", ""):
    print(f"\n✅ PASS: '{expected_coding}' is #1 for 'coding'")
else:
    top_fn = results2[0].get("filename", "?") if results2 else "NO RESULTS"
    print(f"\n❌ FAIL: Expected '{expected_coding}' at #1, got '{top_fn}'")

print("\n" + "=" * 60)
print("TEST 3: Query 'code' (regression check)")
print("=" * 60)
results3, rec3 = engine.hybrid_search("code", top_k=3)
print(f"AI Decider: {rec3.get('label', 'N/A')} | Intent: {rec3.get('intent', 'N/A')}")
for i, r in enumerate(results3):
    fn = r.get("filename", "?")
    score = r.get("final", r.get("similarity", 0))
    print(f"  #{i+1}: {fn}  final={score:.4f}")

if results3 and "code.jpg" in results3[0].get("filename", ""):
    print(f"\n✅ PASS: 'code.jpg' is #1 for 'code'")
else:
    top_fn = results3[0].get("filename", "?") if results3 else "NO RESULTS"
    print(f"\n❌ FAIL: Expected 'code.jpg' at #1, got '{top_fn}'")
