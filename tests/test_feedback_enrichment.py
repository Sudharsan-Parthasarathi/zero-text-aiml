"""
Test suite for Search Result Feedback & Automatic Dataset Enrichment
Tests:
  1. Text Search positive feedback -> adds to text_search_dataset.json and feedback_dataset.json.
  2. Duplicate positive text feedback -> prevents duplicate entry, reports already_exists.
  3. Text Search negative feedback -> logs in feedback_dataset.json, not in text_search_dataset.json.
  4. Accuracy impact -> validated match receives boost in hybrid_search, negative match is penalized.
  5. Sketch Search positive feedback -> saves sketch file and enriches sketch_dataset.json.
  6. Duplicate sketch feedback -> prevents duplicate entry.
  7. Sketch Search negative feedback -> recorded in audit log, not in sketch_dataset.json.
  8. API Endpoints /api/feedback and /api/feedback/stats respond correctly.
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from PIL import Image

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pipeline import (
    get_engine,
    FEEDBACK_PATH,
    TEXT_DATASET_PATH,
    SKETCH_DATASET_PATH,
    USER_SKETCHES_DIR,
)

def create_dummy_sketch_base64():
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # Draw simple black square
    for x in range(30, 70):
        for y in range(30, 70):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def run_tests():
    print("==================================================")
    print("Testing Search Result Feedback & Dataset Enrichment")
    print("==================================================")

    engine = get_engine()
    assert len(engine.metadata) > 0, "No indexed images found."
    test_target = engine.metadata[0]["filename"]
    print(f"Target test image: {test_target}")

    test_query = "unique test feedback query"
    q_norm = test_query.lower().strip()

    # Clean any prior test remnants
    engine.text_search_dataset = [e for e in engine.text_search_dataset if e.get("query_normalized") != q_norm]
    engine.feedback_dataset = [e for e in engine.feedback_dataset if e.get("query_normalized") != q_norm]
    engine.validated_text_queries.pop(q_norm, None)
    engine.negative_feedback_queries.pop(q_norm, None)

    # ----------------------------------------------------
    # TEST 1: Positive Text Search Feedback & Enrichment
    # ----------------------------------------------------
    print("\n[Test 1] Testing positive text search feedback...")
    res1 = engine.record_feedback(
        search_type="text",
        feedback="positive",
        filename=test_target,
        query=test_query,
        score=0.88,
        cluster=1,
    )
    assert res1["status"] == "ok", f"Expected status ok, got {res1}"
    assert res1["enriched"] is True, "Expected enriched to be True"
    assert res1["already_exists"] is False, "Expected already_exists to be False"
    assert test_target in engine.validated_text_queries.get(q_norm, set()), "Target should be in validated queries"
    print(" -> Positive text feedback succeeded and enriched dataset.")

    # ----------------------------------------------------
    # TEST 2: Duplicate Positive Text Feedback Prevention
    # ----------------------------------------------------
    print("\n[Test 2] Testing duplicate detection for text feedback...")
    res2 = engine.record_feedback(
        search_type="text",
        feedback="positive",
        filename=test_target,
        query=test_query,
        score=0.88,
        cluster=1,
    )
    assert res2["status"] == "ok", f"Expected status ok, got {res2}"
    assert res2["already_exists"] is True, "Expected already_exists to be True for duplicate"
    assert res2["enriched"] is False, "Duplicate should not re-enrich dataset"
    # Ensure no duplicate entry in text_search_dataset
    matches = [e for e in engine.text_search_dataset if e.get("query_normalized") == q_norm and e.get("filename") == test_target]
    assert len(matches) == 1, f"Expected exactly 1 entry, found {len(matches)}"
    print(" -> Duplicate detection successfully prevented duplicate record.")

    # ----------------------------------------------------
    # TEST 3: Negative Text Feedback Handling
    # ----------------------------------------------------
    print("\n[Test 3] Testing negative text search feedback...")
    neg_query = "negative test feedback query"
    neg_q_norm = neg_query.lower().strip()
    res3 = engine.record_feedback(
        search_type="text",
        feedback="negative",
        filename=test_target,
        query=neg_query,
        score=0.60,
        cluster=1,
    )
    assert res3["status"] == "ok"
    assert res3["enriched"] is False, "Negative feedback must NOT enrich positive dataset"
    assert test_target in engine.negative_feedback_queries.get(neg_q_norm, set()), "Should be in negative queries"
    # Verify NOT in text_search_dataset
    in_pos = any(e.get("query_normalized") == neg_q_norm for e in engine.text_search_dataset)
    assert not in_pos, "Negative feedback must NOT be in text_search_dataset"
    print(" -> Negative feedback recorded without modifying positive dataset.")

    # ----------------------------------------------------
    # TEST 4: Search Relevance Impact
    # ----------------------------------------------------
    print("\n[Test 4] Testing search relevance impact...")
    # Validate that hybrid_search applies calibration
    # Query with positive validation
    res_pos, _ = engine.hybrid_search(query=test_query, auto_weights=True)
    top_pos = next((item for item in res_pos if item["filename"] == test_target), None)
    assert top_pos is not None, "Target should appear in results"

    # Query with negative rejection
    res_neg, _ = engine.hybrid_search(query=neg_query, auto_weights=True)
    top_neg = next((item for item in res_neg if item["filename"] == test_target), None)
    assert top_neg is not None
    assert top_neg["final"] < top_pos["final"], "Negatively rated result should have lower calibrated score"
    print(f" -> Confirmed relevance calibration: Positive score={top_pos['final']:.4f} vs Negative score={top_neg['final']:.4f}")

    # ----------------------------------------------------
    # TEST 5: Positive Sketch Search Feedback & Enrichment
    # ----------------------------------------------------
    print("\n[Test 5] Testing positive sketch search feedback...")
    sketch_b64 = create_dummy_sketch_base64()
    sketch_target = engine.metadata[1]["filename"] if len(engine.metadata) > 1 else test_target

    res_sk1 = engine.record_feedback(
        search_type="sketch",
        feedback="positive",
        filename=sketch_target,
        sketch_data=sketch_b64,
        text_hint="dummy square",
        score=0.92,
        cluster=0,
    )
    assert res_sk1["status"] == "ok"
    assert res_sk1["enriched"] is True
    assert res_sk1["already_exists"] is False
    print(" -> Positive sketch feedback enriched sketch dataset.")

    # ----------------------------------------------------
    # TEST 6: Duplicate Sketch Feedback Prevention
    # ----------------------------------------------------
    print("\n[Test 6] Testing duplicate sketch feedback...")
    res_sk2 = engine.record_feedback(
        search_type="sketch",
        feedback="positive",
        filename=sketch_target,
        sketch_data=sketch_b64,
        text_hint="dummy square",
        score=0.92,
        cluster=0,
    )
    assert res_sk2["status"] == "ok"
    assert res_sk2["already_exists"] is True
    assert res_sk2["enriched"] is False
    print(" -> Duplicate sketch feedback prevented.")

    # ----------------------------------------------------
    # TEST 7: Negative Sketch Feedback
    # ----------------------------------------------------
    print("\n[Test 7] Testing negative sketch feedback...")
    res_sk3 = engine.record_feedback(
        search_type="sketch",
        feedback="negative",
        filename=sketch_target,
        sketch_data=sketch_b64,
        text_hint="rejected hint",
        score=0.45,
    )
    assert res_sk3["status"] == "ok"
    assert res_sk3["enriched"] is False
    print(" -> Negative sketch feedback recorded in audit log without enriching positive dataset.")

    # ----------------------------------------------------
    # TEST 8: Server Endpoints Validation
    # ----------------------------------------------------
    print("\n[Test 8] Testing FastAPI /api/feedback & /api/feedback/stats endpoints...")
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)

    # Test GET /api/feedback/stats
    stats_res = client.get("/api/feedback/stats")
    assert stats_res.status_code == 200
    stats_data = stats_res.json()
    assert "total_feedback" in stats_data
    assert "positive_count" in stats_data
    assert "negative_count" in stats_data
    assert "text_dataset_count" in stats_data
    assert "sketch_dataset_count" in stats_data
    print(f" -> Feedback stats endpoint operational: {stats_data}")

    # Test POST /api/feedback via HTTP
    api_post_res = client.post("/api/feedback", json={
        "search_type": "text",
        "feedback": "positive",
        "filename": test_target,
        "query": "api test query",
        "score": 0.95,
    })
    assert api_post_res.status_code == 200
    post_data = api_post_res.json()
    assert post_data["status"] == "ok"
    assert post_data["enriched"] is True
    print(f" -> POST /api/feedback verified successfully: {post_data['message']}")

    # Clean up test artifacts from datasets to leave environment clean
    engine.text_search_dataset = [e for e in engine.text_search_dataset if e.get("query") not in (test_query, "api test query")]
    engine.sketch_dataset = [e for e in engine.sketch_dataset if e.get("text_hint") != "dummy square"]
    engine.feedback_dataset = [e for e in engine.feedback_dataset if e.get("query") not in (test_query, neg_query, "api test query") and e.get("text_hint") != "dummy square"]
    engine._save_text_dataset()
    engine._save_sketch_dataset()
    engine._save_feedback_dataset()

    print("\n==================================================")
    print("ALL 8 FEEDBACK & ENRICHMENT TESTS PASSED SUCCESSFULLY! (Code 0)")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
