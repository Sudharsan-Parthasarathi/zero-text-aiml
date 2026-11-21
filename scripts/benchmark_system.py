"""
ZERO-TEXT & ZERO-ARCADE BENCHMARKING SUITE
Quantitatively benchmarks:
1. CLIP Text & Image Encoding Latency (FP32 vs INT8 Dynamic Quantization)
2. FAISS Vector Search Scaling Curve (N=100, N=1,000, N=10,000, N=50,000, N=100,000)
3. Multimodal Search Latency (CLIP + RapidOCR + Color Palette Matching)
4. Memory Footprint and Throughput (Queries per second, P50, P95, P99)
Generates BENCHMARK_REPORT.json and a formatted summary.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Configure thread safety for Windows CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import faiss
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline import get_engine, DIMENSIONS, MODEL_NAME, PRETRAINED


def get_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def benchmark_faiss_scaling():
    print("\n" + "=" * 60)
    print(" 1. FAISS VECTOR RETRIEVAL SCALING BENCHMARK (512-dim)")
    print("=" * 60)

    scales = [100, 1_000, 10_000, 50_000, 100_000]
    num_queries = 20
    k = 15
    dim = DIMENSIONS
    results = {}

    rng = np.random.RandomState(42)
    # Normalized query vectors
    q_vecs = rng.randn(num_queries, dim).astype("float32")
    q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)

    for n in scales:
        print(f"Testing corpus size: {n:,} vectors...")
        # Normalized corpus vectors
        c_vecs = rng.randn(n, dim).astype("float32")
        c_vecs = c_vecs / np.linalg.norm(c_vecs, axis=1, keepdims=True)

        # IndexFlatIP (exact cosine similarity)
        index = faiss.IndexFlatIP(dim)
        t0 = time.perf_counter()
        index.add(c_vecs)
        index_build_ms = (time.perf_counter() - t0) * 1000

        # Query benchmarking
        latencies = []
        for i in range(num_queries):
            q = q_vecs[i : i + 1]
            t_start = time.perf_counter()
            scores, indices = index.search(q, k)
            latencies.append((time.perf_counter() - t_start) * 1000)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        mean_lat = float(np.mean(latencies))
        qps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        results[str(n)] = {
            "corpus_size": n,
            "build_time_ms": round(index_build_ms, 2),
            "mean_latency_ms": round(mean_lat, 3),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
            "qps": round(qps, 1),
            "memory_mb": round((n * dim * 4) / (1024 * 1024), 2),
        }
        print(f"   -> Mean: {mean_lat:.3f} ms | P95: {p95:.3f} ms | QPS: {qps:,.0f} | RAM: {results[str(n)]['memory_mb']} MB")

    return results


def benchmark_model_quantization(engine):
    print("\n" + "=" * 60)
    print(" 2. MODEL INFERENCE & QUANTIZATION BENCHMARK (FP32 vs INT8)")
    print("=" * 60)

    engine.load_clip()
    model_fp32 = engine.model

    test_queries = [
        "a photo of a cute domestic cat",
        "red sports car on highway",
        "deep space cosmic nebula",
        "python code text on monitor",
        "vintage 35mm rangefinder camera"
    ]

    # Pre-tokenize
    tokens = engine.tokenizer(test_queries).to("cpu")

    # Benchmark FP32
    print("Benchmarking FP32 CLIP text encoder...")
    fp32_latencies = []
    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _ = model_fp32.encode_text(tokens[:1])

    for q in test_queries:
        t_tokens = engine.tokenizer([q]).to("cpu")
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model_fp32.encode_text(t_tokens)
                _ = out / out.norm(dim=-1, keepdim=True)
            fp32_latencies.append((time.perf_counter() - t0) * 1000)

    fp32_mean = float(np.mean(fp32_latencies))
    fp32_p95 = float(np.percentile(fp32_latencies, 95))
    fp32_mem = get_process_memory_mb()
    print(f"   FP32 Mean: {fp32_mean:.2f} ms | P95: {fp32_p95:.2f} ms")

    # Benchmark INT8 Dynamic Quantization
    print("Creating INT8 Dynamic Quantized model...")
    try:
        model_int8 = torch.quantization.quantize_dynamic(model_fp32, dtype=torch.qint8)
        if hasattr(model_int8, "transformer"):
            model_int8.transformer.get_cast_dtype = lambda: torch.float32
        if hasattr(model_int8, "visual"):
            model_int8.visual.get_cast_dtype = lambda: torch.float32
        model_int8.eval()

        # Warmup INT8
        with torch.inference_mode():
            for _ in range(3):
                _ = model_int8.encode_text(tokens[:1])

        int8_latencies = []
        for q in test_queries:
            t_tokens = engine.tokenizer([q]).to("cpu")
            for _ in range(5):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model_int8.encode_text(t_tokens)
                    _ = out / out.norm(dim=-1, keepdim=True)
                int8_latencies.append((time.perf_counter() - t0) * 1000)

        int8_mean = float(np.mean(int8_latencies))
        int8_p95 = float(np.percentile(int8_latencies, 95))
        speedup = fp32_mean / int8_mean if int8_mean > 0 else 1.0

        # Verify cosine fidelity between FP32 and INT8 embeddings
        with torch.inference_mode():
            e_fp32 = model_fp32.encode_text(tokens)
            e_fp32 = e_fp32 / e_fp32.norm(dim=-1, keepdim=True)
            e_int8 = model_int8.encode_text(tokens)
            e_int8 = e_int8 / e_int8.norm(dim=-1, keepdim=True)
            cos_sim = float((e_fp32 * e_int8).sum(dim=-1).mean())

        print(f"   INT8 Mean: {int8_mean:.2f} ms | P95: {int8_p95:.2f} ms")
        print(f"   Speedup: {speedup:.2f}x faster | Semantic Cosine Similarity: {cos_sim * 100:.2f}% match")

        quant_results = {
            "supported": True,
            "fp32_mean_ms": round(fp32_mean, 2),
            "fp32_p95_ms": round(fp32_p95, 2),
            "int8_mean_ms": round(int8_mean, 2),
            "int8_p95_ms": round(int8_p95, 2),
            "speedup_factor": round(speedup, 2),
            "cosine_similarity_fidelity": round(cos_sim, 4),
        }
    except Exception as e:
        print(f"   Dynamic quantization notice: {e}")
        quant_results = {"supported": False, "error": str(e), "fp32_mean_ms": round(fp32_mean, 2)}

    return quant_results


def benchmark_multimodal_pipeline(engine):
    print("\n" + "=" * 60)
    print(" 3. MULTIMODAL END-TO-END PIPELINE LATENCY")
    print("=" * 60)

    # 1. Hybrid Search Latency
    hybrid_queries = ["golden retriever dog", "red supercar", "python code calculator", "blue ocean"]
    h_latencies = []
    for q in hybrid_queries:
        for _ in range(3):
            t0 = time.perf_counter()
            _ = engine.hybrid_search(q, auto_weights=True, top_k=15)
            h_latencies.append((time.perf_counter() - t0) * 1000)

    h_mean = float(np.mean(h_latencies))
    h_p95 = float(np.percentile(h_latencies, 95))
    print(f"   Hybrid Search Latency: {h_mean:.2f} ms (P95: {h_p95:.2f} ms)")

    # 2. Sketch Search Latency
    sample_sketch = Image.new("RGB", (224, 224), (255, 255, 255))
    s_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = engine.sketch_search(sample_sketch, top_k=15)
        s_latencies.append((time.perf_counter() - t0) * 1000)

    s_mean = float(np.mean(s_latencies))
    s_p95 = float(np.percentile(s_latencies, 95))
    print(f"   Sketch Search Latency: {s_mean:.2f} ms (P95: {s_p95:.2f} ms)")

    # 3. Image Probe / Drag-to-Search Latency
    p_latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = engine.image_probe_search(sample_sketch, filename="test_probe.png", top_k=15)
        p_latencies.append((time.perf_counter() - t0) * 1000)

    p_mean = float(np.mean(p_latencies))
    p_p95 = float(np.percentile(p_latencies, 95))
    print(f"   Drag-to-Search Latency: {p_mean:.2f} ms (P95: {p_p95:.2f} ms)")

    return {
        "hybrid_search_ms": round(h_mean, 2),
        "hybrid_p95_ms": round(h_p95, 2),
        "sketch_search_ms": round(s_mean, 2),
        "sketch_p95_ms": round(s_p95, 2),
        "image_probe_ms": round(p_mean, 2),
        "image_probe_p95_ms": round(p_p95, 2),
    }


def run_full_benchmark():
    t_start_total = time.time()
    print("=" * 60)
    print(" ZERO-TEXT SYSTEM BENCHMARK & CAPACITY REPORT")
    print(f" Model: {MODEL_NAME} ({PRETRAINED}) | CPU Architecture")
    print("=" * 60)

    engine = get_engine()
    engine.load_clip()

    # Step 1: FAISS Scaling
    faiss_metrics = benchmark_faiss_scaling()

    # Step 2: Model Quantization
    quant_metrics = benchmark_model_quantization(engine)

    # Step 3: Multimodal Pipeline
    pipeline_metrics = benchmark_multimodal_pipeline(engine)

    total_time = round(time.time() - t_start_total, 2)
    ram_mb = round(get_process_memory_mb(), 2)

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu_threads": os.cpu_count(),
            "process_ram_mb": ram_mb,
            "model": MODEL_NAME,
            "dimensions": DIMENSIONS,
        },
        "faiss_scaling": faiss_metrics,
        "quantization_and_inference": quant_metrics,
        "multimodal_pipeline_latency": pipeline_metrics,
        "total_benchmark_duration_seconds": total_time,
    }

    report_path = BASE_DIR / "BENCHMARK_REPORT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    print("\n" + "=" * 60)
    print(f" BENCHMARK COMPLETE ({total_time}s)")
    print(f" Saved full report to: {report_path.name}")
    print("=" * 60)
    return final_report


if __name__ == "__main__":
    run_full_benchmark()
