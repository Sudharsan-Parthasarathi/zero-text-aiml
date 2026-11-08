
import json
import faiss
import umap
import hdbscan
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# ZERO-TEXT: INTELLIGENT IMAGE CLUSTERING
# UMAP + HDBSCAN
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "data" / "image_index.faiss"
METADATA_PATH = BASE_DIR / "data" / "image_metadata.json"

OUTPUT_JSON = BASE_DIR / "data" / "cluster_results.json"
OUTPUT_IMAGE = BASE_DIR / "data" / "cluster_visualization.png"

print("=" * 60)
print("       ZERO-TEXT IMAGE CLUSTERING ENGINE")
print("=" * 60)

# ==========================================
# 1. LOAD FAISS INDEX
# ==========================================

print("\nLoading existing FAISS index...")

if not INDEX_PATH.exists():
    raise FileNotFoundError("FAISS index not found!")

if not METADATA_PATH.exists():
    raise FileNotFoundError("Image metadata not found!")

index = faiss.read_index(str(INDEX_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if index.ntotal != len(metadata):
    raise ValueError("FAISS and metadata mismatch!")

if index.ntotal < 3:
    raise ValueError("At least 3 images are required.")

print("FAISS index loaded!")
print("Total images:", index.ntotal)

# ==========================================
# 2. EXTRACT IMAGE EMBEDDINGS
# ==========================================

print("\nExtracting image embeddings...")

embeddings = index.reconstruct_n(0, index.ntotal)

embeddings = np.asarray(embeddings, dtype=np.float32)

print("Embedding matrix:", embeddings.shape)

# Verify normalized embeddings
norms = np.linalg.norm(embeddings, axis=1)

if not np.allclose(norms, 1.0, atol=1e-3):
    raise ValueError(
        "Embeddings are not normalized. "
        "Check the existing FAISS index."
    )

print("Embedding verification successful!")

# ==========================================
# 3. HDBSCAN CLUSTERING
# ==========================================

print("\nRunning HDBSCAN...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric="euclidean",
    cluster_selection_method="eom"
)

cluster_labels = clusterer.fit_predict(embeddings)

unique_clusters = sorted(
    set(cluster_labels) - {-1}
)

noise_count = int(np.sum(cluster_labels == -1))

print("\nHDBSCAN completed!")

print("Clusters discovered:", len(unique_clusters))
print("Unclustered images:", noise_count)

# ==========================================
# 4. UMAP DIMENSIONALITY REDUCTION
# ==========================================

print("\nRunning UMAP (2D & 3D)...")

n_neighbors = min(5, index.ntotal - 1)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

coordinates = reducer.fit_transform(embeddings)

reducer_3d = umap.UMAP(
    n_components=3,
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
coordinates_3d = reducer_3d.fit_transform(embeddings)

print("UMAP completed!")
print("2D coordinates:", coordinates.shape)
print("3D coordinates:", coordinates_3d.shape)

# ==========================================
# 5. PREPARE CLUSTER RESULTS
# ==========================================

results = []

for i, item in enumerate(metadata):

    results.append({
        "filename": item["filename"],
        "path": item["path"],
        "cluster": int(cluster_labels[i]),
        "x": float(coordinates[i][0]),
        "y": float(coordinates[i][1]),
        "z": float(coordinates_3d[i][2]),
        "x3d": float(coordinates_3d[i][0]),
        "y3d": float(coordinates_3d[i][1]),
        "z3d": float(coordinates_3d[i][2]),
        "membership_probability": float(
            clusterer.probabilities_[i]
        )
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nCluster results saved!")
print("Output:", OUTPUT_JSON)

# ==========================================
# 6. DISPLAY CLUSTER INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("              CLUSTER REPORT")
print("=" * 60)

for cluster_id in unique_clusters:

    print(f"\nCLUSTER {cluster_id}")

    for result in results:

        if result["cluster"] == cluster_id:
            print(" -", result["filename"])

if noise_count:

    print("\nNOISE / UNCLUSTERED IMAGES")

    for result in results:

        if result["cluster"] == -1:
            print(" -", result["filename"])

# ==========================================
# 7. GENERATE 2D VISUALIZATION
# ==========================================

print("\nGenerating cluster visualization...")

plt.figure(figsize=(12, 8))

scatter = plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=cluster_labels,
    cmap="tab20",
    s=140,
    alpha=0.85,
    edgecolors="black"
)

for i, item in enumerate(metadata):

    plt.annotate(
        item["filename"],
        (coordinates[i, 0], coordinates[i, 1]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.title("ZERO-TEXT - UMAP + HDBSCAN Image Clusters")

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nVisualization saved!")
print("Output:", OUTPUT_IMAGE)

# ==========================================
# 8. FINAL REPORT
# ==========================================

print("\n" + "=" * 60)
print("       ZERO-TEXT CLUSTERING COMPLETED")
print("=" * 60)

print("Images processed:", len(metadata))
print("Clusters discovered:", len(unique_clusters))
print("Unclustered images:", noise_count)

print("\nSUCCESS! UMAP + HDBSCAN ENGINE WORKING.")
