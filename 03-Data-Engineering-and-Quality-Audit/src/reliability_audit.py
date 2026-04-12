"""
Module: reliability_audit.py
Description: Performs a comprehensive quality audit of human annotations 
             using Jaccard Similarity (IoU) and Metadata Tag Confirmation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
PATH_TO_DATASET = "MLPC2026_dataset_development"
features_dir = os.path.join(PATH_TO_DATASET, "audio_features")

if not os.path.exists(features_dir):
    print(f"Error: Folder '{features_dir}' not found. Check your path.")
    exit()

# Get all feature files
npz_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npz")])
feature_file_map = {
    os.path.splitext(f)[0]: os.path.join(features_dir, f) for f in npz_files
}

# ==========================================
# 2. DATA PROCESSING & ANALYSIS
# ==========================================
# Trackers for Part 1 & 2: Inter-Annotator Agreement (Human vs Human)
overall_jaccards = []
class_intersections = defaultdict(int)
class_unions = defaultdict(int)

# Trackers for Part 3: Tag Confirmation (Annotator vs Collector)
tag_counts = defaultdict(int)
tag_hits = defaultdict(int)

print(f"Analyzing {len(feature_file_map)} files...")

for file_id in feature_file_map:
    # Load file
    d = np.load(feature_file_map[file_id], allow_pickle=True)

    # Extract data
    ann = d["annotations"]  # Shape: [Time, Class, Annotator]
    class_names = list(d["class_names"])
    target_tags = d["target_classes"]

    # Create binary mask (True where a sound was marked)
    binary = ann > 0

    # --- PART 1 & 2: Inter-Annotator Agreement (Only if >= 2 annotators) ---
    if ann.shape[2] >= 2:
        # Overall Jaccard for this file
        inter_file = binary.all(axis=2).sum()
        union_file = binary.any(axis=2).sum()
        if union_file > 0:
            overall_jaccards.append(inter_file / union_file)

        # Accumulate per-class Jaccard data
        for c_idx, name in enumerate(class_names):
            c_slice = binary[:, c_idx, :]
            class_intersections[name] += c_slice.all(axis=1).sum()
            class_unions[name] += c_slice.any(axis=1).sum()

    # --- PART 3: Tag Confirmation (Collector vs Annotator) ---
    # Find which classes were marked by ANY human annotator
    marked_by_any = binary.any(axis=(0, 2))
    annotated_classes = [class_names[i] for i, val in enumerate(marked_by_any) if val]

    for tag in target_tags:
        tag_counts[tag] += 1
        if tag in annotated_classes:
            tag_hits[tag] += 1

# ==========================================
# 3. STATISTICAL SUMMARIES
# ==========================================
# Calculate results
overall_jaccards = np.array(overall_jaccards)
class_results = sorted(
    [
        (n, (class_intersections[n] / class_unions[n]) * 100)
        for n in class_intersections
        if class_unions[n] > 0
    ],
    key=lambda x: x[1],
    reverse=True,
)
tag_results = sorted(
    [(t, (tag_hits[t] / tag_counts[t]) * 100) for t in tag_counts if tag_counts[t] > 0],
    key=lambda x: x[1],
    reverse=True,
)

print("\n" + "=" * 40)
print(f"OVERALL MEAN AGREEMENT: {overall_jaccards.mean():.1%}")
print(
    f"OVERALL TAG CONFIRMATION: {(sum(tag_hits.values())/sum(tag_counts.values())):.1%}"
)
print("=" * 40 + "\n")

# ==========================================
# 4. GENERATING GRAPHS
# ==========================================

# PLOT 1: Overall Agreement Distribution (Histogram)
plt.figure(figsize=(10, 4))
plt.hist(overall_jaccards * 100, bins=20, color="teal", edgecolor="black", alpha=0.7)
plt.axvline(
    overall_jaccards.mean() * 100,
    color="red",
    linestyle="--",
    label=f"Mean: {overall_jaccards.mean():.1%}",
)
plt.title("Distribution of Inter-Annotator Agreement", fontweight="bold")
plt.xlabel("Jaccard Agreement Score (%)")
plt.ylabel("Number of Files")
plt.legend()
plt.tight_layout()
plt.savefig("2a_1_overall_histogram.png")
plt.show()

# PLOT 2: Class-wise Agreement (Horizontal Bar Chart)
names, scores = zip(*class_results)
plt.figure(figsize=(10, 8))
plt.barh(names, scores, color="skyblue", edgecolor="navy")
plt.xlabel("Jaccard Agreement (%)")
plt.title("Agreement per Sound Class (Human vs Human)", fontweight="bold")
plt.gca().invert_yaxis()
for i, v in enumerate(scores):
    plt.text(v + 1, i, f"{v:.1f}%", va="center")
plt.tight_layout()
plt.savefig("2a_2_class_agreement.png")
plt.show()

# PLOT 3: Tag Confirmation Rate (Horizontal Bar Chart)
t_names, t_scores = zip(*tag_results)
plt.figure(figsize=(10, 8))
plt.barh(t_names, t_scores, color="plum", edgecolor="purple")
plt.xlabel("Confirmation Rate (%)")
plt.title("User Tag Confirmation Rate (Annotator vs Collector)", fontweight="bold")
plt.gca().invert_yaxis()
for i, v in enumerate(t_scores):
    plt.text(v + 1, i, f"{v:.1f}%", va="center")
plt.tight_layout()
plt.savefig("2a_3_tag_confirmation.png")
plt.show()

print("Processing complete. Three PNG graphs saved.")
