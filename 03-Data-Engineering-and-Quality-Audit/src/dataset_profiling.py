"""
Module: dataset_profiling.py
Description: Quantifies key characteristics of the derived ground-truth labels,
             including event frequency, temporal duration statistics, and 
             spatial co-occurrence patterns.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# ==========================================
# 1. SETUP & DATA COLLECTION
# ==========================================
PATH_TO_DATASET = "MLPC2026_dataset_development"
features_dir = os.path.join(PATH_TO_DATASET, "audio_features")

if not os.path.exists(features_dir):
    print(f"Error: Folder '{features_dir}' not found.")
    exit()

npz_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npz")])

# Stats containers
class_event_counts = defaultdict(int)
class_durations = defaultdict(list)
# Get class names from the first file to initialize co-occurrence matrix
d_init = np.load(os.path.join(features_dir, npz_files[0]), allow_pickle=True)
class_names = list(d_init["class_names"])
num_classes = len(class_names)
co_matrix = np.zeros((num_classes, num_classes))

print(f"Analyzing {len(npz_files)} files for label characteristics...")

# ==========================================
# 2. PROCESSING LOOP
# ==========================================
for file_id in npz_files:
    d = np.load(os.path.join(features_dir, file_id), allow_pickle=True)

    # 2.b Logic: Derive binary labels using 0.5 threshold
    # Shape: [Time, Class, Annotator] -> [Time, Class]
    binary_labels = (np.mean(d["annotations"], axis=2) > 0.5).astype(int)

    # --- FREQUENCY & DURATION ANALYSIS ---
    for c_idx, name in enumerate(class_names):
        col = binary_labels[:, c_idx]

        # Find contiguous blocks of 1s (onsets and offsets)
        padded = np.pad(col, (1, 1), mode="constant")
        diff = np.diff(padded)
        onsets = np.where(diff == 1)[0]
        offsets = np.where(diff == -1)[0]

        # Calculate duration in seconds (1 segment = 0.5 seconds hop)
        durations = (offsets - onsets) * 0.5

        if len(durations) > 0:
            class_event_counts[name] += len(durations)
            class_durations[name].extend(durations)

    # --- CO-OCCURRENCE ANALYSIS ---
    # Find which classes are present at least once in this specific file
    present_indices = np.where(binary_labels.any(axis=0))[0]
    for i in present_indices:
        for j in present_indices:
            co_matrix[i, j] += 1

# ==========================================
# 3. GENERATING THE 3 GRAPHS
# ==========================================
names = sorted(class_names)
counts = [class_event_counts[n] for n in names]
avg_durs = [np.mean(class_durations[n]) if n in class_durations else 0 for n in names]

# GRAPH 1: Event Frequency (Bar Chart)
plt.figure(figsize=(12, 6))
plt.bar(names, counts, color="steelblue", edgecolor="black", alpha=0.8)
plt.title("Characteristic 1: Event Frequency", fontsize=14, fontweight="bold")
plt.ylabel("Total Number of Events")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("2c_1_frequency.png")
plt.show()

# GRAPH 2: Average Duration (Line Chart)
plt.figure(figsize=(12, 6))
plt.plot(names, avg_durs, color="crimson", marker="o", linewidth=2, markersize=8)
plt.title("Characteristic 2: Average Event Duration", fontsize=14, fontweight="bold")
plt.ylabel("Average Duration (Seconds)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("2c_2_duration.png")
plt.show()

# GRAPH 3: Co-occurrence Heatmap
plt.figure(figsize=(12, 10))
df_cm = pd.DataFrame(co_matrix, index=class_names, columns=class_names)
sns.heatmap(df_cm, annot=False, cmap="YlGnBu", linewidths=0.5)
plt.title(
    "Characteristic 3: Class Co-occurrence (File Count)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("2c_3_cooccurrence.png")
plt.show()

print(
    "\nAnalysis Complete! Saved: 2c_1_frequency.png, 2c_2_duration.png, 2c_3_cooccurrence.png"
)
