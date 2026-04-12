"""
Module: label_derivation_pipeline.py
Description: A case study visualization of the label binarization process. 
             Compares aggregated mean annotator consensus against final 
             binary ground-truth targets using a majority-rule threshold.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP
PATH_TO_DATASET = "MLPC2026_dataset_development"
FILE_NAME = "000001.npz"
file_path = os.path.join(PATH_TO_DATASET, "audio_features", FILE_NAME)

d = np.load(file_path, allow_pickle=True)
hop_size = 0.5  # Each segment is 0.5 seconds apart
raw_mean = np.mean(d["annotations"], axis=2)
final_binary = (raw_mean > 0.5).astype(int)

# Create a time array in seconds for the X-axis
num_segments = raw_mean.shape[0]
time_in_seconds = np.arange(num_segments) * hop_size

# 2. PLOTTING
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# TOP: Aggregated
sns.heatmap(
    raw_mean.T, ax=axes[0], cmap="YlOrRd", cbar=True, cbar_kws={"label": "Mean Overlap"}
)
axes[0].set_title(
    f"Aggregation: Mean Annotator Consensus ({FILE_NAME})",
    fontsize=14,
    fontweight="bold",
)

# BOTTOM: Binary
sns.heatmap(
    final_binary.T,
    ax=axes[1],
    cmap="Greens",
    cbar=True,
    cbar_kws={"ticks": [0, 1], "label": "Binary Label"},
)
axes[1].set_title(
    "Binarization: Final Labels (Threshold > 0.5)", fontsize=14, fontweight="bold"
)

# FIX THE X-AXIS TO SHOW ACTUAL SECONDS
# We show a label every 10 segments (which is every 5 seconds)
tick_positions = np.arange(0, num_segments, 10)
tick_labels = [f"{int(t)}s" for t in tick_positions * hop_size]

for ax in axes:
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_ylabel("Sound Classes")

axes[1].set_xlabel("Time (Seconds)")

plt.tight_layout()
plt.savefig("2b_final_correct_seconds.png", dpi=300)
plt.show()

print(
    f"Success! The graph now shows the true duration: {time_in_seconds[-1] + 0.5} seconds."
)
