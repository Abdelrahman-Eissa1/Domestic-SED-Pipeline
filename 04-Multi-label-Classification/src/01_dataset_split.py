import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Config - Ensure these files are in the same folder!
FEATURES_DIR = "audio_features"
METADATA_CSV = "metadata.csv"
RANDOM_SEED = 42
os.makedirs("visuals", exist_ok=True)

print("Step 1: Aggregating labels and performing Collector-Level split...")

if not os.path.exists(METADATA_CSV):
    print(f"ERROR: {METADATA_CSV} not found in the current directory!")
    exit()

meta_df = pd.read_csv(METADATA_CSV)
all_features, all_labels, all_filenames = [], [], []

npz_files = sorted(f for f in os.listdir(FEATURES_DIR) if f.endswith(".npz"))
if not npz_files:
    print(f"ERROR: No .npz files found in {FEATURES_DIR}!")
    exit()

for fname in npz_files:
    npz = np.load(os.path.join(FEATURES_DIR, fname), allow_pickle=True)
    ann = npz["annotations"].astype(float)
    T, C, A = ann.shape
    labels = (ann.sum(axis=2) > A / 2).astype(np.int8)
    
    keys = ["mfcc_mean", "mfcc_d_mean", "mfcc_d2_mean", "zcr_mean", "centroid_mean", "flux_mean", "energy_mean", "contrast_mean"]
    feat = np.concatenate([npz[k].reshape(T, -1) for k in keys], axis=1)
    
    all_features.append(feat)
    all_labels.append(labels)
    all_filenames.extend([fname.replace(".npz", ".wav")] * T)

X, y = np.concatenate(all_features), np.concatenate(all_labels)
class_names = list(np.load(os.path.join(FEATURES_DIR, npz_files[0]), allow_pickle=True)["class_names"])

# Save class names immediately
with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

# Split by collector_id
fname_to_col = dict(zip(meta_df["filename"], meta_df["collector_id"]))
indices = np.array([fname_to_col.get(f, "UNK") for f in all_filenames])
u_cols = np.unique(indices)
c_train, c_temp = train_test_split(u_cols, test_size=0.3, random_state=RANDOM_SEED)
c_val, c_test = train_test_split(c_temp, test_size=0.5, random_state=RANDOM_SEED)

# Save splits
masks = [np.isin(indices, c) for c in [c_train, c_val, c_test]]
suffixes = ["_train", "_val", "_test"]
for mask, suff in zip(masks, suffixes):
    np.save(f"X{suff}.npy", X[mask])
    np.save(f"y{suff}.npy", y[mask])

# Visual
plt.figure(figsize=(10, 5))
plt.bar(class_names, y[masks[0]].mean(axis=0)*100, color='#4C72B0')
plt.xticks(rotation=45, ha="right")
plt.title("Class Distribution (Training Set)")
plt.ylabel("Presence Frequency (%)")
plt.tight_layout()
plt.savefig("visuals/01_class_distribution.png")
print("Step 1 Complete. Files generated: X/y train/val/test.npy, class_names.txt, visuals/01_class_distribution.png")
