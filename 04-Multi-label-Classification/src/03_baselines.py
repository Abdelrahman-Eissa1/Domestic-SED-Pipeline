import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

y_train, y_test = np.load("y_train.npy"), np.load("y_test.npy")
with open("class_names.txt", "r") as f: class_names = [l.strip() for l in f.readlines()]

# Baseline: Randomly sample based on training frequency
freqs = y_train.mean(axis=0)
np.random.seed(42)
y_pred_rand = (np.random.rand(*y_test.shape) < freqs).astype(int)
f1_per_class = f1_score(y_test, y_pred_rand, average=None, zero_division=0)

plt.figure(figsize=(10, 5))
plt.bar(class_names, f1_per_class, color='#DD8452')
plt.xticks(rotation=45, ha="right")
plt.title(f"Random Sampling Baseline (Macro F1: {f1_per_class.mean():.4f})")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.savefig("visuals/03_baseline_results.png")
print("Done. Baseline evaluation complete.")
