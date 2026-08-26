import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, confusion_matrix
import os

# Create visuals directory
os.makedirs("visuals", exist_ok=True)

# 1. LOAD DATA
print("Step 6: Final Test Evaluation...")
X_train = np.load("X_train_scaled.npy")
X_val = np.load("X_val_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

with open("class_names.txt", "r") as f:
    class_names = [l.strip() for l in f.readlines()]

# Combine Train + Val for final retraining
X_trainval = np.concatenate([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val], axis=0)

# 2. BASELINE: Random Sampling
class_freq = y_train.mean(axis=0)
np.random.seed(42)
y_pred_base = (np.random.rand(*y_test.shape) < class_freq).astype(int)
f1_base_per = f1_score(y_test, y_pred_base, average=None, zero_division=0)
f1_base_macro = np.mean(f1_base_per)

# 3. RANDOM FOREST: Retrain with best params (n=100, depth=20)
print("  Retraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1)
clf_rf = MultiOutputClassifier(rf, n_jobs=-1).fit(X_trainval, y_trainval)
y_pred_rf = clf_rf.predict(X_test)
f1_rf_per = f1_score(y_test, y_pred_rf, average=None, zero_division=0)
f1_rf_macro = np.mean(f1_rf_per)

# 4. MLP: Retrain with best params (hidden=(128,64), lr=0.001)
print("  Retraining MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), learning_rate_init=0.001, early_stopping=True, random_state=42)
clf_mlp = MultiOutputClassifier(mlp, n_jobs=-1).fit(X_trainval, y_trainval)
y_pred_mlp = clf_mlp.predict(X_test)
f1_mlp_per = f1_score(y_test, y_pred_mlp, average=None, zero_division=0)
f1_mlp_macro = np.mean(f1_mlp_per)

# 5. VISUAL 1: Grouped Bar Chart (Comparison)
print("  Generating Comparison Plot...")
x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width, f1_base_per, width, label=f"Baseline (Macro={f1_base_macro:.4f})", color="#aaaaaa")
ax.bar(x,         f1_mlp_per,  width, label=f"MLP (Macro={f1_mlp_macro:.4f})",      color="#4C72B0")
ax.bar(x + width, f1_rf_per,   width, label=f"RF (Macro={f1_rf_macro:.4f})",       color="#55A868")

ax.set_xticks(x)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_ylabel("F1 Score")
ax.set_title("Test Set Per-Class F1: Baseline vs MLP vs Random Forest")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/06_model_comparison.png", dpi=150)

# 6. VISUAL 2: Confusion Matrices for Best Model (RF)
print("  Generating Confusion Matrices...")
best_pred = y_pred_rf
fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()

for i, cls in enumerate(class_names):
    cm = confusion_matrix(y_test[:, i], best_pred[:, i])
    ax = axes[i]
    ax.imshow(cm, cmap="Blues", alpha=0.7)
    ax.set_title(cls, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Abs", "Pres"], fontsize=8)
    ax.set_yticklabels(["Abs", "Pres"], fontsize=8)
    # Add counts
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="black")

plt.suptitle(f"Per-Class Confusion Matrices (Random Forest Test Macro F1={f1_rf_macro:.4f})", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("visuals/06_confusion_matrices.png", dpi=150)

# 7. SAVE PREDICTIONS
np.save("y_pred_rf.npy", y_pred_rf)
print("Step 6 Complete. Visuals saved to visuals/ folder.")
