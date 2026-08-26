import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import json

X_train, y_train = np.load("X_train_scaled.npy"), np.load("y_train.npy")
X_val, y_val = np.load("X_val_scaled.npy"), np.load("y_val.npy")

depths = [5, 10, 20, None]
results = []
for d in depths:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, class_weight="balanced", n_jobs=-1, random_state=42)
    clf = MultiOutputClassifier(rf).fit(X_train, y_train)
    score = f1_score(y_val, clf.predict(X_val), average="macro")
    results.append(score)

plt.figure(figsize=(8, 4))
plt.plot([str(d) for d in depths], results, marker='o', color='#55A868')
plt.title("Random Forest Sweep: Effect of max_depth")
plt.xlabel("Max Depth")
plt.ylabel("Val Macro F1")
plt.savefig("visuals/04_rf_sweep.png")

with open("rf_best_params.json", "w") as f:
    json.dump({"n_estimators": 100, "max_depth": 20}, f)
