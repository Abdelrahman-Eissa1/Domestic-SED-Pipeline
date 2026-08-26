import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

X_train, y_train = np.load("X_train_scaled.npy"), np.load("y_train.npy")
X_val, y_val = np.load("X_val_scaled.npy"), np.load("y_val.npy")

archs = [(64,), (128, 64), (256, 128)]
lrs = [0.001, 0.01]
score_matrix = np.zeros((len(archs), len(lrs)))

for i, a in enumerate(archs):
    for j, l in enumerate(lrs):
        mlp = MLPClassifier(hidden_layer_sizes=a, learning_rate_init=l, early_stopping=True, random_state=42)
        clf = MultiOutputClassifier(mlp).fit(X_train, y_train)
        score_matrix[i, j] = f1_score(y_val, clf.predict(X_val), average="macro")

plt.figure(figsize=(6, 4))
sns.heatmap(score_matrix, annot=True, xticklabels=lrs, yticklabels=[str(a) for a in archs], cmap="YlGnBu")
plt.title("MLP Hyperparameter Heatmap (Macro F1)")
plt.savefig("visuals/05_mlp_heatmap.png")
