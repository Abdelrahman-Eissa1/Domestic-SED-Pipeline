import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

print("Step 2: Normalizing features...")
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_val_scaled.npy", X_val_scaled)
np.save("X_test_scaled.npy", X_test_scaled)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Done. Scaled features saved.")
