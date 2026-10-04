import os
import pickle
import numpy as np
from simple_model import BernoulliNB, SimpleLabelEncoder

# Feature order MUST match app.py:
# fever, cough, headache, fatigue, body_pain, sore_throat, nausea, rash
FEATURES = ["fever", "cough", "headache", "fatigue",
            "body_pain", "sore_throat", "nausea", "rash"]

PROFILES = {
    "Common Cold":     [0.30, 0.90, 0.30, 0.40, 0.20, 0.85, 0.05, 0.02],
    "Flu":             [0.90, 0.80, 0.70, 0.85, 0.85, 0.50, 0.20, 0.02],
    "COVID-19":        [0.80, 0.85, 0.60, 0.85, 0.65, 0.55, 0.20, 0.03],
    "Malaria":         [0.95, 0.10, 0.85, 0.80, 0.75, 0.05, 0.55, 0.02],
    "Dengue":          [0.95, 0.10, 0.85, 0.85, 0.90, 0.05, 0.50, 0.60],
    "Typhoid":         [0.90, 0.15, 0.70, 0.85, 0.50, 0.10, 0.60, 0.20],
    "Food Poisoning":  [0.40, 0.02, 0.30, 0.60, 0.40, 0.02, 0.95, 0.02],
    "Migraine":        [0.02, 0.02, 0.98, 0.50, 0.15, 0.02, 0.70, 0.02],
    "Chickenpox":      [0.75, 0.20, 0.40, 0.60, 0.40, 0.15, 0.10, 0.98],
}

SAMPLES_PER_DISEASE = 400
rng = np.random.default_rng(42)

X, y = [], []
for disease, probs in PROFILES.items():
    probs = np.array(probs)
    data = (rng.random((SAMPLES_PER_DISEASE, len(FEATURES))) < probs).astype(int)
    X.append(data)
    y += [disease] * SAMPLES_PER_DISEASE

X = np.vstack(X)
y = np.array(y)

label_encoder = SimpleLabelEncoder().fit(y)
y_enc = label_encoder.transform(y)

# simple 80/20 split
perm = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = perm[:split], perm[split:]

model = BernoulliNB().fit(X[train_idx], y_enc[train_idx])
acc = (model.predict(X[test_idx]) == y_enc[test_idx]).mean()
print(f"Test accuracy: {acc:.2%}")

os.makedirs("ml_model", exist_ok=True)
with open("ml_model/disease_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("ml_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Saved: ml_model/disease_model.pkl and ml_model/label_encoder.pkl")