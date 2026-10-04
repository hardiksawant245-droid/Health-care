import os
import pickle
import numpy as np

from simple_model import BernoulliNB, SimpleLabelEncoder


FEATURES = [
    "fever",
    "cough",
    "headache",
    "fatigue",
    "body_pain",
    "sore_throat",
    "nausea",
    "rash"
]


PROFILES = {
    "Common Cold":     [0.35, 0.80, 0.45, 0.55, 0.25, 0.70, 0.15, 0.05],
    "Flu":             [0.85, 0.75, 0.60, 0.85, 0.80, 0.55, 0.35, 0.05],
    "COVID-19":        [0.70, 0.65, 0.55, 0.80, 0.55, 0.45, 0.30, 0.05],
    "Malaria":         [0.90, 0.15, 0.40, 0.75, 0.70, 0.10, 0.45, 0.05],
    "Dengue":          [0.90, 0.10, 0.55, 0.80, 0.85, 0.05, 0.50, 0.20],
    "Typhoid":         [0.75, 0.20, 0.35, 0.70, 0.40, 0.15, 0.60, 0.05],
    "Food Poisoning":  [0.45, 0.10, 0.15, 0.40, 0.20, 0.05, 0.90, 0.05],
    "Migraine":        [0.10, 0.05, 0.90, 0.45, 0.25, 0.05, 0.30, 0.05],
    "Chickenpox":      [0.75, 0.10, 0.20, 0.50, 0.25, 0.05, 0.20, 0.90],
}


rng = np.random.default_rng(42)

X = []
y = []

for disease, probabilities in PROFILES.items():
    probabilities = np.array(probabilities)

    for _ in range(400):
        symptoms = (rng.random(len(FEATURES)) < probabilities).astype(int)
        X.append(symptoms)
        y.append(disease)


X = np.array(X)
y = np.array(y)


encoder = SimpleLabelEncoder()
y_encoded = encoder.fit(y).transform(y)


split = int(0.8 * len(X))

indices = rng.permutation(len(X))

train_idx = indices[:split]
test_idx = indices[split:]


X_train = X[train_idx]
X_test = X[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]


model = BernoulliNB()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


os.makedirs("ml_model", exist_ok=True)


with open("ml_model/disease_model.pkl", "wb") as f:
    pickle.dump(model, f)


with open("ml_model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)


print("Model saved successfully!")
print("Files created inside ml_model folder.")