import numpy as np


class SimpleLabelEncoder:
    """Works like sklearn's LabelEncoder (fit / transform / inverse_transform)."""

    def fit(self, labels):
        self.classes_ = sorted(set(labels))
        return self

    def transform(self, labels):
        idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([idx[l] for l in labels])

    def inverse_transform(self, ids):
        ids = np.asarray(ids).ravel()
        return np.array([str(self.classes_[int(i)]) for i in ids], dtype=object)


class BernoulliNB:
    """Naive Bayes for 0/1 symptom features."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        priors, probs = [], []

        for c in self.classes_:
            Xc = X[y == c]
            priors.append(len(Xc) / len(X))
            probs.append(
                (Xc.sum(axis=0) + self.alpha)
                / (len(Xc) + 2 * self.alpha)
            )

        self.log_prior_ = np.log(np.array(priors))
        self.p_ = np.array(probs)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        log_p = np.log(self.p_)
        log_np = np.log(1 - self.p_)

        scores = (
            X @ log_p.T
            + (1 - X) @ log_np.T
            + self.log_prior_
        )

        return self.classes_[np.argmax(scores, axis=1)]