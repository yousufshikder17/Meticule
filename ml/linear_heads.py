"""Deterministic multinomial logistic regression (softmax) heads over fixed feature vectors (NumPy + SciPy only).

Used by models that put a trained classifier on top of frozen features: transformer embeddings and audio MFCC summaries.
Objective: sum of cross-entropy over the training rows + (l2 / 2) * ||W||^2 (the intercept is not penalised), which is
scikit-learn's multinomial LogisticRegression with C = 1 / l2. Optimised by L-BFGS-B from zero with a fixed iteration cap, so
the same rows give the same head. Features are standardised with statistics fitted on the training rows only.
Serialised as base64 little-endian float64 arrays with their shapes: no pickle.
"""
import base64

import numpy as np
import scipy.optimize
import scipy.special

MAX_FEATURES = 8192
MAX_CLASSES = 1000


def standardizer(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return mean, np.where(std > 0.0, std, 1.0)


def fit(x, y, classes, l2, max_iter):
    """x: (n, d) finite features; y: class indices in [0, classes). Returns the head as plain arrays."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] > MAX_FEATURES or not 2 <= classes <= MAX_CLASSES:
        raise ValueError("Head features are not a finite bounded matrix")
    mean, std = standardizer(x)
    z = (x - mean) / std
    n, d = z.shape
    onehot = np.eye(classes)[np.asarray(y, dtype=np.int64)]

    def objective(theta):
        w, b = theta[: d * classes].reshape(d, classes), theta[d * classes:]
        logits = z @ w + b
        log_p = logits - scipy.special.logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(onehot * log_p) + 0.5 * l2 * np.sum(w * w)
        residual = np.exp(log_p) - onehot
        return loss, np.concatenate([(z.T @ residual + l2 * w).ravel(), residual.sum(axis=0)])

    result = scipy.optimize.minimize(objective, np.zeros(d * classes + classes), jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter)})
    w, b = result.x[: d * classes].reshape(d, classes), result.x[d * classes:]
    if not (np.isfinite(w).all() and np.isfinite(b).all()):
        raise ValueError("Head training diverged")
    return {"mean": mean, "std": std, "weights": w, "bias": b}


def predict_proba(head, x):
    z = (np.asarray(x, dtype=np.float64) - head["mean"]) / head["std"]
    return scipy.special.softmax(z @ head["weights"] + head["bias"], axis=1)


def encode(head):
    return {name: {"shape": list(value.shape), "data": base64.b64encode(np.ascontiguousarray(value, dtype="<f8").tobytes()).decode()}
            for name, value in head.items()}


def decode(payload, features, classes):
    """Rebuilds a head and checks every array's shape against the expected feature and class counts."""
    expected = {"mean": [features], "std": [features], "weights": [features, classes], "bias": [classes]}
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError("Malformed head")
    head = {}
    for name, shape in expected.items():
        entry = payload[name]
        if not isinstance(entry, dict) or entry.get("shape") != shape or not isinstance(entry.get("data"), str):
            raise ValueError("Malformed head")
        raw = base64.b64decode(entry["data"], validate=True)
        if len(raw) != 8 * int(np.prod(shape)):
            raise ValueError("Malformed head")
        value = np.frombuffer(raw, dtype="<f8").reshape(shape).astype(np.float64)
        if not np.isfinite(value).all():
            raise ValueError("Malformed head")
        head[name] = value
    return head
