"""Small numerical helpers for the executable XAI evaluation study.

The helpers specify perturbation semantics explicitly; none is a universal
measure of explanation correctness.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.metrics import roc_auc_score


def make_synthetic(seed=11, correlated=False, n_samples=900):
    """Generate a probabilistic target with known direct functional inputs.

    x1 is a proxy for x0 only in the correlated setting. It never appears in the
    generating logit, but can legitimately help predict y through dependence.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 6))
    if correlated:
        X[:, 1] = 0.97 * X[:, 0] + np.sqrt(1 - 0.97**2) * X[:, 1]
    logit = 1.8 * X[:, 0] + 1.1 * np.sin(X[:, 2]) + 0.8 * X[:, 2] * X[:, 3] - 0.3
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    return X, y, probability


def array_digest(array):
    """Stable fingerprint including shape, dtype, and contiguous array bytes."""
    arr = np.ascontiguousarray(array)
    header = f"{arr.shape}|{arr.dtype.str}|".encode()
    return hashlib.sha256(header + arr.tobytes()).hexdigest()


def top_k_jaccard(left, right, k=3):
    """Per-row overlap of the largest absolute scores; ties use feature order.

    A zero vector is undefined and returns NaN, rather than perfect stability.
    Scores from different estimands must not be compared just because this
    function can accept them.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Scores must be equally shaped two-dimensional arrays.")
    if not 1 <= k <= left.shape[1]:
        raise ValueError("k must lie between 1 and the number of features.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Scores must be finite.")
    values = []
    for a, b in zip(left, right):
        if not np.any(a) or not np.any(b):
            values.append(np.nan)
            continue
        ia = set(np.argsort(-np.abs(a), kind="stable")[:k])
        ib = set(np.argsort(-np.abs(b), kind="stable")[:k])
        values.append(len(ia & ib) / len(ia | ib))
    return np.asarray(values)


def signed_cosine(left, right):
    """Per-row signed cosine, with NaN for undefined zero-norm pairs."""
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Scores must be equally shaped two-dimensional arrays.")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Scores must be finite.")
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide((left * right).sum(axis=1), denominator,
                     out=np.full(len(left), np.nan), where=denominator > 0)


def deletion_curves(predict_probability, X, scores, donors):
    """Replace features in descending absolute score order.

    ``donors`` has shape (n_draws, n_features) and is shared across methods.
    All removed features in a draw come from one donor, preserving their joint
    values; their relation with retained features is still broken. The same
    donor is used along each trajectory. Returns absolute probability change
    averaged across donors at each k, plus raw perturbed rows for diagnostics.
    """
    X, scores, donors = (np.asarray(a, dtype=float) for a in (X, scores, donors))
    if X.ndim != 2 or scores.shape != X.shape:
        raise ValueError("X and scores must be equally shaped two-dimensional arrays.")
    if donors.ndim != 2 or len(donors) == 0 or donors.shape[1] != X.shape[1]:
        raise ValueError("donors must contain at least one matching feature vector.")
    if not all(np.isfinite(a).all() for a in (X, scores, donors)):
        raise ValueError("Inputs must be finite.")
    order = np.argsort(-np.abs(scores), axis=1, kind="stable")
    n, d = X.shape
    perturbed = np.broadcast_to(X[None, None, :, :], (d + 1, len(donors), n, d)).copy()
    for k in range(1, d + 1):
        for row in range(n):
            selected = order[row, :k]
            # Loop over donors to avoid NumPy advanced-index axis reordering.
            for draw, donor in enumerate(donors):
                perturbed[k, draw, row, selected] = donor[selected]
    original = np.asarray(predict_probability(X), dtype=float)
    changed = np.asarray(predict_probability(perturbed.reshape(-1, d)), dtype=float)
    if original.shape != (n,) or changed.shape != ((d + 1) * len(donors) * n,):
        raise ValueError("predict_probability must return one scalar probability per row.")
    if not np.isfinite(original).all() or not np.isfinite(changed).all():
        raise ValueError("Model probabilities must be finite.")
    if np.any((original < 0) | (original > 1)) or np.any((changed < 0) | (changed > 1)):
        raise ValueError("Model probabilities must lie in [0, 1].")
    changed = changed.reshape(d + 1, len(donors), n)
    curve = np.abs(changed - original[None, None, :]).mean(axis=1).T
    area = np.trapezoid(curve, x=np.arange(d + 1) / d, axis=1)
    return curve, area, perturbed


def grouped_permutation_auc(model, X, y, groups, seed=11, n_repeats=8):
    """AUC decrease; grouped columns share one row permutation per repeat.

    The same permutation is used for every group at a given repeat so that
    contrasts are paired. The method breaks dependence with unpermuted columns.
    """
    X = np.asarray(X, dtype=float)
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError("This study requires classes [0, 1].")
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(X)) for _ in range(n_repeats)]
    output = {}
    for name, columns in groups.items():
        columns = list(columns)
        if not columns or len(set(columns)) != len(columns) or any(c < 0 or c >= X.shape[1] for c in columns):
            raise ValueError("Each group must contain valid, unique feature indices.")
        differences = []
        for permutation in permutations:
            shuffled = X.copy()
            shuffled[:, columns] = X[permutation][:, columns]
            differences.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        output[name] = np.asarray(differences)
    return output
