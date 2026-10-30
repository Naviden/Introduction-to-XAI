"""Explicit reference games and resource measurements for regression examples."""
from __future__ import annotations

import gc
from math import factorial
import time
import tracemalloc

import numpy as np


def exact_background_shapley(predict, cases, background):
    """Enumerate the marginal replacement game, batching all model evaluations.

    Returns exact values for the supplied finite background, not a conditional
    or causal explanation. Limited to 10 features to bound exponential work.
    """
    cases, background = np.asarray(cases, dtype=float), np.asarray(background, dtype=float)
    if cases.ndim != 2 or background.ndim != 2 or cases.shape[1] != background.shape[1]:
        raise ValueError("Cases and background must be 2D with matching features.")
    if not len(cases) or not len(background) or not 1 <= cases.shape[1] <= 10:
        raise ValueError("Use nonempty arrays and 1–10 features.")
    if not np.isfinite(cases).all() or not np.isfinite(background).all():
        raise ValueError("Cases and background must be finite.")
    n, d = cases.shape
    masks = np.arange(1 << d)[:, None] & (1 << np.arange(d))[None, :]
    all_rows = np.where(masks[None, :, None, :] != 0, cases[:, None, None, :], background[None, None, :, :])
    prediction = np.asarray(predict(all_rows.reshape(-1, d)), dtype=float)
    if prediction.shape != (n * (1 << d) * len(background),) or not np.isfinite(prediction).all():
        raise ValueError("predict must return one finite regression prediction per row.")
    game = prediction.reshape(n, 1 << d, len(background)).mean(axis=2)
    values = np.zeros((n, d))
    for j in range(d):
        for mask in range(1 << d):
            if not mask & (1 << j):
                size = mask.bit_count()
                weight = factorial(size) * factorial(d-size-1) / factorial(d)
                values[:, j] += weight * (game[:, mask | (1 << j)] - game[:, mask])
    return values, game[:, 0], game[:, -1]


def removal_response(predict, cases, scores, donors):
    """Regression absolute output-change curves under joint-donor replacement."""
    cases, scores, donors = (np.asarray(a, dtype=float) for a in (cases, scores, donors))
    if cases.ndim != 2 or scores.shape != cases.shape or donors.ndim != 2 or donors.shape[1] != cases.shape[1]:
        raise ValueError("Cases/scores must match and donors must share their feature count.")
    if len(donors) == 0 or not all(np.isfinite(a).all() for a in (cases, scores, donors)):
        raise ValueError("Use finite arrays and at least one donor.")
    n, d = cases.shape
    ranking = np.argsort(-np.abs(scores), axis=1, kind='stable')
    changed = np.broadcast_to(cases[None, None], (d+1, len(donors), n, d)).copy()
    for k in range(1, d+1):
        for i in range(n):
            for draw, donor in enumerate(donors):
                changed[k, draw, i, ranking[i, :k]] = donor[ranking[i, :k]]
    original = np.asarray(predict(cases))
    predicted = np.asarray(predict(changed.reshape(-1, d))).reshape(d+1, len(donors), n)
    curve = np.abs(predicted-original[None, None]).mean(axis=1).T
    return curve, np.trapezoid(curve, x=np.arange(d+1)/d, axis=1)


def measured_explanation(model, operation):
    """Measure one timed run plus a separate Python-allocation-profiled rerun.

    Counts calls to this model instance's public predict API and rows passed to
    it. Tree-native internal operations are not represented by those counts.
    tracemalloc does not measure total resident memory or every native buffer.
    """
    original_predict = model.predict
    counts = {'predict_api_calls': 0, 'predict_api_rows': 0}
    def counted_predict(rows, *args, **kwargs):
        counts['predict_api_calls'] += 1
        counts['predict_api_rows'] += len(rows)
        return original_predict(rows, *args, **kwargs)
    model.predict = counted_predict
    try:
        gc.collect()
        started = time.perf_counter()
        result = operation()
        seconds = time.perf_counter()-started
    finally:
        model.predict = original_predict
    # Do not profile while timing: allocation tracing changes execution cost.
    gc.collect()
    tracemalloc.start()
    try:
        memory_result = operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del memory_result
    resource = dict(seconds=seconds, peak_traced_bytes=int(peak_bytes), **counts)
    return result, resource
