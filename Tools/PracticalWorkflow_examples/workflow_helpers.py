"""Raw-feature schema checks and exact finite-background Shapley values."""
from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd


def validate_raw_frame(frame, numerical, categorical):
    """Validate columns/types without learning preprocessing from new rows.

    Missing values are allowed because the fitted pipeline owns imputation.
    Unknown string categories are allowed; report them separately before use.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Supply a pandas DataFrame with named raw features.")
    expected = list(numerical) + list(categorical)
    if not frame.columns.is_unique:
        raise ValueError("Duplicate feature names are not allowed.")
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={missing}; unexpected={extra}")
    result = frame.loc[:, expected].copy()
    for name in numerical:
        try:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain numbers or missing values.") from error
        if np.isinf(result[name].to_numpy()).any():
            raise ValueError(f"{name} must not contain infinity.")
    for name in categorical:
        for value in result[name]:
            if isinstance(value, str):
                if value == "__missing__":
                    raise ValueError("__missing__ is reserved for the pipeline's imputer.")
            elif not (value is None or value is pd.NA or isinstance(value, (float, np.floating)) and np.isnan(value)):
                raise ValueError(f"{name} must contain strings or missing values.")
        # Normalize None/pd.NA to the SimpleImputer missing marker.
        result[name] = result[name].astype(object).where(result[name].notna(), np.nan)
    return result


def unknown_categories(frame, fitted_preprocessor, categorical):
    """Report values absent from fitted encoder vocabularies, without refitting."""
    encoder = fitted_preprocessor.named_transformers_["categorical"].named_steps["encode"]
    return {name: sorted(set(frame[name].dropna()) - set(known))
            for name, known in zip(categorical, encoder.categories_)
            if set(frame[name].dropna()) - set(known)}


def transformed_groups(fitted_preprocessor, numerical, categorical):
    """Map each original feature to its encoded value and missing-indicator columns.

    This mapping supports the notebook's no-drop, no-infrequent-category encoder
    and a numeric imputer with indicators followed by a one-to-one scaler.
    """
    groups = {name: [] for name in list(numerical) + list(categorical)}
    num_start = fitted_preprocessor.output_indices_["numerical"].start
    imputer = fitted_preprocessor.named_transformers_["numerical"].named_steps["impute"]
    if np.isnan(imputer.statistics_).any():
        raise ValueError("All-missing numerical training columns are outside this example's schema.")
    for index, name in enumerate(numerical):
        groups[name].append(num_start + index)
    for indicator_position, source_position in enumerate(imputer.indicator_.features_):
        groups[numerical[source_position]].append(num_start + len(numerical) + indicator_position)
    encoder = fitted_preprocessor.named_transformers_["categorical"].named_steps["encode"]
    if encoder.drop is not None or encoder.min_frequency is not None or encoder.max_categories is not None:
        raise ValueError("The example mapping requires an encoder without dropped/grouped categories.")
    start = fitted_preprocessor.output_indices_["categorical"].start
    for name, vocabulary in zip(categorical, encoder.categories_):
        groups[name].extend(range(start, start + len(vocabulary)))
        start += len(vocabulary)
    return groups


def exact_raw_shapley(model, query, background, target_label):
    """Exact Shapley values for a finite-background raw-feature replacement game.

    v(S) = mean_b model.predict_proba(query[S], b[not S])[target_label].
    Every coalition is evaluated; complexity is 2**d * n_background rows.
    This is marginal background replacement, not conditional or causal SHAP.
    """
    if not isinstance(query, pd.DataFrame) or not isinstance(background, pd.DataFrame):
        raise TypeError("query and background must be raw-feature DataFrames.")
    if len(query) != 1 or len(background) == 0 or list(query.columns) != list(background.columns):
        raise ValueError("Use one query row and a nonempty background with identical ordered columns.")
    columns = list(query.columns)
    d = len(columns)
    if not 1 <= d <= 12:
        raise ValueError("Exact enumeration is restricted to 1–12 raw features.")
    labels = list(model.classes_)
    if target_label not in labels:
        raise ValueError("Requested target label is absent from the model's classes_.")
    target_index = labels.index(target_label)
    coalitions = []
    for mask in range(1 << d):
        rows = background.copy()
        for j, column in enumerate(columns):
            if mask & (1 << j):
                rows[column] = query.iloc[0][column]
        coalitions.append(rows)
    probabilities = model.predict_proba(pd.concat(coalitions, ignore_index=True))[:, target_index]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("The model must return finite class probabilities.")
    game = probabilities.reshape(1 << d, len(background)).mean(axis=1)
    values = np.zeros(d)
    for j in range(d):
        for mask in range(1 << d):
            if not mask & (1 << j):
                size = mask.bit_count()
                weight = factorial(size) * factorial(d - size - 1) / factorial(d)
                values[j] += weight * (game[mask | (1 << j)] - game[mask])
    return dict(values=values, base_value=game[0], prediction=game[-1],
                coalition_values=game, target_index=target_index,
                evaluated_rows=len(probabilities))
