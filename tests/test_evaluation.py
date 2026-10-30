"""Numerical checks on evaluation metrics, not expected library rankings."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "Tools/Evaluation_examples/evaluation_helpers.py"
SPEC = importlib.util.spec_from_file_location("evaluation_helpers", PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def test_deletion_respects_absolute_ranking_and_has_shared_endpoint():
    # Only feature 0 changes this probability. Negative attribution still ranks by magnitude.
    probability = lambda x: x[:, 0] / 10
    X = np.array([[8., 2.]])
    faithful, faithful_area, _ = helpers.deletion_curves(probability, X, [[-2., 0.]], [[0., 0.]])
    wrong, wrong_area, _ = helpers.deletion_curves(probability, X, [[0., 2.]], [[0., 0.]])
    np.testing.assert_allclose(faithful, [[0., .8, .8]])
    np.testing.assert_allclose(wrong, [[0., 0., .8]])
    assert faithful_area[0] == pytest.approx(.6)
    assert wrong_area[0] == pytest.approx(.2)


def test_empirical_donors_are_joint_vectors_and_absolute_before_averaging():
    probability = lambda x: x[:, 0]
    curve, _, changed = helpers.deletion_curves(probability, [[.5, .5]], [[2., 1.]], [[0., 0.], [1., 1.]])
    np.testing.assert_allclose(curve, [[0., .5, .5]])
    np.testing.assert_allclose(changed[-1, :, 0], [[0., 0.], [1., 1.]])


def test_similarity_handles_sign_and_undefined_zero_vectors():
    a = np.array([[3., -2., 0., 0.], [0., 0., 0., 0.]])
    b = -a
    np.testing.assert_allclose(helpers.top_k_jaccard(a, b, k=2)[0], 1)
    np.testing.assert_allclose(helpers.signed_cosine(a, b)[0], -1)
    assert np.isnan(helpers.top_k_jaccard(a, b)[1])
    assert np.isnan(helpers.signed_cosine(a, b)[1])


def test_group_permutation_preserves_within_group_relationship():
    class IdenticalColumnsModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            # Duplicated features must remain identical whenever permuted together.
            assert np.array_equal(X[:, 0], X[:, 1])
            p = np.where(X[:, 0] > 0, .9, .1)
            return np.column_stack([1 - p, p])
    values = np.arange(-10, 10)
    X = np.column_stack([values, values])
    result = helpers.grouped_permutation_auc(IdenticalColumnsModel(), X, values > 0,
                                             {"pair": [0, 1]}, n_repeats=5)
    assert len(result["pair"]) == 5
    assert (result["pair"] >= 0).all()


def test_rejects_invalid_probability_and_feature_dimensions():
    with pytest.raises(ValueError, match="probabilities"):
        helpers.deletion_curves(lambda x: np.full(len(x), 2.), [[1., 1.]], [[1., 1.]], [[0., 0.]])
    with pytest.raises(ValueError, match="donors"):
        helpers.deletion_curves(lambda x: x[:, 0], [[1., 1.]], [[1., 1.]], [[0.]])


def test_synthetic_proxy_has_no_direct_generating_effect():
    independent, y_ind, p_ind = helpers.make_synthetic(seed=7, correlated=False)
    correlated, y_cor, p_cor = helpers.make_synthetic(seed=7, correlated=True)
    np.testing.assert_allclose(p_ind, p_cor)
    np.testing.assert_array_equal(y_ind, y_cor)
    assert abs(np.corrcoef(independent[:, :2].T)[0, 1]) < .15
    assert np.corrcoef(correlated[:, :2].T)[0, 1] > .95
