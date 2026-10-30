"""Reference-game and instrumentation tests independent of method rankings."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / 'Tools/Evaluation_examples/regression_helpers.py'
SPEC = importlib.util.spec_from_file_location('regression_helpers', PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def test_exact_reference_matches_linear_game_and_dummy_feature():
    prediction = lambda x: 4 + 2*x[:,0] - 3*x[:,1]
    cases = np.array([[2.,1.,8.], [-1.,3.,5.]])
    background = np.array([[0.,0.,0.], [1.,2.,3.]])
    values, baseline, predicted = helpers.exact_background_shapley(prediction, cases, background)
    np.testing.assert_allclose(values, (cases-background.mean(axis=0))*[2.,-3.,0.], atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=1)+baseline, prediction(cases), atol=1e-12)
    np.testing.assert_allclose(predicted, prediction(cases))


def test_interaction_is_shared_symmetrically():
    values, _, _ = helpers.exact_background_shapley(lambda x:x[:,0]*x[:,1], [[2.,2.]], [[0.,0.]])
    np.testing.assert_allclose(values, [[2.,2.]])


def test_regression_removal_supports_unbounded_outputs():
    curve, area = helpers.removal_response(lambda x:10*x[:,0], [[8.,2.]], [[4.,0.]], [[0.,0.]])
    np.testing.assert_allclose(curve, [[0.,80.,80.]])
    np.testing.assert_allclose(area, [60.])


def test_resource_measurement_counts_prediction_rows_and_restores_method():
    class Model:
        def predict(self, X):
            return np.asarray(X).sum(axis=1)
    model = Model()
    original = model.predict
    result, resource = helpers.measured_explanation(model, lambda:model.predict(np.ones((7,3))))
    np.testing.assert_allclose(result, 3)
    assert resource['predict_api_calls'] == 1 and resource['predict_api_rows'] == 7
    assert resource['seconds'] >= 0 and resource['peak_traced_bytes'] > 0
    assert model.predict == original


def test_instrumentation_restores_prediction_after_failure():
    class Model:
        def predict(self, X):
            return np.zeros(len(X))
    model = Model()
    original = model.predict
    def fail():
        raise RuntimeError('deliberate')
    with pytest.raises(RuntimeError, match='deliberate'):
        helpers.measured_explanation(model, fail)
    assert model.predict == original
