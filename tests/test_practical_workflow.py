"""Check raw-feature explanations and schema failure behavior."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1] / "Tools/PracticalWorkflow_examples/workflow_helpers.py"
SPEC = importlib.util.spec_from_file_location("workflow_helpers", PATH)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)


def test_exact_shapley_additive_model_and_target_label_index():
    class Model:
        classes_ = np.array(["returned", "kept"])
        def predict_proba(self, X):
            p = .2 + .1 * X.a.to_numpy() + .2 * X.b.to_numpy()
            return np.column_stack([p, 1-p])
    background = pd.DataFrame({'a': [0., 1.], 'b': [0., 1.]})
    query = pd.DataFrame({'a': [2.], 'b': [1.]})
    result = helpers.exact_raw_shapley(Model(), query, background, 'returned')
    assert result['target_index'] == 0
    assert result['base_value'] == pytest.approx(.35)
    np.testing.assert_allclose(result['values'], [.15, .1])
    assert result['base_value'] + result['values'].sum() == pytest.approx(.6)
    opposite = helpers.exact_raw_shapley(Model(), query, background, 'kept')
    np.testing.assert_allclose(opposite['values'], -result['values'])


def test_raw_shapley_dummy_and_interaction_symmetry():
    class Interaction:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            p = .8 * X.a.to_numpy() * X.b.to_numpy()
            return np.column_stack([1-p, p])
    background = pd.DataFrame({'a': [0.], 'b': [0.], 'unused': [9.]})
    query = pd.DataFrame({'a': [1.], 'b': [1.], 'unused': [4.]})
    result = helpers.exact_raw_shapley(Interaction(), query, background, 1)
    np.testing.assert_allclose(result['values'], [.4, .4, 0.])


def test_schema_reorders_columns_and_normalizes_missing():
    frame = pd.DataFrame({'category': [None, 'known'], 'value': ['1.5', np.nan]})
    result = helpers.validate_raw_frame(frame, ['value'], ['category'])
    assert list(result) == ['value', 'category']
    assert result.value.iloc[0] == 1.5
    assert pd.isna(result.category.iloc[0])


@pytest.mark.parametrize('frame', [pd.DataFrame({'wrong': [1.], 'category': ['x']}),
                                  pd.DataFrame({'value': [np.inf], 'category': ['x']}),
                                  pd.DataFrame({'value': [1.], 'category': [3]}),
                                  pd.DataFrame({'value': ['bad'], 'category': ['x']})])
def test_schema_rejects_invalid_inputs(frame):
    with pytest.raises(ValueError):
        helpers.validate_raw_frame(frame, ['value'], ['category'])
