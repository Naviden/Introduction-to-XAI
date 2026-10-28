"""Numerical regression checks for the attribution implementations in notebooks."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")
ROOT = Path(__file__).resolve().parents[1]


def notebook_function(path, name, **namespace):
    """Load the actual notebook function without training or downloading models."""
    notebook = json.loads((ROOT / path).read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                scope = {"tf": tf, "np": np, **namespace}
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, str(path), "exec"), scope)
                return scope[name]
    raise AssertionError(f"Function {name} was not found in {path}")


def linear_classifier():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2,)),
        tf.keras.layers.Dense(2),
    ])
    # Class 0 wins at the baseline; class 1 wins at the input [2, 3].
    model.layers[-1].set_weights([
        np.array([[-1, 2], [0, 3]], dtype=np.float32),
        np.array([4, 0], dtype=np.float32),
    ])
    return model


def test_integrated_gradients_keeps_target_fixed_and_is_complete():
    attribute = notebook_function(
        "Tools/IntegratedGradients_examples/IntegratedGradients_example.ipynb",
        "integrated_gradients",
    )
    model = linear_classifier()
    sample = np.array([2, 3], dtype=np.float32)
    baseline = np.zeros_like(sample)
    values = attribute(model, sample, baseline, num_steps=20)
    np.testing.assert_allclose(values, [4, 9], atol=1e-6)
    expected_change = model(sample[None, :])[0, 1] - model(baseline[None, :])[0, 1]
    np.testing.assert_allclose(values.sum(), expected_change, atol=1e-6)
    np.testing.assert_allclose(
        attribute(model, sample, baseline, num_steps=20, target_class=0), [-2, 0]
    )


def test_integrated_gradients_rejects_zero_steps():
    attribute = notebook_function(
        "Tools/IntegratedGradients_examples/IntegratedGradients_example.ipynb",
        "integrated_gradients",
    )
    with pytest.raises(ValueError, match="num_steps"):
        attribute(linear_classifier(), [1, 2], [0, 0], num_steps=0)


def test_integrated_gradients_trapezoidal_rule_includes_both_endpoints():
    attribute = notebook_function(
        "Tools/IntegratedGradients_examples/IntegratedGradients_example.ipynb",
        "integrated_gradients",
    )
    inputs = tf.keras.Input(shape=(1,))
    model = tf.keras.Model(inputs, tf.keras.layers.Lambda(lambda x: x**2)(inputs))
    np.testing.assert_allclose(
        attribute(model, [3], [1], num_steps=1, target_class=0), [8], atol=1e-6
    )


@pytest.mark.parametrize("class_weight, expected", [(0.0, 0.0), (1.0, 1.0)])
def test_gradcam_produces_finite_normalized_heatmap(class_weight, expected):
    compute = notebook_function(
        "Tools/GradCAM_examples/GradCAM_example.ipynb", "compute_gradcam"
    )
    inputs = tf.keras.Input(shape=(4, 4, 1))
    conv = tf.keras.layers.Conv2D(1, 1, use_bias=False, name="features")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(conv)
    outputs = tf.keras.layers.Dense(1, use_bias=False)(pooled)
    model = tf.keras.Model(inputs, outputs)
    model.get_layer("features").set_weights([np.ones((1, 1, 1, 1), np.float32)])
    model.layers[-1].set_weights([np.array([[class_weight]], np.float32)])
    heatmap = compute(model, np.ones((1, 4, 4, 1), np.float32), 0, "features")
    assert heatmap.shape == (4, 4)
    assert np.isfinite(heatmap).all()
    np.testing.assert_allclose(heatmap, expected)


def test_lime_callback_returns_probabilities():
    model = linear_classifier()
    predict = notebook_function(
        "Tools/LIME_examples/LIME_image.ipynb", "predict_probabilities", model=model
    )
    values = predict(np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float64))
    assert values.shape == (2, 2)
    assert ((0 <= values) & (values <= 1)).all()
    np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(values.argmax(axis=1), [1, 0])
