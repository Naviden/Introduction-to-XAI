"""Analytic checks for the notebook's actual PyTorch attribution implementation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
path = Path(__file__).resolve().parents[1] / 'Tools/NeuralComparison_examples/attribution_helpers.py'
spec = importlib.util.spec_from_file_location('image_attribution_helpers', path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class LinearClassifier(torch.nn.Module):
    def forward(self, x):
        flat = x.flatten(1)
        return torch.stack([4 - flat[:, 0], 2 * flat[:, 0] + 3 * flat[:, 1]], dim=1)


def test_ig_matches_linear_coefficients_and_keeps_class_fixed():
    sample = torch.tensor([[2., 3.]])
    baseline = torch.zeros_like(sample)
    model = LinearClassifier()
    # Class 0 wins at baseline; class 1 wins at sample. Attribute class 1 throughout.
    values = helpers.integrated_gradients(model, sample, baseline, target=1, num_steps=8)
    torch.testing.assert_close(values, torch.tensor([[4., 9.]]))
    torch.testing.assert_close(values.sum(), model(sample)[0, 1] - model(baseline)[0, 1])
    torch.testing.assert_close(helpers.integrated_gradients(model, sample, baseline, 0), torch.tensor([[-2., 0.]]))


def test_ig_trapezoid_uses_endpoints_and_rejects_zero_steps():
    class Square(torch.nn.Module):
        def forward(self, x):
            return x ** 2
    model = Square()
    sample, baseline = torch.tensor([[3.]]), torch.tensor([[1.]])
    torch.testing.assert_close(helpers.integrated_gradients(model, sample, baseline, 0, 1), torch.tensor([[8.]]))
    with pytest.raises(ValueError, match='num_steps'):
        helpers.integrated_gradients(model, sample, baseline, 0, 0)


def test_smoothgrad_preserves_linear_derivative():
    actual = helpers.smooth_gradients(LinearClassifier(), torch.tensor([[2., 3.]]), 1, samples=16, noise_std=0.2)
    torch.testing.assert_close(actual, torch.tensor([[2., 3.]]))


def test_region_masking_and_occlusion_preserve_the_fixed_target():
    class SumClassifier(torch.nn.Module):
        def forward(self, x):
            summed = x.flatten(1).sum(dim=1)
            return torch.stack([summed, -summed], dim=1)
    sample = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    baseline = torch.zeros_like(sample)
    regions = helpers.grid_regions(4, 4, 2)
    restored = helpers.masked_images(sample, baseline, [[1, 1, 1, 1], [0, 0, 0, 0]], regions)
    torch.testing.assert_close(restored[:1], sample)
    torch.testing.assert_close(restored[1:], baseline)
    expected = np.array([sample.numpy()[0, 0][regions == j].sum() for j in range(4)])
    np.testing.assert_allclose(helpers.occlusion_scores(SumClassifier(), sample, baseline, 0, regions), expected)
    np.testing.assert_allclose(helpers.occlusion_scores(SumClassifier(), sample, baseline, 1, regions), -expected)


@pytest.mark.parametrize('weight', [0., 1., -1.])
def test_gradcam_is_finite_and_handles_zero_or_negative_evidence(weight):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()
            self.classifier = torch.nn.Linear(4, 1, bias=False)
            with torch.no_grad():
                self.classifier.weight.fill_(weight)
    native, expanded = helpers.grad_cam(TinyModel(), torch.ones(1, 1, 2, 2), 0)
    assert np.isfinite(expanded).all()
    np.testing.assert_allclose(native, max(weight, 0.))
    np.testing.assert_allclose(expanded, native)
