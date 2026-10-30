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


def affine_module(bias=0.):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[-1., 0.], [2., 3.]]))
        model[0].bias.copy_(torch.tensor([4., bias]))
    return model.eval()


def test_deeplift_rescale_reconstructs_a_fixed_target_reference_difference():
    pytest.importorskip('captum')
    model = affine_module(bias=5.)
    sample, reference = torch.tensor([[2., 3.]]), torch.tensor([[1., 1.]])
    original = {key: value.clone() for key, value in model.state_dict().items()}
    actual, delta = helpers.deep_lift_rescale(model, sample, reference, target=1)
    torch.testing.assert_close(actual, torch.tensor([[2., 6.]]))
    torch.testing.assert_close(actual.sum(), model(sample)[0, 1]-model(reference)[0, 1])
    torch.testing.assert_close(delta, torch.zeros(1), atol=1e-6, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    assert all(not module._forward_hooks and not module._backward_hooks for module in model.modules())


def test_deeplift_rescale_crosses_relu_threshold_instead_of_using_endpoint_gradient():
    pytest.importorskip('captum')
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1.)
        model[0].bias.fill_(-1.)
        model[2].weight.fill_(2.)
    sample, reference = torch.tensor([[2.]]), torch.tensor([[0.]])
    actual, delta = helpers.deep_lift_rescale(model, sample, reference, 0)
    # Endpoint gradient * (x-reference) would incorrectly give 4; Rescale gives 2.
    torch.testing.assert_close(actual, torch.tensor([[2.]]))
    torch.testing.assert_close(delta, torch.zeros(1), atol=1e-6, rtol=0)


def test_lrp_returns_logit_scaled_relevance_without_double_scaling():
    pytest.importorskip('captum')
    model = affine_module(bias=0.)
    sample = torch.tensor([[2., 3.]])
    actual, delta = helpers.lrp_epsilon(model, sample, 1, epsilon=0.)
    torch.testing.assert_close(actual, torch.tensor([[4., 9.]]))
    torch.testing.assert_close(actual.sum(), model(sample)[0, 1])
    torch.testing.assert_close(delta, torch.zeros(1), atol=1e-6, rtol=0)
    # A second output multiplication would yield [52, 117] and fail this test.
    assert not hasattr(model[0], 'rule')
    assert not model[0]._forward_hooks and not model[0]._backward_hooks


def test_lrp_exposes_bias_and_epsilon_relevance_residuals():
    pytest.importorskip('captum')
    model = affine_module(bias=5.)
    sample = torch.tensor([[2., 3.]])
    actual, delta = helpers.lrp_epsilon(model, sample, 1, epsilon=1.)
    # z = 18, epsilon = 1: each relevance is x_i*w_i * 18/(18+1).
    expected = torch.tensor([[4., 9.]]) * 18 / 19
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(delta, torch.tensor([18.])-expected.sum())
    assert float(delta[0]) > 5  # unallocated bias plus epsilon absorption


def test_lrp_zero_output_is_finite():
    pytest.importorskip('captum')
    model = affine_module(bias=0.)
    actual, delta = helpers.lrp_epsilon(model, torch.zeros(1, 2), 1, epsilon=1e-6)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros(1, 2))
    torch.testing.assert_close(delta, torch.zeros(1))


def test_deeplift_maxpool_restores_strict_deterministic_state():
    pytest.importorskip('captum')
    model = torch.nn.Sequential(torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[2].weight.fill_(2.)
    sample = torch.tensor([[[[1., 2.], [3., 4.]]]])
    baseline = torch.zeros_like(sample)
    original_mode = torch.are_deterministic_algorithms_enabled()
    original_warning = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        values, delta = helpers.deep_lift_rescale(model, sample, baseline, 0)
        torch.testing.assert_close(values.sum(), torch.tensor(8.))
        torch.testing.assert_close(delta, torch.zeros(1), atol=1e-6, rtol=0)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(original_mode, warn_only=original_warning)
