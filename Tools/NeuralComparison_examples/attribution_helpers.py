"""Small CPU-friendly attribution routines used by Shared_Image_Attribution.ipynb.

Targets are fixed output-column indices. Model outputs must be class logits and
models must already be in evaluation mode. No routine changes model parameters.
"""

import numpy as np
import torch
import torch.nn.functional as F


def input_gradients(model, inputs, target):
    """Derivative of one fixed class logit for each independent batch example."""
    batch = inputs.detach().clone().requires_grad_(True)
    logits = model(batch)
    if not 0 <= target < logits.shape[1]:
        raise ValueError("target must identify an output column")
    return torch.autograd.grad(logits[:, target].sum(), batch)[0].detach()


def integrated_gradients(model, sample, baseline, target, num_steps=256):
    """Straight-line IG with a trapezoidal rule that includes both endpoints."""
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if sample.shape != baseline.shape or sample.shape[0] != 1:
        raise ValueError("sample and baseline must have the same single-example shape")
    alpha = torch.linspace(0, 1, num_steps + 1, dtype=sample.dtype, device=sample.device)
    alpha = alpha.reshape((-1,) + (1,) * (sample.ndim - 1))
    path = baseline + alpha * (sample - baseline)
    gradients = input_gradients(model, path, target)
    average_gradient = 0.5 * (gradients[:-1] + gradients[1:]).mean(dim=0, keepdim=True)
    return ((sample - baseline) * average_gradient).detach()


def smooth_gradients(model, sample, target, samples=64, noise_std=0.1, seed=42):
    """Mean signed gradient under independent Gaussian input noise, without clipping."""
    if sample.shape[0] != 1 or samples < 1 or noise_std < 0:
        raise ValueError("use one sample, positive samples, and nonnegative noise_std")
    generator = torch.Generator(device=sample.device).manual_seed(seed)
    noise = torch.randn((samples,) + sample.shape[1:], generator=generator,
                        dtype=sample.dtype, device=sample.device) * noise_std
    return input_gradients(model, sample + noise, target).mean(dim=0, keepdim=True)


def grad_cam(model, sample, target):
    """Grad-CAM for models exposing features and a classifier of flattened maps.

    Return raw nonnegative native and bilinearly upsampled maps. Do not claim
    additive completeness or compare their scale to pixel contributions.
    """
    if sample.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single example")
    maps = model.features(sample)
    # Permit parameter-free toy feature extractors as well as trained CNNs.
    if not maps.requires_grad:
        maps = maps.detach().requires_grad_(True)
    logits = model.classifier(maps.flatten(1))
    gradients = torch.autograd.grad(logits[0, target], maps)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    native = torch.relu((weights * maps).sum(dim=1, keepdim=True))
    upsampled = F.interpolate(native, size=sample.shape[-2:], mode='bilinear', align_corners=False)
    return native[0, 0].detach().numpy(), upsampled[0, 0].detach().numpy()


def grid_regions(height=8, width=8, patch_size=2):
    """Return contiguous non-overlapping square-region labels, starting at zero."""
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by positive patch_size")
    labels = np.arange((height // patch_size) * (width // patch_size)).reshape(height // patch_size, width // patch_size)
    return labels.repeat(patch_size, axis=0).repeat(patch_size, axis=1)


def masked_images(sample, baseline, presence, regions):
    """Restore present regions of sample; replace absent regions by baseline.

    `presence` has one column for each contiguous label in the 2D region map.
    Continuous values in [0, 1] also support interpolation-based probes.
    """
    regions = np.asarray(regions)
    if sample.shape != baseline.shape or sample.shape[0] != 1 or tuple(regions.shape) != tuple(sample.shape[-2:]):
        raise ValueError("sample, baseline and region dimensions must agree")
    if not np.array_equal(np.unique(regions), np.arange(len(np.unique(regions)))):
        raise ValueError("region labels must be contiguous and start at zero")
    presence = torch.as_tensor(presence, dtype=sample.dtype, device=sample.device)
    if presence.ndim != 2 or presence.shape[1] != int(regions.max()) + 1:
        raise ValueError("presence must have one column per region")
    if not bool(torch.isfinite(presence).all()) or not bool(((presence >= 0) & (presence <= 1)).all()):
        raise ValueError("presence values must lie in [0, 1]")
    region_tensor = torch.as_tensor(regions, dtype=torch.long, device=sample.device)
    pixel_mask = presence[:, region_tensor].unsqueeze(1)
    return baseline + pixel_mask * (sample - baseline)


def occlusion_scores(model, sample, baseline, target, regions):
    """Selected-logit drop when each region is removed separately; not additive."""
    count = len(np.unique(regions))
    presence = 1 - np.eye(count, dtype=np.float32)
    with torch.no_grad():
        original = model(sample)[0, target]
        masked = model(masked_images(sample, baseline, presence, regions))[:, target]
    return (original - masked).numpy()


def deep_lift_rescale(model, sample, baseline, target):
    """Captum DeepLIFT Rescale on an isolated copy of the fixed fitted model.

    Return contributions (including input-minus-baseline multipliers) and
    Captum's sum(contributions) - (f(sample) - f(baseline)) residual.
    The caller must use separately instantiated supported nonlinear modules.
    """
    from copy import deepcopy
    from captum.attr import DeepLift

    if sample.shape != baseline.shape or sample.shape[0] != 1:
        raise ValueError("sample and baseline must have the same single-example shape")
    copied_model = torch.nn.Sequential(deepcopy(model)).eval()
    inputs = sample.detach().clone().requires_grad_(True)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # Captum's MaxPool rule uses max_unpool2d, which PyTorch does not register
    # as deterministic. Keep the warning visible and restore the caller's state.
    uses_maxpool = any(isinstance(layer, torch.nn.MaxPool2d) for layer in copied_model.modules())
    try:
        if deterministic and uses_maxpool:
            torch.use_deterministic_algorithms(True, warn_only=True)
        values, delta = DeepLift(copied_model, multiply_by_inputs=True).attribute(
            inputs, baselines=baseline.detach().clone(), target=target,
            return_convergence_delta=True,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return values.detach(), delta.detach()


def lrp_epsilon(model, sample, target, epsilon=1e-6):
    """Captum LRP with explicit epsilon rules on affine and pooling modules.

    This intentionally supports only the simple feed-forward architectures used
    here: Conv2d, Linear, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d and ReLU. ReLU
    propagates relevance through Captum's supported activation hooks. Biases
    remain unchanged. Epsilon stabilization and unallocated bias relevance mean
    the input relevances need not sum to the selected logit.

    Captum 0.9.0 returns output-scaled relevance, despite contrary wording in
    its public docstring. Do not multiply it by the output again. The returned
    residual is f(sample) - sum(input relevances), unlike DeepLIFT's sign.
    """
    from copy import deepcopy
    from captum.attr import LRP
    from captum.attr._utils.lrp_rules import EpsilonRule

    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if sample.shape[0] != 1:
        raise ValueError("LRP expects one input example")
    copied_model = torch.nn.Sequential(deepcopy(model)).eval()
    rule_layers = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.MaxPool2d,
                   torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d)
    for module in copied_model.modules():
        if len(list(module.children())):
            continue
        if isinstance(module, rule_layers):
            module.rule = EpsilonRule(epsilon=epsilon)
        elif not isinstance(module, torch.nn.ReLU):
            raise TypeError(f"No specified LRP rule for {type(module).__name__}")
    values, delta = LRP(copied_model).attribute(
        sample.detach().clone().requires_grad_(True), target=target,
        return_convergence_delta=True,
    )
    return values.detach(), delta.detach()
