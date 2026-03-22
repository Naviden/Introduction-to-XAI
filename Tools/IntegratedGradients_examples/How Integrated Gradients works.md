# How Integrated Gradients Works

Integrated Gradients (IG) is an attribution method for differentiable models. It assigns an importance score to each input feature by accumulating gradients along a path from a baseline to the actual input.

## Algorithm

1. **Choose a baseline**: A reference input (e.g., all zeros) representing "absence of information."
2. **Interpolate**: Create m intermediate inputs along the straight line from baseline x' to input x.
3. **Compute gradients**: For each intermediate input, compute the gradient of the output w.r.t. the input.
4. **Integrate**: Average all the gradients and multiply element-wise by (x - x').

## Formula

IG_i(x) = (x_i - x'_i) × ∫₀¹ (∂F(x' + α(x - x')) / ∂x_i) dα

Approximated numerically as:

IG_i(x) ≈ (x_i - x'_i) × (1/m) Σ_{k=1}^{m} (∂F(x' + k/m × (x - x')) / ∂x_i)

## Key Axioms

- **Completeness**: The attributions sum to the difference between the output at the input and the output at the baseline: Σ_i IG_i(x) = F(x) - F(x')
- **Sensitivity**: If a feature changes between baseline and input and the output changes, that feature gets non-zero attribution.
- **Implementation Invariance**: Two functionally equivalent models always produce the same attributions.
