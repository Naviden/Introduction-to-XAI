# How Integrated Gradients works

**Question:** how can a fixed differentiable model output's change from a chosen baseline to one input be attributed to input coordinates?

For scalar output $F$, baseline $x'$ and input $x$,

$$IG_i(x;x')=(x_i-x'_i)\int_0^1\frac{\partial F(x'+\alpha(x-x'))}{\partial x_i}\,d\alpha.$$

Choose a class/score once and keep it fixed along the path. Numerical quadrature approximates the integral; the notebook uses the trapezoidal rule with both endpoints. [Sundararajan, Taly and Yan (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

## Units and assumptions

Attributions have the scalar output's units. Explaining a probability and explaining its pre-softmax logit are different questions. The baseline is a reference input, not universally “absence of information.” Zero after standardization means the training-feature means; zero pixels mean something different after image normalization. The straight path can cross unrealistic inputs.

## Axioms and numerical checks

- **Completeness:** for the exact integral under its regularity assumptions, $\sum_i IG_i=F(x)-F(x')$. A finite-step calculation has numerical error; report the residual and check convergence with more steps.
- **Sensitivity(a):** if input and baseline differ in **one feature only** and their outputs differ, that feature must receive nonzero attribution. This does not say every changed feature has nonzero attribution when several change.
- **Implementation invariance:** functionally equivalent models give the same exact IG values under the same target, baseline and path. Finite precision and quadrature can introduce small numerical discrepancies.

## Limitations and interpretation

A positive value supports this output relative to this baseline; a negative value opposes it. Compare defensible baselines and inspect numerical convergence. Completeness does not establish causality, correct predictions, robustness or human usefulness. Gradient saturation and path selection deserve scrutiny even when the residual is small. The [Iris example](IntegratedGradients_example.ipynb) explains a fixed class probability from the training-mean baseline.
