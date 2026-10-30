# How SHAP works

**Question:** how can the difference between one model output and a specified reference be allocated among input features?

For a chosen coalition value function $v$, Shapley values satisfy

$$\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(|N|-|S|-1)!}{|N|!}\,[v(S\cup\{i\})-v(S)],\qquad f(x)=\phi_0+\sum_i\phi_i.$$

Here $\phi_0=v(\emptyset)$. A coalition is a subset of available features; the definition of a “missing” feature is part of the explanation, not a property of the model alone. Conditional expectations, interventional background samples and tree-path-dependent calculations can allocate credit differently when features depend on each other. TreeExplainer exploits tree structure; other SHAP explainers have different requirements and approximation errors. [Lundberg and Lee (2017)](https://arxiv.org/abs/1705.07874), [TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).

## Units and assumptions

Fix the target class, output transformation and background before interpreting a plot. Attributions add in the explained output's units: regression response, probability, or logit. A logit attribution is **not** a probability change. The baseline generally is an expected model output under the chosen reference, not necessarily the observed class prevalence or the model intercept. Positive and negative values refer to this specified output and reference.

## Checked example: an explicit zero-reference game

This synthetic logistic model uses glucose, BMI and age only to illustrate arithmetic; it is not a fitted medical model. Define $v(S)$ as the **logit** obtained by setting absent inputs to zero. Zero is unrealistic for these measurements, so this game is useful for calculation rather than clinical interpretation. The following code enumerates every feature order to compute the average marginal contributions.

<!-- checked-example: shap-logit -->
```python
from itertools import permutations
from math import exp, isclose

values = {"glucose": 148.0, "BMI": 33.6, "age": 50.0}
coefficients = {"glucose": 0.05, "BMI": 0.01, "age": 0.02}

def value(present):
    return -6.0 + sum(coefficients[name] * values[name] for name in present)

orders = list(permutations(values))
contributions = dict.fromkeys(values, 0.0)
for order in orders:
    present = []
    for name in order:
        before = value(present)
        present.append(name)
        contributions[name] += (value(present) - before) / len(orders)

baseline = value([])
full_logit = value(list(values))
assert isclose(baseline + sum(contributions.values()), full_logit)
print(f"Baseline logit: {baseline:.3f}")
for name, contribution in contributions.items():
    print(f"{name}: {contribution:.3f} logit units")
print(f"Full logit: {full_logit:.3f}")
print(f"Full probability: {1 / (1 + exp(-full_logit)):.6f}")
```
```text
Baseline logit: -6.000
glucose: 7.400 logit units
BMI: 0.336 logit units
age: 1.000 logit units
Full logit: 2.736
Full probability: 0.939118
```
<!-- /checked-example -->

Each feature has the same marginal contribution in every order because the logit is additive. Applying the sigmoid separately to each attribution would destroy additivity. To explain probability instead, define a probability-valued game and recompute its Shapley values.

## Limitations and checks

Additivity checks numerical reconstruction, not causality or explanation usefulness. Background selection, correlated features, feature grouping and masking choices can change results. Compare sensible backgrounds and inspect approximation error. Mean absolute SHAP is a dataset summary of attribution magnitude; it is not a measure of held-out accuracy loss or proof of a feature's real-world effect.
