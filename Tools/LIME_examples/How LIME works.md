# How LIME works

**Question:** which simple model approximates a selected prediction in a chosen neighborhood?

LIME samples perturbations, queries the fitted model, weights samples by proximity, and fits an interpretable surrogate. Its objective balances local fit and explanation complexity:

$$\xi(x)=\arg\min_{g\in G}\mathcal L(f,g,\pi_x)+\Omega(g).$$

For a weighted least-squares surrogate, $\mathcal L=\sum_i\pi_x(z_i)[f(z_i)-g(z_i')]^2$. Here $g$ includes an intercept, $\pi_x$ supplies neighborhood weights, and $\Omega$ penalizes explanation complexity, for example by limiting the number of selected features.

The surrogate may use an interpretable representation $z'$ such as word presence or discretized feature conditions, rather than the model's original input coordinates. The loss measures agreement with the model's outputs, not agreement with the true labels. [Ribeiro, Singh and Guestrin (2016)](https://arxiv.org/abs/1602.04938).

## Units and assumptions

Specify the target class, model response, perturbation distribution, feature representation, distance and kernel width. For class-probability explanations, the surrogate predicts that class's probability. A coefficient is a change in that approximation per unit of its encoded feature; it is not a causal effect or automatically an additive SHAP value. The intercept plus active terms gives the surrogate prediction, which can differ from the original prediction.

## Checked example: neighborhood weights

For this example choose $w(d)=\exp(-d^2/\sigma^2)$ with $\sigma=3$. Distances below are illustrative, already computed in an agreed scaled representation. They are not raw distances between age and education categories. Library kernels can use a different convention, so inspect the implementation before interpreting a numeric width.

<!-- checked-example: lime-kernel -->
```python
from math import exp

sigma = 3.0
distances = [1.0, 4.0, 2.5, 6.0, 3.5, 2.0]
print("distance weight")
for distance in distances:
    weight = exp(-(distance / sigma) ** 2)
    print(f"{distance:g} {weight:.6f}")
```
```text
distance weight
1 0.894839
4 0.169013
2.5 0.499352
6 0.018316
3.5 0.256376
2 0.641180
```
<!-- /checked-example -->

A point twice as far away does not get half the weight. These weights specify a regression problem only after model outputs, the intercept, encoded features and regularization are supplied; they do not by themselves establish any fitted coefficient.

## Limitations and checks

A sparse explanation can fit its sampled neighborhood poorly. Report local weighted fit and the gap at the query, and repeat sampling with multiple seeds/kernel widths. Perturbations can create unrealistic correlated-feature combinations, unnatural text, or masked images unlike training data. A stable approximation in this artificial neighborhood need not remain faithful elsewhere. The notebooks provide full fitted examples in [tabular](LIME_tabular.ipynb), [text](LIME_textual.ipynb), and [image](LIME_image.ipynb) settings.
