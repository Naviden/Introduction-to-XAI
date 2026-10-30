# How accumulated local effects work

**Question:** how does a model's output change locally with one feature, averaged over nearby observed cases?

For a continuous feature, partition its values into bins. In each bin, replace that feature by its lower and upper boundaries for the observations inside the bin; average the prediction difference. Accumulate those differences and subtract a sample-weighted centering constant:

$$\widehat{ALE}_j(x)=\sum_{k=1}^{k(x)}\frac{1}{n_k}\sum_{i:x_{ij}\in B_k}[f(z_k,x_{i,-j})-f(z_{k-1},x_{i,-j})]-C.$$

This finite-bin calculation estimates accumulated conditional local effects. Two-feature ALE additionally removes lower-order effects to isolate the interaction component. [Apley and Zhu (2020)](https://arxiv.org/abs/1612.08468).

## Units and assumptions

The vertical scale is the chosen model response's units, **centered around zero**. A classifier must expose a specified class probability or score rather than unexplained integer class labels. An ALE value of zero is not a probability of zero or a decision threshold. State the reference sample, bin count and response transformation.

## ALE versus PDP

PDP substitutes a feature value across all reference rows. ALE makes local substitutions within occupied bins, reducing long-range extrapolation for dependent features. Finite bins still introduce synthetic boundary combinations. Sparse regions, sharp nonlinearities, bin choices and dependence structure affect the estimated curve; “unbiased under all correlations” is not a valid general claim.

## Limitations and checks

Check bin counts and compare several bin resolutions. Centering removes absolute prediction level, and averaging hides individual heterogeneity. A curve describes model behavior under these local substitutions, not an intervention's causal effect. It does not explain one selected case. In the [California housing notebook](ALE_example.ipynb), the model response is median house value in units of 100000 dollars; the curves therefore show centered changes in those units.
