# How DiCE works

**Question:** which alternative input vectors receive a desired prediction, and how can we show several different candidates?

DiCE supports several searches, including random sampling, genetic search, nearest observed candidates via KD-tree, and differentiable optimization. The original optimization balances desired-output loss, proximity and diversity, schematically:

$$\mathcal L=\lambda_y L_{target}+\lambda_d d(x,cf)-\lambda_{div}\operatorname{diversity}(cf_1,\ldots,cf_k).$$

This objective describes one family of algorithms, not a claim that every DiCE backend solves the same optimization. [Mothilal, Sharma and Tan (2020)](https://arxiv.org/abs/1905.07697), [DiCE documentation](https://interpret.ml/DiCE/).

## Units and assumptions

Validity is checked against a specified model output and target/threshold. Distance depends on feature scaling, categorical encoding and the chosen cost definition; a small numeric distance need not be a small human effort. Fix immutable features and permitted ranges explicitly. Model access differs by backend: gradient optimization needs a compatible differentiable model, while other searches can use prediction queries.

## Interpretation and limitations

A candidate that flips the prediction is not necessarily globally minimal, diverse, plausible or actionable. Random search can fail to find any candidate within its budget. A nearest training example supplies an observed feature vector but does not establish that the person can reach it. Constraints on individual coordinates do not capture causal dependencies or downstream consequences.

Report the fraction of queries with valid candidates, actual prediction flips, normalized distances, changed-feature counts and constraint violations. Recheck candidates after rounding or inverse preprocessing. Interpret results as what-if inputs to a predictor; a causal action plan needs additional assumptions about how actions change features and outcomes. The [example](DiCE_example.ipynb) contrasts unconstrained and constrained random search.
