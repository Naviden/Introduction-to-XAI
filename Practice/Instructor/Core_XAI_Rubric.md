# Core XAI exercise rubric

Use with the [student notebook](../Core_XAI_Exercises.ipynb) and [reference solutions](Core_XAI_Solutions.ipynb). Total: **100 points**. The student template deliberately leaves answers unfinished; a passing setup run is not a passing assignment.

## Exercise 1: additive units and reconstruction — 30 points

| Criterion | Points |
|---|---:|
| Recover per-feature spline terms; center using training rows and adjust the intercept | 8 |
| Verify both the logit decomposition and sigmoid probability against the fitted model | 8 |
| Plot signed contributions with the correct class, reference and log-odds units | 6 |
| Transfer to binary EBM: intercept plus terms gives the score; an interaction term is not univariate | 4 |
| Explain reference dependence and why exact decomposition is not a causal claim | 4 |

Give credit for an equivalent mathematically correct centering convention if it is stated and reconstruction holds. Summing logit contributions as probability increments loses the unit/reconstruction credit even if the chart looks plausible.

## Exercise 2: randomness, correlation and masking — 35 points

| Criterion | Points |
|---|---:|
| Keep model/split fixed and control at least five shuffle seeds | 7 |
| Report an accurate mean and clearly defined variability summary | 8 |
| Implement both independent and joint shuffling; preserve the pair with one shared row permutation | 8 |
| Fit the support-band threshold on training data and report its limitations | 6 |
| Distinguish redundancy, unrealistic samples, algorithm randomness and causal effects | 6 |

Do not require one ordering of feature importance or an asserted direction for every AUC change. Reward a correct explanation of the observed result. Shuffle standard deviation is not automatically a population confidence interval.

## Exercise 3: counterfactual validation — 35 points

| Criterion | Points |
|---|---:|
| Check prediction flip, immutability, numeric bounds and allowed directions separately | 14 |
| Filter admissible candidates before ranking distance; state the scaling and finite search scope | 8 |
| Compute and interpret diversity among admissible candidates | 5 |
| Explain why model validity and supplied constraints do not establish real feasibility or causal benefit | 8 |

The provided pool has two admissible rows, A and B; B is closest under the requested normalized L1 metric. A different explicitly justified cost metric can change that preference. Calling a prediction flip alone “actionable” loses the constraint/interpretation credit.

## Feedback format

For each exercise, record: score, one correct reasoning step, one specific correction, and the evidence the student should inspect next. Separate programming mistakes from interpretation mistakes. Do not reward selecting a prettier case, hiding an inconvenient result or silently dropping failed diagnostics.

Suggested extension: ask students to change the random seed, reference, cost metric or feature dependence and predict which conclusions should remain valid. Use the [human-study materials](../../docs/HUMAN_STUDY.md) for a separate discussion of whether a technically correct explanation helps a reader perform a task.
