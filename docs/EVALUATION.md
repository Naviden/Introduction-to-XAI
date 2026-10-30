# Evaluate the explanation as well as the model

Start with the executable [evaluation and sensitivity lab](../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb).
It deliberately uses controlled data, repeated runs and imperfect explanations.
The data-generating mechanism is known only because the examples are synthetic.
It must not be confused with the fitted predictor or a real-world causal process.

## Protocol

| Dimension | Report | Important limit |
| --- | --- | --- |
| Predictive quality | Held-out discrimination, probability loss/calibration where relevant, errors by class or subgroup | A faithful explanation can describe a bad model |
| Algebraic checks | SHAP reconstruction or IG completeness in stated units | Reconstruction is not proof of causal correctness, stability or usefulness |
| Approximation fidelity | LIME local fit and query error; surrogate held-out agreement/error | A local neighborhood or global aggregate can hide serious errors elsewhere |
| Perturbation response | Score changes after masking/removing ranked features, against random rankings | Edited inputs may be out of distribution; repeat with alternative maskers |
| Stability | Variation over explainer seeds, neighborhoods/backgrounds, small input changes and model seeds | Stable explanations can still be wrong; some changes should change the explanation |
| Dependence on the learned model | Parameter or label-randomization controls where meaningful | Passing a sanity check is necessary evidence for some tasks, not a sufficient universal validation |
| Rules | Precision, coverage, rule complexity, uncertainty and perturbation distribution | Sampled precision is not a guarantee over all possible inputs |
| Counterfactuals / recourse | Validity, distance/cost, sparsity, diversity, constraints, failure rate, update robustness | Prediction flips do not establish feasible interventions; see the causal lab |
| Practical cost | Elapsed time, model calls, peak memory when measured, hardware and batch size | One warm-cache laptop run is not a deployment latency guarantee |
| Human utility | Task accuracy, error detection, time and appropriate reliance vs a no-explanation baseline | Visual appeal and self-reported trust do not demonstrate usefulness |

Do not combine these dimensions into a universal “best explainer” number.
Choose metrics for the intended question and document their weaknesses.
Agreement between methods is not ground truth when they estimate different quantities.

## Sampling and uncertainty

- Fix the train/validation/test split and model when comparing explainers. Select
  cases by a declared rule before inspecting explanations; include errors and
  multiple classes rather than just one persuasive example.
- Separate variation from data resampling, model fitting, explanation sampling,
  and methodological choices. A standard deviation across shuffles is not a
  confidence interval across patients or future datasets.
- Keep model/explainer tuning out of the final evaluation data. Report paired
  results on the same cases, settings, seeds and perturbation budgets.
- Study correlated features and alternative backgrounds. A low individual
  permutation score can reflect substitution by a correlated feature.
- Report unsuccessful searches and undefined metrics explicitly. Do not silently
  discard failures or replace them with favorable values.

## A minimal result record

Record source commit/hash, software versions, dataset/model identifiers, split
and case IDs, seeds, target class/output units, reference/background, settings,
metrics with units, failures, runtime and hardware. Retain raw per-case/per-run
measurements so aggregate tables can be regenerated. The runner writes an
environment manifest and source hashes; the evaluation lab writes its own
experiment manifest and CSV results.

## Human evaluation template

Describe **who** will use the explanation, **which task** they perform, and the
decision it should improve. Compare with a baseline without explanations, define
an observable outcome (for example detecting a planted model error), and document
how cases and participants are sampled. Report accuracy/time as well as perceived
helpfulness. Avoid using explanation quality as evidence of fairness or safety
without separately evaluating those properties. This repository supplies a
study-design template, not results from a human study.

## References

- [Doshi-Velez & Kim: Towards a Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608).
- [Adebayo et al.: Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).
- [Hedström et al.: Quantus](https://jmlr.org/papers/v24/22-0142.html).
- [Agarwal et al.: OpenXAI](https://arxiv.org/abs/2206.11104).
- [Hooker et al.: A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758).
- [Karimi et al.: Algorithmic Recourse from Counterfactual Explanations to Interventions](https://arxiv.org/abs/2002.06278).
