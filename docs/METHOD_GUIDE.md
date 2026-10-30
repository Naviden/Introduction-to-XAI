# Choose an explanation by the question

First specify the audience, decision, model output and reference population.
An explanation of a model prediction does not establish a cause of the real-world
outcome. An interpretable model can be a useful starting point before a post-hoc
explainer is needed.

| Your question | Start with | Access / scope | Check before interpreting |
| --- | --- | --- | --- |
| Can a simpler model solve this task? | [Logistic regression, tree, spline GAM and EBM](../Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) · [read outputs](executed/Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) | Training data; intrinsic, local and global | Held-out performance, complexity, scaling, shape functions, interactions and exact term/logit reconstruction |
| Which inputs help predictive performance? | [Permutation / drop-column](../Tools/FeatureImportanceComparison_examples/FeatureImportanceComparison.ipynb) | Predictions + labeled evaluation data; global; drop-column refits | Metric, data split, feature dependencies, grouped importance, repeat variability |
| Which features contribute to this prediction? | [SHAP and LIME starter](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Model predictions; SHAP implementation may need internals | Target class, probability vs logit, SHAP background, LIME neighborhood and local fit |
| How do I explain my mixed-data pipeline? | [Mixed-data workflow](../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb) | Raw rows, fitted preprocessing and prediction function | Train-only fitting, missingness, unseen categories, original-feature groups and class-index mapping |
| How does model response vary with an input? | [PDP / ICE](../Tools/Visual%20Explanation_example/PDP_ICE.ipynb), [ALE](../Tools/ALE_examples/ALE_example.ipynb) | Prediction function + representative features; global or conditional curves | Support of edited inputs, correlated variables, averaging, ALE bins and centering |
| Which conditions usually preserve this decision? | [Anchors](../Tools/ANCHOR_examples/Anchor_tabular.ipynb) | Prediction function + perturbation distribution; local rule | Precision, coverage, uncertainty and the sampling distribution; no universal guarantee |
| Which other inputs get a different decision? | [DiCE](../Tools/DiCE_examples/DiCE_example.ipynb) | Prediction function + feature constraints; local counterfactuals | Validity, proximity, diversity, feasibility, immutable attributes and search failures |
| Which actions could change the decision? | [Causal recourse](../Tools/CausalRecourse_examples/Causal_Recourse.ipynb) | An explicit causal model + actionable variables | Identifiability/assumptions, downstream effects, cost, constraints and model changes |
| Can a small model approximate the predictor? | [Tree / RuleFit comparison](../Tools/MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb) | Predictions used as surrogate labels | Fidelity to the predictor separately from accuracy against observed outcomes |
| What changed between model versions? | [MERLIN](../Tools/Merlin_examples/20NG_example.ipynb) | Two models + representative cases | Fidelity of both explanatory surrogates and coverage of the comparison population |
| Which image regions affect a neural score? | [Shared image attribution](../Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) · [read outputs](executed/Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) | Gradients/rules for gradient, DeepLIFT and LRP methods; predictions for perturbations | Same model/input/target, baseline, propagation rule, units, reconstruction residuals, map resolution, masking and randomization controls |
| How do words affect a text classifier? | [Shared text comparison](../Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) · [read outputs](executed/Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) | One TF-IDF classifier, word-type groups and an explicit deletion distribution | Selected class/log-odds, repeated words, TF-IDF renormalization, held-out local fit, masking artifacts and Anchors precision/coverage |
| How reliable and costly are explanations across tasks? | [Evaluation and sensitivity](../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb), [regression benchmark](../Tools/Evaluation_examples/Regression_Benchmark.ipynb) · [read regression outputs](executed/Tools/Evaluation_examples/Regression_Benchmark.ipynb) | Repeated splits/cases, model predictions and reference explanations | Paired controls, perturbation assumptions, SD versus confidence intervals, measured prediction calls/rows, time and traced allocation limits |
| Do explanations help intended users? | [Human-study materials](HUMAN_STUDY.md) | Study protocol, preregistration and reporting templates | Appropriate participants/tasks, model errors, counterbalancing, participant dependence, decision quality and reliance; no empirical human results are included |
| What do attention heads attend to? | [Attention visualization](../Tools/Attention_examples/Attention_visualization.ipynb) | Attention weights | Attention weights alone do not establish faithful or causal importance |
| Which features affect a survival-model score? | [Survival SHAP](../Tools/SHAP_examples/SHAP_survival_model.ipynb) | Compatible survival model; target score | Censoring, held-out concordance, log-hazard vs survival probability, prediction horizon |

The [complete tabular comparison](../Tools/MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb)
is useful after the starter. Most sections explain one forest; Integrated Gradients
explains a separately trained neural network. Raw attribution magnitudes across
different models, units and reference distributions are not interchangeable.

## What to record for every explanation

1. The concrete question and intended reader.
2. The model, preprocessing, class/output, output units and data split.
3. The explanation definition: reference values, background distribution,
   perturbations, neighborhood, feature grouping and algorithm settings.
4. Model performance, relevant explanation checks, variability and computation cost.
5. The interpretation the result supports, a failure case, and what it cannot establish.

Use the [evaluation protocol](EVALUATION.md) and
[reproducibility guide](REPRODUCIBILITY.md) alongside a method notebook.
The [core exercises](../Practice/Core_XAI_Exercises.ipynb) ask learners to check
output units, dependent features and counterfactual validity. Complete
[instructor solutions](../Practice/Instructor/Core_XAI_Solutions.ipynb) and a
[grading rubric](../Practice/Instructor/Core_XAI_Rubric.md) are separate from the
student template.

## Coverage boundaries

The collection covers classification, regression and a survival example with
tabular, image and text-classification data. The intrinsic-model comparison
includes both a spline GAM and native Explainable Boosting Machines, with and
without interactions. Exact additive reconstruction describes the fitted model;
it does not turn its terms into causal effects.

The shared CNN comparison includes DeepLIFT Rescale and epsilon-LRP. It records
reference/rule choices and residuals, retains a failed DeepLIFT reference check,
and does not cover every variant or architecture. The text comparison uses one
word-type deletion protocol; its artificial deletions and Anchors precision
estimates do not establish behavior on natural-language paraphrases. The
regression benchmark distinguishes a known generating function from its fitted
predictor, and measures public prediction calls and traced allocations rather
than total arithmetic work or full process memory.

These compact experiments teach a protocol, not a universal method ranking.
Bike-sharing uses a chronological split but is not a sequence-attribution
example. Attention visualization is not a generative-language-model explanation.
Concept explanations (TCAV/concept bottlenecks), prototypes/criticisms,
training-data influence, GNNs, temporal attribution and generative or mechanistic
interpretability remain extension topics. Human-study templates are available;
participant studies have not been conducted as part of this collection. See the
[roadmap](ROADMAP.md) for inclusion criteria and remaining work.
