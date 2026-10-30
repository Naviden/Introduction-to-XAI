# Choose an explanation from context to message

This guide applies **Figure 1 and Sections 3–5 of Cambria et al. (2023)**,
[*A survey on XAI and natural language explanations*](https://doi.org/10.1016/j.ipm.2022.103111).
Its three stages are **Context definition → Explanation generation → Message
generation**, with six dimensions developed below. See the updated
[Dimensions notebook](../Theory/Dimensions.ipynb) for the original Figure 1 and
worked examples, or [read it with outputs](executed/Theory/Dimensions.ipynb).

The paper studies XAI systems with an emphasis on natural-language explanations.
The method mappings here are **our applications of its framework**, not the
paper's coding or ranking of these experiments. Its categories describe different
choices in a workflow and can overlap; a method name alone does not settle them.

## 1. Context definition

### Explanation Goal

| Goal in Figure 1 / §3.1 | What to select and verify |
| --- | --- |
| **Model Explanation** | Fit an interpretable surrogate to mimic the black box. Assess fidelity to that model over a stated population, separately from accuracy against observed outcomes. |
| **Outcome Explanation** | Explain a specified output for a selected input. Record the target, reference and question being answered. |
| **Model Inspection** | Examine aspects of model behaviour or mechanism through a representation, without necessarily fitting a surrogate. State what the inspection actually reveals. |

Local/global scope does not replace these goals: a global importance summary is
not automatically a Model Explanation in the paper's surrogate sense. A limited
forest impurity-importance chart is an inspection; a small tree fitted to the
forest's predictions is a surrogate explanation. The [Dimensions notebook](../Theory/Dimensions.ipynb)
implements both and measures fidelity separately from predictive accuracy.

Training an interpretable model directly on observed outcomes is a separate
**intrinsic-model design** choice. Section 3.1 discusses transparent-box design as
background; Figure 1 organizes an explanation pipeline around a black box.

### Audience

Figure 1, drawing on Tomsett et al., shows six roles: **Creator, Operator,
Executor, Decision-subject, Data-subject and Examiner**. Identify the actual role
and task: debugging a pipeline, acting on a recommendation, understanding a
decision that affects them, or auditing the system can require different evidence.
Roles can overlap. Section 3.2 and Table 1 operationalize audiences more broadly
as **End users, Developers and Decision-makers**; these are not one-to-one
replacements for the six roles in the figure.

Specify what the reader should be able to understand or do after seeing the
explanation. Appropriate reliance and task performance are more informative
evaluation goals than maximizing agreement with the model.

## 2. Explanation generation

| Dimension | Figure 1 / relevant section | Selection implication |
| --- | --- | --- |
| **Explanator Type** | Figure 1 lists decision trees, decision rules, salient masks, prototype selection and neuron activation. Section 4.1 / Table 1 also use **Feature Inspection (FIN)** alongside trees, rules and masks. | Choose an explanation mechanism or artefact compatible with the model, data, access and goal. Not every type pictured is implemented here. |
| **Structure** | Feature importance; Sensitivity analysis; Direct Answers (§4.2). | Decide whether to organize the answer as contributions, responses to input/model changes, or an explicit what/how/why answer. The authors allow the first two to be restricted forms of direct answers. |
| **Explanation Type** | Plain-fact; Contrastive; Counterfactual (§4.3). | State the question. For a contrastive question, identify the **fact**, **foil** and **contrast class**. A counterfactual can answer a contrastive question. |

**Feature Inspection** is an explanator category in the prose; **Feature
importance** is a structure. A salient mask likewise differs from the choice to
display an image heatmap. The figure, prose and survey table use somewhat
different illustrative/operational categories; do not silently treat them as
one identical list.

Contrastive questions may compare properties of the same object (**P-contrast**),
different objects (**O-contrast**), or an object over time (**T-contrast**). When
comparing times or model versions, state whether inputs, preprocessing or the
model changed. A model-level counterfactual input is not by itself a feasible or
causally effective action.

### Match the question to an implementation

The groups below indicate the goal of the stated use, not an exclusive label
for the algorithm. For example, SHAP can explain one outcome or be aggregated
into an inspection summary. The **Access / scope** and checking columns add
repository guidance beyond Figure 1's six dimensions.

#### Model Explanation: approximate the black box

| Your question | Start with | Access / scope | Check before interpreting |
| --- | --- | --- | --- |
| Can a small model approximate the predictor? | [Tree / RuleFit comparison](../Tools/MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb) | Predictions used as surrogate labels | Fidelity to the predictor separately from accuracy against observed outcomes |
| What changed between model versions? | [MERLIN](../Tools/Merlin_examples/20NG_example.ipynb) | Two models + representative cases | Fidelity of both explanatory surrogates and coverage of the comparison population |

#### Outcome Explanation: a selected prediction or case

| Your question | Start with | Access / scope | Check before interpreting |
| --- | --- | --- | --- |
| Which features contribute to this prediction? | [SHAP and LIME starter](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Model predictions; SHAP implementation may need internals | Target class, probability vs logit, SHAP background, LIME neighborhood and local fit |
| Which conditions usually preserve this decision? | [Anchors](../Tools/ANCHOR_examples/Anchor_tabular.ipynb) | Prediction function + perturbation distribution; local rule | Precision, coverage, uncertainty and the sampling distribution; no universal guarantee |
| Which other inputs get a different decision? | [DiCE](../Tools/DiCE_examples/DiCE_example.ipynb) | Prediction function + feature constraints; local counterfactuals | Validity, proximity, diversity, feasibility, immutable attributes and search failures |
| Which image regions affect a neural score? | [Shared image attribution](../Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) · [read outputs](executed/Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) | Gradients/rules for gradient, DeepLIFT and LRP methods; predictions for perturbations | Same model/input/target, baseline, propagation rule, units, reconstruction residuals, map resolution, masking and randomization controls |
| Which words contribute to this text prediction? | [Shared text comparison](../Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) · [read outputs](executed/Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) | One TF-IDF classifier, word-type groups and an explicit deletion distribution | Selected class/log-odds, repeated words, TF-IDF renormalization, held-out local fit, masking artifacts and Anchors precision/coverage |

#### Model Inspection: examine model behaviour or mechanisms

| Your question | Start with | Access / scope | Check before interpreting |
| --- | --- | --- | --- |
| Which inputs help predictive performance? | [Permutation / drop-column](../Tools/FeatureImportanceComparison_examples/FeatureImportanceComparison.ipynb) | Predictions + labeled evaluation data; global; drop-column refits | Metric, data split, feature dependencies, grouped importance, repeat variability |
| How does model response vary with an input? | [PDP / ICE](../Tools/Visual%20Explanation_example/PDP_ICE.ipynb), [ALE](../Tools/ALE_examples/ALE_example.ipynb) | Prediction function + representative features; population or individual response/effect curves | Support of edited inputs, correlated variables, averaging, ALE bins and centering |
| What do attention heads attend to? | [Attention visualization](../Tools/Attention_examples/Attention_visualization.ipynb) | Attention weights | Attention weights alone do not establish faithful or causal importance |
| Which features and interactions contribute to survival-model scores across cases? | [Survival SHAP](../Tools/SHAP_examples/SHAP_survival_model.ipynb) | Compatible survival model; aggregated per-case attributions and interaction summaries | Censoring, held-out concordance, explained full-data model versus held-out validation model, log-hazard vs survival probability, prediction horizon |

#### Additional design, adaptation and evaluation questions

These questions support the workflow but are not additional Explanation Goal
categories in Figure 1. Intrinsic design, adaptation, causal recourse and
evaluation need their own assumptions.

| Your question | Start with | Access / scope | Check before interpreting |
| --- | --- | --- | --- |
| Can a simpler model solve this task? | [Logistic regression, tree, spline GAM and EBM](../Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) · [read outputs](executed/Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) | Training data; intrinsic, local and global | Held-out performance, complexity, scaling, shape functions, interactions and exact term/logit reconstruction |
| How do I explain my mixed-data pipeline? | [Mixed-data workflow](../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb) | Raw rows, fitted preprocessing and prediction function | Train-only fitting, missingness, unseen categories, original-feature groups and class-index mapping |
| Which actions could change the decision? | [Causal recourse](../Tools/CausalRecourse_examples/Causal_Recourse.ipynb) | An explicit causal model + actionable variables | Identifiability/assumptions, downstream effects, cost, constraints and model changes |
| How reliable and costly are explanations across tasks? | [Evaluation and sensitivity](../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb), [regression benchmark](../Tools/Evaluation_examples/Regression_Benchmark.ipynb) · [read regression outputs](executed/Tools/Evaluation_examples/Regression_Benchmark.ipynb) | Repeated splits/cases, model predictions and reference explanations | Paired controls, perturbation assumptions, SD versus confidence intervals, measured prediction calls/rows, time and traced allocation limits |
| Do explanations help intended users? | [Human-study materials](HUMAN_STUDY.md) | Study protocol, preregistration and reporting templates | Appropriate participants/tasks, model errors, counterbalancing, participant dependence, decision quality and reliance; no empirical human results are included |

The [complete tabular comparison](../Tools/MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb)
is useful after the starter. Most sections explain one forest; Integrated Gradients
explains a separately trained neural network. Raw attribution magnitudes across
different models, units and reference distributions are not interchangeable.

## 3. Message generation

**Presentation** is the sixth dimension: how the raw explanation is communicated
to the chosen audience (§5). Figure 1 shows **Graphics, Images, Reports, Texts and
Dialogue System**. Section 5.1 groups four families (Graphics/plots, Texts, Images,
Reports) and discusses dialogue within its natural-language treatment; Table 1
uses Graphic, Image, Rule, Text and Dialogue System. These differences do not
make presentation interchangeable with the explanation algorithm.

| Presentation choice | Practical application and check |
| --- | --- |
| Graphics/plots | Use a bar chart, waterfall, tree or response curve with the target, units and reference clearly labelled. |
| Images | Display a mask or heatmap over the input; preserve the selected class, resolution and masking assumptions. An image heatmap is distinct from a numeric heatmap plot. |
| Reports | Show contributions, surrogate fidelity or counterfactual candidates in a table with the relevant context and checks. |
| Texts | Distinguish rules and word annotations from natural-language sentences. Ground every stated value and conclusion in the underlying explanation. |
| Dialogue System | Support user questions and follow-up responses through an actual interaction mechanism. A fixed paragraph alone is not dialogue. |

The [Dimensions notebook](../Theory/Dimensions.ipynb) presents the same SHAP result
as a waterfall, table and audience-specific template text. This is a small
**template-based NLG** example, distinguished in §5.1 from end-to-end generation.
The [shared text comparison](../Tools/TextComparison_examples/Shared_Text_Comparison.ipynb)
explains text inputs; that task is distinct from generating explanations in
natural language. The repository does not provide a general dialogue manager.

Check that shorter messages preserve the target, reference, units and limits of
the numerical result. Presentation quality or fluency does not demonstrate
fidelity, correctness of the prediction or usefulness to people. Use the
[human-study materials](HUMAN_STUDY.md) to plan an evaluation of comprehension,
decision quality and appropriate reliance; no human-study results are claimed.

## Record the six dimensions and the implementation checks

1. **Context:** Explanation Goal and Audience, with a concrete question and task.
2. **Generation:** Explanator Type, Structure and Explanation Type; state the fact,
   foil and contrast class when applicable.
3. **Message:** Presentation, the evidence it communicates and any information
   omitted or available through follow-up.
4. **Implementation:** model, preprocessing, output/class, units, data split and
   population; intrinsic/post-hoc construction and model access where relevant.
5. **Definition:** reference values, background distribution, perturbations,
   neighborhood, feature grouping and algorithm settings.
6. **Evaluation:** model performance, explanation fidelity/stability, variability
   and cost, plus human utility where actually measured.
7. **Interpretation:** supported conclusions, a failure case and what the result
   cannot establish, particularly any predictive-versus-causal distinction.

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
The paper's AHP selection model (§6) is conditional on user preferences, criteria
weights and survey coding; it is not an empirical leaderboard. We do not
reproduce that ranking here.
Bike-sharing uses a chronological split but is not a sequence-attribution
example. Attention visualization is not a generative-language-model explanation.
Concept explanations (TCAV/concept bottlenecks), prototypes/criticisms,
training-data influence, GNNs, temporal attribution and generative or mechanistic
interpretability remain extension topics. Human-study templates are available;
participant studies have not been conducted as part of this collection. See the
[roadmap](ROADMAP.md) for inclusion criteria and remaining work.

## Reference

Cambria, E., Malandri, L., Mercorio, F., Mezzanzanica, M., & Nobani, N. (2023).
*A survey on XAI and natural language explanations*. **Information Processing &
Management, 60**(1), 103111.
[doi:10.1016/j.ipm.2022.103111](https://doi.org/10.1016/j.ipm.2022.103111).
Figure 1 is on p. 4; Sections 3–5 develop the context, generation and presentation
dimensions. [BibTeX](../Theory/references.bib) ·
[Dimensions notebook](../Theory/Dimensions.ipynb) ·
[Notebook with outputs](executed/Theory/Dimensions.ipynb).
