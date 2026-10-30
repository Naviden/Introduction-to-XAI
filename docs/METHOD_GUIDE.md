# Choose an XAI method: a beginner's path

Explainable AI (XAI) helps you examine how a model makes predictions. Start with
one model and one question. By the end of this path, you should be able to read
three common explanations, check what they actually support, and choose your next
notebook.

**Start now:** [read the first comparison with its plots and results](executed/Tools/Starter_examples/First_XAI_Comparison.ipynb).
You do not need to install anything to follow it. Keep this guide open alongside
the notebook and use the walkthrough below.

[Before you start](#before-you-start) · [First walkthrough](#your-first-walkthrough) ·
[Choose your next method](#choose-your-next-method) · [Learning sequence](#continue-in-this-order) ·
[Document your explanation](#document-your-explanation)

Already familiar with SHAP, LIME and explanation evaluation? Go to the
[detailed method reference](METHOD_REFERENCE.md), which lists the
full method tables, access requirements, evaluation checks and coverage limits.

## Before you start

For **reading**, you need only the ideas in this table. For **running and editing**
the notebook, you should also be comfortable running Python cells and reading a
pandas table. You do not need neural networks or the SHAP mathematics first.

| Term | Meaning in the first notebook |
| --- | --- |
| Feature | An input column, such as a measured cell radius. One row is one case. |
| Model and prediction | A fitted random forest takes those measurements and produces a score for a class. |
| Target class and probability | We explain the model's probability for **malignant**, encoded as class 1. A score of 0.70 means the model outputs 70% for that class; it does not establish that the estimate is well calibrated. |
| Training and test data | The model learns from training rows. Separate test rows let us check predictions on cases it did not learn from. |
| Global and local | Global describes behaviour over a set of cases. Local concerns one selected case. Neither automatically tells you what happens for every case. |
| Reference or baseline | The starting value against which a contribution is measured. Its definition is part of the explanation. |
| Surrogate and fidelity | A surrogate is a simpler model fitted to imitate another model. Fidelity is how well it matches that model, which differs from accuracy against the true labels. |

The starter uses a bundled historical breast-cancer dataset as a teaching example.
Its explanations describe this fitted classifier; they are not clinical advice or
evidence that changing a measurement would change a diagnosis.

### Read, run, or adapt

| What you want to do | Where to go |
| --- | --- |
| Understand the example first | [Notebook with saved outputs](executed/Tools/Starter_examples/First_XAI_Comparison.ipynb). Read the text and plots in order; no setup is needed. |
| Run it on your computer | Follow [Start here](../README.md#start-here), using `requirements-tabular.txt`, then open the [source notebook](../Tools/Starter_examples/First_XAI_Comparison.ipynb). Choose **Restart Kernel and Run All**. |
| Try it in a browser | Use **Open in Colab** on the source notebook and follow the [Colab setup instructions](REPRODUCIBILITY.md#google-colab). The badge opens the notebook; dependencies still need setup. |
| Use your own table and model | Finish the walkthrough first, then follow the [mixed-data workflow](../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb) for missing values, categories and preprocessing. |

The starter runs on CPU with the lightweight environment. Its data are bundled;
there are no data or model downloads after installing the packages. For platform
support and exact environments, see [reproducibility](REPRODUCIBILITY.md).

## Your first walkthrough

Keep the model, test data and selected class fixed. The notebook asks **three
different questions**, so its charts need not rank the features in the same order.
The section names below match the notebook.

### 1. Check the predictor before explaining it

Read **“Fit once and keep the test set separate.”** Find the number of training
and test cases and the two reported metrics. Accuracy measures the fraction of
correct class predictions. ROC AUC measures how well scores rank positive cases
above negative cases across thresholds; it is not percentage accuracy.

Then read **“Fix one case and one output.”** The notebook deliberately selects a
correctly classified malignant test case with a score near 0.70. This makes a
readable illustration, not a representative sample of all predictions.

**Checkpoint:** finish the sentence: “I am explaining this forest's probability
of ___ for row ___.” If you cannot name the model, case and output, pause before
interpreting an attribution plot.

### 2. Read a global ranking: permutation importance

Read **“Global question: what happens when a feature is shuffled?”** The notebook
shuffles one column among test rows and measures how much ROC AUC falls. A larger
positive bar means the model's ranking performance suffered more under that
shuffle. Error bars describe variation across repeated shuffles.

**Checkpoint:** identify the horizontal-axis units: **ROC AUC loss**, not a change
in this one case's probability. A small bar does not prove the feature is unused;
another correlated measurement can carry similar information. Shuffling can
also create unrealistic combinations of measurements.

### 3. Read one prediction: SHAP

Read **“SHAP: an additive explanation of this prediction.”** SHAP assigns signed
contributions to features under a specified reference. Here the reference uses
training sample counts stored in the trees. Follow the waterfall from its
reference value to the selected class probability: positive contributions add
to the reference and negative contributions subtract from it in that calculation.

The [saved starter run](executed/Tools/Starter_examples/First_XAI_Comparison.ipynb)
reports these rounded values:

```text
reference probability + sum of SHAP contributions = model probability
        0.3746        +          0.3299           =       0.7045
```

**Checkpoint:** locate the notebook's reconstruction check. It verifies that the
contributions add up to the explained output. It does not prove that the model
is correct, that the reference is the only sensible choice, or that a feature
caused the outcome. Other notebooks can explain **log-odds** instead of
probability, so always read the units before interpreting the bars.

**Optional browser exercise:** open the
[SHAP Explainer](https://nobani.it/tools/shap-explainer), change one feature and
select **Compute SHAP**. Follow the waterfall, inspect a feature's coalition
contributions, and check that the base value plus the contributions reconstructs
the prediction. The demo uses its own toy model and reference.

### 4. Check a local approximation: LIME

Read **“LIME: how well does a simple local story fit?”** LIME creates modified
inputs around the case, asks the forest for their predictions, and fits a small
weighted linear model to approximate them. The displayed conditions belong to
this local approximation.

Inspect **both** diagnostics:

| Diagnostic | What to ask | Saved starter run |
| --- | --- | --- |
| Weighted local R² | How well does the small model match the forest across LIME's sampled neighbourhood? Values closer to 1 indicate a better fit on those samples. | **0.205**: the local summary fits poorly. |
| Case approximation gap | How far apart are the two predictions at the selected case? | LIME **0.6735**, forest **0.7045**; a gap of **0.0310**, or 3.10 percentage points. |

**Checkpoint:** explain why a fairly close prediction at one point does not
rescue a poor fit across the sampled neighbourhood. LIME's sampled inputs can
also be unrealistic. Its weights are not SHAP values and should not be averaged
with them. These numbers describe the saved run; reruns can vary.

**Optional browser exercise:** in the
[LIME Explainer](https://nobani.it/tools/lime-explainer), keep the toy classifier
and selected point fixed, change the kernel width, and select **Explain selected
point**. Compare the local line, coefficients and R². Freshly sampled neighbours
can also change the result. This two-feature example helps visualize the local
fit before working with the notebook's larger table.

### 5. Write a conclusion that the evidence supports

A suitable conclusion for this run is:

> For the selected case, the forest assigns a malignant-class probability of
> 0.7045. SHAP reconstructs that prediction relative to its stated reference.
> LIME's approximation is close at the case but fits its sampled neighbourhood
> poorly, so its displayed conditions need further checking. Neither result
> establishes what would happen after a real-world intervention.

**Try one change:** in the LIME code cell, change `random_state=SEED` to
`random_state=123` inside `LimeTabularExplainer`. Rerun that cell and the final
diagnostics cell, keeping the global `SEED`, fitted forest and query unchanged.
Compare the selected conditions, local R² and case gap. Explain why a change in
the explanation need not mean the forest's prediction changed. Then attempt the
notebook's other next experiments.

## Choose your next method

Choose the **question** first. The links below are starting points, not a ranking
of methods. If you are unsure, complete the three-method starter before branching.

| Your question | Start with | Check before trusting the answer |
| --- | --- | --- |
| Can I use a model whose own rules or terms I can inspect? | [Interpretable models](../Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb): small tree, logistic regression, additive models and EBM | Check performance on unseen data; a readable model can still be wrong. |
| Which inputs help performance across many cases? | Permutation importance in the [starter](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Name the metric, use held-out data, and consider correlated inputs. |
| How are contributions assigned to this one prediction? | SHAP in the [starter](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Fix the class, units and reference; check reconstruction. |
| Can a simple approximation summarize behaviour near this case? | LIME in the [starter](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Inspect neighbourhood fit and the case gap; repeat the sampling. |
| How does the model's prediction vary as an input changes? | [PDP and ICE](../Tools/Visual%20Explanation_example/PDP_ICE.ipynb); later [ALE](../Tools/ALE_examples/ALE_example.ipynb) | Check whether the edited inputs are plausible. A response curve is not a causal effect. |
| Which changed inputs receive a different model decision? | [DiCE](../Tools/DiCE_examples/DiCE_example.ipynb); later [causal recourse](../Tools/CausalRecourse_examples/Causal_Recourse.ipynb) | Check the new prediction and allowed changes separately. An impossible change is not useful advice. |
| Which words or image regions matter to a selected score? | [Read the text comparison](executed/Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) or [image comparison](executed/Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) | Keep the input, model and target fixed; inspect what deletion, masking or a reference image means. |

PDP means **partial dependence plot**: a model response averaged over cases.
ICE means **individual conditional expectation**: a response curve for each
case. ALE means **accumulated local effects**: accumulated local model changes
over intervals supported by the data. DiCE generates diverse counterfactual
inputs. The [detailed reference](METHOD_REFERENCE.md#match-the-question-to-an-implementation)
adds rules, surrogates, attention, survival models and evaluation methods.

The table mixes lightweight and optional examples. Check the
[notebook environment table](../README.md#core-comparisons-and-laboratories) and
[dependency profiles](REPRODUCIBILITY.md) before running a new branch. The full
shared-tabular comparison and neural examples are later steps.

## Continue in this order

| Step | Activity | You are ready to move on when… |
| --- | --- | --- |
| 1 | Complete the starter walkthrough above. | You can distinguish a performance drop, an additive contribution and a local approximation, and name one limitation of each. |
| 2 | [Read Dimensions with outputs](executed/Theory/Dimensions.ipynb) to explore goals, audiences and ways to present an explanation. | You can state your question and audience, choose the kind of evidence to generate, and describe how to present it. Reading requires no extra environment. |
| 3 | [Read](executed/Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) or [run the interpretable-model comparison](../Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb). Start with the small tree and logistic regression, then the additive models. | You can separate model accuracy from readability, and reconstruct a prediction in the right units. It uses the same lightweight profile. |
| 4 | Attempt [Core XAI exercises](../Practice/Core_XAI_Exercises.ipynb), then consult the separate [solutions](../Practice/Instructor/Core_XAI_Solutions.ipynb). | You can check units, dependent features and allowed counterfactual changes. The TODO cells are intentionally unfinished; running the template is not completing the exercises. |
| 5 | Adapt the [mixed-data workflow](../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb), or explore the [evaluation and sensitivity lab](../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb). | You can explain your preprocessing, test more than one case, and report when an explanation changes or fails. |

Before the additive-model sections, learn one additional scale: **log-odds** is
`log(p / (1 - p))`. A log-odds value of 0 corresponds to probability 0.5. Add the
model's terms on the log-odds scale to get a total `z`, then convert it with
`p = 1 / (1 + exp(-z))` (the sigmoid function). Adding individual probability
changes is not the same calculation. **GAM** means
generalized additive model; **EBM** means Explainable Boosting Machine. The
comparison notebook explains and checks their fitted terms.

[Basics](../Theory/Basics.ipynb) is optional background reading. Running all of it
requires the neural/full environment because it includes TensorFlow. Running
[Dimensions](../Theory/Dimensions.ipynb) also needs packages beyond the starter's
lightweight profile. Neither full execution is a prerequisite for this path.

## Document your explanation

Before sharing a result, answer three questions:

1. **What does the reader need to understand?** Name the question, audience,
   model and output. For the starter, a developer is examining one forest
   prediction for one selected case.
2. **What evidence answers that question?** State the method, reference, units
   and checks. For SHAP here, that means signed probability contributions under
   the tree-path reference, with a reconstruction check.
3. **How will you communicate the result?** Choose a plot, table or text that
   the reader can interpret, and state the limits. A waterfall and a short
   explanation can show how contributions add up without claiming causation.

For a more formal vocabulary, the [Dimensions notebook](../Theory/Dimensions.ipynb)
and [detailed reference](METHOD_REFERENCE.md) discuss the context, explanation
and message stages of [Cambria et al. (2023)](https://doi.org/10.1016/j.ipm.2022.103111).
This is one organizing framework for these choices; the method tutorials also
provide their own primary references and checks.

For a project report, use the [evaluation protocol](EVALUATION.md) and
[reproducibility guide](REPRODUCIBILITY.md). For methods outside this collection,
see the [coverage boundaries](METHOD_REFERENCE.md#coverage-boundaries) and
[roadmap](ROADMAP.md). There is no single best explainer for every question.
