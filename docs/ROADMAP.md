# Scope and next extensions

This repository aims to let readers understand, choose, run, evaluate, adapt and
cite representative XAI methods. Inclusion is based on the distinct question a
method answers, a maintained implementation, an executable example and a way to
evaluate its limitations—not on collecting every named variation.

## Core foundation implemented

| Area | Executable or reusable evidence | Scope |
| --- | --- | --- |
| Theory and selection | Checked SHAP/LIME/RuleFit arithmetic, [method guide](METHOD_GUIDE.md), [evaluation protocol](EVALUATION.md), [contribution checklist](../CONTRIBUTING.md) | Questions, assumptions, output units, interpretation and failure cases |
| Lightweight entry point | [First XAI comparison](../Tools/Starter_examples/First_XAI_Comparison.ipynb) | Small CPU example using bundled data |
| Intrinsic models | [Source](../Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) · [read outputs](executed/Tools/InterpretableModels_examples/Interpretable_Models_Comparison.ipynb) | Logistic regression, tree, spline GAM, native EBM with/without interactions and forest; exact term/logit reconstruction |
| Explanation evaluation | [Sensitivity lab](../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb), [regression source](../Tools/Evaluation_examples/Regression_Benchmark.ipynb) · [read regression outputs](executed/Tools/Evaluation_examples/Regression_Benchmark.ipynb) | Repeated cases/seeds, random controls, two removal baselines, correlation, failure examples; three regression tasks and measured resource use |
| Practical adaptation and recourse | [Mixed-data workflow](../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb), [causal recourse](../Tools/CausalRecourse_examples/Causal_Recourse.ipynb) | Missingness, unseen categories, raw feature groups; interventions under an explicit synthetic causal model |
| Shared neural comparison | [Source](../Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) · [read outputs](executed/Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb) | One CNN/input/target; gradient, reference, rule and perturbation methods including DeepLIFT Rescale and epsilon-LRP; reconstruction and randomization checks |
| Shared text comparison | [Source](../Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) · [read outputs](executed/Tools/TextComparison_examples/Shared_Text_Comparison.ipynb) | One TF-IDF classifier and word-type deletion protocol for SHAP, LIME, occlusion and Anchors; cohort, fit and stability diagnostics |
| Teaching and human evaluation | [Exercises](../Practice/Core_XAI_Exercises.ipynb), [separate solutions](../Practice/Instructor/Core_XAI_Solutions.ipynb), [rubric](../Practice/Instructor/Core_XAI_Rubric.md), [human-study materials](HUMAN_STUDY.md) | Student templates, executable solutions and reusable study design; no participant data or empirical human-study results |
| Reproduction and durable records | [Executed reading copies](executed/README.md), [raw records](results/README.md), [environment guide](REPRODUCIBILITY.md) | Source/helper hashes and manifests; tabular, neural and full transitive locks for Python 3.11 on macOS 14+ ARM and Linux x86-64/glibc 2.28+ CPU; Java 21 remains external |
| Asset identity and reuse | [Asset registry and verification](ASSETS.md), [provenance and notices](PROVENANCE.md) | File checksums, immutable Hugging Face revisions, dataset fingerprints and cache preparation; original code MIT and original teaching content CC BY 4.0 |

## Remaining work within the core scope

- **Historical asset permissions:** some upstream data and image redistribution
  grants remain unverified. Track that due diligence in the provenance ledger;
  obtain a documented grant, replace the asset, or require an authorized external
  download as appropriate. Checksums establish identity, not permission to reuse.
- **Broader empirical coverage:** add independent datasets, model families and
  application settings with predeclared cases/splits. The current compact repeated
  studies teach evaluation protocols rather than support a field-wide ranking.
  Overlapping splits and synthetic regenerated datasets require different
  uncertainty interpretations.
- **Resource accounting at larger scale:** current regression records measure wall
  time, public prediction API calls/rows and traced Python allocations. Dedicated
  isolated-process measurements of native/accelerator memory and end-to-end cold
  runs remain future work.
- **Human evidence:** the protocol, preregistration and reporting materials are
  ready to adapt. Actual participant recruitment, context-appropriate review,
  study execution and analysis remain to be done before claiming human utility.
- **Wider platform assurance:** the locks target the documented platforms. Other
  operating systems/architectures and future dependency versions need their own
  installation and execution evidence. Asset verification and native library
  cache preparation are separate steps for some legacy examples.

## Research extensions

Concept explanations (TCAV and concept bottlenecks), prototypes/criticisms,
training-data influence, GNNs, temporal models, generative model attribution and
mechanistic interpretability. These remain outside the current implementation.

Each addition should meet [the contribution checklist](../CONTRIBUTING.md): a
precise question, assumptions, units, reproducible code, meaningful checks,
failure case, interpretation and primary sources. Report model quality separately
from explanation fidelity or human utility.
