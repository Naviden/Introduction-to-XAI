# Contributing an XAI example

If an explainer is missing, open an issue or contribute a pull request. Describe
the explanation question it answers and how it differs from existing coverage.

## Required notebook structure

1. **Question and audience:** prerequisites, learning outcome, modality and model access.
2. **Setup:** dependency profile, downloads, seeds and approximate CPU/resource cost.
3. **Definition:** intuition, equations where useful, output units and assumptions.
4. **Runnable example:** explicit split, preprocessing fitted on training data,
   held-out model quality, fixed output/class and declared case selection.
5. **Interpretation:** connect the actual computed output to the question.
6. **Evaluation:** relevant numerical checks, variability, a failure case and limits.
7. **References and provenance:** primary method paper, implementation and dataset/model sources.

Keep names descriptive and links relative. Prefer bundled or synthetic small data
for the first example. Make expensive training/downloads explicit. Missing
dependencies should fail with useful instructions rather than silently skipping
the method. Do not use test data for model selection.

## Validation

```bash
python scripts/check_worked_examples.py
python scripts/check_notebooks.py --validate-only
python -m pytest -q
python scripts/check_notebooks.py --notebook Tools/YourMethod/Example.ipynb
```

Install the appropriate dependency profile first. The full test suite needs the
full environment; the lightweight environment intentionally omits neural tests.
If changing shared dependencies, revalidate the full suite. Check code in a fresh
kernel, inspect plots, and clear stale source outputs after edits. Preserve student
exercises as templates; put solutions in clearly separated instructor material.

For arithmetic teaching examples, use the executable Markdown block convention
in the SHAP/LIME/RuleFit guides and run the worked-example checker. Numerical
checks should catch meaningful errors, not merely reproduce the implementation.

The runner discovers notebooks in `Theory`, `Tools` and `Practice`. Add heavyweight
notebooks to the appropriate group in `scripts/check_notebooks.py` so they are not
accidentally placed in a lightweight job. Generated local results belong in
ignored `validation/` or notebook-local `results/` directories. Publish selected
executed copies using the export script only after successful validation.

## Reuse terms

Original contributed code is MIT licensed; original teaching prose and figures
are CC BY 4.0. Retain upstream notices for adapted material and record third-party
exceptions in [provenance](docs/PROVENANCE.md). Do not claim ownership of datasets,
model weights, screenshots or illustrations supplied by others.
