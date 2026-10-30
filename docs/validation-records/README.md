# Validation record — 30 October 2026

The core-expansion release was checked locally on Apple silicon with Python
3.11.16, the full macOS hash lock and Java 21. All **32 source notebooks** executed
in fresh kernels with no skipped code cells. All **95 automated tests** and the
three worked mathematical examples passed. EBM and both new practice notebooks
also ran in the lightweight profile, which has neither TensorFlow nor PyTorch.
Student TODO cells intentionally remain exercises.

- [All notebook execution receipts](results.json): source/helper/asset hashes,
  measured duration, status and executed-artifact identity.
- [Installed environment](environment.json): package versions and platform.
- [Independent upstream file checks](assets-upstream-review.json): all five pinned
  SHAP/seaborn downloads match their recorded hashes. Eighteen cached file/model/
  bundled entries also passed the offline verification CLI; native loader data
  fingerprints were checked during execution.
- [Readable outputs](../executed/README.md) and [raw study tables](../results/README.md).

The combined receipts include final targeted reruns for Grad-CAM, Anchors and the
regression manifest after their last edits. Every receipt was checked against the
current source and dependencies before publication. Legacy notebook output files
are kept as local/CI artifacts; selected reading copies are committed.

Commands: `python -m pytest -q`, `python scripts/check_worked_examples.py`,
`python scripts/check_links.py`, `python scripts/check_notebooks.py --group all`,
and `python scripts/manage_assets.py verify <registered keys>`.

This records local checks, not an assertion that all future platforms, datasets
or scientific uses work. GitHub Actions independently runs the Linux profiles on
each push. Its status belongs to the particular commit/run shown by GitHub.
