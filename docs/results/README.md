# Recorded core-study results

These are measured results from the current validated teaching runs, not a leaderboard or estimates of real-world effects. Raw per-case/per-run CSVs and JSON manifests retain the settings, units, failures and variability needed to interpret the figures.

`git_commit` in a study manifest is the checkout base when `git_dirty` is true; the recorded source/helper hashes identify the exact executed changes. Runtime depends on hardware and concurrent work. Read the notebook's assumptions before comparing numbers.

- [regression](regression/): [source notebook](../../Tools/Evaluation_examples/Regression_Benchmark.ipynb); 8 data/manifest files plus the execution receipt.
- [text](text/): [source notebook](../../Tools/TextComparison_examples/Shared_Text_Comparison.ipynb); 4 data/manifest files plus the execution receipt.
- [evaluation](evaluation/): [source notebook](../../Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb); 10 data/manifest files plus the execution receipt.
- [mixed-data](mixed-data/): [source notebook](../../Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb); 5 data/manifest files plus the execution receipt.
- [recourse](recourse/): [source notebook](../../Tools/CausalRecourse_examples/Causal_Recourse.ipynb); 4 data/manifest files plus the execution receipt.

Regenerate with `python scripts/check_notebooks.py --group all --output-dir validation/foundations-final`, then `python scripts/publish_results.py --input validation/foundations-final`. [Reading copies](../executed/README.md) include the plotted results.
