# Reproduce an example or a comparison

Use Python 3.11. The full neural environment targets Linux and Apple silicon macOS
14+. The lightweight path avoids Java and neural frameworks. Windows/Intel Mac
are not validated targets for this release.

## Lightweight environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tabular.txt -r requirements-validation.txt
python scripts/check_notebooks.py --group foundations
```

For a resolved environment, replace the install line with one of:

```bash
# Apple silicon macOS, Python 3.11:
python -m pip install --require-hashes -r requirements-tabular-macos.lock
# Linux x86-64, Python 3.11:
python -m pip install --require-hashes -r requirements-tabular-linux.lock
```

These platform-specific lock snapshots include validation tools and transitive
dependencies. They are not promises about other Python/platform combinations.
They can be regenerated with `uv pip compile --python-version 3.11
--python-platform <platform> --generate-hashes requirements-tabular.txt
requirements-validation.txt --output-file <lockfile>`; review and execute the
profile after regenerating. Use `aarch64-apple-darwin` or `x86_64-unknown-linux-gnu`.
Full-suite direct pins remain in `requirements.txt`; full transitive locks are a
separate extension rather than implied by these lightweight locks.

## Execution records

The strict runner writes `results.json`, `environment.json`, and executed notebook
copies under the chosen `--output-dir`. Each record includes source SHA-256,
notebook-local Python helper hashes, execution status, elapsed seconds, and the
executed artifact hash. Package versions, Python and platform are recorded once
per run. All code cells execute, including cells tagged to skip or expect errors.

The evaluation lab additionally records the synthetic generation, split IDs,
selected cases, model/explainer seeds, background/settings and per-case/per-run
results under `validation/evaluation-study/`. If `git_dirty` is true, `git_commit`
is the checkout base; the source/helper hashes identify the executed changes. A successful run is evidence of
execution and declared numerical checks, not validation of all scientific claims.

Helper hashes cover `.py` files beneath the source notebook's directory. External
files, data downloads and imported packages require their own manifests; the
environment and [provenance register](PROVENANCE.md) document that boundary.

## Read validated outputs without installing

[Executed reading copies](executed/README.md) preserve selected foundational
examples and their plots in the repository. They are snapshots for reading;
run the linked source in its original directory. To refresh:

```bash
python scripts/check_notebooks.py --group foundations --output-dir validation/foundations
python scripts/publish_executed.py --input validation/foundations
```

The publisher rejects failed/non-executed records, changed sources or local
helpers, modified execution artifacts and unexecuted code cells. Published
manifests identify the exact source and environment. Review the resulting plots
before committing. [Recorded evaluation tables](results/README.md) are retained
separately for inspection and regeneration.

## Runtime and network

All foundation notebooks use bundled or generated data. The new image comparison
also uses bundled digits and trains a small CPU model. Other neural/data examples
download assets on first use. Caches are placed under `validation/cache` where
supported; some libraries use their own paths as described in the README.

CI separates the CIFAR training notebook from other neural examples. It uses a
10,000-image teaching subset, five epochs and bounded CPU threads; its held-out
accuracy is printed so explanation plots are not mistaken for model validation.
All CI groups run strictly and upload their outputs for 30 days. Only committed
reading copies/result tables are durable beyond that retention window.
