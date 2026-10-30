# Reproduce an example or a comparison

Use Python 3.11. The resolved environments target Linux x86-64 (glibc 2.28+) and
Apple silicon macOS 14+. Windows and Intel Mac are not validated targets.

| Profile | Direct requirements | Hash locks | Coverage |
|---|---|---|---|
| Lightweight | `requirements-tabular.txt` + `requirements-validation.txt` | `requirements-tabular-{linux,macos}.lock` | Foundations, EBM, regression and core exercises; no neural frameworks/Java |
| Neural | `requirements-neural.txt` + `requirements-validation.txt` | `requirements-neural-{linux,macos}.lock` | Lightweight plus neural/image/text examples, Captum and Anchor |
| Full optional stack | `requirements-full.txt` | `requirements-full-{linux,macos}.lock` | All examples, H2O, MERLIN, RuleFit and validation tools |

The full profile also needs **Java 21** installed separately for H2O, and a C
compiler for PyEDA. Python locks do not install or pin the operating system, Java
runtime, compiler or BLAS implementation. Linux locks select CPU PyTorch wheels
from its official index. GPU execution is outside the tested profile.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Choose the appropriate platform/profile lock:
python -m pip install --require-hashes -r requirements-full-linux.lock
python -m pip check
python scripts/check_notebooks.py --group all --output-dir validation/all
```

On Apple silicon, use the matching `-macos.lock`. PyEDA 0.29's legacy C extension
needs this scoped compiler compatibility flag with recent Apple Clang:

```bash
CFLAGS=-Wno-error=incompatible-function-pointer-types python -m pip install \
  --require-hashes -r requirements-full-macos.lock
```

For a small installation, select the tabular lock and `--group foundations`.
The shared text comparison needs `anchor-exp` too, supplied by the neural/full
profiles. `requirements.txt` remains the normal full teaching requirements;
`requirements-full.txt` adds MERLIN and validation. Hash locks include transitive
packages, but source distributions may use platform build tooling.

## Refresh locks deliberately

Use a current `uv`; review dependency changes and rerun the associated notebooks.
Example macOS neural resolution:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile --python-version 3.11.16 \
  --python-platform aarch64-apple-darwin --generate-hashes \
  requirements-neural.txt requirements-validation.txt \
  --output-file requirements-neural-macos.lock
```

For Linux, use `--python-platform x86_64-manylinux_2_28 --torch-backend cpu
--emit-index-url`. Include `--extra-index-url https://download.pytorch.org/whl/cpu`
in the resulting neural/full lock so pip can fetch the pinned `torch==...+cpu`
wheels: uv's automatic backend selection does not emit that index. Tabular
profiles do not need the torch option/index. Full profiles take only
`requirements-full.txt` as their input. These are platform snapshots, not promises
about other Python/platform combinations.

## Execution records

The strict runner writes `results.json`, `environment.json`, and executed copies
under `--output-dir`. Records contain source SHA-256, notebook-local helper hashes,
status, elapsed seconds and executed-artifact hash. Notebooks importing
`xai_assets` also fingerprint the shared helper and asset manifest. The installed
package versions, Python and platform are recorded once per run. All code cells
execute, including cells tagged to skip or expect errors.

Study manifests retain task generation, split/case IDs, seeds, references,
settings and per-case/run results. If `git_dirty` is true, `git_commit` is the
checkout base; source/helper hashes identify the executed changes. Regression
resources measure wall time, observed public prediction calls/rows and a separate
Python-allocation peak run. That peak is **not** total process/native/GPU memory.
A successful run supports the declared checks, not every possible scientific claim.

Helper hashes cover `.py` files beneath the source notebook's directory; checked
external inputs are governed by [the asset manifest and offline guidance](ASSETS.md).

## Durable reading copies and results

[Executed reading copies](executed/README.md) preserve selected examples and plots
for reading on GitHub. Run the source linked inside each snapshot in its original
directory. [Raw study tables](results/README.md) retain measured results separately.

```bash
python scripts/check_notebooks.py --group all --output-dir validation/all
python scripts/publish_executed.py --input validation/all \
  --notebook Tools/TextComparison_examples/Shared_Text_Comparison.ipynb
python scripts/publish_results.py --input validation/all
```

The publisher rejects failed runs, changed sources/helpers, modified execution
artifacts and unexecuted code cells. Inspect plots before committing. Large legacy
JavaScript-heavy outputs are left as CI artifacts rather than copied into Git.

## Runtime and CI

CI installs the corresponding Linux hash lock, checks dependency consistency and
executes separate foundations, core, neural, CIFAR-training and special groups.
Public asset caches are keyed by group and manifest hash. Initial external
downloads can take considerably longer than computation. The CIFAR teaching
example uses 10,000 images, five epochs and bounded CPU threads, with held-out
accuracy reported. CI uploads outputs for 30 days; committed reading copies and
study tables remain available beyond that window.
