# Introduction to Explainable AI

Practical Jupyter notebooks for understanding and comparing explainable AI methods:
LIME, SHAP, Anchors, Grad-CAM, Integrated Gradients, ALE, DiCE, RuleFit, MERLIN,
partial dependence, and feature importance.

## Getting started

Use **Python 3.11** on Linux or Apple silicon macOS 14+. The pinned deep-learning
packages do not support Intel Macs. The dependency versions in `requirements.txt` are a compatible
set used for validation; they are not a promise that every newer release works.
The full installation includes both TensorFlow and PyTorch because different
notebooks use different backends.

```bash
git clone https://github.com/Naviden/Introduction-to-XAI.git
cd Introduction-to-XAI
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# On Linux, install the CPU build first to avoid unnecessary CUDA packages:
# python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m notebook
```

Open a notebook and use **Restart Kernel and Run All**. Run notebooks from their
own directories so relative data paths resolve. The Anchors notebook uses spaCy's
English tokenizer; no separate spaCy language-model download is required.

## Notebook guide

| Directory | What it covers |
| --- | --- |
| `Theory/` | White-box and black-box models, surrogates, local/global explanations, and counterfactuals |
| `Tools/LIME_examples/` | Tabular classification/regression, text, and images |
| `Tools/SHAP_examples/` | Tabular explanations, summary plots, and survival models |
| `Tools/SHAP_text_image_examples/` | Transformer text explanations and ImageNet image explanations |
| `Tools/ANCHOR_examples/` | **Text** anchors on bundled movie reviews (the historical filename is `Anchor_tabular.ipynb`) |
| Other `Tools/` folders | The individual XAI methods named above |
| `Practice/` | An intentionally unfinished student LIME exercise; setup cells run, and students complete the marked tasks |

Several notebooks download public datasets or pretrained models on first use.
Internet access, disk space for model weights, and several minutes of CPU time
are needed for the complete suite. Survival SHAP interaction values, BERT anchor
sampling, and image explanations are the most expensive examples. No GPU is
required. Explanations illustrate model behavior, not causal effects.

## RuleFit and MERLIN

**RuleFit** uses H2O and needs a Java runtime (tested with Java 21). Check it with
`java -version` before opening its notebook. The notebook starts a local H2O
server, limited to two threads and 2 GB of heap when launched here. The final cell shuts
down only a server started by this notebook; a pre-existing server is left running.

**MERLIN** has extra dependencies and is installed separately:

```bash
python -m pip install -r requirements-merlin.txt
```

The package name is `merlinxai`, while the Python import is `merlin`. PyEDA supplies
Espresso internally, and may need a C compiler to build. On recent macOS Clang,
PyEDA 0.29 can fail on legacy C function-pointer warnings. This command-scoped
workaround was used for local validation:

```bash
CFLAGS="-Wno-error=incompatible-function-pointer-types" python -m pip install -r requirements-merlin.txt
```

The optional requirements pin the legacy `pkg_resources` provider needed by
MERLIN's `stopit` dependency. Graphviz is only needed if you enable saved diagrams;
the example leaves that feature disabled.

## Validation

Install the test tools, and the MERLIN extras to run **all** notebooks:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-merlin.txt
python -m pytest -q
python scripts/check_notebooks.py --validate-only
python scripts/check_notebooks.py --group all
```

The runner starts a fresh kernel per notebook using the active Python interpreter,
executes every code cell, and exits with a failure status if any notebook fails.
It writes executed copies and `results.json` under `validation/`, leaving the
source notebooks unchanged. Supported dataset/model caches and plotting configuration are redirected there;
use `--cache-dir PATH` to select another location. Some libraries manage their own
files: SHAP caches packaged datasets in its installation, DiCE saves `adult.zip`
in the notebook directory, and MERLIN creates local `logs/` and `results/`.
These generated files are ignored by Git. The student exercise is checked
as a template, not graded as a completed solution.

For a smaller run, select `--group core`, `--group deep`, or `--group special`,
or use `--notebook Tools/DiCE_examples/DiCE_example.ipynb`. `core` covers 12
notebooks, `deep` covers the seven TensorFlow/transformer notebooks, and `special`
covers MERLIN and RuleFit. `--timeout` controls the time limit per cell in seconds
(default 1200).

GitHub Actions runs all three groups on pushes and pull requests, along with
schema/syntax checks and numerical regression tests. Executed notebooks and
failure details are uploaded as workflow artifacts. External download failures
remain visible as failures; the runner does not silently skip examples.

## Contributing

Submit a pull request with the notebook change and its validation results. Clear
stale outputs after editing code, keep student exercises unfinished, and rerun
the affected notebook from a fresh kernel. Revalidate the full suite when changing
shared dependencies.
