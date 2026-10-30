"""Validate or execute the teaching notebooks without modifying their saved output."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
DEEP = {
    "Theory/Basics.ipynb",
    "Tools/ANCHOR_examples/Anchor_tabular.ipynb",
    "Tools/Attention_examples/Attention_visualization.ipynb",
    "Tools/GradCAM_examples/GradCAM_example.ipynb",
    "Tools/IntegratedGradients_examples/IntegratedGradients_example.ipynb",
    "Tools/LIME_examples/LIME_image.ipynb",
    "Tools/SHAP_text_image_examples/SHAP_text_image.ipynb",
}
SPECIAL = {
    "Tools/MethodComparison_examples/Shared_Tabular_Case_Comparison.ipynb",
    "Tools/Merlin_examples/20NG_example.ipynb",
    "Tools/RuleFit_examples/RuleFit_example.ipynb",
}
FOUNDATION_FOLDERS = {
    "Starter_examples", "InterpretableModels_examples", "Evaluation_examples",
    "CausalRecourse_examples", "PracticalWorkflow_examples",
}
IMAGE_TRAINING = {"Tools/LIME_examples/LIME_image.ipynb"}
NEURAL_COMPARISONS = {"Tools/NeuralComparison_examples/Shared_Image_Attribution.ipynb"}
DEEP |= NEURAL_COMPARISONS


def local_dependencies(path):
    """Fingerprint notebook-local Python helpers, including newly added files."""
    dependencies = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.parent.rglob("*.py"))
            if not any(part in {"__pycache__", "results", "logs"} for part in p.parts)}

    if "xai_assets" in path.read_text():
        for relative in ("xai_assets.py", "assets/manifest.json"):
            dependencies[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return dependencies


def notebook_paths(group="all"):
    for folder in ("Theory", "Tools", "Practice"):
        for path in sorted((ROOT / folder).rglob("*.ipynb")):
            if ".ipynb_checkpoints" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            category = ("foundations" if path.parent.name in FOUNDATION_FOLDERS else
                        "deep" if relative in DEEP else "special" if relative in SPECIAL else "core")
            selected = group in ("all", category)
            selected |= group == "neural" and relative in DEEP - IMAGE_TRAINING
            selected |= group == "image-training" and relative in IMAGE_TRAINING
            if selected:
                yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "core", "deep", "special", "foundations", "neural", "image-training"), default="all")
    parser.add_argument("--notebook", action="append", help="Repository-relative notebook path; repeat to select several.")
    parser.add_argument("--validate-only", action="store_true", help="Check notebook schemas and code syntax, without executing.")
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per cell (default: 1200).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "validation" / "cache",
                        help="Isolated cache for downloaded models, datasets and plotting configuration.")
    args = parser.parse_args()
    paths = [ROOT / p for p in args.notebook] if args.notebook else list(notebook_paths(args.group))
    if not paths:
        parser.error("No notebooks selected.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.json").write_text(json.dumps({
        "schema_version": 1, "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"])),
    }, indent=2) + "\n")
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for variable, subdirectory in {
        "HF_HOME": "huggingface", "HF_HUB_CACHE": "huggingface/hub",
        "TRANSFORMERS_CACHE": "huggingface/hub", "KERAS_HOME": "keras",
        "SCIKIT_LEARN_DATA": "sklearn", "SEABORN_DATA": "seaborn", "XDG_CACHE_HOME": "xdg",
        "MPLCONFIGDIR": "matplotlib",
    }.items():
        os.environ[variable] = str(cache / subdirectory)
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    results = []
    # Always use this interpreter, even if a user's global python3 kernel points elsewhere.
    with tempfile.TemporaryDirectory(prefix="xai-kernel-") as kernel_dir:
        spec = Path(kernel_dir) / "xai"
        spec.mkdir()
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "XAI validation", "language": "python",
        }))
        for path in paths:
            relative = path.relative_to(ROOT)
            started = time.monotonic()
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            result = {"notebook": relative.as_posix(), "mode": "validate" if args.validate_only else "execute",
                      "source_sha256": source_hash, "local_dependencies": local_dependencies(path)}
            notebook = None
            print(f"RUN  {relative}", flush=True)
            try:
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                from IPython.core.inputtransformer2 import TransformerManager
                transformer = TransformerManager()
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        compile(transformer.transform_cell(cell.source), f"{relative}:cell-{index + 1}", "exec")
                if not args.validate_only:
                    manager = KernelManager(kernel_name="xai", kernel_spec_manager=KernelSpecManager(kernel_dirs=[kernel_dir]))
                    client = NotebookClient(notebook, km=manager, timeout=args.timeout, allow_errors=False,
                                            force_raise_errors=True, skip_cells_with_tag="",
                                            resources={"metadata": {"path": str(path.parent)}})
                    # With an explicitly supplied manager, request cleanup on completion/failure.
                    client.execute(cleanup_kc=True)
                result["status"] = "passed"
            except Exception as error:
                result.update(status="failed", error=f"{type(error).__name__}: {error}")
            if notebook is not None and not args.validate_only:
                notebook.metadata["validation"] = {
                    "source_sha256": source_hash, "status": result["status"],
                    "local_dependencies": result["local_dependencies"],
                    "python": platform.python_version(),
                }
                destination = args.output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                nbformat.write(notebook, destination)
                result["executed_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            result["seconds"] = round(time.monotonic() - started, 2)
            results.append(result)
            print(f"{result['status'].upper()} {relative} ({result['seconds']}s)", flush=True)
            if "error" in result:
                print(result["error"], flush=True)
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = sum(item["status"] == "failed" for item in results)
    print(f"{len(results) - failed}/{len(results)} notebooks passed.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
