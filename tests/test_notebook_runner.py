"""Check the validation command against actual kernels, including error tags."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "check_notebooks", ROOT / "scripts/check_notebooks.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "name,tags,should_pass",
    [
        ("success", [], True),
        ("ordinary_error", [], False),
        ("tagged_error", ["raises-exception"], False),
        ("skip_tagged_error", ["skip-execution"], False),
    ],
)
def test_runner_executes_every_cell_and_reports_errors(
    tmp_path, monkeypatch, name, tags, should_pass
):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    # Track the environment changes made inside main so each test restores them.
    for variable in (
        "HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "KERAS_HOME",
        "SCIKIT_LEARN_DATA", "SEABORN_DATA", "XDG_CACHE_HOME", "MPLCONFIGDIR",
        "MPLBACKEND", "TOKENIZERS_PARALLELISM", "TF_CPP_MIN_LOG_LEVEL",
    ):
        if variable in os.environ:
            monkeypatch.setenv(variable, os.environ[variable])
        else:
            # Register the absent key, then leave it absent for setdefault.
            monkeypatch.setenv(variable, "")
            monkeypatch.delenv(variable)

    notebook_directory = tmp_path / "Tools" / "Fixture"
    notebook_directory.mkdir(parents=True)
    source = (
        "from pathlib import Path\n"
        "import sys\n"
        f"assert Path.cwd() == Path({str(notebook_directory)!r})\n"
        f"assert sys.executable == {sys.executable!r}\n"
        "print('executed fixture')"
        if should_pass else "raise RuntimeError('runner regression sentinel')"
    )
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(source, metadata={"tags": tags})
    ])
    relative = Path("Tools") / "Fixture" / f"{name}.ipynb"
    original = tmp_path / relative
    nbformat.write(notebook, original)
    original_bytes = original.read_bytes()
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "check_notebooks.py", "--notebook", str(relative),
        "--output-dir", str(output), "--cache-dir", str(tmp_path / "cache"),
        "--timeout", "30",
    ])

    exit_code = runner.main()

    assert exit_code == (0 if should_pass else 1)
    results = json.loads((output / "results.json").read_text())
    assert len(results) == 1
    assert results[0]["mode"] == "execute"
    assert results[0]["status"] == ("passed" if should_pass else "failed")
    executed = nbformat.read(output / relative, as_version=4)
    assert executed.cells[0].execution_count == 1
    if should_pass:
        assert any(
            item.output_type == "stream" and "executed fixture" in item.text
            for item in executed.cells[0].outputs
        )
    else:
        # Kernel startup or fixture failures must not masquerade as the expected error.
        assert "RuntimeError" in results[0]["error"]
        assert "runner regression sentinel" in results[0]["error"]
        assert any(
            item.output_type == "error" and item.ename == "RuntimeError"
            for item in executed.cells[0].outputs
        )
    assert original.read_bytes() == original_bytes
