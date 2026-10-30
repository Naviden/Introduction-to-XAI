"""Prevent stale or incomplete executions from becoming trusted reading copies."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "publish_executed", ROOT / "scripts/publish_executed.py"
)
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def completed_run(tmp_path):
    root = tmp_path / "repository"
    directory = tmp_path / "execution"
    relative = Path("Tools/Example/example.ipynb")
    source_path = root / relative
    source_path.parent.mkdir(parents=True)
    artifact_path = directory / relative
    artifact_path.parent.mkdir(parents=True)
    helper = source_path.parent / "helper.py"
    helper.write_text("VALUE = 2\n")
    source = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("An example with a local dependency."),
        nbformat.v4.new_code_cell("from helper import VALUE\nprint(VALUE)"),
    ])
    nbformat.write(source, source_path)
    record = {
        "notebook": relative.as_posix(), "mode": "execute", "status": "passed",
        "source_sha256": digest(source_path),
        "local_dependencies": {helper.relative_to(root).as_posix(): digest(helper)},
    }
    executed = copy.deepcopy(source)
    executed.cells[1].execution_count = 1
    executed.cells[1].outputs = [nbformat.v4.new_output("stream", name="stdout", text="2\n")]
    executed.metadata["validation"] = {
        "source_sha256": record["source_sha256"], "status": "passed",
        "local_dependencies": record["local_dependencies"],
    }
    nbformat.write(executed, artifact_path)
    record["executed_sha256"] = digest(artifact_path)
    return root, directory, record, source_path, artifact_path, helper


def test_complete_matching_run_can_be_published(completed_run):
    root, directory, record, *_ = completed_run
    notebook = PUBLISHER.checked_notebook(directory, record, root=root)
    assert notebook.cells[1].outputs[0].text == "2\n"


def test_helper_change_invalidates_unchanged_notebook(completed_run):
    root, directory, record, _, _, helper = completed_run
    helper.write_text("VALUE = 3\n")
    with pytest.raises(ValueError, match="Local helper dependencies changed"):
        PUBLISHER.checked_notebook(directory, record, root=root)


def test_changed_artifact_is_rejected_even_with_valid_source_hash(completed_run):
    root, directory, record, _, artifact_path, _ = completed_run
    artifact = nbformat.read(artifact_path, as_version=4)
    artifact.cells[1].outputs[0].text = "invented result\n"
    nbformat.write(artifact, artifact_path)
    with pytest.raises(ValueError, match="Executed artifact changed"):
        PUBLISHER.checked_notebook(directory, record, root=root)


def test_changed_source_invalidates_old_execution(completed_run):
    root, directory, record, source_path, *_ = completed_run
    source = nbformat.read(source_path, as_version=4)
    source.cells[1].source += "\nprint('new computation')"
    nbformat.write(source, source_path)
    with pytest.raises(ValueError, match="Source changed since execution"):
        PUBLISHER.checked_notebook(directory, record, root=root)


@pytest.mark.parametrize("problem, message", [
    ("not_run", "Unexecuted code"),
    ("error", "Error output"),
    ("different_code", "Executed cell sources do not match"),
])
def test_matching_hashes_do_not_override_execution_checks(completed_run, problem, message):
    root, directory, record, _, artifact_path, _ = completed_run
    artifact = nbformat.read(artifact_path, as_version=4)
    if problem == "not_run":
        artifact.cells[1].execution_count = None
        artifact.cells[1].outputs = []
    elif problem == "error":
        artifact.cells[1].outputs = [nbformat.v4.new_output(
            "error", ename="RuntimeError", evalue="failed", traceback=["failed"]
        )]
    else:
        artifact.cells[1].source = "print('different code')"
    nbformat.write(artifact, artifact_path)
    # Model a damaged execution receipt with a matching artifact checksum. The
    # independent semantic checks must still reject it rather than trust hashes.
    record["executed_sha256"] = digest(artifact_path)
    with pytest.raises(ValueError, match=message):
        PUBLISHER.checked_notebook(directory, record, root=root)
