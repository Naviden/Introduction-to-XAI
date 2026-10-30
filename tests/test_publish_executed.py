"""Shared asset definitions must invalidate otherwise unchanged reading copies."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_script("check_notebooks")
PUBLISHER = load_script("publish_executed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed_dependency", ["xai_assets.py", "assets/manifest.json"])
@pytest.mark.parametrize("reference_location", ["code", "markdown"])
def test_changed_shared_asset_definition_blocks_stale_publication(
    tmp_path, monkeypatch, changed_dependency, reference_location
):
    root = tmp_path / "repository"
    execution = tmp_path / "execution"
    relative = Path("Tools/Example/example.ipynb")
    source_path = root / relative
    executed_path = execution / relative
    source_path.parent.mkdir(parents=True)
    executed_path.parent.mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "xai_assets.py").write_text("VALUE = 2\n")
    (root / "assets/manifest.json").write_text('{"schema_version": 1, "files": {}}\n')
    cells = [nbformat.v4.new_code_cell("print(2)")]
    if reference_location == "code":
        cells[0].source = "import xai_assets\nprint(xai_assets.VALUE)"
    else:
        # The text notebook loads xai_assets indirectly through a local helper
        # and explicitly identifies that shared dependency in its prose.
        cells.insert(0, nbformat.v4.new_markdown_cell("A local helper verifies inputs with `xai_assets`."))
    source = nbformat.v4.new_notebook(cells=cells)
    nbformat.write(source, source_path)
    monkeypatch.setattr(RUNNER, "ROOT", root)
    dependencies = RUNNER.local_dependencies(source_path)
    assert dependencies == {
        "xai_assets.py": digest(root / "xai_assets.py"),
        "assets/manifest.json": digest(root / "assets/manifest.json"),
    }
    record = {
        "notebook": relative.as_posix(), "mode": "execute", "status": "passed",
        "source_sha256": digest(source_path), "local_dependencies": dependencies,
    }
    executed = copy.deepcopy(source)
    executed.cells[-1].execution_count = 1
    executed.cells[-1].outputs = [nbformat.v4.new_output("stream", name="stdout", text="2\n")]
    executed.metadata["validation"] = {
        "source_sha256": record["source_sha256"], "status": "passed",
        "local_dependencies": dependencies,
    }
    nbformat.write(executed, executed_path)
    record["executed_sha256"] = digest(executed_path)
    assert PUBLISHER.checked_notebook(execution, record, root=root).cells[-1].outputs[0].text == "2\n"

    dependency = root / changed_dependency
    if changed_dependency.endswith(".json"):
        dependency.write_text('{"schema_version": 2, "files": {}}\n')
    else:
        dependency.write_text("VALUE = 3\n")
    # Neither the notebook nor its saved outputs changed. The dependency receipt
    # is the only evidence capable of catching this stale execution.
    assert digest(source_path) == record["source_sha256"]
    assert digest(executed_path) == record["executed_sha256"]
    with pytest.raises(ValueError, match="Local helper dependencies changed"):
        PUBLISHER.checked_notebook(execution, record, root=root)
