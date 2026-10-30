"""Save verified executed notebooks as durable GitHub-readable teaching snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit, quote

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def checked_notebook(directory, record, root=ROOT):
    relative = Path(record["notebook"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Notebook path must be inside the repository")
    if record.get("mode") != "execute" or record.get("status") != "passed":
        raise ValueError(f"Cannot publish an unsuccessful/non-execution record: {relative}")
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if digest != record.get("source_sha256"):
        raise ValueError(f"Source changed since execution: {relative}")
    dependencies = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root / relative).parent.rglob("*.py"))
                    if not any(part in {"__pycache__", "results", "logs"} for part in p.parts)}
    if "xai_assets" in (root / relative).read_text():
        for dependency in ("xai_assets.py", "assets/manifest.json"):
            dependencies[dependency] = hashlib.sha256((root / dependency).read_bytes()).hexdigest()
    if dependencies != record.get("local_dependencies"):
        raise ValueError(f"Local helper dependencies changed since execution: {relative}")
    if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != record.get("executed_sha256"):
        raise ValueError(f"Executed artifact changed since validation: {relative}")
    notebook = nbformat.read(directory / relative, as_version=4)
    nbformat.validate(notebook)
    if notebook.metadata.get("validation", {}).get("source_sha256") != digest:
        raise ValueError(f"Executed notebook has no matching validation metadata: {relative}")
    source = nbformat.read(root / relative, as_version=4)
    if [(c.cell_type, c.source) for c in source.cells] != [(c.cell_type, c.source) for c in notebook.cells]:
        raise ValueError(f"Executed cell sources do not match the original: {relative}")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if cell.source.strip() and cell.execution_count is None:
                raise ValueError(f"Unexecuted code in {relative}")
            if any(o.output_type == "error" for o in cell.get("outputs", [])):
                raise ValueError(f"Error output in {relative}")
    return notebook


def relocate_links(text, source, destination):
    def replace(match):
        target = match.group(1)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        original = (source.parent / unquote(parsed.path)).resolve()
        relocated = quote(os.path.relpath(original, destination.parent), safe="/.-_")
        if parsed.fragment:
            relocated += "#" + parsed.fragment
        return "](" + relocated + ")"
    return re.sub(r"\]\(([^\s)]+)\)", replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--notebook", action="append", help="Publish only these repository-relative notebooks; repeat to select several")
    parser.add_argument("--destination", type=Path, default=ROOT / "docs/executed")
    args = parser.parse_args()
    records = json.loads((args.input / "results.json").read_text())
    if args.notebook:
        selected = set(args.notebook)
        if selected - {r["notebook"] for r in records}:
            raise ValueError("A selected notebook is missing from the execution records")
        records = [r for r in records if r["notebook"] in selected]
    # Verify the entire batch before writing a single reading copy.
    prepared = [(r, checked_notebook(args.input, r)) for r in records]
    args.destination.mkdir(parents=True, exist_ok=True)
    for record, notebook in prepared:
        relative = Path(record["notebook"])
        destination = args.destination / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                cell.source = relocate_links(cell.source, ROOT / relative, destination)
        nbformat.write(notebook, destination)
    # Multiple validation batches can be published without erasing prior snapshots.
    batch = args.input.name
    manifests = args.destination / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / f"{batch}-results.json").write_text(json.dumps(records, indent=2) + "\n")
    shutil.copyfile(args.input / "environment.json", manifests / f"{batch}-environment.json")
    lines = ["# Notebooks with outputs", "", "These notebooks can be read on GitHub without installing Python. "
             "Use the source links below to run or adapt them. "
             "Relative imports in the source assume its original directory.", "",
             "The manifests record source hashes, measured times and software/platform versions. "
             "They describe a particular run; outputs can change with data, seeds, versions or hardware.", ""]
    for path in sorted(args.destination.rglob("*.ipynb")):
        relative = path.relative_to(args.destination)
        title = path.stem.replace("_", " ")
        source_link = quote(os.path.relpath(ROOT / relative, args.destination), safe="/.-_")
        lines.append(f"- [{title}]({quote(relative.as_posix(), safe='/.-_')}) "
                     f"· [source notebook]({source_link})")
    lines += ["", "Regenerate using `scripts/check_notebooks.py`, then "
              "`scripts/publish_executed.py --input validation/<run>`. "
              "Publication rejects changed sources, failed runs and unexecuted cells.", ""]
    (args.destination / "README.md").write_text("\n".join(lines))
    print(f"Published {len(prepared)} verified reading copies to {args.destination}")


if __name__ == "__main__":
    main()
