"""Save verified executed notebooks as durable GitHub-readable teaching snapshots."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit, quote

import nbformat

ROOT = Path(__file__).resolve().parents[1]

# Only this optional notebook-widget warning is omitted. Numerical and model
# warnings must remain visible, even when their local filenames are normalized.
WIDGET_WARNING = re.compile(
    r"^.*\bTqdmWarning: IProgress not found\. Please update jupyter and ipywidgets\."
    r" See https://ipywidgets\.readthedocs\.io/en/stable/user_install\.html\r?\n"
    r"(?:[ \t]+from \.autonotebook import tqdm as notebook_tqdm\r?\n)?",
    re.MULTILINE,
)
LOCAL_PATH = re.compile(
    r"(?:/(?:Users|home)/|[A-Za-z]:[\\/]Users[\\/]|/(?:private/)?(?:tmp|var/folders)/)"
    r"[^\s<>\"']+"
)


def portable_text(text, root=ROOT):
    """Normalize machine-specific paths without changing diagnostic messages."""
    root_text = str(root.resolve())
    def replace(match):
        path = match.group(0).replace("\\", "/")
        if "/site-packages/" in path:
            return "site-packages/" + path.split("/site-packages/", 1)[1]
        # Keep line numbers and sentence delimiters around a redacted path.
        suffix = re.search(r"(?::\d+(?::\d+)?)?[.,;:()\[\]{}]*$", path).group()
        return "[local-path]" + suffix

    # URL paths such as https://example.org/home/tutorial are not local paths.
    parts = re.split(r"(https?://[^\s<>\"']+)", text)
    for index in range(0, len(parts), 2):
        # Handle the known checkout first, including checkout paths with spaces.
        part = parts[index].replace(root_text + os.sep, "./")
        part = re.sub(re.escape(root_text) + r"(?=$|[\s<>\"'])", ".", part)
        part = re.sub(r"\./[^\s<>\"']*?site-packages/", "site-packages/", part)
        parts[index] = LOCAL_PATH.sub(replace, part)
    return "".join(parts)


def portable_value(value, root=ROOT):
    """Normalize paths in nested output data/metadata without touching numbers."""
    if isinstance(value, str):
        return portable_text(value, root)
    if isinstance(value, list):
        return [portable_value(item, root) for item in value]
    if isinstance(value, dict):
        return {key: portable_value(item, root) for key, item in value.items()}
    return value


def prepare_public_notebook(notebook, root=ROOT):
    """Clean a copy for publication; retain the original execution as evidence."""
    public = copy.deepcopy(notebook)
    public.metadata = portable_value(public.metadata, root)
    for cell in public.cells:
        cell.metadata = portable_value(cell.metadata, root)
        if cell.cell_type != "code":
            continue
        outputs = []
        for output in cell.get("outputs", []):
            # A warning can be split across consecutive stderr stream messages.
            if (output.output_type == "stream" and output.name == "stderr"
                    and outputs and outputs[-1].output_type == "stream"
                    and outputs[-1].name == "stderr"):
                outputs[-1].text += output.text
            else:
                outputs.append(output)
        cleaned = []
        for output in outputs:
            if output.output_type == "stream" and output.name == "stderr":
                output.text = WIDGET_WARNING.sub("", output.text)
                if not output.text:
                    continue
            for key in ("text", "traceback", "evalue", "metadata"):
                if key in output:
                    output[key] = portable_value(output[key], root)
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/") or mime == "application/json":
                    output.data[mime] = portable_value(value, root)
            cleaned.append(output)
        cell.outputs = cleaned
    return public


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
        notebook = prepare_public_notebook(notebook)
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
             "They describe a particular run; outputs can change with data, seeds, versions or hardware.", "",
             "Published outputs use portable paths and omit the optional tqdm widget-availability warning. "
             "Other warnings and computed results are retained. Executed-artifact hashes refer to the "
             "original, unmodified validation files.", ""]
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
