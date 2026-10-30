"""Check local Markdown/HTML links in tracked teaching documents (no network)."""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
HTML_LINK = re.compile(r'''(?:href|src)=["']([^"']+)["']''')


def local_targets(text):
    # Code examples are not actual links rendered by Markdown.
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    for raw in MARKDOWN_LINK.findall(text) + HTML_LINK.findall(text):
        target = raw.strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        yield unquote(parsed.path)


def documents(root=ROOT):
    for folder in (root, root / "docs", root / "Theory", root / "Tools", root / "Practice"):
        paths = folder.glob("*.md") if folder == root else folder.rglob("*")
        for path in paths:
            if path.suffix not in {".md", ".ipynb"} or any(
                p in {"results", "logs", ".ipynb_checkpoints"} for p in path.parts
            ):
                continue
            if path.suffix == ".ipynb":
                notebook = json.loads(path.read_text())
                text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
            else:
                text = path.read_text()
            yield path, text


def main():
    errors = []
    count = 0
    for path, text in documents():
        for target in local_targets(text):
            count += 1
            resolved = ROOT / target.lstrip("/") if target.startswith("/") else path.parent / target
            if not resolved.exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {target}")
    for error in errors:
        print(error)
    print(f"Checked {count} local links; {len(errors)} missing targets.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
