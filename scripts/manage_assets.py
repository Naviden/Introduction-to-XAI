"""List, prefetch and verify the repository's registered teaching assets.

Examples (from the repository root):
  python scripts/manage_assets.py list
  python scripts/manage_assets.py prefetch nhanesi_X nhanesi_y
  python scripts/manage_assets.py prefetch hf:bert_attention
  python scripts/manage_assets.py verify --all

Prefetch never selects everything implicitly. `prefetch --all` explicitly opts
into every registered file and model, potentially several GB of downloads.
Verification never downloads missing bytes. XAI_OFFLINE=1 also makes prefetch
cache-only. File assets use XAI_ASSET_CACHE (default: validation/assets); Hub
assets use the Hugging Face cache (HF_HOME/HF_HUB_CACHE).

CIFAR, California housing, 20 Newsgroups and OpenML archives are listed by name
but their download here does not populate Keras/scikit-learn/OpenML caches.
Prepare those native caches with the relevant library loaders before offline
notebook execution. Decoded-dataset fingerprints are checked by the notebooks
after loading; this command verifies registered file bytes, not decoded arrays.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct script execution puts scripts/, rather than the repository, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import xai_assets  # noqa: E402


CACHE_NOTE = (
    "File assets: XAI_ASSET_CACHE, default validation/assets. Hugging Face: "
    "HF_HOME/HF_HUB_CACHE. Fetching native archives (cifar10_archive, "
    "california_archive, newsgroups_archive, bike_arff) does not populate "
    "Keras/scikit-learn/OpenML caches; prepare those separately with their "
    "library loaders. Decoded dataset fingerprints require loaded arrays and "
    "are checked inside the notebooks."
)


def inventory(spec):
    """Return explicitly namespaced assets, without touching any cache/network."""
    result = {}
    for section, prefix in [("files", "file"), ("huggingface", "hf"), ("bundled", "bundled")]:
        for key, details in spec.get(section, {}).items():
            result[f"{prefix}:{key}"] = (prefix, key, details)
    return result


def resolve_keys(assets, requested, all_assets):
    """Validate all requested names before performing any downloads."""
    if all_assets:
        if requested:
            raise ValueError("Use named assets or --all, not both")
        return list(assets)
    if not requested:
        raise ValueError("Name one or more assets, or explicitly pass --all")
    selected = []
    for key in requested:
        matches = [name for name in assets if name == key or name.split(":", 1)[1] == key]
        if len(matches) != 1:
            reason = "Unknown" if not matches else "Ambiguous"
            raise ValueError(f"{reason} asset {key!r}; use a namespaced key from the list command")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def handle_asset(asset, offline):
    kind, key, details = asset
    if kind == "file":
        return xai_assets.asset_file(key, offline=offline)
    if kind == "hf":
        return xai_assets.hf_model_path(key, offline=offline)
    # Bundled material is already in the checkout; "prefetch" verifies it too.
    path = (ROOT / details["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Bundled asset path is outside the repository: {key}")
    return xai_assets.verify_file(path, details["sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List registered files, models and cache requirements")
    listing.add_argument("--json", action="store_true", help="Emit the inventory as JSON")
    for command in ["prefetch", "verify"]:
        operation = commands.add_parser(command, help="Download and verify selected assets" if command == "prefetch" else "Verify cached assets without downloading")
        operation.add_argument("keys", nargs="*", help="Names from list; file:, hf: and bundled: prefixes are accepted")
        operation.add_argument("--all", action="store_true", help="Explicitly select every asset; prefetch can download several GB")
    args = parser.parse_args(argv)
    spec = xai_assets.manifest()
    assets = inventory(spec)
    if args.command == "list":
        if args.json:
            print(json.dumps({"assets": {name: details for name, (_, _, details) in assets.items()},
                              "decoded_datasets": spec.get("decoded_datasets", {}),
                              "cache_note": CACHE_NOTE}, indent=2))
        else:
            for name, (kind, _, details) in assets.items():
                location = details["repo_id"] + "@" + details["revision"] if kind == "hf" else details.get("filename", details.get("path", ""))
                print(f"{name}\t{location}")
            print("\n" + CACHE_NOTE)
        return 0
    try:
        selected = resolve_keys(assets, args.keys, args.all)
    except ValueError as error:
        parser.error(str(error))
    failed = False
    for name in selected:
        try:
            path = handle_asset(assets[name], offline=args.command == "verify")
            print(f"VERIFIED {name}: {path}")
        except Exception as error:
            failed = True
            print(f"FAILED {name}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
