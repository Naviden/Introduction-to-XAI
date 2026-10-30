"""Archive raw core-study tables only after matching successful execution receipts."""
from pathlib import Path
import argparse
import json
import shutil

from publish_executed import ROOT, checked_notebook


STUDIES = {
    "evaluation": ("Tools/Evaluation_examples/Evaluation_and_Sensitivity.ipynb", "validation/evaluation-study"),
    "mixed-data": ("Tools/PracticalWorkflow_examples/Mixed_Data_Workflow.ipynb", "validation/mixed-data-workflow"),
    "recourse": ("Tools/CausalRecourse_examples/Causal_Recourse.ipynb", "Tools/CausalRecourse_examples/results"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Foundations runner output directory")
    args = parser.parse_args()
    records = {r["notebook"]: r for r in json.loads((args.input / "results.json").read_text())}
    for notebook, _ in STUDIES.values():
        checked_notebook(args.input, records[notebook])
    destination = ROOT / "docs/results"
    destination.mkdir(parents=True, exist_ok=True)
    lines = ["# Recorded core-study results", "", "These are measured results from the current validated teaching runs, not "
             "a leaderboard or estimates of real-world effects. Raw per-case/per-run CSVs and JSON manifests retain "
             "the settings, units, failures and variability needed to interpret the figures.", "",
             "`git_commit` in a study manifest is the checkout base when `git_dirty` is true; "
             "the recorded source/helper hashes identify the exact executed changes. Runtime depends on hardware "
             "and concurrent work. Read the notebook's assumptions before comparing numbers.", ""]
    for name, (notebook, directory) in STUDIES.items():
        target = destination / name
        target.mkdir(exist_ok=True)
        files = sorted(p for p in (ROOT / directory).iterdir() if p.suffix in {".csv", ".json"})
        if not files:
            raise ValueError(f"No results found for {name}; execute the source notebook first")
        for path in files:
            if path.name == "results.json":
                report = json.loads(path.read_text())
                # CSVs retain every measurement. Avoid duplicating those tables
                # as tens of thousands of JSON lines in the manifest.
                report.pop("tables", None)
                report["table_files"] = [p.name for p in files if p.suffix == ".csv"]
                (target / path.name).write_text(json.dumps(report, indent=2) + "\n")
            else:
                shutil.copyfile(path, target / path.name)
        (target / "execution.json").write_text(json.dumps(records[notebook], indent=2) + "\n")
        lines.append(f"- [{name}]({name}/): [source notebook](../../{notebook}); "
                     f"{len(files)} data/manifest files plus the execution receipt.")
    lines += ["", "Regenerate with `python scripts/check_notebooks.py --group foundations "
              "--output-dir validation/foundations-final`, then "
              "`python scripts/publish_results.py --input validation/foundations-final`. "
              "[Reading copies](../executed/README.md) include the plotted results.", ""]
    (destination / "README.md").write_text("\n".join(lines))
    print(f"Archived {len(STUDIES)} core studies under {destination}")


if __name__ == "__main__":
    main()
