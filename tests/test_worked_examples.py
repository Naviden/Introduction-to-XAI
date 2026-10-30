"""Check published arithmetic and ensure stale or broken examples fail visibly."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "worked_examples", ROOT / "scripts/check_worked_examples.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def test_published_worked_examples_match_computed_values():
    assert set(CHECKER.check_repository()) == {
        "shap-logit", "lime-kernel", "rulefit-rules",
    }


def test_old_shap_arithmetic_is_rejected(tmp_path):
    source = (ROOT / "Tools/SHAP_examples/How SHAP works.md").read_text()
    document = tmp_path / "SHAP.md"
    document.write_text(source.replace("glucose: 7.400 logit units",
                                       "glucose: 7.100 logit units"))
    with pytest.raises(ValueError, match="output mismatch"):
        CHECKER.check_document(document)


def test_required_example_cannot_disappear(tmp_path):
    with pytest.raises(ValueError, match="Missing required worked examples"):
        CHECKER.check_repository(tmp_path)


@pytest.mark.parametrize("code, expected, message", [
    ("raise RuntimeError('broken')", "", "execution failed"),
    ("import time; time.sleep(30)", "", "exceeded"),
    ("print(2)", "3", "output mismatch"),
])
def test_broken_examples_fail(tmp_path, code, expected, message):
    document = tmp_path / "example.md"
    document.write_text(
        "<!-- checked-example: example -->\n```python\n" + code
        + "\n```\n```text\n" + expected
        + "\n```\n<!-- /checked-example -->\n"
    )
    with pytest.raises(ValueError, match=message):
        CHECKER.check_document(document, timeout=0.2 if message == "exceeded" else 10.0)


def test_unclosed_example_cannot_be_silently_skipped(tmp_path):
    document = tmp_path / "example.md"
    document.write_text("<!-- checked-example: incomplete -->\n```python\nprint(1)\n```")
    with pytest.raises(ValueError, match="malformed"):
        CHECKER.check_document(document)
