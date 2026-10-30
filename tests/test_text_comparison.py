"""Check the shared text masking game against fitted and analytic references."""
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "text_helpers", ROOT / "Tools/TextComparison_examples/text_helpers.py"
)
HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPERS)


def fitted_pipeline(norm="l2"):
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=HELPERS.tokenize, token_pattern=None,
                                lowercase=False, norm=norm)),
        ("classifier", LogisticRegression(C=5., random_state=42)),
    ]).fit(
        ["good good film", "great good movie", "good wonderful café",
         "bad bad film", "awful bad movie", "bad terrible café"],
        ["positive"] * 3 + ["negative"] * 3,
    )


def test_word_types_remove_every_occurrence_and_preserve_counts():
    masker = HELPERS.WordMasker("Good, GOOD! not bad café.")
    assert masker.words == ["good", "not", "bad", "café"]
    assert masker.texts([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]) == [
        "good good not bad café", "not bad café", "",
    ]
    with pytest.raises(ValueError, match="binary"):
        masker.texts([[0.5, 1, 1, 1]])


def test_original_and_rejoined_text_have_identical_model_outputs():
    pipeline = fitted_pipeline()
    text = "GOOD, good... bad café!"
    masker = HELPERS.WordMasker(text)
    reconstructed = masker.texts(np.ones((1, len(masker.words))))
    np.testing.assert_allclose(pipeline.predict_proba([text]), pipeline.predict_proba(reconstructed))


@pytest.mark.parametrize("target", ["negative", "positive"])
def test_coordinate_accounting_and_margin_use_the_requested_class(target):
    pipeline = fitted_pipeline()
    text = "good good bad unknownword"
    words, contributions, intercept = HELPERS.coordinate_contributions(pipeline, text, target)
    index = list(pipeline.classes_).index(target)
    margin = HELPERS.target_margin(pipeline, [text], target)[0]
    np.testing.assert_allclose(expit(margin), pipeline.predict_proba([text])[0, index])
    np.testing.assert_allclose(intercept + contributions.sum(), margin)
    np.testing.assert_allclose(intercept, HELPERS.target_margin(pipeline, [""], target)[0])
    assert contributions[words.index("unknownword")] == 0


@pytest.mark.parametrize("norm", [None, "l2"])
def test_normalization_changes_deletion_semantics(norm):
    pipeline = fitted_pipeline(norm=norm)
    text = "good good bad"
    masker = HELPERS.WordMasker(text)
    _, terms, _ = HELPERS.coordinate_contributions(pipeline, text, "positive")
    original = HELPERS.target_margin(pipeline, [text], "positive")[0]
    drops = original - HELPERS.target_margin(
        pipeline, masker.texts(1 - np.eye(len(masker.words), dtype=int)), "positive"
    )
    if norm is None:
        np.testing.assert_allclose(drops, terms, atol=1e-12)
    else:
        # Removing one raw word rescales the other coordinate. A linear
        # transformed model therefore need not be additive in word presence.
        assert np.max(np.abs(drops - terms)) > 0.05


def test_deletion_curve_preserves_negative_evidence_and_shared_endpoint():
    def margin(masks):
        return 3 + masks @ np.array([1., -2., 4.])
    np.testing.assert_allclose(HELPERS.deletion_curve(margin, [2, 0, 1]), [0, 4, 5, 3])
    with pytest.raises(ValueError, match="permutation"):
        HELPERS.deletion_curve(margin, [1, 1, 2])


def test_zero_similarity_is_undefined_and_sign_reversal_is_visible():
    assert all(np.isnan(value) for value in HELPERS.similarity([0, 0], [1, 2], k=1))
    jaccard, cosine = HELPERS.similarity([1, -3], [-1, 3], k=1)
    assert jaccard == 1
    assert cosine == pytest.approx(-1)


def test_loader_keeps_accented_lines_and_label_order_and_rejects_changed_bytes(tmp_path, monkeypatch):
    (tmp_path / "rt-polarity.neg").write_bytes("bad café\n\n".encode("latin-1"))
    (tmp_path / "rt-polarity.pos").write_bytes("très\x85 good\n".encode("latin-1"))
    expected = {"bundled": {
        key: {"sha256": hashlib.sha256((tmp_path / f"rt-polarity.{suffix}").read_bytes()).hexdigest()}
        for suffix, key in [("neg", "polarity_negative"), ("pos", "polarity_positive")]
    }}
    monkeypatch.setattr(HELPERS.xai_assets, "manifest", lambda: expected)
    texts, labels = HELPERS.load_reviews(tmp_path)
    assert texts.tolist() == ["bad café", "très\x85 good"]
    assert labels.tolist() == [0, 1]
    (tmp_path / "rt-polarity.pos").write_bytes(b"Changed corpus\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        HELPERS.load_reviews(tmp_path)
