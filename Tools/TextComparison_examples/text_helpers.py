"""Shared bag-of-word-type masking for the text comparison.

Every explainer calls the same fitted raw-text pipeline. These semantics are
specific to its unigram tokenizer; they are not a claim about contextual models.
"""
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import xai_assets  # noqa: E402


def tokenize(text):
    """Lowercase Unicode word tokens, including one-letter words and numbers."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return re.findall(r"(?u)\b\w+\b", text.casefold())


def load_reviews(directory):
    """Verify and read the bundled corpus without dropping accented rows."""
    texts, labels = [], []
    for label, (suffix, key) in enumerate([("neg", "polarity_negative"), ("pos", "polarity_positive")]):
        path = Path(directory) / f"rt-polarity.{suffix}"
        xai_assets.verify_file(path, xai_assets.manifest()["bundled"][key]["sha256"])
        # Records are LF-delimited. splitlines() also treats legacy 0x85
        # punctuation inside reviews as a line break, inventing extra records.
        lines = [line.strip() for line in path.read_text(encoding="latin-1").split("\n") if line.strip()]
        texts.extend(lines)
        labels.extend([label] * len(lines))
    return np.asarray(texts, dtype=object), np.asarray(labels)


class WordMasker:
    """One binary variable per distinct normalized word type.

    Removing a type removes all its occurrences; retaining it preserves its
    original count. Rejoining tokens is prediction-equivalent only for the
    notebook's unigram model, which already discards punctuation and order.
    """
    def __init__(self, text):
        self.tokens = tokenize(text)
        self.words = list(dict.fromkeys(self.tokens))
        if not self.words:
            raise ValueError("The selected text must contain at least one word token")
        positions = {word: j for j, word in enumerate(self.words)}
        self.token_groups = np.asarray([positions[word] for word in self.tokens])

    def texts(self, presence):
        presence = np.asarray(presence)
        if presence.ndim != 2 or presence.shape[1] != len(self.words):
            raise ValueError("presence must have one column per word type")
        if not np.isin(presence, [0, 1]).all():
            raise ValueError("Word masks must be binary")
        return [" ".join(token for token, group in zip(self.tokens, self.token_groups) if mask[group])
                for mask in presence]


def target_margin(pipeline, texts, target_label):
    """Log odds of target_label against the other class in a binary pipeline."""
    classes = list(pipeline.classes_)
    if len(classes) != 2 or target_label not in classes:
        raise ValueError("Require a binary classifier and a known target label")
    margin = np.asarray(pipeline.decision_function(texts), dtype=float)
    if margin.shape != (len(texts),) or not np.isfinite(margin).all():
        raise ValueError("Require one finite binary decision margin per text")
    return margin if classes.index(target_label) == 1 else -margin


def coordinate_contributions(pipeline, text, target_label):
    """Exact fitted TF-IDF coordinate decomposition, not deletion-game Shapley.

    Deletion re-normalizes the remaining TF-IDF coordinates, so these products
    are not the marginal changes caused by deleting raw words.
    """
    masker = WordMasker(text)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    classes = list(pipeline.classes_)
    if len(classes) != 2 or target_label not in classes:
        raise ValueError("Require a binary classifier and a known target label")
    sign = 1 if classes.index(target_label) == 1 else -1
    row = vectorizer.transform([text])
    weighted = np.asarray(row.multiply(classifier.coef_[0]).toarray())[0] * sign
    values = np.asarray([weighted[vectorizer.vocabulary_[word]] if word in vectorizer.vocabulary_ else 0.
                         for word in masker.words])
    baseline = float(sign * classifier.intercept_[0])
    np.testing.assert_allclose(baseline + values.sum(), target_margin(pipeline, [text], target_label)[0], atol=1e-10)
    return masker.words, values, baseline


def deletion_curve(predict_margin, order):
    """Original-minus-masked target margin as each word type is removed."""
    order = np.asarray(order)
    d = len(order)
    if d == 0 or not np.array_equal(np.sort(order), np.arange(d)):
        raise ValueError("order must be a permutation of all word indices")
    masks = np.ones((d + 1, d), dtype=int)
    for k in range(1, d + 1):
        masks[k, order[:k]] = 0
    margin = np.asarray(predict_margin(masks), dtype=float)
    if margin.shape != (d + 1,) or not np.isfinite(margin).all():
        raise ValueError("Require one finite target margin per mask")
    return margin[0] - margin


def similarity(scores, reference, k=5):
    """Within-method absolute top-k Jaccard and signed cosine; zero is undefined."""
    a, b = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("Require matching finite score vectors")
    if not 1 <= k <= len(a):
        raise ValueError("k must lie between 1 and the number of words")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return np.nan, np.nan
    left = set(np.argsort(-np.abs(a), kind="stable")[:k])
    right = set(np.argsort(-np.abs(b), kind="stable")[:k])
    return len(left & right) / len(left | right), float(np.clip(a @ b / norm, -1, 1))
