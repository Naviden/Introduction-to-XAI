"""Regression checks for the text corpus loader used by the Anchor tutorial."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def polarity_loader():
    path = ROOT / 'Tools/ANCHOR_examples/Anchor_tabular.ipynb'
    notebook = json.loads(path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.FunctionDef) and node.name == 'load_polarity':
                namespace = {'Path': Path}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
                return namespace['load_polarity']
    raise AssertionError('load_polarity not found')


class PolarityLoaderTests(unittest.TestCase):
    def test_latin1_rows_and_labels_are_preserved(self):
        # The old loader silently dropped non-UTF-8 reviews and returned bytes.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'rt-polarity.neg').write_bytes('bad caf\xe9\n\n'.encode('latin-1'))
            (path / 'rt-polarity.pos').write_bytes('tr\xe8s good\n'.encode('latin-1'))
            fixture_manifest = {"bundled": {
                key: {"path": name, "sha256": hashlib.sha256((path / name).read_bytes()).hexdigest()}
                for key, name in [("polarity_negative", "rt-polarity.neg"),
                                  ("polarity_positive", "rt-polarity.pos")]
            }}
            # This small fixture has its own declared identity. Production uses
            # the checked-in manifest; keep real checksum verification enabled.
            with patch("xai_assets.manifest", return_value=fixture_manifest):
                texts, labels = polarity_loader()(path)
        self.assertEqual(texts, ['bad caf\xe9', 'tr\xe8s good'])
        self.assertEqual(labels, [0, 1])

    def test_all_bundled_reviews_are_loaded(self):
        texts, labels = polarity_loader()(ROOT / 'Tools/ANCHOR_examples/rt-polaritydata')
        self.assertEqual(len(texts), 10662)
        self.assertEqual(labels.count(0), 5331)
        self.assertEqual(labels.count(1), 5331)
        self.assertTrue(all(isinstance(text, str) for text in texts))


class MerlinRuleTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from scipy.sparse import csr_matrix
        path = ROOT / 'Tools/Merlin_examples/20NG_example.ipynb'
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.FunctionDef) and node.name == 'text_rule_occurrence':
                    namespace = {}
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
                    self.evaluate = namespace['text_rule_occurrence']
        # Vocabularies intentionally differ from rule insertion order.
        self.manager = SimpleNamespace(
            feature_names=['Beta', 'Alpha'],
            sparse_samples=csr_matrix([[0, 1], [1, 0], [1, 1], [0, 0]]),
        )

    def test_positive_and_negative_conditions_use_feature_names(self):
        actual = self.evaluate(self.manager, '{Alpha: 1, Beta: 0}')
        self.assertEqual(actual.tolist(), [True, False, False, False])

    def test_unseen_positive_word_matches_nothing(self):
        self.assertEqual(self.evaluate(self.manager, '{NewWord: 1}').tolist(), [False] * 4)
        self.assertEqual(self.evaluate(self.manager, "{'NewWord': 1, 'Alpha': 1}").tolist(), [False] * 4)

    def test_unseen_negative_word_is_absent_everywhere(self):
        actual = self.evaluate(self.manager, '{NewWord: 0, Alpha: 1}')
        self.assertEqual(actual.tolist(), [True, False, True, False])

    def test_empty_conjunction_matches_every_row(self):
        self.assertEqual(self.evaluate(self.manager, '{}').tolist(), [True] * 4)

    def test_nonbinary_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.manager, '{Alpha: 2}')


if __name__ == '__main__':
    unittest.main()
