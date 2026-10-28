"""Regression checks for numerical helpers embedded in teaching notebooks."""
import ast
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def notebook_function(path, name):
    notebook = json.loads((ROOT / path).read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {'np': np}
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
                return namespace[name]
    raise AssertionError(f'{name} not found in {path}')


class SurvivalConcordanceTests(unittest.TestCase):
    def setUp(self):
        self.c_index = notebook_function(
            'Tools/SHAP_examples/SHAP_survival_model.ipynb', 'c_statistic_harrell'
        )

    def test_risk_order_and_ties(self):
        self.assertEqual(self.c_index([3, 2, 1], [1, 2, 3]), 1.0)
        self.assertEqual(self.c_index([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertEqual(self.c_index([1, 1, 1], [1, 2, 3]), 0.5)

    def test_censoring_only_counts_known_orderings(self):
        # Person 0 is censored before person 1's event: their pair is unknowable.
        # Person 2 survives longer than person 1's observed event.
        self.assertEqual(self.c_index([100, 2, 1], [-1, 2, -3]), 1.0)

    def test_no_comparable_pairs_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'No comparable'):
            self.c_index([2, 1], [-1, -2])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.c_index([1], [1, 2])
        with self.assertRaises(ValueError):
            self.c_index([np.nan, 1], [1, 2])


if __name__ == '__main__':
    unittest.main()
