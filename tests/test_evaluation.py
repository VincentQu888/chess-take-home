"""Metric definitions, training boundaries and baseline comparisons."""

import unittest

import numpy as np
import pandas as pd

from chess_pipeline.config import CLASS_IDS, FEATURES
from chess_pipeline.evaluation import deterministic_rating_rule, evaluate_models, evaluate_probabilities
from chess_pipeline.modeling import get_gain_importance, train_models, training_class_prior


class EvaluationTests(unittest.TestCase):
    def test_deterministic_rule_and_probability_metrics(self):
        examples = pd.DataFrame({"white_rating": [2500, 2400, 2300], "black_rating": [2400]*3})
        self.assertEqual(deterministic_rating_rule(examples).tolist(), [2, 1, 0])
        perfect = np.eye(3)*0.98 + np.ones((3, 3))*(0.02/3)
        result = evaluate_probabilities(np.arange(3), perfect)
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["roc_auc_ovo_macro"], 1)
        self.assertAlmostEqual(result["log_loss"], -np.log(perfect[0, 0]))
        self.assertAlmostEqual(result["brier_multiclass"], ((perfect[0]-np.eye(3)[0])**2).sum())
        self.assertLess(result["log_loss"], evaluate_probabilities(CLASS_IDS, np.ones((3, 3))/3)["log_loss"])
        self.assertTrue(np.isnan(evaluate_probabilities(np.array([0, 2]), perfect[[0, 2]])["roc_auc_ovo_macro"]))

    def test_invalid_probabilities(self):
        for probs in [np.ones((3, 3)), np.ones((3, 2)), np.full((3, 3), np.nan),
                      np.tile([-1, 1, 1], (3, 1))]:
            with self.assertRaises(ValueError):
                evaluate_probabilities(CLASS_IDS, probs)

    def test_train_and_evaluate_without_test_fitting(self):
        frame = pd.DataFrame({name: np.linspace(0, 1, 12) for name in FEATURES})
        frame = frame.assign(y=np.tile(np.arange(3), 4), white_rating=2500,
                             black_rating=2400, event_id="test_event")
        parts = {"train": frame.iloc[:6].copy(), "validation": frame.iloc[6:9].copy(),
                 "test": frame.iloc[9:].copy()}
        models = train_models(parts["train"], parts["validation"])
        prior = training_class_prior(parts["train"])
        result = evaluate_models(models, prior, parts)
        self.assertEqual(len(result.metrics), 6)
        self.assertEqual(len(result.accuracy_comparison), 8)
        self.assertNotIn("higher_rating_rule", result.metrics.index.get_level_values("model"))
        for model, columns in models.values():
            self.assertEqual(model.booster_.feature_name(), list(columns))
            self.assertEqual(len(get_gain_importance(model)), len(columns))
        np.testing.assert_allclose(prior, [1/3]*3)
        self.assertEqual(len(result.per_event), 2)
        with self.assertRaises(ValueError):
            train_models(parts["train"].assign(y=0), parts["validation"])


if __name__ == "__main__":
    unittest.main()
