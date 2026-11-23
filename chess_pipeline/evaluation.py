"""
Probability metrics and deterministic baseline; no fitting or network I/O.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .config import CLASS_IDS


@dataclass(frozen=True)
class EvaluationResult:
    metrics: pd.DataFrame
    accuracy_comparison: pd.DataFrame
    predictions: dict
    per_event: pd.DataFrame


def deterministic_rating_rule(frame: pd.DataFrame) -> np.ndarray:
    difference = frame["white_rating"] - frame["black_rating"]
    # black win=0, draw=1, white win=2; equality predicts draw.
    return np.select([difference > 0, difference < 0], [2, 0], default=1)


def evaluate_probabilities(y, probabilities) -> dict:
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(y), 3):
        raise ValueError("Expected probabilities in black_win / draw / white_win order")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid class probabilities")
    auc = (roc_auc_score(y, probabilities, labels=CLASS_IDS, multi_class="ovo", average="macro")
           if set(np.unique(y)) == set(CLASS_IDS) else np.nan)
    return {
        "log_loss": log_loss(y, probabilities, labels=CLASS_IDS),
        "accuracy": accuracy_score(y, probabilities.argmax(axis=1)),
        "roc_auc_ovo_macro": auc,
        "brier_multiclass": np.square(probabilities-np.eye(3)[np.asarray(y)]).sum(axis=1).mean(),
    }


def evaluate_models(models: dict, prior: np.ndarray, parts: dict) -> EvaluationResult:
    """Evaluate every predictor on identical rows; the hard rule has accuracy only."""
    reports, accuracy_reports, predictions = [], [], {}
    for split_name in ("validation", "test"):
        frame = parts[split_name]
        probabilities_by_model = {
            "class_prior": np.tile(prior, (len(frame), 1)),
            **{name: model.predict_proba(frame[list(columns)])
               for name, (model, columns) in models.items()},
        }
        predictions[split_name] = probabilities_by_model
        for name, probabilities in probabilities_by_model.items():
            reports.append({"split": split_name, "model": name,
                            **evaluate_probabilities(frame["y"], probabilities)})
            accuracy_reports.append({"split": split_name, "model": name,
                                     "accuracy": reports[-1]["accuracy"]})
        accuracy_reports.append({"split": split_name, "model": "higher_rating_rule",
                                 "accuracy": accuracy_score(frame["y"], deterministic_rating_rule(frame))})
    per_event = []
    for event_id, frame in parts["test"].groupby("event_id"):
        for name, (model, columns) in models.items():
            per_event.append({"event": event_id, "model": name, "games": len(frame),
                              **evaluate_probabilities(frame["y"], model.predict_proba(frame[list(columns)]))})
    return EvaluationResult(
        metrics=pd.DataFrame(reports).set_index(["split", "model"]),
        accuracy_comparison=pd.DataFrame(accuracy_reports).set_index(["split", "model"]),
        predictions=predictions, per_event=pd.DataFrame(per_event),
    )
