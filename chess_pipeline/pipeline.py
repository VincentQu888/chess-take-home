"""Compose the independently testable data and experiment stages.

run_data_pipeline: API facts -> prior ratings -> features -> labeled time splits.
run_experiment: data pipeline -> validation-stopped models -> frozen evaluation.
Neither function writes a dataset or model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .api import ChessComClient
from .collection import collect_games
from .config import CLASS_NAMES, DEFAULT_CACHE, DEFAULT_END, DEFAULT_START, PipelineConfig
from .features import add_pregame_ratings, build_feature_table
from .splitting import split_chronologically, split_frames

if TYPE_CHECKING:
    from .evaluation import EvaluationResult


@dataclass(frozen=True)
class DataResult:
    raw: pd.DataFrame
    table: pd.DataFrame
    coverage: pd.DataFrame

    @property
    def parts(self) -> dict[str, pd.DataFrame]:
        return split_frames(self.table)


@dataclass(frozen=True)
class ExperimentResult:
    data: DataResult
    models: dict
    prior: np.ndarray
    evaluation: EvaluationResult
    importance: pd.DataFrame


def collect_dataset(start: str = DEFAULT_START, end: str = DEFAULT_END,
                    cache_dir: Path = DEFAULT_CACHE, *, fetch=None,
                    client: ChessComClient | None = None) -> pd.DataFrame:
    """Collection-stage entry point, preserving the original CLI's output schema."""
    client = client if client is not None else ChessComClient(cache_dir=cache_dir, fetch=fetch)
    facts = collect_games(start, end, client)
    frame = add_pregame_ratings(facts)
    frame.attrs["audit"] = {
        **facts.attrs["audit"],
        "missing_prior_rating_games": int(frame[["white_rating", "black_rating"]].isna().any(axis=1).sum()),
    }
    return frame


def prepare_dataset(raw: pd.DataFrame, config: PipelineConfig) -> DataResult:
    """Prepare cached games within the inclusive UTC event-date window.

    If the source pool changes, rebuild prior ratings rather than retaining
    references to removed games. Preserve the original collection audit and
    record the narrower preparation window separately.
    """
    if raw["event_start"].isna().any():
        raise ValueError("Missing chronology")
    first = int(pd.Timestamp(config.history_start, tz="UTC").timestamp())
    stop = int((pd.Timestamp(config.end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    within_window = raw["event_start"].ge(first) & raw["event_start"].lt(stop)
    if not within_window.all():
        prior_columns = [f"{color}_{suffix}" for color in ("white", "black")
                         for suffix in ("rating", "rating_time", "rating_game_id")]
        facts = raw.loc[within_window].drop(columns=prior_columns, errors="ignore").reset_index(drop=True)
        bounded = add_pregame_ratings(facts)
        bounded.attrs = {**raw.attrs, "preparation_window": {
            "history_start": config.history_start, "end_date": config.end_date, "games": len(bounded)}}
        raw = bounded
    target_mask = raw["event_start"] >= int(pd.Timestamp(config.target_start, tz="UTC").timestamp())
    coverage = raw.loc[target_mask].assign(
        missing_prior_rating=raw.loc[target_mask, ["white_rating", "black_rating"]].isna().any(axis=1))
    features = build_feature_table(raw, config.target_start, lookback_days=config.lookback_days,
                                   prior_games=config.prior_games, draw_prior=config.draw_prior)
    table = split_chronologically(features)
    label_map = {label: i for i, label in enumerate(CLASS_NAMES)}
    return DataResult(raw=raw, table=table.assign(y=table["outcome"].map(label_map)), coverage=coverage)


def run_data_pipeline(config: PipelineConfig = PipelineConfig(),
                      client: ChessComClient | None = None) -> DataResult:
    raw = collect_dataset(config.history_start, config.end_date, client=client)
    return prepare_dataset(raw, config)


def run_experiment(config: PipelineConfig = PipelineConfig(),
                   client: ChessComClient | None = None) -> ExperimentResult:
    # Keep ML dependencies out of the collection-only CLI's import path.
    from .evaluation import evaluate_models
    from .modeling import get_gain_importance, train_models, training_class_prior

    data = run_data_pipeline(config, client)
    parts = data.parts
    models = train_models(parts["train"], parts["validation"], seed=config.seed)
    prior = training_class_prior(parts["train"])
    return ExperimentResult(data=data, models=models, prior=prior,
                            evaluation=evaluate_models(models, prior, parts),
                            importance=get_gain_importance(models["historical_features"][0]))
