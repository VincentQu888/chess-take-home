"""
Shared defaults; configuration contains values, not pipeline behavior.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tempfile

API = "https://api.chess.com/pub"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "chess-take-home-api-v1"
DEFAULT_START = "2026-01-01"
DEFAULT_TARGET_START = "2026-02-03"
DEFAULT_END = "2026-03-10"
SEEDS = ("hikaru", "magnuscarlsen")
CLASS_NAMES = ("black_win", "draw", "white_win")
CLASS_IDS = (0, 1, 2)
RATING_FEATURES = ("rating_diff", "mean_rating")
HISTORY_FEATURES = tuple(f"{color}_{name}" for color in ("white", "black")
                         for name in ("form", "draw_rate"))
FEATURES = RATING_FEATURES + ("round",) + HISTORY_FEATURES


@dataclass(frozen=True)
class PipelineConfig:
    """Experiment settings; model hyperparameters stay fixed in modeling.py."""

    history_start: str = DEFAULT_START
    target_start: str = DEFAULT_TARGET_START
    end_date: str = DEFAULT_END
    lookback_days: int = 60
    prior_games: int = 10
    draw_prior: float = 0.15
    seed: int = 42

    def __post_init__(self):
        start, target, end = (date.fromisoformat(value) for value in
                              (self.history_start, self.target_start, self.end_date))
        if not start <= target <= end:
            raise ValueError("Require history_start <= target_start <= end_date")
        if self.lookback_days <= 0 or self.prior_games <= 0:
            raise ValueError("lookback_days and prior_games must be positive")
        if not 0 <= self.draw_prior <= 1:
            raise ValueError("draw_prior must be between zero and one")
