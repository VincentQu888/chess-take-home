"""
Pure, leakage-safe feature construction; this module performs no I/O.
"""

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, DEFAULT_TARGET_START, FEATURES


def add_pregame_ratings(raw: pd.DataFrame) -> pd.DataFrame:
    """Latest collected post-game rating completed STRICTLY before each game.

    Pool colors per player, retain source ID/time, and leave absent history
    missing. Uncollected intervening games can make the proxy stale. Equal
    completion timestamps are broken deterministically by source game ID.
    """
    if raw.empty or raw["game_id"].duplicated().any():
        raise ValueError("Expected nonempty, deduplicated games")
    if (raw["start_time"].isna().any() or raw["end_time"].isna().any()
            or (raw["end_time"] < raw["start_time"]).any()):
        raise ValueError("Invalid game chronology")
    players = pd.concat([
        raw[["game_id", "start_time", "end_time", color, f"{color}_post_rating"]]
        .rename(columns={color: "player", f"{color}_post_rating": "post_rating"})
        .assign(color=color)
        for color in ("white", "black")
    ], ignore_index=True)
    queries = players[["game_id", "start_time", "player", "color"]]
    sources = players[["player", "game_id", "end_time", "post_rating"]].rename(
        columns={"game_id": "rating_game_id", "end_time": "rating_time", "post_rating": "rating"})
    matched = pd.merge_asof(
        queries.sort_values(["start_time", "game_id"]),
        sources.sort_values(["rating_time", "rating_game_id"]),
        left_on="start_time", right_on="rating_time", by="player",
        direction="backward", allow_exact_matches=False,
    )
    result = raw.copy()
    for color in ("white", "black"):
        prior = matched.loc[matched["color"].eq(color)].set_index("game_id")
        result = result.join(prior[["rating", "rating_time", "rating_game_id"]]
                             .add_prefix(color + "_"), on="game_id")
        known = result[f"{color}_rating"].notna()
        assert (result.loc[known, f"{color}_rating_time"] < result.loc[known, "start_time"]).all()
        assert result.loc[known, f"{color}_rating_game_id"].ne(result.loc[known, "game_id"]).all()
    return result


def add_rating_features(games: pd.DataFrame) -> pd.DataFrame:
    """Use resolved prior ratings, never the API's post-game columns."""
    return games.assign(rating_diff=games["white_rating"]-games["black_rating"],
                        mean_rating=(games["white_rating"]+games["black_rating"])/2)


def build_player_history(raw: pd.DataFrame) -> pd.DataFrame:
    """One observation per player/game; form is NaN without both prior ratings."""
    score = raw["outcome"].map({"white_win": 1.0, "draw": 0.5, "black_win": 0.0})
    difference = raw["white_rating"] - raw["black_rating"]
    white_expected = 1 / (1 + 10.0 ** (-difference / 400))
    columns = ["player", "event_start", "end_time", "residual", "is_draw"]
    return pd.concat([raw.assign(
        player=raw[color],
        residual=(score-white_expected) * (1 if color == "white" else -1),
        is_draw=raw["outcome"].eq("draw").astype(float)
    )[columns] for color in ("white", "black")], ignore_index=True)


def summarize_player_history(history: pd.DataFrame, event_start: int, *,
                             lookback_days: int = 60, prior_games: int = 10,
                             draw_prior: float = 0.15) -> pd.DataFrame:
    """Freeze all historical features before the event, even for later rounds."""
    available = history.loc[(history["event_start"] < event_start)
                            & (history["end_time"] < event_start)
                            & (history["end_time"] >= event_start-lookback_days*86400)]
    summary = available.groupby("player").agg(
        residual_sum=("residual", "sum"), draws=("is_draw", "sum"),
        history_count=("residual", "size"), form_count=("residual", "count"),
        history_end=("end_time", "max"))
    return summary.assign(
        form=summary["residual_sum"] / (summary["form_count"] + prior_games),
        draw_rate=(summary["draws"] + prior_games*draw_prior)
                  / (summary["history_count"] + prior_games)
    ).drop(columns=["residual_sum", "draws"])


def build_feature_table(raw: pd.DataFrame, target_start: str = DEFAULT_TARGET_START, *,
                        lookback_days: int = 60, prior_games: int = 10,
                        draw_prior: float = 0.15) -> pd.DataFrame:
    """Eligible target games with seven features and separate provenance columns."""
    if raw.empty or raw["game_id"].duplicated().any():
        raise ValueError("Expected nonempty, deduplicated raw games")
    if not raw["outcome"].isin(CLASS_NAMES).all():
        raise ValueError("Unexpected outcome label")
    if raw["event_start"].isna().any() or raw["end_time"].isna().any():
        raise ValueError("Missing chronology")
    history = build_player_history(raw)
    cutoff = int(pd.Timestamp(target_start, tz="UTC").timestamp())
    has_prior = raw[["white_rating", "black_rating"]].notna().all(axis=1)
    targets = raw.loc[(raw["event_start"] >= cutoff) & has_prior]
    if targets.empty:
        raise ValueError("No target games with both prior ratings on/after TARGET_START")
    tables = []
    for (_, event_start), games in targets.groupby(["event_id", "event_start"], sort=True):
        summary = summarize_player_history(history, event_start, lookback_days=lookback_days,
                                           prior_games=prior_games, draw_prior=draw_prior)
        enriched = add_rating_features(games)
        for color in ("white", "black"):
            defaults = {"form": 0.0, "draw_rate": draw_prior,
                        "history_count": 0, "form_count": 0, "history_end": 0}
            joined = games[[color]].join(summary, on=color).drop(columns=color)
            enriched = enriched.join(joined.fillna(defaults).add_prefix(color + "_"))
        tables.append(enriched)
    result = pd.concat(tables).sort_values(["event_start", "round", "game_id"]).reset_index(drop=True)
    assert np.isfinite(result[list(FEATURES)].to_numpy()).all()
    for color in ("white", "black"):
        assert (result[f"{color}_history_end"] < result["event_start"]).all()
        assert (result[f"{color}_rating_time"] < result["start_time"]).all()
        assert result[f"{color}_rating_game_id"].ne(result["game_id"]).all()
    return result
