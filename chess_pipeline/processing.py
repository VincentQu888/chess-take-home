"""
Normalize API facts and reject conflicting data; no engineered features.
"""

from io import StringIO

import chess.pgn
import pandas as pd

from .helpers import utc_timestamp

RESULTS = {"1-0": "white_win", "0-1": "black_win", "1/2-1/2": "draw"}
DRAW_RESULTS = {"agreed", "repetition", "stalemate", "insufficient", "50move", "timevsinsufficient"}


def normalize_game(game: dict, event: dict, round_number: int) -> dict | None:
    """Keep explicitly POST-game ratings, chronology and outcomes; discard moves."""
    if (game.get("rules") != "chess" or game.get("time_class") != "blitz"
            or not game.get("rated")):
        return None
    headers = chess.pgn.read_headers(StringIO(game["pgn"]))
    if headers is None:
        raise ValueError(f"Missing PGN headers: {game.get('url')}")
    result = headers.get("Result")
    if result == "*":
        return None
    if result not in RESULTS:
        raise ValueError(f"Unknown PGN result {result!r}: {game.get('url')}")
    try:
        start_time = utc_timestamp(headers["UTCDate"], headers["UTCTime"])
        ratings = [int(game[color]["rating"]) for color in ("white", "black")]
        end_time = int(game["end_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp/rating: {game.get('url')}") from exc
    if (min(ratings) <= 0 or end_time < start_time
            or start_time < event["start_time"] or round_number < 1):
        raise ValueError(f"Inconsistent game metadata: {game.get('url')}")
    white_result, black_result = game["white"]["result"], game["black"]["result"]
    consistent = (
        (result == "1-0" and white_result == "win" and black_result != "win")
        or (result == "0-1" and black_result == "win" and white_result != "win")
        or (result == "1/2-1/2" and white_result in DRAW_RESULTS and black_result in DRAW_RESULTS)
    )
    if not consistent:
        raise ValueError(f"PGN/API result mismatch: {game['url']}")
    return {
        "game_id": game.get("uuid") or game["url"].rstrip("/").rsplit("/", 1)[-1],
        "game_url": game["url"], "event_id": event["id"],
        "event_start": int(event["start_time"]), "round": round_number,
        "start_time": start_time, "end_time": end_time,
        "white": game["white"]["username"].lower(),
        "black": game["black"]["username"].lower(),
        "white_post_rating": ratings[0], "black_post_rating": ratings[1],
        "time_control": game["time_control"], "outcome": RESULTS[result],
    }


def deduplicate_games(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No eligible completed games collected")
    distinct = frame.drop_duplicates()
    if distinct["game_id"].duplicated().any() or distinct["game_url"].duplicated().any():
        raise ValueError("Conflicting copies of a game; inspect source data")
    return distinct.sort_values(["event_start", "round", "game_id"]).reset_index(drop=True)
