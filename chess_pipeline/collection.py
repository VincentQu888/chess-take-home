"""
Discover verified tournaments and traverse their official round/group records.
"""

from datetime import date, datetime, timezone

import pandas as pd

from .api import ChessComClient
from .config import SEEDS
from .helpers import validate_api_url
from .processing import deduplicate_games, normalize_game


def discover_tournaments(start: date, end: date, client: ChessComClient) -> list[dict]:
    """Seeds discover event URLs only; game collection is not restricted to seeds."""
    urls = set()
    for username in SEEDS:
        for month in pd.period_range(start, end, freq="M"):
            for game in client.player_games(username, month.year, month.month):
                url = game.get("tournament", "")
                if "titled-tuesday" in url:
                    validate_api_url(url)
                    urls.add(url)
    events = []
    for url in sorted(urls):
        event = client.tournament(url)
        played_on = datetime.fromtimestamp(event["start_time"], timezone.utc).date()
        if not start <= played_on <= end:
            continue
        settings = event["settings"]
        if (event["creator"].lower() != "chesscom" or not settings["is_official"]
                or settings["rules"] != "chess" or settings["time_class"] != "blitz"
                or event["status"] != "finished"):
            raise ValueError(f"Unexpected tournament metadata: {url}")
        events.append({**event, "id": url.rsplit("/", 1)[-1], "api_url": url})
    if not events:
        raise ValueError("No verified Titled Tuesdays found in the requested date range")
    return sorted(events, key=lambda event: (event["start_time"], event["id"]))


def extract_round_number(round_url: str, tournament_url: str) -> int:
    """Derive a pre-game round number from its verified API path, never PGN."""
    validate_api_url(round_url)
    prefix = tournament_url + "/"
    if not round_url.startswith(prefix) or not round_url[len(prefix):].isdigit():
        raise ValueError(f"Unexpected round URL: {round_url}")
    number = int(round_url[len(prefix):])
    if number < 1:
        raise ValueError(f"Unexpected round URL: {round_url}")
    return number


def collect_tournament_games(event: dict, client: ChessComClient) -> tuple[list[dict], int]:
    expected = {event["api_url"] + f"/{i}"
                for i in range(1, event["settings"]["total_rounds"] + 1)}
    if len(event["rounds"]) != len(expected) or set(event["rounds"]) != expected:
        raise ValueError(f"Incomplete/duplicate round listing: {event['id']}")
    rows, skipped = [], 0
    for round_url in event["rounds"]:
        number = extract_round_number(round_url, event["api_url"])
        groups = client.round_groups(round_url)
        if not groups:
            raise ValueError(f"No groups in {round_url}")
        for group_url in groups:
            validate_api_url(group_url)
            if not group_url.startswith(round_url + "/"):
                raise ValueError(f"Unexpected group URL: {group_url}")
            for game in client.group_games(group_url):
                row = normalize_game(game, event, number)
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)
    return rows, skipped


def collect_games(start: str, end: str, client: ChessComClient) -> pd.DataFrame:
    """Collect normalized API facts, before prior-rating/feature construction."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("start must not be after end")
    events = discover_tournaments(first, last, client)
    rows, skipped = [], 0
    for event in events:
        games, excluded = collect_tournament_games(event, client)
        rows.extend(games)
        skipped += excluded
        print(f"Collected {event['id']}", flush=True)
    frame = deduplicate_games(rows)
    frame.attrs["audit"] = {
        "source": "Chess.com Published Data API tournament round/group games",
        "discovery_seeds": list(SEEDS), "start": start, "end": end,
        "events": len(events), "rows_before_dedup": len(rows),
        "duplicates_removed": len(rows)-len(frame), "skipped_games": skipped,
        "cache_dir": str(client.cache_dir),
    }
    return frame
