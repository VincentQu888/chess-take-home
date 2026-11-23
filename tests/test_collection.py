"""Offline API/collection boundary tests; run with unittest discovery."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from chess_pipeline.api import ChessComClient
from chess_pipeline.collection import extract_round_number
from chess_pipeline.config import API, SEEDS, PipelineConfig
from chess_pipeline.helpers import utc_timestamp, validate_api_url
from chess_pipeline.pipeline import collect_dataset, run_experiment
from chess_pipeline.processing import deduplicate_games, normalize_game


def example_game():
    timestamp = utc_timestamp("2026.02.03", "16:00:00")
    return {
        "url": "https://www.chess.com/game/live/123", "rated": True,
        "rules": "chess", "time_class": "blitz", "time_control": "300",
        "end_time": timestamp + 300,
        "white": {"username": "Alice", "rating": 2500, "result": "win"},
        "black": {"username": "Bob", "rating": 2400, "result": "resigned"},
        "pgn": '[UTCDate "2026.02.03"]\n[UTCTime "16:00:00"]\n'
               '[Result "1-0"]\n[ECO "C20"]\n\n1. e4 e5 1-0',
    }


def response(status, body):
    value = requests.Response()
    value.status_code = status
    value._content = body.encode()
    value.url = API + "/tournament/titled-tuesday-example"
    return value


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.game = example_game()
        self.event = {"id": "titled-tuesday-example",
                      "start_time": utc_timestamp("2026.02.03", "16:00:00")}

    def test_normalization_keeps_only_explicit_post_ratings(self):
        original = deepcopy(self.game)
        row = normalize_game(self.game, self.event, 1)
        self.assertEqual(row["outcome"], "white_win")
        self.assertEqual(row["white"], "alice")
        self.assertEqual(row["white_post_rating"], 2500)
        self.assertEqual(row["round"], 1)
        self.assertEqual(row["start_time"], self.event["start_time"])
        self.assertNotIn("white_rating", row)
        self.assertNotIn("eco_family", row)
        self.assertEqual(original, self.game)
        no_eco = {**self.game, "pgn": self.game["pgn"].replace('[ECO "C20"]', '')}
        self.assertEqual(normalize_game(no_eco, self.event, 1), row)

    def test_filters_and_result_mapping(self):
        for changed in [{"rules": "chess960"}, {"rated": False},
                        {"pgn": self.game["pgn"].replace('"1-0"', '"*"')}]:
            self.assertIsNone(normalize_game({**self.game, **changed}, self.event, 1))
        for result, white, black, label in [
            ("0-1", "timeout", "win", "black_win"),
            ("1/2-1/2", "repetition", "repetition", "draw"),
            ("1/2-1/2", "timevsinsufficient", "timevsinsufficient", "draw"),
        ]:
            game = {**self.game, "pgn": self.game["pgn"].replace('"1-0"', f'"{result}"'),
                    "white": {**self.game["white"], "result": white},
                    "black": {**self.game["black"], "result": black}}
            self.assertEqual(normalize_game(game, self.event, 1)["outcome"], label)

    def test_rejects_invalid_metadata_and_conflicting_duplicates(self):
        for changed in [{"end_time": 0}, {"pgn": '[Result "1-0"]'},
                        {"white": {**self.game["white"], "rating": -1}},
                        {"white": {**self.game["white"], "result": "resigned"}}]:
            with self.assertRaises(ValueError):
                normalize_game({**self.game, **changed}, self.event, 1)
        row = normalize_game(self.game, self.event, 1)
        self.assertEqual(len(deduplicate_games([row, dict(row)])), 1)
        with self.assertRaises(ValueError):
            deduplicate_games([row, {**row, "round": 2}])
        with self.assertRaises(ValueError):
            deduplicate_games([])

    def test_round_and_url_validation(self):
        event_url = API + "/tournament/titled-tuesday-example"
        self.assertEqual(extract_round_number(event_url + "/11", event_url), 11)
        for url in [event_url + "/0", event_url + "/x", event_url + "/1/2",
                    API + "/tournament/other/1"]:
            with self.assertRaises(ValueError):
                extract_round_number(url, event_url)
        for url in ["http://api.chess.com/pub/player/a", "https://example.com/pub/a"]:
            with self.assertRaises(ValueError):
                validate_api_url(url)
        with self.assertRaises(ValueError):
            PipelineConfig(history_start="2026-04-01")

    def test_mocked_collection_deduplicates_and_keeps_missing_prior_missing(self):
        url = API + "/tournament/titled-tuesday-example"
        metadata = {**self.event, "creator": "CHESScom", "status": "finished",
                    "settings": {"is_official": True, "rules": "chess", "time_class": "blitz",
                                 "total_rounds": 1}, "rounds": [url + "/1"]}
        sources = {**{f"{API}/player/{seed}/games/2026/02": {"games": [{"tournament": url}]}
                      for seed in SEEDS},
                   url: metadata, url + "/1": {"groups": [url + "/1/1"]},
                   url + "/1/1": {"games": [self.game, dict(self.game), {**self.game, "rated": False}]}}
        with patch("builtins.print"):
            frame = collect_dataset("2026-02-03", "2026-02-03", fetch=sources.__getitem__)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.attrs["audit"]["duplicates_removed"], 1)
        self.assertEqual(frame.attrs["audit"]["skipped_games"], 1)
        self.assertTrue(pd.isna(frame.iloc[0].white_rating))
        for start, end in [("2026-02-04", "2026-02-03"), ("2026-02-01", "2026-02-02")]:
            with self.assertRaises(ValueError):
                collect_dataset(start, end, fetch=sources.__getitem__)

    def test_complete_experiment_with_injected_api(self):
        sources = {}
        archives = {month: [] for month in (1, 2, 3)}
        first = utc_timestamp("2026.01.20", "16:00:00")
        for i in range(8):  # Two warm-up dates, then six target dates.
            started = first + i*7*86400
            stamp = datetime.fromtimestamp(started, timezone.utc)
            url = API + f"/tournament/titled-tuesday-fixture-{i}"
            archives[stamp.month].append({"tournament": url})
            outcome = i % 3
            result = ("1-0", "1/2-1/2", "0-1")[outcome]
            white_result = ("win", "agreed", "resigned")[outcome]
            black_result = ("resigned", "agreed", "win")[outcome]
            base = example_game()
            game = {**base, "url": f"https://www.chess.com/game/live/{i}",
                    "end_time": started+300,
                    "white": {**base["white"], "result": white_result},
                    "black": {**base["black"], "result": black_result},
                    "pgn": f'[UTCDate "{stamp:%Y.%m.%d}"]\n[UTCTime "16:00:00"]\n[Result "{result}"]'}
            sources[url] = {"creator": "CHESScom", "status": "finished", "start_time": started,
                            "settings": {"is_official": True, "rules": "chess", "time_class": "blitz",
                                         "total_rounds": 1}, "rounds": [url+"/1"]}
            sources[url+"/1"] = {"groups": [url+"/1/1"]}
            sources[url+"/1/1"] = {"games": [game]}
        for seed in SEEDS:
            for month, games in archives.items():
                sources[f"{API}/player/{seed}/games/2026/{month:02d}"] = {"games": games}
        client = ChessComClient(fetch=sources.__getitem__)
        with patch("requests.get", side_effect=AssertionError("Unexpected network")), patch("builtins.print"):
            result = run_experiment(client=client)
        self.assertEqual(len(result.data.raw), 8)
        self.assertEqual(len(result.data.table), 6)
        self.assertEqual(len(result.evaluation.accuracy_comparison), 8)
        self.assertEqual(len(result.importance), 7)

    def test_cache_retries_and_error_responses(self):
        url = API + "/tournament/titled-tuesday-example"
        with tempfile.TemporaryDirectory(prefix="chess-api-test-") as directory:
            client = ChessComClient(cache_dir=Path(directory))
            with patch("requests.get", return_value=response(200, '{"games": []}')) as get:
                with patch("time.sleep"):
                    self.assertEqual(client.get_json(url), {"games": []})
                    self.assertEqual(client.get_json(url), {"games": []})
                self.assertEqual(get.call_count, 1)
            for status, exception in [(404, RuntimeError), (429, RuntimeError), (302, ValueError)]:
                with patch("requests.get", return_value=response(status, 'error')) as get:
                    with patch("time.sleep"), self.assertRaises(exception):
                        client.get_json(url + f"/{status}")
                    self.assertEqual(get.call_count, 4 if status == 429 else 1)
            with patch("requests.get", side_effect=[response(200, 'not json'), response(200, '{"ok": true}')]):
                with patch("time.sleep"):
                    self.assertEqual(client.get_json(url + "/retry"), {"ok": True})


if __name__ == "__main__":
    unittest.main()
