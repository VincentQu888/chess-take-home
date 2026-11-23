"""Synthetic regression tests for time availability, provenance and splits."""

import unittest

import numpy as np
import pandas as pd

from chess_pipeline.config import FEATURES, PipelineConfig
from chess_pipeline.features import add_pregame_ratings, build_feature_table
from chess_pipeline.pipeline import prepare_dataset
from chess_pipeline.splitting import split_chronologically


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.start = int(pd.Timestamp("2026-02-03", tz="UTC").timestamp())
        week, start = 7*86400, self.start
        base = dict(game_id="history", game_url="history", event_id="warmup",
                    event_start=start-week, start_time=start-week+1, end_time=start-week+300,
                    white="alice", black="bob", white_post_rating=2500, black_post_rating=2400,
                    round=1, time_control="300", outcome="white_win")
        seed = dict(base, game_id="seed", game_url="seed", event_id="seed_event",
                    event_start=start-2*week, start_time=start-2*week+1,
                    end_time=start-2*week+300, white_post_rating=2490, black_post_rating=2410)
        self.examples = [seed, base] + [dict(
            base, game_id=f"g{i}", game_url=f"g{i}", event_id=f"e{i}",
            event_start=start+i*week, start_time=start+i*week+1, end_time=start+i*week+300,
            white_post_rating=2700+i, black_post_rating=2200-i,
            outcome=["white_win", "draw", "black_win"][i % 3]) for i in range(6)]
        self.observations = pd.DataFrame(self.examples)
        self.raw = add_pregame_ratings(self.observations)
        self.built = build_feature_table(self.raw, "2026-02-03")

    def test_feature_values_and_input_immutability(self):
        original = self.raw.copy(deep=True)
        built = build_feature_table(self.raw, "2026-02-03")
        pd.testing.assert_frame_equal(self.raw, original)
        first = built.iloc[0]
        self.assertEqual(first.rating_diff, 100)
        self.assertEqual(first.mean_rating, 2450)
        self.assertEqual(first.white_rating, 2500)
        self.assertEqual(first.white_post_rating, 2700)
        self.assertEqual(first.white_history_count, 2)
        self.assertEqual(first.white_form_count, 1)
        self.assertAlmostEqual(first.white_draw_rate, 1.5/12)
        self.assertAlmostEqual(first.white_form, (1-1/(1+10**(-80/400)))/11)
        self.assertLess(first.white_history_end, first.event_start)
        self.assertEqual(len(FEATURES), 7)
        self.assertFalse(any("opening" in name or "eco" in name for name in built.columns))
        self.assertFalse(any("post_rating" in name or "rating_time" in name for name in FEATURES))

    def test_future_labels_and_post_ratings_do_not_change_available_features(self):
        future = self.observations.event_start >= self.start
        poisoned = self.observations.assign(
            outcome=self.observations.outcome.where(~future, "black_win"),
            white_post_rating=self.observations.white_post_rating.where(~future, 9999),
            black_post_rating=self.observations.black_post_rating.where(~future, 1))
        rebuilt = build_feature_table(add_pregame_ratings(poisoned), "2026-02-03")
        pd.testing.assert_frame_equal(self.built.iloc[:1][list(FEATURES)], rebuilt.iloc[:1][list(FEATURES)])

    def test_overlapping_stale_and_missing_histories(self):
        unfinished = self.observations.assign(end_time=np.where(
            self.observations.game_id == "history", self.start+100, self.observations.end_time))
        first = build_feature_table(add_pregame_ratings(unfinished), "2026-02-03").iloc[0]
        self.assertEqual(first.white_history_count, 1)
        self.assertEqual(first.white_form, 0)
        self.assertEqual(first.white_rating, 2490)
        self.assertAlmostEqual(first.white_draw_rate, 1.5/11)
        warmup = self.observations.event_start < self.start
        stale = self.observations.assign(**{name: self.observations[name].where(
            ~warmup, self.observations[name]-61*86400)
            for name in ("event_start", "start_time", "end_time")})
        self.assertEqual(build_feature_table(add_pregame_ratings(stale), "2026-02-03")
                         .iloc[0].white_history_count, 0)
        eligible = build_feature_table(add_pregame_ratings(self.observations.loc[~warmup]), "2026-02-03")
        self.assertNotIn("g0", set(eligible.game_id))
        self.assertEqual(len(eligible), 5)

    def test_same_event_history_freezes_but_prior_round_rating_is_available(self):
        later = dict(self.examples[2], game_id="later", game_url="later", round=2,
                     start_time=self.start+1000, end_time=self.start+1300)
        raw = add_pregame_ratings(pd.concat([self.observations, pd.DataFrame([later])], ignore_index=True))
        built = build_feature_table(raw, "2026-02-03")
        self.assertEqual(built.loc[built.event_id.eq("e0"), "white_history_count"].tolist(), [2, 2])
        self.assertEqual(built.loc[built.game_id.eq("later"), "white_rating"].iloc[0], 2700)

    def test_color_switch_exact_timestamps_and_missing_rating(self):
        samples = pd.DataFrame([
            dict(game_id="first", start_time=100, end_time=200, white="alice", black="bob",
                 white_post_rating=2500, black_post_rating=2400),
            dict(game_id="second", start_time=300, end_time=400, white="bob", black="alice",
                 white_post_rating=2390, black_post_rating=2510),
            dict(game_id="third", start_time=500, end_time=600, white="carol", black="alice",
                 white_post_rating=2300, black_post_rating=2520)])
        original = samples.copy(deep=True)
        rated = add_pregame_ratings(samples).set_index("game_id")
        pd.testing.assert_frame_equal(samples, original)
        self.assertTrue(pd.isna(rated.loc["first", "white_rating"]))
        self.assertEqual(rated.loc["second", "white_rating"], 2400)
        self.assertEqual(rated.loc["second", "black_rating"], 2500)
        self.assertEqual(rated.loc["third", "black_rating"], 2510)
        self.assertEqual(rated.loc["third", "black_rating_game_id"], "second")
        self.assertTrue(pd.isna(rated.loc["third", "white_rating"]))
        for start in (150, 200):
            changed = samples.iloc[:2].assign(start_time=[100, start])
            self.assertTrue(pd.isna(add_pregame_ratings(changed).iloc[1].white_rating))
        shuffled = add_pregame_ratings(samples.sample(frac=1, random_state=7)).set_index("game_id")
        pd.testing.assert_frame_equal(rated.sort_index(), shuffled.sort_index())

    def test_preparation_and_date_splits(self):
        data = prepare_dataset(self.raw, PipelineConfig())
        self.assertEqual(len(data.table), 6)
        self.assertEqual(set(data.parts), {"train", "validation", "test"})
        self.assertEqual(data.table.groupby("event_id")["split"].nunique().max(), 1)
        self.assertLess(data.parts["train"].event_start.max(), data.parts["validation"].event_start.min())
        self.assertLess(data.parts["validation"].event_start.max(), data.parts["test"].event_start.min())
        self.assertFalse(data.coverage.missing_prior_rating.any())
        with self.assertRaises(ValueError):
            split_chronologically(self.built.iloc[:2])

    def test_preparation_respects_inclusive_end_date(self):
        config = PipelineConfig(end_date="2026-03-03")
        data = prepare_dataset(self.raw, config)
        self.assertEqual(len(data.table), 5)
        self.assertEqual(len(data.coverage), 5)
        self.assertNotIn("g5", set(data.raw.game_id))
        boundary = int(pd.Timestamp("2026-03-04", tz="UTC").timestamp())
        self.assertTrue(data.raw.event_start.lt(boundary).all())
        self.assertIn("g4", set(data.table.game_id))
        # The event's date determines inclusion, even for a game after midnight.
        changed = self.raw.assign(
            event_start=self.raw.event_start.where(self.raw.game_id.ne("g4"), boundary-1),
            start_time=self.raw.start_time.where(self.raw.game_id.ne("g4"), boundary),
            end_time=self.raw.end_time.where(self.raw.game_id.ne("g4"), boundary+300))
        self.assertIn("g4", set(prepare_dataset(changed, config).table.game_id))

    def test_preparation_recomputes_ratings_after_removing_earlier_games(self):
        original = self.raw.copy(deep=True)
        config = PipelineConfig(history_start="2026-02-03")
        data = prepare_dataset(self.raw, config)
        self.assertNotIn("seed", set(data.raw.game_id))
        self.assertNotIn("history", set(data.raw.game_id))
        self.assertNotIn("g0", set(data.table.game_id))
        self.assertTrue(pd.isna(data.raw.loc[data.raw.game_id.eq("g0"), "white_rating"].iloc[0]))
        self.assertEqual(len(data.table), 5)
        self.assertEqual(data.coverage.missing_prior_rating.sum(), 1)
        pd.testing.assert_frame_equal(self.raw, original)

    def test_invalid_feature_inputs(self):
        for invalid in [self.raw.iloc[:0], pd.concat([self.raw, self.raw.iloc[:1]]),
                        self.raw.assign(outcome="unknown"), self.raw.assign(event_start=np.nan)]:
            with self.assertRaises(ValueError):
                build_feature_table(invalid)
        with self.assertRaises(ValueError):
            build_feature_table(self.raw, "2030-01-01")
        with self.assertRaises(ValueError):
            add_pregame_ratings(self.observations.assign(end_time=0))


if __name__ == "__main__":
    unittest.main()
