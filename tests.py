"""
Unit tests for the Betpawa Tracker project.

Tests cover:
  - Result string parsing (scraper.parse_result)
  - DataManager CRUD operations (in-memory SQLite)
  - AI model feature engineering and basic GA/QL mechanics
  - CLI argument parsing
"""

import sys
import os
import math
import random
import tempfile
import unittest
import pickle

# Add the project root to sys.path so imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from scraper import parse_result, _parse_row, parse_matchday_text
from data_manager import DataManager
from ai_model import (
    _build_features,
    _feature_size,
    GeneticAlgorithm,
    QLearning,
    BetpawaAI,
)
import cli as cli_module


# ── Result parsing ─────────────────────────────────────────────────────────────

class TestParseResult(unittest.TestCase):

    def test_standard_format(self):
        r = parse_result("(1 - 0)2 - 1")
        self.assertEqual(r, (1, 0, 2, 1))

    def test_zero_zero(self):
        r = parse_result("(0 - 0)0 - 0")
        self.assertEqual(r, (0, 0, 0, 0))

    def test_spaces_variants(self):
        r = parse_result("( 2 - 1 ) 3 - 2")
        self.assertEqual(r, (2, 1, 3, 2))

    def test_embedded_in_text(self):
        r = parse_result("MUN vs ARS  (0 - 0)0 - 3  FT")
        self.assertEqual(r, (0, 0, 0, 3))

    def test_no_match_returns_none(self):
        self.assertIsNone(parse_result("no score here"))

    def test_partial_returns_none(self):
        self.assertIsNone(parse_result("MUN ARS"))

    def test_large_scores(self):
        r = parse_result("(10 - 9)12 - 11")
        self.assertEqual(r, (10, 9, 12, 11))


class TestParseRow(unittest.TestCase):

    def test_mun_home(self):
        row = "MUN ARS (1 - 0)2 - 0"
        result = _parse_row(row)
        self.assertIsNotNone(result)
        self.assertIn("MUN", (result["team_home"], result["team_away"]))
        self.assertEqual(result["ft_home"], 2)
        self.assertEqual(result["ft_away"], 0)

    def test_row_without_mun_returns_none(self):
        row = "ARS CHE (0 - 1)1 - 2"
        self.assertIsNone(_parse_row(row))

    def test_row_without_score_returns_none(self):
        self.assertIsNone(_parse_row("MUN vs ARS"))


class TestParseMatchdayText(unittest.TestCase):

    def test_extracts_mun_fixture_from_matchday_page(self):
        text = """
        Back
        MATCH DAY: 01
        English League
        ARS - WHU
        (2 - 0)2 - 1
        LEE - MUN
        (1 - 0)1 - 0
        TOT - BRE
        (0 - 0)0 - 1
        """
        matches = parse_matchday_text(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["team_home"], "LEE")
        self.assertEqual(matches[0]["team_away"], "MUN")
        self.assertEqual(matches[0]["ft_home"], 1)
        self.assertEqual(matches[0]["ft_away"], 0)

    def test_returns_empty_when_team_not_present(self):
        text = "ARS - WHU (2 - 0)2 - 1 TOT - BRE (0 - 0)0 - 1"
        self.assertEqual(parse_matchday_text(text), [])


# ── DataManager ───────────────────────────────────────────────────────────────

class TestDataManager(unittest.TestCase):

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.db = DataManager(db_path=self._tmpfile.name)

    def tearDown(self):
        # Close any lingering SQLite connections before deleting on Windows
        del self.db
        try:
            os.unlink(self._tmpfile.name)
        except PermissionError:
            pass  # Windows file-locking; temp dir will clean up later

    def _insert_win(self):
        return self.db.insert_result(
            team_home="MUN", team_away="ARS",
            ht_home=1, ht_away=0,
            ft_home=2, ft_away=0,
            raw_result="(1 - 0)2 - 0",
        )

    def test_insert_and_count(self):
        self.assertEqual(self.db.count_results(), 0)
        row_id = self._insert_win()
        self.assertIsNotNone(row_id)
        self.assertEqual(self.db.count_results(), 1)

    def test_deduplication(self):
        id1 = self._insert_win()
        id2 = self._insert_win()  # same raw_result within 5 min
        self.assertIsNotNone(id1)
        self.assertIsNone(id2)
        self.assertEqual(self.db.count_results(), 1)

    def test_insert_without_deduplication(self):
        id1 = self._insert_win()
        id2 = self.db.insert_result(
            team_home="MUN", team_away="ARS",
            ht_home=1, ht_away=0,
            ft_home=2, ft_away=0,
            raw_result="(1 - 0)2 - 0",
            deduplicate=False,
        )
        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertEqual(self.db.count_results(), 2)

    def test_result_source_is_stored(self):
        self.db.insert_result(
            team_home="MUN", team_away="CHE",
            ht_home=0, ht_away=0,
            ft_home=1, ft_away=0,
            raw_result="(0 - 0)1 - 0",
            source="historical",
            deduplicate=False,
        )
        row = self.db.get_recent_results(1)[0]
        self.assertEqual(row["source"], "historical")

    def test_count_results_by_source(self):
        self.db.insert_result(
            team_home="MUN", team_away="ARS",
            ht_home=1, ht_away=0,
            ft_home=2, ft_away=0,
            raw_result="(1 - 0)2 - 0",
            source="live",
        )
        self.db.insert_result(
            team_home="MUN", team_away="CHE",
            ht_home=0, ht_away=1,
            ft_home=0, ft_away=2,
            raw_result="(0 - 1)0 - 2",
            source="historical",
            deduplicate=False,
        )
        counts = self.db.count_results_by_source()
        self.assertEqual(counts["live"], 1)
        self.assertEqual(counts["historical"], 1)

    def test_outcome_computed_correctly_win(self):
        self.db.insert_result(
            team_home="MUN", team_away="ARS",
            ht_home=1, ht_away=0,
            ft_home=3, ft_away=1,
            raw_result="(1 - 0)3 - 1",
        )
        outcomes = self.db.get_outcomes()
        self.assertEqual(outcomes, [config.RESULT_WIN])

    def test_outcome_draw(self):
        self.db.insert_result(
            team_home="ARS", team_away="MUN",
            ht_home=0, ht_away=0,
            ft_home=1, ft_away=1,
            raw_result="(0 - 0)1 - 1",
        )
        outcomes = self.db.get_outcomes()
        self.assertEqual(outcomes, [config.RESULT_DRAW])

    def test_outcome_loss(self):
        self.db.insert_result(
            team_home="MUN", team_away="CHE",
            ht_home=0, ht_away=1,
            ft_home=0, ft_away=2,
            raw_result="(0 - 1)0 - 2",
        )
        outcomes = self.db.get_outcomes()
        self.assertEqual(outcomes, [config.RESULT_LOSS])

    def test_clear_all_results(self):
        self._insert_win()
        # Insert a different match so it's not deduplicated
        self.db.insert_result(
            team_home="MUN", team_away="CHE",
            ht_home=0, ht_away=1,
            ft_home=0, ft_away=2,
            raw_result="(0 - 1)0 - 2",
        )
        self.assertEqual(self.db.count_results(), 2)
        
        self.db.clear_all_results()
        self.assertEqual(self.db.count_results(), 0)

    def test_goals_series(self):
        self.db.insert_result("MUN", "ARS", 1, 0, 3, 1, "(1 - 0)3 - 1")
        series = self.db.get_goals_series()
        self.assertEqual(series, [(3, 1)])

    def test_get_recent_results_order(self):
        self.db.insert_result("MUN", "A", 0, 0, 1, 0, "(0 - 0)1 - 0")
        self.db.insert_result("MUN", "B", 0, 0, 2, 0, "(0 - 0)2 - 0")
        rows = self.db.get_recent_results(10)
        # Should be oldest → newest
        self.assertEqual(rows[0]["team_away"], "A")
        self.assertEqual(rows[1]["team_away"], "B")

    def test_log_scrape(self):
        self.db.log_scrape("INFO", "test message")
        log = self.db.get_scrape_log(10)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["message"], "test message")


# ── AI model ──────────────────────────────────────────────────────────────────

class TestFeatureEngineering(unittest.TestCase):

    def test_feature_size(self):
        self.assertEqual(_feature_size(5), 24)   # 5*4 + 4
        self.assertEqual(_feature_size(3), 16)   # 3*4 + 4

    def test_build_features_shape(self):
        w = 5
        outcomes = [config.RESULT_WIN] * 10
        goals = [(2, 1)] * 10
        feats = _build_features(outcomes, goals, w)
        self.assertEqual(feats.shape, (_feature_size(w),))

    def test_build_features_padding(self):
        # Even with fewer outcomes than window, should not raise
        w = 5
        feats = _build_features([config.RESULT_WIN], [(1, 0)], w)
        self.assertEqual(feats.shape, (_feature_size(w),))

    def test_win_rate_all_wins(self):
        w = 5
        outcomes = [config.RESULT_WIN] * w
        goals = [(1, 0)] * w
        feats = _build_features(outcomes, goals, w)
        # win_rate is after the one-hot outcomes and goal series
        win_rate_idx = w * 4 + 2
        self.assertAlmostEqual(feats[win_rate_idx], 1.0)

    def test_build_features_includes_odds_flag(self):
        feats = _build_features(
            [config.RESULT_WIN],
            [(2, 1)],
            window=1,
            odds=(2.5, 3.0, 2.9),
            include_odds=True,
        )
        self.assertEqual(feats.shape, (_feature_size(1, include_odds=True),))
        self.assertEqual(feats[-4], 1.0)

    def test_build_features_missing_odds_flag(self):
        feats = _build_features(
            [config.RESULT_WIN],
            [(2, 1)],
            window=1,
            odds=None,
            include_odds=True,
        )
        self.assertEqual(feats[-4], 0.0)


class TestGeneticAlgorithm(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        self.ga = GeneticAlgorithm(
            population_size=10,
            generations=5,
            feature_window=3,
        )
        # Generate synthetic outcomes and goals
        self.outcomes = [random.choice([0, 1, 2]) for _ in range(20)]
        self.goals = [(random.randint(0, 4), random.randint(0, 4)) for _ in range(20)]

    def test_population_size(self):
        self.assertEqual(len(self.ga.population), 10)

    def test_gene_length(self):
        n_feats = _feature_size(3)
        expected = 3 * (n_feats + 1)  # N_ACTIONS * (n_features + 1)
        self.assertEqual(self.ga.gene_length, expected)

    def test_train_increases_generation(self):
        self.ga.train(self.outcomes, self.goals, generations=3)
        self.assertEqual(self.ga.generation, 3)

    def test_best_fitness_in_range(self):
        self.ga.train(self.outcomes, self.goals, generations=5)
        self.assertGreaterEqual(self.ga.best_fitness, 0.0)
        self.assertLessEqual(self.ga.best_fitness, 1.0)

    def test_predict_returns_valid_action(self):
        self.ga.train(self.outcomes, self.goals, generations=3)
        pred = self.ga.predict_next(self.outcomes, self.goals)
        self.assertIn(pred, [config.RESULT_LOSS, config.RESULT_DRAW, config.RESULT_WIN])

    def test_fitness_history_length(self):
        n_gens = 4
        self.ga.train(self.outcomes, self.goals, generations=n_gens)
        self.assertEqual(len(self.ga.fitness_history), n_gens)


class TestQLearning(unittest.TestCase):

    def setUp(self):
        random.seed(42)
        self.ql = QLearning(state_window=3)
        self.outcomes = [random.choice([0, 1, 2]) for _ in range(30)]

    def test_train_increments_steps(self):
        self.ql.train(self.outcomes)
        # Should have trained on outcomes[3..29] = 27 steps
        self.assertEqual(self.ql.total_steps, len(self.outcomes) - 3)

    def test_q_table_populated(self):
        self.ql.train(self.outcomes)
        self.assertGreater(len(self.ql.q_table), 0)

    def test_accuracy_in_range(self):
        self.ql.train(self.outcomes)
        self.assertGreaterEqual(self.ql.accuracy, 0.0)
        self.assertLessEqual(self.ql.accuracy, 1.0)

    def test_epsilon_decreases(self):
        initial_eps = self.ql.epsilon
        self.ql.train(self.outcomes)
        self.assertLess(self.ql.epsilon, initial_eps)

    def test_predict_returns_valid_action(self):
        self.ql.train(self.outcomes)
        pred = self.ql.predict_next(self.outcomes)
        self.assertIn(pred, [config.RESULT_LOSS, config.RESULT_DRAW, config.RESULT_WIN])


class TestBetpawaAI(unittest.TestCase):

    def setUp(self):
        random.seed(0)
        self.ai = BetpawaAI()
        self.outcomes = [random.choice([0, 1, 2]) for _ in range(15)]
        self.goals = [(random.randint(0, 3), random.randint(0, 3)) for _ in range(15)]

    def test_train_and_predict(self):
        self.ai.train(self.outcomes, self.goals, ga_generations=5)
        result = self.ai.predict(self.outcomes, self.goals)
        self.assertIn("ga_prediction", result)
        self.assertIn("ql_prediction", result)
        self.assertIn("consensus", result)
        self.assertIn(result["consensus"], [0, 1, 2])

    def test_summary_keys(self):
        summary = self.ai.summary()
        for key in ("ga_generation", "ga_best_fitness", "ql_steps", "ql_accuracy"):
            self.assertIn(key, summary)

    def test_insufficient_data_no_crash(self):
        ai = BetpawaAI()
        # Should not raise even with 1 data point
        ai.train([config.RESULT_WIN], [(2, 0)])


class TestModelPersistence(unittest.TestCase):

    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self._tmpfile.close()

    def tearDown(self):
        try:
            os.unlink(self._tmpfile.name)
        except PermissionError:
            pass

    def test_ai_save_and_load_round_trip(self):
        random.seed(123)
        outcomes = [random.choice([0, 1, 2]) for _ in range(12)]
        goals = [(random.randint(0, 3), random.randint(0, 3)) for _ in range(12)]

        ai = BetpawaAI(checkpoint_path=self._tmpfile.name, auto_load=False)
        ai.train(outcomes, goals, ga_generations=2, lstm_epochs=1)
        original = ai.summary()
        self.assertTrue(ai.save())

        restored = BetpawaAI(checkpoint_path=self._tmpfile.name, auto_load=True)
        self.assertEqual(restored.summary()["ga_generation"], original["ga_generation"])
        self.assertEqual(restored.summary()["ql_steps"], original["ql_steps"])
        self.assertEqual(restored.summary()["ql_q_table_size"], original["ql_q_table_size"])

    def test_legacy_checkpoint_restores_non_ga_state(self):
        random.seed(321)
        outcomes = [random.choice([0, 1, 2]) for _ in range(12)]
        goals = [(random.randint(0, 3), random.randint(0, 3)) for _ in range(12)]

        ai = BetpawaAI(checkpoint_path=self._tmpfile.name, auto_load=False)
        ai.train(outcomes, goals, ga_generations=1, lstm_epochs=1)

        legacy_state = ai.state_dict()
        legacy_state["version"] = 1
        legacy_state["ga"]["version"] = 1

        with open(self._tmpfile.name, "wb") as handle:
            pickle.dump(legacy_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        restored = BetpawaAI(checkpoint_path=self._tmpfile.name, auto_load=True)
        self.assertEqual(restored.ql.total_steps, ai.ql.total_steps)
        self.assertEqual(restored.ql.epsilon, ai.ql.epsilon)
        self.assertEqual(restored.ql.q_table, ai.ql.q_table)
        self.assertEqual(restored.ga.generation, 0)


# ── CLI argument parsing ───────────────────────────────────────────────────────

class TestCLIParser(unittest.TestCase):

    def _parse(self, argv):
        return cli_module.build_parser().parse_args(argv)

    def test_start_command(self):
        args = self._parse(["start"])
        self.assertEqual(args.command, "start")

    def test_scrape_command(self):
        args = self._parse(["scrape"])
        self.assertEqual(args.command, "scrape")

    def test_results_default_count(self):
        args = self._parse(["results"])
        self.assertEqual(args.count, 20)

    def test_results_custom_count(self):
        args = self._parse(["results", "--count", "5"])
        self.assertEqual(args.count, 5)

    def test_train_default_generations(self):
        args = self._parse(["train"])
        self.assertEqual(args.ga_generations, config.GA_GENERATIONS)

    def test_train_custom_generations(self):
        args = self._parse(["train", "--ga-generations", "25"])
        self.assertEqual(args.ga_generations, 25)

    def test_predict_command(self):
        args = self._parse(["predict"])
        self.assertEqual(args.command, "predict")

    def test_backfill_defaults(self):
        args = self._parse(["backfill"])
        self.assertEqual(args.command, "backfill")
        self.assertEqual(args.seasons, config.HISTORICAL_SEASONS_LIMIT)
        self.assertEqual(args.matchdays, config.HISTORICAL_MATCHDAYS)
        self.assertIsNone(args.start_season_id)

    def test_backfill_custom_args(self):
        args = self._parse([
            "backfill",
            "--seasons", "3",
            "--matchdays", "12",
            "--start-season-id", "137462",
        ])
        self.assertEqual(args.seasons, 3)
        self.assertEqual(args.matchdays, 12)
        self.assertEqual(args.start_season_id, 137462)

    def test_status_command(self):
        args = self._parse(["status"])
        self.assertEqual(args.command, "status")

    def test_save_ai_command(self):
        args = self._parse(["save-ai"])
        self.assertEqual(args.command, "save-ai")

    def test_load_ai_command(self):
        args = self._parse(["load-ai"])
        self.assertEqual(args.command, "load-ai")

    def test_checkpoint_status_command(self):
        args = self._parse(["checkpoint-status"])
        self.assertEqual(args.command, "checkpoint-status")

    def test_no_command_raises(self):
        with self.assertRaises(SystemExit):
            self._parse([])


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
