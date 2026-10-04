"""Unit tests for Phase 3: Seed Scheduling."""

import random
import tempfile
import unittest
from pathlib import Path
import numpy as np

from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.scheduler import (
    HeuristicScheduler,
    LinearBanditScheduler,
    RandomScheduler,
)


class TestSeedRecord(unittest.TestCase):
    """Tests for SeedRecord metadata tracking."""

    def _make_record(self, data=b"SEED", name="test_seed"):
        return SeedRecord(id=0, data=data, name=name)

    def test_initial_state(self):
        """Verify SeedRecord starts with all counters zeroed."""
        r = self._make_record()
        self.assertEqual(r.times_selected, 0)
        self.assertEqual(r.times_mutated, 0)
        self.assertEqual(r.times_produced_coverage, 0)
        self.assertEqual(r.total_coverage_discovered, 0)
        self.assertEqual(r.last_iteration_selected, 0)
        self.assertEqual(r.size_bytes, 4)

    def test_record_selection(self):
        """Verify record_selection increments counters correctly."""
        r = self._make_record()
        r.record_selection(iteration=42)
        self.assertEqual(r.times_selected, 1)
        self.assertEqual(r.times_mutated, 1)
        self.assertEqual(r.last_iteration_selected, 42)

    def test_record_reward_positive(self):
        """Verify positive reward increments yield counters."""
        r = self._make_record()
        r.record_reward(reward=3)
        self.assertEqual(r.times_produced_coverage, 1)
        self.assertEqual(r.total_coverage_discovered, 3)

    def test_record_reward_zero(self):
        """Verify zero reward does not increment yield counters."""
        r = self._make_record()
        r.record_reward(reward=0)
        self.assertEqual(r.times_produced_coverage, 0)
        self.assertEqual(r.total_coverage_discovered, 0)

    def test_record_reward_accumulation(self):
        """Verify multiple rewards accumulate correctly."""
        r = self._make_record()
        r.record_reward(2)
        r.record_reward(0)
        r.record_reward(5)
        self.assertEqual(r.total_coverage_discovered, 7)
        self.assertEqual(r.times_produced_coverage, 2)


class TestCorpusWithSeedRecord(unittest.TestCase):
    """Tests for Corpus using SeedRecord backend."""

    def test_add_seed_creates_record(self):
        """Verify add_seed returns a SeedRecord and appends to records list."""
        c = Corpus()
        rec = c.add_seed(b"HELLO")
        self.assertIsInstance(rec, SeedRecord)
        self.assertEqual(len(c), 1)
        self.assertEqual(c.records[0].data, b"HELLO")

    def test_seeds_property_backwards_compatible(self):
        """Verify .seeds property exposes raw bytes list for legacy test compatibility."""
        c = Corpus()
        c.add_seed(b"AAA")
        c.add_seed(b"BBB")
        self.assertEqual(c.seeds, [b"AAA", b"BBB"])

    def test_select_seed_returns_bytes(self):
        """Verify select_seed() returns bytes (backwards compatible)."""
        c = Corpus(rng=random.Random(0))
        c.add_seed(b"A")
        result = c.select_seed()
        self.assertIsInstance(result, bytes)

    def test_getitem_returns_bytes(self):
        """Verify corpus[i] returns bytes."""
        c = Corpus()
        c.add_seed(b"DATA")
        self.assertEqual(c[0], b"DATA")

    def test_get_record_by_index(self):
        """Verify get_record(i) returns the SeedRecord."""
        c = Corpus()
        c.add_seed(b"REC")
        rec = c.get_record(0)
        self.assertIsInstance(rec, SeedRecord)
        self.assertEqual(rec.data, b"REC")

    def test_add_interesting_input_creates_record(self):
        """Verify add_interesting_input adds a SeedRecord with coverage metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "s.txt").write_bytes(b"SEED")
            c = Corpus(corpus_dir=tmp_path)
            self.assertEqual(len(c), 1)

            c.add_interesting_input(b"NEW_INPUT", {"BRANCH_X"}, iteration=7, save_to_disk=False)
            self.assertEqual(len(c), 2)
            rec = c.get_record(1)
            self.assertIn("BRANCH_X", rec.coverage_units)
            self.assertEqual(rec.metadata["iteration"], 7)


class TestRandomScheduler(unittest.TestCase):
    """Tests for RandomScheduler baseline."""

    def setUp(self):
        self.rng = random.Random(42)
        self.scheduler = RandomScheduler(rng=self.rng)
        self.corpus = Corpus(rng=random.Random(42))
        self.corpus.add_seed(b"A")
        self.corpus.add_seed(b"B")
        self.corpus.add_seed(b"C")

    def test_select_returns_seed_record(self):
        """Verify select_seed returns a SeedRecord."""
        rec = self.scheduler.select_seed(self.corpus, iteration=1)
        self.assertIsInstance(rec, SeedRecord)

    def test_selection_increments_record(self):
        """Verify selection increments SeedRecord.times_selected."""
        rec = self.scheduler.select_seed(self.corpus, iteration=5)
        self.assertEqual(rec.times_selected, 1)
        self.assertEqual(rec.last_iteration_selected, 5)

    def test_update_accumulates_reward(self):
        """Verify update accumulates reward in scheduler and record."""
        rec = self.scheduler.select_seed(self.corpus, iteration=1)
        self.scheduler.update(rec, reward=2, iteration=1)
        self.assertEqual(self.scheduler.total_reward, 2)
        self.assertEqual(rec.total_coverage_discovered, 2)

    def test_exploration_rate_is_one(self):
        """Verify random scheduler reports exploration rate of 1.0."""
        telemetry = self.scheduler.get_telemetry()
        self.assertEqual(telemetry["exploration_rate"], 1.0)

    def test_empty_corpus_raises(self):
        """Verify selection from empty corpus raises ValueError."""
        empty = Corpus()
        with self.assertRaises(ValueError):
            self.scheduler.select_seed(empty, iteration=1)

    def test_all_seeds_selectable(self):
        """Verify all seeds can be selected by the random scheduler over many trials."""
        selected_data = set()
        for i in range(100):
            rec = self.scheduler.select_seed(self.corpus, iteration=i)
            selected_data.add(rec.data)
        self.assertEqual(selected_data, {b"A", b"B", b"C"})


class TestHeuristicScheduler(unittest.TestCase):
    """Tests for HeuristicScheduler scoring and epsilon-greedy exploration."""

    def setUp(self):
        self.rng = random.Random(99)
        self.corpus = Corpus(rng=random.Random(99))
        self.corpus.add_seed(b"LOW_YIELD")   # id=0
        self.corpus.add_seed(b"HIGH_YIELD")  # id=1

    def test_invalid_epsilon_raises(self):
        """Verify epsilon outside [0.0, 1.0] raises ValueError."""
        with self.assertRaises(ValueError):
            HeuristicScheduler(epsilon=1.5)
        with self.assertRaises(ValueError):
            HeuristicScheduler(epsilon=-0.1)

    def test_compute_score_new_seed(self):
        """Verify new seed (0 selections) gets neutral score of 1.0."""
        rec = SeedRecord(id=0, data=b"X", name="fresh")
        score = HeuristicScheduler.compute_score(rec)
        self.assertAlmostEqual(score, 1.0)

    def test_compute_score_improves_with_reward(self):
        """Verify score increases when a seed discovers more coverage."""
        rec = SeedRecord(id=0, data=b"X", name="seed")
        score_before = HeuristicScheduler.compute_score(rec)
        rec.record_selection(1)
        rec.record_reward(5)
        score_after = HeuristicScheduler.compute_score(rec)
        self.assertGreater(score_after, score_before)

    def test_compute_score_decays_with_selections(self):
        """Verify score decreases when a seed is selected repeatedly without reward."""
        rec = SeedRecord(id=0, data=b"X", name="seed")
        for i in range(10):
            rec.record_selection(i)
            rec.record_reward(0)
        score = HeuristicScheduler.compute_score(rec)
        self.assertLess(score, 1.0)

    def test_exploitation_favors_high_yield_seed(self):
        """Verify exploitation (epsilon=0) picks seed with higher initial score.

        We directly boost total_coverage_discovered on one seed without extra
        selections, giving it score=(1+20)/(1+0)=21.0 vs the other at 1.0.
        A single epsilon=0 pick must choose the high-scoring seed.
        """
        sched = HeuristicScheduler(epsilon=0.0, rng=random.Random(0))
        high_rec = self.corpus.get_record(1)
        high_rec.total_coverage_discovered = 20  # score=21.0 vs low=1.0

        chosen = sched.select_seed(self.corpus, iteration=1)
        self.assertEqual(chosen.id, high_rec.id,
                         "Exploitation should pick the highest-scoring seed")

    def test_exploration_selects_non_best_seed(self):
        """Verify that with epsilon=1.0 (pure exploration), any seed may be chosen."""
        sched = HeuristicScheduler(epsilon=1.0, rng=random.Random(7))
        selected = set()
        for i in range(50):
            rec = sched.select_seed(self.corpus, iteration=i)
            selected.add(rec.data)
        self.assertGreater(len(selected), 1, "Pure exploration should select multiple seeds")

    def test_exploration_counter_increments(self):
        """Verify explorations counter is updated correctly."""
        sched = HeuristicScheduler(epsilon=1.0, rng=random.Random(1))
        # With epsilon=1 and multiple seeds, every pick is exploration
        for i in range(10):
            sched.select_seed(self.corpus, iteration=i)
        self.assertEqual(sched.explorations, 10)
        self.assertEqual(sched.exploitations, 0)

    def test_exploitation_counter_increments(self):
        """Verify exploitations counter is updated correctly."""
        sched = HeuristicScheduler(epsilon=0.0, rng=random.Random(1))
        for i in range(10):
            sched.select_seed(self.corpus, iteration=i)
        self.assertEqual(sched.exploitations, 10)
        self.assertEqual(sched.explorations, 0)

    def test_starvation_prevention_with_epsilon(self):
        """Verify every seed gets selected at least once across enough iterations."""
        # 5 seeds, epsilon=0.3: over 200 iterations, all should be selected
        sched = HeuristicScheduler(epsilon=0.3, rng=random.Random(42))
        c = Corpus(rng=random.Random(42))
        for i in range(5):
            c.add_seed(f"seed_{i}".encode())

        # Give seed 0 a huge score bias
        for i in range(50):
            c.get_record(0).record_selection(i)
            c.get_record(0).record_reward(10)

        selected_ids = set()
        for i in range(200):
            rec = sched.select_seed(c, iteration=i)
            selected_ids.add(rec.id)

        self.assertEqual(len(selected_ids), 5,
                         "All seeds should be selected at least once (no starvation)")

    def test_get_telemetry_keys(self):
        """Verify telemetry dictionary contains all expected keys."""
        sched = HeuristicScheduler(epsilon=0.15, rng=random.Random(0))
        t = sched.get_telemetry()
        for key in ("strategy", "epsilon", "total_selections", "total_reward",
                    "average_reward", "explorations", "exploitations",
                    "empirical_exploration_rate"):
            self.assertIn(key, t, f"Missing telemetry key: {key}")
        self.assertEqual(t["epsilon"], 0.15)

    def test_average_reward_calculation(self):
        """Verify average reward is computed correctly."""
        sched = HeuristicScheduler(epsilon=0.0, rng=random.Random(0))
        rec = self.corpus.get_record(0)
        sched.select_seed(self.corpus, iteration=1)
        sched.update(rec, reward=4, iteration=1)
        sched.select_seed(self.corpus, iteration=2)
        sched.update(rec, reward=0, iteration=2)
        self.assertAlmostEqual(sched.average_reward, 2.0)

    def test_scheduler_swap_does_not_break_corpus(self):
        """Verify swapping schedulers at runtime works correctly."""
        rnd = RandomScheduler(rng=random.Random(0))
        heu = HeuristicScheduler(epsilon=0.2, rng=random.Random(0))

        # Use both schedulers against the same corpus
        for i in range(5):
            rec = rnd.select_seed(self.corpus, iteration=i)
            rnd.update(rec, reward=0, iteration=i)
        for i in range(5):
            rec = heu.select_seed(self.corpus, iteration=i)
            heu.update(rec, reward=1, iteration=i)

        # Corpus integrity: all seeds still present
        self.assertEqual(len(self.corpus), 2)


class TestLinearBanditScheduler(unittest.TestCase):
    """Unit tests for LinearBanditScheduler."""

    def setUp(self):
        self.rng = random.Random(42)
        self.corpus = Corpus(rng=random.Random(42))
        self.corpus.add_seed(b"SEED_LOW")   # id=0
        self.corpus.add_seed(b"SEED_HIGH")  # id=1

    def test_invalid_epsilon_raises(self):
        """Verify invalid epsilon outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            LinearBanditScheduler(epsilon=-0.1)
        with self.assertRaises(ValueError):
            LinearBanditScheduler(epsilon=1.1)

    def test_select_returns_seed_record(self):
        """Verify select_seed returns a SeedRecord."""
        sched = LinearBanditScheduler(rng=random.Random(0))
        rec = sched.select_seed(self.corpus, iteration=1)
        self.assertIsInstance(rec, SeedRecord)

    def test_exploration_counter_increments(self):
        """Verify epsilon=1.0 performs only explorations."""
        sched = LinearBanditScheduler(epsilon=1.0, rng=random.Random(1))
        for i in range(10):
            sched.select_seed(self.corpus, iteration=i)
        self.assertEqual(sched.explorations, 10)
        self.assertEqual(sched.exploitations, 0)

    def test_exploitation_counter_increments(self):
        """Verify epsilon=0.0 performs only exploitations."""
        sched = LinearBanditScheduler(epsilon=0.0, rng=random.Random(1))
        for i in range(10):
            sched.select_seed(self.corpus, iteration=i)
        self.assertEqual(sched.exploitations, 10)
        self.assertEqual(sched.explorations, 0)

    def test_exploitation_prefers_highest_predicted_seed(self):
        """Verify that when weights strongly favor a feature, the seed with that feature is picked."""
        sched = LinearBanditScheduler(epsilon=0.0, rng=random.Random(0))

        # Give SEED_HIGH (id=1) a strong positive yield history
        high_rec = self.corpus.get_record(1)
        high_rec.total_coverage_discovered = 15
        high_rec.times_produced_coverage = 5
        high_rec.times_selected = 5

        # Train the model with a strong positive reward for high yield
        feat_high = sched.feature_extractor.extract(high_rec, current_iteration=5)
        for _ in range(5):
            sched.model.update(feat_high, reward=3.0)

        # Under epsilon=0, the scheduler must exploit SEED_HIGH
        chosen = sched.select_seed(self.corpus, iteration=6)
        self.assertEqual(chosen.id, high_rec.id)

    def test_update_modifies_model_weights_and_stats(self):
        """Verify update calls model.update and adjusts telemetry."""
        sched = LinearBanditScheduler(epsilon=0.0, rng=random.Random(0))
        rec = sched.select_seed(self.corpus, iteration=1)

        weights_before = sched.model.weights.copy()
        sched.update(rec, reward=4, iteration=1)
        weights_after = sched.model.weights

        self.assertEqual(sched.total_reward, 4)
        self.assertEqual(rec.total_coverage_discovered, 4)
        self.assertEqual(sched.model.total_updates, 1)
        self.assertFalse(np.array_equal(weights_before, weights_after))

    def test_anti_starvation_with_epsilon(self):
        """Verify all seeds get explored when epsilon > 0 even if one has huge reward."""
        sched = LinearBanditScheduler(epsilon=0.3, rng=random.Random(42))
        c = Corpus(rng=random.Random(42))
        for i in range(5):
            c.add_seed(f"seed_{i}".encode())

        # Give seed 0 a huge reward to dominate exploitation
        c.get_record(0).total_coverage_discovered = 100
        feat0 = sched.feature_extractor.extract(c.get_record(0), 10)
        for _ in range(10):
            sched.model.update(feat0, reward=10.0)

        selected_ids = set()
        for i in range(200):
            rec = sched.select_seed(c, iteration=i)
            sched.update(rec, reward=0, iteration=i)
            selected_ids.add(rec.id)

        self.assertEqual(len(selected_ids), 5, "Exploration should ensure no seed is starved")

    def test_telemetry_keys(self):
        """Verify telemetry dictionary contains bandit-specific keys."""
        sched = LinearBanditScheduler(epsilon=0.25, rng=random.Random(0))
        t = sched.get_telemetry()
        for key in (
            "strategy",
            "epsilon",
            "total_selections",
            "total_reward",
            "average_reward",
            "explorations",
            "exploitations",
            "empirical_exploration_rate",
            "model_updates",
            "model_mse",
            "model_mae",
            "learned_weights",
        ):
            self.assertIn(key, t, f"Missing telemetry key: {key}")
        self.assertEqual(t["strategy"], "linear_bandit")
        self.assertEqual(t["epsilon"], 0.25)

    def test_reward_propagation_and_model_persistence_in_fuzzer(self):
        """Verify reward propagation and model weight evolution during an actual Fuzzer run."""
        from fuzzer.fuzzer import Fuzzer
        import io, contextlib

        with tempfile.TemporaryDirectory() as tmp_c, tempfile.TemporaryDirectory() as tmp_cr:
            # Seed file
            (Path(tmp_c) / "s1.txt").write_bytes(b"FUZZ_ECHO\n")

            target_exe = Path("targets/vulnerable_target.exe")
            if not target_exe.exists():
                return  # Skip if binary not built

            fuzzer = Fuzzer(
                target_path=target_exe,
                corpus_dir=tmp_c,
                crashes_dir=tmp_cr,
                iterations=15,
                scheduler_type="linear_bandit",
                epsilon=0.2,
                learning_rate=0.05,
                l2_reg=0.001,
                seed=42,
                calibrate=False,
            )

            # Initial model weights are all zeros
            initial_weights = fuzzer.scheduler.model.weights.copy()
            np.testing.assert_allclose(initial_weights, np.zeros(8))

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats = fuzzer.run()

            # Verifications:
            # 1. Scheduler integration
            self.assertEqual(fuzzer.scheduler.name, "linear_bandit")
            # 2. Update count matches total executions
            self.assertEqual(fuzzer.scheduler.model.total_updates, 15)
            self.assertEqual(stats.total_executions, 15)
            # 3. Model weights evolved and persisted throughout the run
            final_weights = fuzzer.scheduler.model.weights
            self.assertFalse(np.array_equal(initial_weights, final_weights))
            # 4. Telemetry reflects updates and valid MSE
            telemetry = fuzzer.scheduler.get_telemetry()
            self.assertEqual(telemetry["model_updates"], 15)
            self.assertGreaterEqual(telemetry["model_mse"], 0.0)


if __name__ == "__main__":
    unittest.main()
