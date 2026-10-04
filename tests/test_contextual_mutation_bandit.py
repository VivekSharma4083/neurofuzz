"""Unit and integration tests for ContextualMutationBandit (Phase 6D)."""

from pathlib import Path
import random
import tempfile
import unittest
import numpy as np

from fuzzer.contextual_mutation_bandit import ContextualMutationBandit
from fuzzer.corpus import Corpus
from fuzzer.fuzzer import Fuzzer
from fuzzer.mutation_bandit import get_default_mutation_arms
from fuzzer.mutation_features import MutationFeatureExtractor


class TestContextualMutationBandit(unittest.TestCase):
    """Test suite for ContextualMutationBandit."""

    def setUp(self) -> None:
        self.operators = ["flip_bit", "replace_byte", "insert_byte", "delete_byte", "dictionary_insert_boundary", "dictionary_replace"]
        self.feature_dim = 6
        self.rng = random.Random(42)
        self.bandit = ContextualMutationBandit(
            operators=self.operators,
            feature_dim=self.feature_dim,
            epsilon=0.20,
            learning_rate=0.05,
            l2_reg=0.001,
            rng=self.rng,
        )

    def test_initialization_validation(self) -> None:
        """Validate input parameters during construction."""
        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=[])

        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=self.operators, feature_dim=0)

        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=self.operators, epsilon=-0.1)

        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=self.operators, epsilon=1.5)

        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=self.operators, learning_rate=0.0)

        with self.assertRaises(ValueError):
            ContextualMutationBandit(operators=self.operators, l2_reg=-0.01)

    def test_exploration_vs_exploitation_counts(self) -> None:
        """Verify epsilon=1.0 explores always and epsilon=0.0 exploits always."""
        context = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.1], dtype=np.float64)

        # Pure exploration
        explore_bandit = ContextualMutationBandit(
            operators=self.operators,
            feature_dim=6,
            epsilon=1.0,
            rng=random.Random(123),
        )
        for _ in range(20):
            explore_bandit.select_operator(context)
        self.assertEqual(explore_bandit.explorations, 20)
        self.assertEqual(explore_bandit.exploitations, 0)

        # Pure exploitation
        exploit_bandit = ContextualMutationBandit(
            operators=self.operators,
            feature_dim=6,
            epsilon=0.0,
            rng=random.Random(123),
        )
        for _ in range(20):
            exploit_bandit.select_operator(context)
        self.assertEqual(exploit_bandit.exploitations, 20)
        self.assertEqual(exploit_bandit.explorations, 0)

    def test_unpulled_arm_weights_isolation(self) -> None:
        """Updating arm A must not alter weights of arm B."""
        context = np.array([1.0, 0.8, 0.6, 0.5, 0.5, 0.2], dtype=np.float64)
        pulled_arm = "dictionary_insert_boundary"
        unpulled_arm = "delete_byte"

        initial_unpulled_weights = np.copy(self.bandit.models[unpulled_arm].weights)
        self.bandit.update(pulled_arm, context, reward=5.0)

        # Pulled arm weights changed
        self.assertFalse(np.array_equal(self.bandit.models[pulled_arm].weights, np.zeros(6)))
        # Unpulled arm weights remain strictly untouched
        self.assertTrue(np.array_equal(self.bandit.models[unpulled_arm].weights, initial_unpulled_weights))

    def test_contextual_discrimination_synthetic(self) -> None:
        """Bandit should learn distinct operator preferences conditioned on different contexts."""
        bandit = ContextualMutationBandit(
            operators=["flip_bit", "replace_byte", "dictionary_insert_boundary"],
            feature_dim=3,
            epsilon=0.0,  # Pure greedy evaluation after updates
            learning_rate=0.1,
            l2_reg=0.0001,
            rng=random.Random(999),
        )

        # Context A: Protocol diagnostic context (feature 1 is high)
        # Context B: Byte replacement context (feature 2 is high)
        ctx_a = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        ctx_b = np.array([1.0, 0.0, 1.0], dtype=np.float64)

        # Train: In Context A, dictionary_insert_boundary produces reward 10.0
        # In Context B, replace_byte produces reward 10.0
        for _ in range(15):
            bandit.update("dictionary_insert_boundary", ctx_a, reward=10.0)
            bandit.update("replace_byte", ctx_a, reward=0.0)
            bandit.update("flip_bit", ctx_a, reward=0.0)

            bandit.update("replace_byte", ctx_b, reward=10.0)
            bandit.update("dictionary_insert_boundary", ctx_b, reward=0.0)
            bandit.update("flip_bit", ctx_b, reward=0.0)

        # Evaluation: When given ctx_a, it must prefer dictionary_insert_boundary
        chosen_a = bandit.select_operator(ctx_a)
        self.assertEqual(chosen_a, "dictionary_insert_boundary")

        # When given ctx_b, it must prefer replace_byte
        chosen_b = bandit.select_operator(ctx_b)
        self.assertEqual(chosen_b, "replace_byte")

    def test_reset_functionality(self) -> None:
        """Reset should clear counts, rewards, and model weights."""
        context = np.array([1.0, 0.5, 0.3, 0.2, 0.4, 0.1], dtype=np.float64)
        self.bandit.update("flip_bit", context, reward=3.0)
        self.assertEqual(self.bandit.counts["flip_bit"], 1)

        self.bandit.reset()
        self.assertEqual(self.bandit.counts["flip_bit"], 0)
        self.assertEqual(self.bandit.total_selections, 0)
        self.assertTrue(np.all(self.bandit.models["flip_bit"].weights == 0.0))

    def test_statistics_structure(self) -> None:
        """get_statistics must return expected structure and telemetry."""
        stats = self.bandit.get_statistics()
        self.assertEqual(stats["strategy"], "contextual_mutation_bandit")
        self.assertEqual(stats["feature_dim"], 6)
        self.assertEqual(stats["epsilon"], 0.20)
        self.assertIn("operators", stats)
        for op in self.operators:
            self.assertIn(op, stats["operators"])
            self.assertIn("selections", stats["operators"][op])
            self.assertIn("model_mse", stats["operators"][op])
            self.assertIn("weights", stats["operators"][op])


class TestFuzzerContextualIntegration(unittest.TestCase):
    """Integration test suite for Fuzzer with Contextual Mutation Bandit."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.corpus_dir = Path(self.temp_dir.name) / "corpus"
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.crashes_dir = Path(self.temp_dir.name) / "crashes"
        self.crashes_dir.mkdir(parents=True, exist_ok=True)

        # Seed with initial inputs
        (self.corpus_dir / "seed1.bin").write_bytes(b"NF01|PING|HELLO")
        (self.corpus_dir / "seed2.bin").write_bytes(b"NF01|DIAG|STEP=1")

        self.target = Path("targets/structured_target.exe")
        self.dict_path = Path("dictionaries/structured_protocol.dict")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fuzzer_runs_with_contextual_mutation_random_scheduler(self) -> None:
        """Fuzzer executes with RandomScheduler and Contextual Mutation Policy."""
        if not self.target.exists():
            self.skipTest("Target binary not built")

        fuzzer = Fuzzer(
            target_path=self.target,
            corpus_dir=self.corpus_dir,
            crashes_dir=self.crashes_dir,
            iterations=15,
            timeout=1.0,
            seed=101,
            stats_interval=10,
            calibrate=True,
            scheduler_type="random",
            dictionary_path=self.dict_path,
            boundary_aware=True,
            mutation_policy="contextual",
            mutation_epsilon=0.20,
        )
        stats = fuzzer.run()

        self.assertEqual(stats.total_executions, 15)
        self.assertEqual(stats.mutation_policy, "contextual")
        self.assertIn("operators", stats.contextual_mutation_bandit_telemetry)
        # Verify learned weights table populated
        for op_info in stats.contextual_mutation_bandit_telemetry["operators"].values():
            self.assertIn("weights", op_info)

    def test_fuzzer_runs_with_contextual_mutation_linear_bandit_scheduler(self) -> None:
        """Fuzzer executes with LinearBanditScheduler and Contextual Mutation Policy simultaneously."""
        if not self.target.exists():
            self.skipTest("Target binary not built")

        fuzzer = Fuzzer(
            target_path=self.target,
            corpus_dir=self.corpus_dir,
            crashes_dir=self.crashes_dir,
            iterations=15,
            timeout=1.0,
            seed=202,
            stats_interval=10,
            calibrate=True,
            scheduler_type="linear_bandit",
            dictionary_path=self.dict_path,
            boundary_aware=True,
            mutation_policy="contextual",
            mutation_epsilon=0.20,
        )
        stats = fuzzer.run()

        self.assertEqual(stats.total_executions, 15)
        self.assertEqual(stats.mutation_policy, "contextual")
        self.assertIn("learned_weights", fuzzer.scheduler.get_telemetry())
        self.assertIn("operators", stats.contextual_mutation_bandit_telemetry)


if __name__ == "__main__":
    unittest.main()
