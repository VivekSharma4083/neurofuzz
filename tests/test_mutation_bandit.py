"""Unit and integration tests for Phase 6C: UCB1 Mutation-Operator Selection."""

import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from fuzzer.dictionary import Dictionary
from fuzzer.fuzzer import Fuzzer
from fuzzer.mutation_bandit import MutationBandit, get_default_mutation_arms
from fuzzer.scheduler import HeuristicScheduler, LinearBanditScheduler, RandomScheduler


class TestUCB1Formula(unittest.TestCase):
    """Direct mathematical unit tests for UCB1 formula and arm mechanics."""

    def setUp(self) -> None:
        self.arms = ["flip_bit", "replace_byte", "insert_byte", "delete_byte"]
        self.bandit = MutationBandit(operators=self.arms, exploration_constant=1.0, rng=random.Random(42))

    def test_empty_operators_raises(self) -> None:
        """Empty operators list must raise ValueError."""
        with self.assertRaises(ValueError):
            MutationBandit(operators=[])

    def test_negative_c_raises(self) -> None:
        """Negative exploration constant must raise ValueError."""
        with self.assertRaises(ValueError):
            MutationBandit(operators=["flip_bit"], exploration_constant=-0.5)

    def test_unvisited_arms_infinite_score(self) -> None:
        """Before any pulls, all arms must have UCB score = +infinity."""
        for arm in self.arms:
            self.assertEqual(self.bandit.get_ucb_score(arm), float("inf"))
            self.assertEqual(self.bandit.get_mean_reward(arm), 0.0)

    def test_initial_round_robin_exploration(self) -> None:
        """First K pulls must select every unvisited arm exactly once in order."""
        selected_arms = []
        for _ in range(len(self.arms)):
            arm = self.bandit.select_operator()
            self.bandit.update(arm, reward=0.0)
            selected_arms.append(arm)

        self.assertEqual(selected_arms, self.arms)
        for arm in self.arms:
            self.assertEqual(self.bandit.counts[arm], 1)
        self.assertEqual(self.bandit.total_selections, len(self.arms))

    def test_exact_ucb_calculation(self) -> None:
        """Verify exact manual calculation: UCB = Q + c * sqrt(ln(N) / n).

        Given: Q = 2.0, N = 10, n = 5, c = 1.0
        expected = 2.0 + sqrt(ln(10) / 5) = 2.0 + sqrt(2.30258509 / 5) = 2.0 + 0.6786141 = 2.6786141
        """
        bandit = MutationBandit(operators=["arm_a", "arm_b"], exploration_constant=1.0)
        # Artificially populate arm_a with 5 pulls and total reward 10.0 (Q = 2.0)
        bandit.counts["arm_a"] = 5
        bandit.total_rewards["arm_a"] = 10.0
        # Populate arm_b with 5 pulls and total reward 5.0
        bandit.counts["arm_b"] = 5
        bandit.total_rewards["arm_b"] = 5.0
        bandit.total_selections = 10

        expected_ucb_a = 2.0 + math.sqrt(math.log(10) / 5)
        calculated_ucb_a = bandit.get_ucb_score("arm_a")
        self.assertTrue(
            math.isclose(calculated_ucb_a, expected_ucb_a, rel_tol=1e-5),
            f"Expected {expected_ucb_a}, got {calculated_ucb_a}",
        )
        self.assertTrue(math.isclose(calculated_ucb_a, 2.678614, rel_tol=1e-5))

    def test_exploration_constant_scaling(self) -> None:
        """Exploration bonus must scale linearly with parameter c."""
        bandit_c1 = MutationBandit(operators=["op"], exploration_constant=1.0)
        bandit_c2 = MutationBandit(operators=["op"], exploration_constant=2.0)

        for b in [bandit_c1, bandit_c2]:
            b.counts["op"] = 4
            b.total_rewards["op"] = 4.0  # Q = 1.0
            b.total_selections = 16

        bonus_1 = bandit_c1.get_ucb_score("op") - 1.0
        bonus_2 = bandit_c2.get_ucb_score("op") - 1.0
        self.assertTrue(math.isclose(bonus_2, 2.0 * bonus_1, rel_tol=1e-5))

    def test_exploitation_after_initialization(self) -> None:
        """Arm with high reward should be selected for exploitation immediately after round-robin."""
        for arm in self.arms:
            self.bandit.select_operator()
            # Give high reward to replace_byte, zero to others
            reward = 5.0 if arm == "replace_byte" else 0.0
            self.bandit.update(arm, reward)

        # 5th selection must choose replace_byte
        next_op = self.bandit.select_operator()
        self.assertEqual(next_op, "replace_byte")

    def test_zero_reward_handling(self) -> None:
        """Zero reward must correctly keep mean reward at 0.0 with finite valid UCB score."""
        self.bandit.update("flip_bit", 0.0)
        self.assertEqual(self.bandit.get_mean_reward("flip_bit"), 0.0)
        self.assertEqual(self.bandit.nonzero_rewards["flip_bit"], 0)

        # After all arms visited with 0
        for arm in self.arms[1:]:
            self.bandit.update(arm, 0.0)

        score = self.bandit.get_ucb_score("flip_bit")
        self.assertGreater(score, 0.0)
        self.assertFalse(math.isinf(score))

    def test_deterministic_tie_breaking(self) -> None:
        """Seeded RNG produces deterministic selections when scores tie."""
        b1 = MutationBandit(operators=self.arms, rng=random.Random(999))
        b2 = MutationBandit(operators=self.arms, rng=random.Random(999))

        # Perform 20 selections with equal zero rewards
        seq1 = [b1.select_operator() for _ in range(20)]
        seq2 = [b2.select_operator() for _ in range(20)]
        self.assertEqual(seq1, seq2)

    def test_reset(self) -> None:
        """reset() must restore bandit to initial state."""
        for arm in self.arms:
            self.bandit.update(arm, 2.0)
        self.assertEqual(self.bandit.total_selections, len(self.arms))

        self.bandit.reset()
        self.assertEqual(self.bandit.total_selections, 0)
        for arm in self.arms:
            self.assertEqual(self.bandit.counts[arm], 0)
            self.assertEqual(self.bandit.total_rewards[arm], 0.0)
            self.assertEqual(self.bandit.get_ucb_score(arm), float("inf"))

    def test_unknown_operator_raises(self) -> None:
        """Methods must raise KeyError when passed an unregistered operator."""
        with self.assertRaises(KeyError):
            self.bandit.get_mean_reward("invalid_operator")
        with self.assertRaises(KeyError):
            self.bandit.get_ucb_score("invalid_operator")
        with self.assertRaises(KeyError):
            self.bandit.update("invalid_operator", 1.0)


class TestSyntheticBanditEnvironment(unittest.TestCase):
    """Test UCB1 learning in a controlled synthetic multi-armed bandit environment."""

    def test_ucb1_converges_to_high_reward_arm(self) -> None:
        """Verify UCB1 pulls the clearly superior arm far more often than inferior arms."""
        arms = [
            "flip_bit",
            "replace_byte",
            "insert_byte",
            "delete_byte",
            "dictionary_insert_boundary",
            "dictionary_replace",
        ]
        # True expected rewards: dictionary_insert_boundary is ~10x better than others
        expected_rewards = {
            "flip_bit": 0.05,
            "replace_byte": 0.05,
            "insert_byte": 0.05,
            "delete_byte": 0.01,
            "dictionary_insert_boundary": 0.80,
            "dictionary_replace": 0.10,
        }

        rng = random.Random(12345)
        bandit = MutationBandit(operators=arms, exploration_constant=1.0, rng=rng)

        # Run 1000 simulated pulls
        for _ in range(1000):
            arm = bandit.select_operator()
            p = expected_rewards[arm]
            reward = 1.0 if rng.random() < p else 0.0
            bandit.update(arm, reward)

        stats = bandit.get_statistics()
        best_arm_pulls = stats["operators"]["dictionary_insert_boundary"]["selections"]
        best_arm_share = stats["operators"]["dictionary_insert_boundary"]["selection_rate"]

        # 1. Superior arm should receive the majority of pulls (> 50%)
        self.assertGreater(best_arm_share, 0.50, f"Expected >50% share, got {best_arm_share:.1%}")

        # 2. Every arm must have been explored at least once
        for arm in arms:
            self.assertGreaterEqual(stats["operators"][arm]["selections"], 1)

        # 3. Empirical mean for dictionary_insert_boundary should approximate true mean (~0.80)
        empirical_mean = stats["operators"]["dictionary_insert_boundary"]["mean_reward"]
        self.assertAlmostEqual(empirical_mean, 0.80, delta=0.15)


class TestFuzzerMutationBanditIntegration(unittest.TestCase):
    """Integration tests verifying MutationBandit operates with Fuzzer and Schedulers."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.corpus_dir = Path(self.tmp_dir) / "corpus"
        self.crashes_dir = Path(self.tmp_dir) / "crashes"
        self.corpus_dir.mkdir(parents=True)
        self.crashes_dir.mkdir(parents=True)

        # Create minimal seed
        (self.corpus_dir / "seed1.txt").write_bytes(b"NF01|PING")

        self.target = Path("targets/structured_target.exe")
        self.dict_file = Path("dictionaries/structured_protocol.dict")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_default_mutation_arms_helper(self) -> None:
        """Verify arm configuration helper without and with dictionary."""
        arms_no_dict = get_default_mutation_arms(has_dictionary=False)
        self.assertEqual(len(arms_no_dict), 4)
        self.assertNotIn("dictionary_insert_boundary", arms_no_dict)

        arms_dict_boundary = get_default_mutation_arms(has_dictionary=True, boundary_aware=True)
        self.assertEqual(len(arms_dict_boundary), 6)
        self.assertIn("dictionary_insert_boundary", arms_dict_boundary)

        arms_dict_random = get_default_mutation_arms(has_dictionary=True, boundary_aware=False)
        self.assertEqual(len(arms_dict_random), 6)
        self.assertIn("dictionary_insert_random", arms_dict_random)

    def test_fuzzer_mutation_policy_fixed_backward_compatibility(self) -> None:
        """When mutation_policy='fixed', mutation_bandit must be None and runs normally."""
        fuzzer = Fuzzer(
            target_path=self.target,
            corpus_dir=self.corpus_dir,
            crashes_dir=self.crashes_dir,
            iterations=10,
            calibrate=False,
            mutation_policy="fixed",
        )
        self.assertIsNone(fuzzer.mutation_bandit)
        self.assertEqual(fuzzer.mutation_policy, "fixed")
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 10)
        self.assertEqual(stats.mutation_policy, "fixed")

    def test_fuzzer_mutation_policy_ucb1_without_dictionary(self) -> None:
        """When mutation_policy='ucb1' without dictionary, operates on 4 byte arms."""
        fuzzer = Fuzzer(
            target_path=self.target,
            corpus_dir=self.corpus_dir,
            crashes_dir=self.crashes_dir,
            iterations=20,
            calibrate=False,
            mutation_policy="ucb1",
            ucb_c=1.0,
        )
        self.assertIsNotNone(fuzzer.mutation_bandit)
        self.assertEqual(fuzzer.mutation_policy, "ucb1")
        self.assertEqual(len(fuzzer.mutation_bandit.operators), 4)
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 20)
        self.assertEqual(fuzzer.mutation_bandit.total_selections, 20)

    def test_fuzzer_mutation_policy_ucb1_with_dictionary_and_boundary(self) -> None:
        """When mutation_policy='ucb1' with dictionary, operates on 6 arms."""
        fuzzer = Fuzzer(
            target_path=self.target,
            corpus_dir=self.corpus_dir,
            crashes_dir=self.crashes_dir,
            iterations=30,
            calibrate=False,
            dictionary_path=self.dict_file,
            boundary_aware=True,
            mutation_policy="ucb1",
            ucb_c=1.0,
            seed=42,
        )
        self.assertIsNotNone(fuzzer.mutation_bandit)
        self.assertEqual(len(fuzzer.mutation_bandit.operators), 6)
        self.assertIn("dictionary_insert_boundary", fuzzer.mutation_bandit.operators)

        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 30)
        self.assertEqual(fuzzer.mutation_bandit.total_selections, 30)
        # Telemetry should be populated in stats
        self.assertIn("operators", stats.mutation_bandit_telemetry)

    def test_interoperability_with_all_seed_schedulers(self) -> None:
        """Verify UCB1 mutation selection operates cleanly alongside Random, Heuristic, and LinearBandit."""
        schedulers = ["random", "heuristic", "linear_bandit"]
        for sched in schedulers:
            fuzzer = Fuzzer(
                target_path=self.target,
                corpus_dir=self.corpus_dir,
                crashes_dir=self.crashes_dir,
                iterations=15,
                calibrate=False,
                scheduler_type=sched,
                dictionary_path=self.dict_file,
                boundary_aware=True,
                mutation_policy="ucb1",
                seed=101,
            )
            stats = fuzzer.run()
            self.assertEqual(stats.total_executions, 15)
            self.assertEqual(fuzzer.mutation_bandit.total_selections, 15)
            self.assertEqual(stats.scheduler_name, sched)


if __name__ == "__main__":
    unittest.main()
