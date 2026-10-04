"""Unit tests for Adaptive Frontier Rate Decay and StateFrontier (Phase 7D)."""

import random
import unittest

from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.dictionary import Dictionary
from fuzzer.state_frontier import FrontierCandidate, StateFrontier


class TestAdaptiveFrontier(unittest.TestCase):
    """Test adaptive frontier rate decay and replenishment."""

    def setUp(self) -> None:
        self.dictionary = Dictionary(tokens=[b"STEP=1", b"STEP=2", b"STEP=3", b"STEP=4"])
        self.dummy_candidate = FrontierCandidate(
            seed_id=1,
            seed_name="seed1",
            seed_hash="hash123",
            parent_depth=14,
            boundary_pos=16,
            token=b"STEP=2",
            mutated_data=b"NF01|DIAG|STEP=1|STEP=2",
            mutation_type="boundary_insert",
        )

    def test_fixed_mode_preservation(self) -> None:
        """Fixed rate mode maintains nominal rate regardless of outcomes."""
        sf = StateFrontier(
            dictionary=self.dictionary,
            rate_mode="fixed",
            frontier_rate=0.20,
            min_frontier_rate=0.02,
            max_frontier_rate=0.30,
        )
        self.assertEqual(sf.get_current_rate(), 0.20)

        # Simulate 100 failed attempts
        for i in range(100):
            sf.on_execution_result(
                self.dummy_candidate,
                result_coverage=["DEPTH_14"],
                is_new_cov=False,
                new_units=set(),
                iteration=i,
            )

        self.assertEqual(sf.get_current_rate(), 0.20)
        self.assertEqual(sf.telemetry.rate_changes_count, 0)

        # Simulate depth advance
        sf.on_depth_advanced(new_depth=15, iteration=101)
        self.assertEqual(sf.get_current_rate(), 0.20)
        self.assertEqual(sf.telemetry.rate_changes_count, 0)

    def test_adaptive_initial_rate(self) -> None:
        """Adaptive rate mode initializes at frontier rate."""
        sf = StateFrontier(
            dictionary=self.dictionary,
            rate_mode="adaptive",
            frontier_rate=0.20,
            min_frontier_rate=0.02,
            max_frontier_rate=0.30,
        )
        self.assertEqual(sf.get_current_rate(), 0.20)
        self.assertEqual(sf.telemetry.rate_mode, "adaptive")

    def test_adaptive_stagnation_decay(self) -> None:
        """Rate decays after sustained unproductivity down to floor."""
        sf = StateFrontier(
            dictionary=self.dictionary,
            rate_mode="adaptive",
            frontier_rate=0.20,
            min_frontier_rate=0.02,
            max_frontier_rate=0.30,
            decay_factor=0.80,
            stagnation_threshold=25,
        )
        # Register a seed so queue isn't considered permanently exhausted
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1", name="s1", coverage_units={"DEPTH_14"})
        sf.register_seed(rec, depth=14)

        # 24 failed attempts: no decay yet
        for i in range(24):
            sf.on_execution_result(
                self.dummy_candidate,
                result_coverage=["DEPTH_14"],
                is_new_cov=False,
                new_units=set(),
                iteration=i + 1,
            )
        self.assertAlmostEqual(sf.get_current_rate(), 0.20, places=4)
        self.assertEqual(sf.telemetry.rate_changes_count, 0)

        # 25th failed attempt triggers first decay: 0.20 * 0.80 = 0.16
        sf.on_execution_result(
            self.dummy_candidate,
            result_coverage=["DEPTH_14"],
            is_new_cov=False,
            new_units=set(),
            iteration=25,
        )
        self.assertAlmostEqual(sf.get_current_rate(), 0.16, places=4)
        self.assertEqual(sf.telemetry.rate_changes_count, 1)

        # Multiple decays to reach floor (needs >= 11 decay windows: 11 * 25 = 275 attempts)
        for i in range(26, 500):
            sf.on_execution_result(
                self.dummy_candidate,
                result_coverage=["DEPTH_14"],
                is_new_cov=False,
                new_units=set(),
                iteration=i,
            )

        # Rate must floor at min_frontier_rate (0.02)
        self.assertAlmostEqual(sf.get_current_rate(), 0.02, places=4)
        self.assertGreater(sf.telemetry.rate_changes_count, 1)

    def test_adaptive_replenishment_on_depth_advance(self) -> None:
        """Discovering a new depth level replenishes rate toward ceiling."""
        sf = StateFrontier(
            dictionary=self.dictionary,
            rate_mode="adaptive",
            frontier_rate=0.20,
            min_frontier_rate=0.02,
            max_frontier_rate=0.30,
            decay_factor=0.50,
            stagnation_threshold=10,
        )
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1", name="s1", coverage_units={"DEPTH_14"})
        sf.register_seed(rec, depth=14)

        # Decay the rate with failed attempts
        for i in range(25):
            sf.on_execution_result(
                self.dummy_candidate,
                result_coverage=["DEPTH_14"],
                is_new_cov=False,
                new_units=set(),
                iteration=i + 1,
            )
        self.assertLess(sf.get_current_rate(), 0.20)

        # Discover new depth
        sf.on_depth_advanced(new_depth=15, iteration=26)

        # Rate should replenish to max_frontier_rate (or >= 0.25)
        self.assertGreaterEqual(sf.get_current_rate(), 0.25)
        self.assertLessEqual(sf.get_current_rate(), 0.30)
        self.assertGreaterEqual(sf.telemetry.rate_changes_count, 2)

    def test_adaptive_coverage_gain_resets_stagnation(self) -> None:
        """Gaining coverage resets stagnation counter without decaying."""
        sf = StateFrontier(
            dictionary=self.dictionary,
            rate_mode="adaptive",
            frontier_rate=0.20,
            stagnation_threshold=25,
        )
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1", name="s1", coverage_units={"DEPTH_14"})
        sf.register_seed(rec, depth=14)

        # 20 failed attempts
        for i in range(20):
            sf.on_execution_result(
                self.dummy_candidate,
                result_coverage=["DEPTH_14"],
                is_new_cov=False,
                new_units=set(),
                iteration=i + 1,
            )
        self.assertEqual(sf._stagnant_attempts, 20)

        # Coverage gained resets stagnant attempts
        sf.on_execution_result(
            self.dummy_candidate,
            result_coverage=["DEPTH_14", "EXTRA_BRANCH"],
            is_new_cov=True,
            new_units={"EXTRA_BRANCH"},
            iteration=21,
        )
        self.assertEqual(sf._stagnant_attempts, 0)
        self.assertAlmostEqual(sf.get_current_rate(), 0.20, places=4)

    def test_delimiter_update_and_extension(self) -> None:
        """Updating delimiter changes sequence synthesis formatting."""
        sf = StateFrontier(delimiter=b"|", dictionary=self.dictionary)
        self.assertEqual(sf.delimiter, b"|")

        sf.set_delimiter(b";")
        self.assertEqual(sf.delimiter, b";")

        # Register seed with semicolon format
        rec = SeedRecord(id=1, data=b"NF01;DIAG;STEP=1", name="s1", coverage_units={"DEPTH_14"})
        sf.register_seed(rec, depth=14)

        # Next candidate should exist and contain semicolon
        self.assertTrue(sf.has_candidate())
        _, cand_bytes, op, cand = sf.next_candidate()
        self.assertEqual(op, "state_frontier_extend")
        self.assertIn(b";", cand_bytes)

    def test_telemetry_serialization(self) -> None:
        """Telemetry includes Phase 7D adaptive rate metrics in dictionary."""
        sf = StateFrontier(rate_mode="adaptive", frontier_rate=0.20)
        sf.on_execution_result(
            self.dummy_candidate,
            result_coverage=["DEPTH_14"],
            is_new_cov=False,
            new_units=set(),
            iteration=1,
        )
        sf.on_depth_advanced(new_depth=15, iteration=2)

        data = sf.telemetry.to_dict()
        self.assertEqual(data["rate_mode"], "adaptive")
        self.assertIn("current_frontier_rate", data)
        self.assertIn("rate_changes_count", data)
        self.assertIn("min_frontier_rate", data)
        self.assertIn("max_frontier_rate", data)


if __name__ == "__main__":
    unittest.main()
