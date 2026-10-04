"""Unit and integration tests for Phase 6F Compatibility-Aware Parent Selection."""

import random
import unittest
from typing import List, Set

from fuzzer.corpus import SeedRecord
from fuzzer.donor_selector import (
    DonorCompatibility,
    DonorSelectionRecord,
    DonorSelector,
    DonorSelectorTelemetry,
)
from fuzzer.fuzzer import Fuzzer
from fuzzer.splicer import FieldSplicer


class TestDonorSelector(unittest.TestCase):
    """Test suite for generic structural donor compatibility evaluation and selection."""

    def setUp(self) -> None:
        self.rng = random.Random(42)
        self.selector = DonorSelector(
            mode="compatible",
            delimiter=b"|",
            prefix_weight=3.0,
            command_weight=2.0,
            protocol_weight=1.0,
            extra_field_weight=1.0,
            field_diff_penalty=0.5,
            depth_diff_penalty=0.25,
            rng=self.rng,
        )

    # 1. Empty corpus
    def test_01_empty_corpus(self) -> None:
        parent_a = b"NF01|DIAG|STEP=1"
        chosen = self.selector.select_donor(parent_a, [])
        self.assertIsNone(chosen)
        self.assertEqual(self.selector.telemetry.total_attempts, 1)
        self.assertEqual(self.selector.telemetry.single_seed_fallbacks, 1)

    # 2. One-seed corpus (self-donor prevention fallback)
    def test_02_one_seed_corpus(self) -> None:
        parent_a = b"NF01|DIAG|STEP=1"
        corpus = [parent_a]
        chosen = self.selector.select_donor(parent_a, corpus)
        # Should gracefully return None because the only seed is identical to parent A
        self.assertIsNone(chosen)
        self.assertEqual(self.selector.telemetry.single_seed_fallbacks, 1)

    # 3. Two-seed corpus
    def test_03_two_seed_corpus(self) -> None:
        parent_a = b"NF01|DIAG|STEP=1"
        seed_b = b"NF01|AUTH|USER=admin"
        corpus = [parent_a, seed_b]
        chosen = self.selector.select_donor(parent_a, corpus)
        self.assertEqual(chosen, seed_b)
        self.assertEqual(self.selector.telemetry.successful_selections, 1)

    # 4. Self-donor prevention (B != A)
    def test_04_self_donor_prevention(self) -> None:
        parent_a = b"NF01|DIAG|STEP=1"
        corpus = [
            b"NF01|DIAG|STEP=1",  # exact duplicate
            b"NF01|DIAG|STEP=1",  # exact duplicate
            b"NF01|DIAG|STEP=1|STEP=2",  # different candidate
        ]
        chosen = self.selector.select_donor(parent_a, corpus)
        self.assertEqual(chosen, b"NF01|DIAG|STEP=1|STEP=2")

    # 5. Common prefix calculation
    def test_05_common_prefix_calculation(self) -> None:
        p_a = b"NF01|DIAG|STEP=1|EXTRA"
        p_b = b"NF01|DIAG|STEP=1|STEP=2"
        compat = self.selector.evaluate_compatibility(p_a, p_b)
        # Matching fields: NF01, DIAG, STEP=1 -> length 3
        self.assertEqual(compat.common_prefix_fields, 3)

    # 6. Same protocol prefix
    def test_06_same_protocol_prefix(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        p_b_same = b"NF01|CALC|OP=ADD"
        p_b_diff = b"CUSTOM|DIAG|STEP=1"
        compat_same = self.selector.evaluate_compatibility(p_a, p_b_same)
        compat_diff = self.selector.evaluate_compatibility(p_a, p_b_diff)
        self.assertTrue(compat_same.same_protocol_prefix)
        self.assertFalse(compat_diff.same_protocol_prefix)

    # 7. Same command
    def test_07_same_command(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        p_b_same = b"NF01|DIAG|STEP=1|STEP=2"
        p_b_diff = b"NF01|CALC|OP=DIV"
        compat_same = self.selector.evaluate_compatibility(p_a, p_b_same)
        compat_diff = self.selector.evaluate_compatibility(p_a, p_b_diff)
        self.assertTrue(compat_same.same_command)
        self.assertFalse(compat_diff.same_command)

    # 8. Field count difference
    def test_08_field_count_difference(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"  # 3 fields
        p_b = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"  # 5 fields
        compat = self.selector.evaluate_compatibility(p_a, p_b)
        self.assertEqual(compat.field_count_difference, 2)

    # 9. Depth difference
    def test_09_depth_difference(self) -> None:
        rec_a = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1", name="a", coverage_units={"DEPTH_1", "state_1"})
        rec_b = SeedRecord(id=2, data=b"NF01|DIAG|STEP=1|STEP=2", name="b", coverage_units={"DEPTH_5", "state_2"})
        compat = self.selector.evaluate_compatibility(rec_a, rec_b)
        self.assertEqual(compat.depth_difference, 4)

    # 10. Donor extra fields
    def test_10_donor_extra_fields(self) -> None:
        # Prefix length = 3, Donor total fields = 5 -> extra = 2
        p_a = b"NF01|DIAG|STEP=1"
        p_b = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"
        compat = self.selector.evaluate_compatibility(p_a, p_b)
        self.assertEqual(compat.donor_extra_fields, 2)

        # Donor has fewer fields than prefix (donor is shorter)
        p_a_long = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"
        p_b_short = b"NF01|DIAG|STEP=1"
        compat_short = self.selector.evaluate_compatibility(p_a_long, p_b_short)
        self.assertEqual(compat_short.donor_extra_fields, 0)

    # 11. Compatibility score calculation
    def test_11_compatibility_score_calculation(self) -> None:
        # Parent A: NF01|DIAG|STEP=1 (3 fields)
        # Donor B: NF01|DIAG|STEP=1|STEP=2|STEP=3 (5 fields)
        # prefix=3, same_cmd=1, same_proto=1, extra=2, field_diff=2, depth_diff=0
        # Score = 3.0*3 + 2.0*1 + 1.0*1 + 1.0*2 - 0.5*2 - 0.25*0 = 9 + 2 + 1 + 2 - 1 = 13.0
        p_a = b"NF01|DIAG|STEP=1"
        p_b = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"
        compat = self.selector.evaluate_compatibility(p_a, p_b)
        self.assertAlmostEqual(compat.compatibility_score, 13.0, places=4)

    # 12. Highest-score selection
    def test_12_highest_score_selection(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        low_donor = b"NF01|AUTH|USER=admin"
        high_donor = b"NF01|DIAG|STEP=1|STEP=2"
        corpus = [low_donor, high_donor]
        chosen = self.selector.select_donor(p_a, corpus)
        self.assertEqual(chosen, high_donor)

    # 13. Tie handling
    def test_13_tie_handling_deterministic(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        # Two donors with identical structural profiles
        cand1 = b"NF01|DIAG|STEP=X"
        cand2 = b"NF01|DIAG|STEP=Y"
        corpus = [cand1, cand2]

        sel1 = DonorSelector(mode="compatible", rng=random.Random(123))
        sel2 = DonorSelector(mode="compatible", rng=random.Random(123))
        chosen1 = sel1.select_donor(p_a, corpus)
        chosen2 = sel2.select_donor(p_a, corpus)
        self.assertEqual(chosen1, chosen2)

    # 14. Deterministic RNG reproducibility
    def test_14_deterministic_reproducibility(self) -> None:
        p_a = b"NF01|DATA|VAL=1"
        corpus = [
            b"NF01|CALC|OP=ADD",
            b"NF01|AUTH|USER=root",
            b"NF01|DATA|VAL=2|EXTRA=1",
            b"NF01|PING|SEQ=1",
        ]
        sel1 = DonorSelector(mode="compatible", rng=random.Random(999))
        sel2 = DonorSelector(mode="compatible", rng=random.Random(999))
        history1 = [sel1.select_donor(p_a, corpus) for _ in range(10)]
        history2 = [sel2.select_donor(p_a, corpus) for _ in range(10)]
        self.assertEqual(history1, history2)

    # 15. Random policy behavior
    def test_15_random_policy_behavior(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        corpus = [
            b"NF01|CALC|OP=ADD",
            b"NF01|AUTH|USER=root",
            b"NF01|DIAG|STEP=1|STEP=2",
        ]
        rand_sel = DonorSelector(mode="random", rng=random.Random(42))
        selections = set(rand_sel.select_donor(p_a, corpus) for _ in range(50))
        # In random mode, all valid alternative seeds should be sampled
        self.assertEqual(len(selections), 3)
        self.assertNotIn(p_a, selections)

    # 16. Compatible policy behavior
    def test_16_compatible_policy_behavior(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        corpus = [
            b"NF01|CALC|OP=ADD",
            b"NF01|AUTH|USER=root",
            b"NF01|DIAG|STEP=1|STEP=2",
        ]
        comp_sel = DonorSelector(mode="compatible", rng=random.Random(42))
        # In compatible mode, the DIAG seed is consistently preferred
        for _ in range(20):
            chosen = comp_sel.select_donor(p_a, corpus)
            self.assertEqual(chosen, b"NF01|DIAG|STEP=1|STEP=2")

    # 17. Useful-donor preference (donor_extra_fields > 0)
    def test_17_useful_donor_preference(self) -> None:
        p_a = b"NF01|DIAG|STEP=1|STEP=2"
        # Donor 1: shorter than parent A (0 extra fields)
        d_short = b"NF01|DIAG|STEP=1"
        # Donor 2: longer than parent A (1 extra field)
        d_long = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"
        corpus = [d_short, d_long]

        chosen = self.selector.select_donor(p_a, corpus)
        self.assertEqual(chosen, d_long)

    # 18. Malformed / binary input handling
    def test_18_malformed_binary_input(self) -> None:
        p_a = b"\x00\xff\xfe|RANDOM\x00BYTES"
        corpus = [
            b"",
            b"|||",
            b"\x80\x90\xaa",
            b"VALID|FORMAT|A",
        ]
        # Must not raise exceptions
        chosen = self.selector.select_donor(p_a, corpus)
        self.assertIsNotNone(chosen)
        compat = self.selector.evaluate_compatibility(p_a, b"\x00\x01\x02")
        self.assertIsInstance(compat.compatibility_score, float)

    # 19. No NaN / infinite score
    def test_19_no_nan_or_infinite_score(self) -> None:
        p_a = b"A" * 1000
        p_b = b"B" * 2000
        compat = self.selector.evaluate_compatibility(p_a, p_b)
        self.assertFalse(compat.compatibility_score != compat.compatibility_score)  # not NaN
        self.assertFalse(abs(compat.compatibility_score) == float("inf"))

    # 20. Integration with field_splice
    def test_20_integration_with_field_splice(self) -> None:
        p_a = b"NF01|DIAG|STEP=1"
        corpus = [
            b"NF01|AUTH|USER=admin",
            b"NF01|DIAG|STEP=1|STEP=2",
        ]
        splicer = FieldSplicer(delimiter=b"|", rng=random.Random(42))
        chosen_donor = self.selector.select_donor(p_a, corpus)
        self.assertEqual(chosen_donor, b"NF01|DIAG|STEP=1|STEP=2")

        spliced = splicer.splice(p_a, chosen_donor)
        self.assertIsNotNone(spliced)
        self.assertTrue(spliced.startswith(b"NF01|DIAG|STEP=1"))

    # 21. Synthetic Donor Selection Test
    def test_21_synthetic_donor_selection_scenario(self) -> None:
        """Synthetic benchmark scenario: Parent A is NF01|DIAG|STEP=1.

        Corpus candidates:
        - B1: NF01|CALC|OP=ADD|NUM=10
        - B2: NF01|DIAG|STEP=1|STEP=2|STEP=3
        - B3: NF01|AUTH|USER=admin

        Assert that compatible mode chooses B2 decisively without hardcoding command names.
        """
        parent_a = b"NF01|DIAG|STEP=1"
        b1 = b"NF01|CALC|OP=ADD|NUM=10"
        b2 = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"
        b3 = b"NF01|AUTH|USER=admin"

        corpus = [b1, b2, b3]

        c_b1 = self.selector.evaluate_compatibility(parent_a, b1)
        c_b2 = self.selector.evaluate_compatibility(parent_a, b2)
        c_b3 = self.selector.evaluate_compatibility(parent_a, b3)

        # Verify score ordering
        self.assertGreater(c_b2.compatibility_score, c_b1.compatibility_score)
        self.assertGreater(c_b2.compatibility_score, c_b3.compatibility_score)

        # Verify selection
        chosen = self.selector.select_donor(parent_a, corpus)
        self.assertEqual(chosen, b2)

        # Verify telemetry
        stats = self.selector.get_telemetry()
        self.assertEqual(stats["total_attempts"], 1)
        self.assertEqual(stats["successful_selections"], 1)
        self.assertEqual(stats["diag_diag_pairings"], 1)
        self.assertEqual(stats["same_command_pairing_rate"], 1.0)
        self.assertEqual(stats["useful_donor_rate"], 1.0)

    # 22. End-to-end integration test with Fuzzer
    def test_22_fuzzer_end_to_end_compatible(self) -> None:
        """Verify Fuzzer runs cleanly with compatible donor policy and structured target."""
        from pathlib import Path
        import sys

        target_path = Path("targets/structured_target")
        if sys.platform.startswith("win"):
            target_path = target_path.with_suffix(".exe")

        if not target_path.exists():
            self.skipTest(f"Target binary {target_path} not found.")

        fuzzer = Fuzzer(
            target_path=target_path,
            corpus_dir="corpus_structured",
            iterations=40,
            seed=42,
            stats_interval=20,
            calibrate=True,
            scheduler_type="random",
            enable_splicing=True,
            splice_probability=0.5,
            donor_policy="compatible",
        )
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 40)
        self.assertEqual(stats.donor_policy, "compatible")
        self.assertIn("total_attempts", stats.donor_telemetry)
        self.assertGreaterEqual(stats.donor_telemetry["total_attempts"], 0)


if __name__ == "__main__":
    unittest.main()
