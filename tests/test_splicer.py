"""Unit and integration tests for Phase 6E: Structured Crossover & Field Splicing."""

from pathlib import Path
import random
import sys
import unittest

from fuzzer.corpus import Corpus
from fuzzer.executor import Executor
from fuzzer.fuzzer import Fuzzer
from fuzzer.mutation_bandit import get_default_mutation_arms
from fuzzer.mutator import Mutator, op_field_splice
from fuzzer.splicer import FieldSplicer, SplicerTelemetry


class TestFieldSplicer(unittest.TestCase):
    """20 detailed unit tests for FieldSplicer components and edge cases."""

    def setUp(self) -> None:
        self.rng = random.Random(42)
        self.splicer = FieldSplicer(delimiter=b"|", max_splice_size=512, rng=self.rng)

    def test_01_basic_field_parsing(self) -> None:
        """1. Basic field parsing: splits on delimiter and discards empty parts."""
        data = b"NF01|DIAG|STEP=1"
        fields = self.splicer.split_fields(data)
        self.assertEqual(fields, [b"NF01", b"DIAG", b"STEP=1"])

    def test_02_empty_input(self) -> None:
        """2. Empty input: returns empty list."""
        self.assertEqual(self.splicer.split_fields(b""), [])

    def test_03_single_field_input(self) -> None:
        """3. Single field input: returns single element list."""
        self.assertEqual(self.splicer.split_fields(b"NF01"), [b"NF01"])

    def test_04_two_compatible_diag_seeds(self) -> None:
        """4. Splicing two compatible DIAG seeds produces valid structured output."""
        parent_a = b"NF01|DIAG|STEP=1"
        parent_b = b"NF01|DIAG|STEP=2"
        candidate = self.splicer.splice(parent_a, parent_b)
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.startswith(b"NF01|DIAG"))
        self.assertNotEqual(candidate, parent_a)

    def test_05_common_prefix_detection(self) -> None:
        """5. Common prefix detection finds identical leading fields."""
        fields_a = [b"NF01", b"DIAG", b"STEP=1"]
        fields_b = [b"NF01", b"DIAG", b"STEP=2"]
        prefix = self.splicer.find_common_prefix(fields_a, fields_b)
        self.assertEqual(prefix, [b"NF01", b"DIAG"])

        # No common prefix
        fields_c = [b"PING", b"123"]
        self.assertEqual(self.splicer.find_common_prefix(fields_a, fields_c), [])

    def test_06_prefix_suffix_splice(self) -> None:
        """6. Strategy A: Prefix/suffix splice combines prefix with donor suffix."""
        fields_a = [b"NF01", b"DIAG", b"STEP=1"]
        fields_b = [b"NF01", b"DIAG", b"STEP=2", b"STEP=3"]
        result = self.splicer.splice_prefix_suffix(fields_a, fields_b)
        self.assertIsNotNone(result)
        # Prefix is [NF01, DIAG], suffix in B is [STEP=2, STEP=3]
        self.assertEqual(result[:2], [b"NF01", b"DIAG"])
        self.assertTrue(result[2] in (b"STEP=2", b"STEP=3"))

    def test_07_single_field_append(self) -> None:
        """7. Strategy B: Single field append appends a field without immediate duplicate."""
        fields_a = [b"NF01", b"CALC", b"OP=ADD"]
        fields_b = [b"NF01", b"CALC", b"NUM=10", b"DEN=2"]
        result = self.splicer.splice_single_field_append(fields_a, fields_b)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[:3], fields_a)
        self.assertIn(result[3], fields_b)
        self.assertNotEqual(result[3], result[2])

    def test_08_segment_splice(self) -> None:
        """8. Strategy C: Field segment splice extracts contiguous donor segment."""
        fields_a = [b"NF01", b"WRITE", b"MODE=APPEND"]
        fields_b = [b"NF01", b"WRITE", b"FILE=log.txt", b"SIZE=64"]
        result = self.splicer.splice_field_segment(fields_a, fields_b)
        self.assertIsNotNone(result)
        self.assertTrue(len(result) >= 3)
        self.assertEqual(result[0], b"NF01")
        self.assertEqual(result[1], b"WRITE")

    def test_09_duplicate_prevention(self) -> None:
        """9. Adjacent duplicate fields are collapsed and recorded in telemetry."""
        fields = [b"NF01", b"DIAG", b"STEP=1", b"STEP=1", b"STEP=2"]
        deduped = self.splicer.deduplicate_adjacent(fields)
        self.assertEqual(deduped, [b"NF01", b"DIAG", b"STEP=1", b"STEP=2"])
        self.assertGreaterEqual(self.splicer.telemetry.duplicate_prevention_events, 1)

    def test_10_different_command_types(self) -> None:
        """10. Splicing across different commands (e.g. PING and CALC) succeeds gracefully."""
        parent_a = b"NF01|PING|hello"
        parent_b = b"NF01|CALC|OP=DIV|NUM=10|DEN=2"
        candidate = self.splicer.splice(parent_a, parent_b)
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.startswith(b"NF01"))
        self.assertNotEqual(candidate, parent_a)

    def test_11_parent_a_equals_parent_b(self) -> None:
        """11. Splicing identical parents is rejected and returns None."""
        parent = b"NF01|DIAG|STEP=1"
        self.assertIsNone(self.splicer.splice(parent, parent))
        self.assertGreaterEqual(self.splicer.telemetry.rejected_splices, 1)

    def test_12_one_seed_corpus_fallback(self) -> None:
        """12. One-seed corpus returns None for donor selection."""
        parent_a = b"NF01|DIAG|STEP=1"
        corpus = [parent_a]
        donor = self.splicer.select_donor(corpus, parent_a)
        self.assertIsNone(donor)

    def test_13_empty_corpus_fallback(self) -> None:
        """13. Empty corpus returns None for donor selection."""
        donor = self.splicer.select_donor([], b"NF01|DIAG|STEP=1")
        self.assertIsNone(donor)

    def test_14_maximum_size_rejection(self) -> None:
        """14. Candidates exceeding max_splice_size are rejected."""
        strict_splicer = FieldSplicer(max_splice_size=15, rng=random.Random(1))
        # 16 bytes payload candidate
        parent_a = b"NF01|DIAG|STEP=1"  # 16 bytes
        parent_b = b"NF01|DIAG|STEP=2"
        candidate = strict_splicer.splice(parent_a, parent_b)
        self.assertIsNone(candidate)
        self.assertGreaterEqual(strict_splicer.telemetry.rejected_splices, 1)

    def test_15_deterministic_rng_reproducibility(self) -> None:
        """15. Deterministic RNG produces identical output across separate runs."""
        s1 = FieldSplicer(rng=random.Random(12345))
        s2 = FieldSplicer(rng=random.Random(12345))
        p_a = b"NF01|DIAG|STEP=1"
        p_b = b"NF01|DIAG|STEP=2|STEP=3"
        c1 = s1.splice(p_a, p_b)
        c2 = s2.splice(p_a, p_b)
        self.assertEqual(c1, c2)

    def test_16_binary_non_utf8_input(self) -> None:
        """16. Non-UTF8 binary byte fields are handled without decode errors."""
        p_a = b"NF01|\xff\xfe\xfd|DATA"
        p_b = b"NF01|\xaa\xbb\xcc|DATA"
        candidate = self.splicer.splice(p_a, p_b)
        self.assertIsNotNone(candidate)
        self.assertIsInstance(candidate, bytes)

    def test_17_no_accidental_delimiter_duplication(self) -> None:
        """17. No accidental consecutive delimiters (b'||') are generated."""
        p_a = b"||NF01|||DIAG||STEP=1||"
        p_b = b"|NF01|DIAG|STEP=2||"
        candidate = self.splicer.splice(p_a, p_b)
        if candidate is not None:
            self.assertNotIn(b"||", candidate)

    def test_18_candidate_differs_from_parent_a(self) -> None:
        """18. Successful candidate is guaranteed to differ from Parent A."""
        p_a = b"NF01|INFO|SYSTEM"
        p_b = b"NF01|INFO|SYSTEM|CPU"
        for _ in range(10):
            res = self.splicer.splice(p_a, p_b)
            if res is not None:
                self.assertNotEqual(res, p_a)

    def test_19_field_ordering_preservation(self) -> None:
        """19. Field ordering is preserved across splicing strategies."""
        fields_a = [b"NF01", b"DIAG", b"STEP=1"]
        fields_b = [b"NF01", b"DIAG", b"STEP=2", b"STEP=3"]
        res = self.splicer.splice_prefix_suffix(fields_a, fields_b)
        self.assertIsNotNone(res)
        # Prefix remains at indices 0 and 1
        self.assertEqual(res[0], b"NF01")
        self.assertEqual(res[1], b"DIAG")

    def test_20_serialization_round_trip(self) -> None:
        """20. Serialization round-trip join_fields(split_fields(x)) == x for clean data."""
        data = b"NF01|DIAG|STEP=1|EXTRA"
        self.assertEqual(self.splicer.join_fields(self.splicer.split_fields(data)), data)


class TestSyntheticSplicing(unittest.TestCase):
    """Synthetic test proving deterministic depth progression via structured splicing."""

    def test_synthetic_diag_depth_progression(self) -> None:
        """Verify Parent A (STEP=1) + Donor B (STEP=1|STEP=2|STEP=3) produces STEP=1|STEP=2."""
        splicer = FieldSplicer(rng=random.Random(10))
        parent_a = b"NF01|DIAG|STEP=1"
        donor_b = b"NF01|DIAG|STEP=1|STEP=2|STEP=3"

        # Explicitly verify prefix_suffix strategy
        fields_a = splicer.split_fields(parent_a)
        fields_b = splicer.split_fields(donor_b)

        # Force k=1 slice of donor suffix [STEP=2, STEP=3]
        prefix = splicer.find_common_prefix(fields_a, fields_b)  # [NF01, DIAG, STEP=1]
        self.assertEqual(prefix, [b"NF01", b"DIAG", b"STEP=1"])

        donor_suffix = fields_b[len(prefix):]  # [STEP=2, STEP=3]
        self.assertEqual(donor_suffix, [b"STEP=2", b"STEP=3"])

        # Combining prefix with first suffix field
        combined = prefix + [donor_suffix[0]]
        result_bytes = splicer.join_fields(splicer.deduplicate_adjacent(combined))
        self.assertEqual(result_bytes, b"NF01|DIAG|STEP=1|STEP=2")

        # Now test through the public splice() interface
        found_target = False
        for s in range(50):
            test_splicer = FieldSplicer(rng=random.Random(s))
            cand = test_splicer.splice(parent_a, donor_b)
            if cand == b"NF01|DIAG|STEP=1|STEP=2":
                found_target = True
                break
        self.assertTrue(found_target, "Splice produced exactly NF01|DIAG|STEP=1|STEP=2")


class TestMutatorSplicingIntegration(unittest.TestCase):
    """Integration of FieldSplicer within Mutator and Fuzzer."""

    def test_op_field_splice_function(self) -> None:
        """Standalone op_field_splice function works correctly."""
        splicer = FieldSplicer(rng=random.Random(42))
        res = op_field_splice(b"NF01|PING|1", random.Random(42), splicer, b"NF01|PING|2")
        self.assertIsNotNone(res)
        self.assertNotEqual(res, b"NF01|PING|1")

    def test_mutator_splicing_operator_registration(self) -> None:
        """Mutator registers field_splice when enable_splicing is True."""
        mut_no_splice = Mutator(enable_splicing=False)
        self.assertNotIn("field_splice", mut_no_splice.operators)

        mut_splice = Mutator(enable_splicing=True)
        self.assertIn("field_splice", mut_splice.operators)
        self.assertIsNotNone(mut_splice.splicer)

    def test_mutator_mutate_with_operator_field_splice(self) -> None:
        """Calling mutate_with_operator with field_splice returns modified bytes."""
        seeds = [b"NF01|DIAG|STEP=1", b"NF01|DIAG|STEP=2"]
        mutator = Mutator(
            enable_splicing=True,
            corpus_provider=lambda: seeds,
            rng=random.Random(42),
        )
        mutated, op_used = mutator.mutate_with_operator(seeds[0], operator_name="field_splice")
        self.assertEqual(op_used, "field_splice")
        self.assertNotEqual(mutated, seeds[0])

    def test_default_mutation_arms_with_splicing(self) -> None:
        """get_default_mutation_arms includes field_splice when requested."""
        arms_without = get_default_mutation_arms(has_dictionary=True, boundary_aware=True, include_splicing=False)
        self.assertEqual(len(arms_without), 6)
        self.assertNotIn("field_splice", arms_without)

        arms_with = get_default_mutation_arms(has_dictionary=True, boundary_aware=True, include_splicing=True)
        self.assertEqual(len(arms_with), 7)
        self.assertIn("field_splice", arms_with)


class TestFuzzerSplicingEndToEnd(unittest.TestCase):
    """End-to-end execution of Fuzzer with splicing enabled across policies."""

    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        target_name = "structured_target.exe" if sys.platform.startswith("win") else "structured_target"
        cls.target_path = base_dir / "targets" / target_name
        cls.corpus_dir = base_dir / "corpus_structured"
        cls.dict_path = base_dir / "dictionaries" / "structured_protocol.dict"

    def test_fuzzer_splicing_fixed_policy(self) -> None:
        """End-to-end run under fixed policy with splicing enabled."""
        if not self.target_path.exists():
            self.skipTest(f"Target binary not found at {self.target_path}")

        fuzzer = Fuzzer(
            target_path=self.target_path,
            corpus_dir=self.corpus_dir,
            iterations=25,
            seed=42,
            dictionary_path=self.dict_path,
            dictionary_probability=0.3,
            boundary_aware=True,
            mutation_policy="fixed",
            enable_splicing=True,
            splice_probability=0.2,
        )
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 25)
        self.assertTrue(stats.enable_splicing)
        self.assertGreaterEqual(stats.splice_mutations, 1)

    def test_fuzzer_splicing_ucb1_policy(self) -> None:
        """End-to-end run under UCB1 bandit policy with 7 arms."""
        if not self.target_path.exists():
            self.skipTest(f"Target binary not found at {self.target_path}")

        fuzzer = Fuzzer(
            target_path=self.target_path,
            corpus_dir=self.corpus_dir,
            iterations=25,
            seed=42,
            dictionary_path=self.dict_path,
            boundary_aware=True,
            mutation_policy="ucb1",
            enable_splicing=True,
        )
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 25)
        self.assertIn("field_splice", fuzzer.mutation_bandit.operators)
        self.assertGreaterEqual(fuzzer.mutation_bandit.counts["field_splice"], 1)

    def test_fuzzer_splicing_contextual_policy(self) -> None:
        """End-to-end run under Contextual bandit policy with 7 arms."""
        if not self.target_path.exists():
            self.skipTest(f"Target binary not found at {self.target_path}")

        fuzzer = Fuzzer(
            target_path=self.target_path,
            corpus_dir=self.corpus_dir,
            iterations=25,
            seed=42,
            dictionary_path=self.dict_path,
            boundary_aware=True,
            mutation_policy="contextual",
            enable_splicing=True,
        )
        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 25)
        self.assertIn("field_splice", fuzzer.contextual_mutation_bandit.operators)
        self.assertGreaterEqual(fuzzer.contextual_mutation_bandit.counts["field_splice"], 1)


if __name__ == "__main__":
    unittest.main()
