"""Unit tests for Phase 7C: State-Sequence-Aware Frontier Mutation."""

from collections import deque
import random
import unittest
from pathlib import Path

from fuzzer.boundary import BoundaryDetector
from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.coverage import CoverageTracker
from fuzzer.dictionary import Dictionary
from fuzzer.mutator import Mutator, op_state_frontier_extend
from fuzzer.scheduler import RandomScheduler, HeuristicScheduler, LinearBanditScheduler
from fuzzer.state_frontier import (
    FrontierCandidate,
    FrontierSeed,
    FrontierTelemetry,
    StateFrontier,
)
from fuzzer.fuzzer import Fuzzer


class TestStateFrontierCore(unittest.TestCase):
    """Core tests for StateFrontier tracking, candidates, and bookkeeping."""

    def setUp(self) -> None:
        self.dict_tokens = [
            b"STEP=1",
            b"STEP=2",
            b"STEP=3",
            b"STEP=4",
            b"|STEP=1",
            b"|STEP=2",
            b"|STEP=3",
            b"|STEP=4",
            b"EXEC=CRASH",
            b"NEST=L2",
        ]
        self.dictionary = Dictionary(tokens=self.dict_tokens)
        self.detector = BoundaryDetector(delimiter=b"|")
        self.frontier = StateFrontier(
            dictionary=self.dictionary,
            delimiter=b"|",
            boundary_detector=self.detector,
            rng=random.Random(42),
            max_candidates_per_seed=100,
            priority_burst_budget=20,
        )

    def test_frontier_creation(self) -> None:
        """1. Frontier creation: verify initial empty state and properties."""
        self.assertEqual(self.frontier.max_depth, 0)
        self.assertIsNone(self.frontier.deepest_seed)
        self.assertFalse(self.frontier.has_candidate())
        self.assertEqual(self.frontier.queue_length, 0)
        self.assertEqual(self.frontier.telemetry.deepest_frontier_reached, 0)

    def test_deepest_seed_tracking(self) -> None:
        """2. Deepest-seed tracking: frontier tracks max depth across registered seeds."""
        rec1 = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        rec2 = SeedRecord(id=2, data=b"NF01|PING\n", name="seed2", coverage_units={"DEPTH_3"})

        # Register lower depth first
        self.frontier.register_seed(rec2, depth=3)
        self.assertEqual(self.frontier.max_depth, 3)
        self.assertEqual(self.frontier.deepest_seed.name, "seed2")

        # Register higher depth
        self.frontier.register_seed(rec1, depth=14)
        self.assertEqual(self.frontier.max_depth, 14)
        self.assertEqual(self.frontier.deepest_seed.name, "seed1")

    def test_frontier_update_after_new_depth(self) -> None:
        """3. Frontier update after new depth: depth increases frontier and triggers priority burst."""
        rec1 = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        self.frontier.register_seed(rec1, depth=14, iteration=10)

        self.assertTrue(self.frontier.has_candidate())
        self.assertTrue(self.frontier.is_priority_active())
        self.assertGreater(self.frontier.queue_length, 0)

    def test_candidate_generation(self) -> None:
        """4. Candidate generation: generates valid untried candidates."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        f_seed = FrontierSeed(record=rec, depth=14, discovery_iteration=0)

        candidates = self.frontier.generate_candidates(f_seed)
        self.assertGreater(len(candidates), 0)
        for cand in candidates:
            self.assertEqual(cand.parent_depth, 14)
            self.assertEqual(cand.seed_name, "seed1")
            self.assertNotEqual(cand.mutated_data, rec.data)
            self.assertTrue(cand.candidate_key)

    def test_boundary_aware_insertion(self) -> None:
        """5. Boundary-aware insertion: candidate mutations preserve prefix and insert properly."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        self.frontier.register_seed(rec, depth=14)

        # Inspect generated candidates
        candidates = list(self.frontier._candidate_queue)
        found_step2_trailing = False
        for c in candidates:
            # Trailing insertions before newline should yield proper sequence extension
            if c.mutated_data == b"NF01|DIAG|STEP=1|STEP=2\n":
                found_step2_trailing = True
            # Verify candidate is a non-trivial mutation
            self.assertNotEqual(c.mutated_data, rec.data)
        self.assertTrue(found_step2_trailing, "Expected candidate: NF01|DIAG|STEP=1|STEP=2\\n")

    def test_duplicate_candidate_prevention(self) -> None:
        """6. Duplicate candidate prevention: avoids duplicate keys or payloads."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        f_seed = FrontierSeed(record=rec, depth=14, discovery_iteration=0)

        cands1 = self.frontier.generate_candidates(f_seed)
        count1 = len(cands1)

        # Generating again on same seed should yield 0 new candidates due to deduplication
        cands2 = self.frontier.generate_candidates(f_seed)
        self.assertEqual(len(cands2), 0)
        self.assertGreater(count1, 0)

    def test_candidate_bookkeeping(self) -> None:
        """7. Candidate bookkeeping: tracks attempts, discoveries, and telemetry."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        self.frontier.register_seed(rec, depth=14)

        # Pop candidate
        parent_rec, mutated, op, cand = self.frontier.next_candidate()
        self.assertEqual(op, "state_frontier_extend")
        self.assertEqual(self.frontier.telemetry.frontier_extension_attempts, 1)

        # Report feedback: new coverage and new depth
        self.frontier.on_execution_result(
            cand,
            result_coverage=["DEPTH_15", "PATH_DEEP_STATE_2"],
            is_new_cov=True,
            new_units={"DEPTH_15", "PATH_DEEP_STATE_2"},
            iteration=5,
        )

        self.assertEqual(self.frontier.telemetry.frontier_discoveries, 1)
        self.assertEqual(self.frontier.telemetry.frontier_coverage_discovered, 2)
        self.assertEqual(self.frontier.telemetry.frontier_depth_discoveries, 1)
        self.assertIn("14->15", self.frontier.telemetry.depth_transitions)

    def test_frontier_exhaustion(self) -> None:
        """8. Frontier exhaustion: pops all candidates gracefully until queue empty."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG\n", name="seed1", coverage_units={"DEPTH_6"})
        self.frontier.register_seed(rec, depth=6)

        popped = 0
        while self.frontier.has_candidate():
            res = self.frontier.next_candidate()
            self.assertIsNotNone(res)
            popped += 1

        self.assertFalse(self.frontier.has_candidate())
        self.assertIsNone(self.frontier.next_candidate())
        self.assertGreater(popped, 0)
        self.assertGreaterEqual(self.frontier.telemetry.frontier_candidates_exhausted, 1)

    def test_new_deeper_seed_becoming_new_frontier(self) -> None:
        """9. New deeper seed becoming the new frontier: queue shifts to new frontier."""
        rec14 = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed14", coverage_units={"DEPTH_14"})
        self.frontier.register_seed(rec14, depth=14)
        self.assertEqual(self.frontier.max_depth, 14)

        # Simulate discovery of depth 15
        rec15 = SeedRecord(
            id=2,
            data=b"NF01|DIAG|STEP=1|STEP=2\n",
            name="seed15",
            coverage_units={"DEPTH_15", "PATH_DEEP_STATE_2"},
        )
        self.frontier.register_seed(rec15, depth=15, iteration=25)
        self.assertEqual(self.frontier.max_depth, 15)
        self.assertEqual(self.frontier.deepest_seed.name, "seed15")

        # Next popped candidate should belong to parent depth 15
        p_rec, mutated, op, cand = self.frontier.next_candidate()
        self.assertEqual(cand.parent_depth, 15)
        self.assertEqual(cand.seed_name, "seed15")

    def test_reset_isolation(self) -> None:
        """10. Reset/isolation: clears all internal queues and state."""
        rec = SeedRecord(id=1, data=b"NF01|DIAG|STEP=1\n", name="seed1", coverage_units={"DEPTH_14"})
        self.frontier.register_seed(rec, depth=14)
        self.frontier.next_candidate()

        self.frontier.reset()
        self.assertEqual(self.frontier.max_depth, 0)
        self.assertIsNone(self.frontier.deepest_seed)
        self.assertFalse(self.frontier.has_candidate())
        self.assertEqual(self.frontier.queue_length, 0)
        self.assertEqual(self.frontier.telemetry.frontier_extension_attempts, 0)


class TestMutatorFrontierIntegration(unittest.TestCase):
    """Verify mutator registration and fallback operator behavior."""

    def test_existing_mutation_operators_still_work(self) -> None:
        """11. Existing mutation operators still work."""
        mutator = Mutator(rng=random.Random(42))
        self.assertIn("flip_bit", mutator.operators)
        self.assertIn("replace_byte", mutator.operators)
        self.assertIn("insert_byte", mutator.operators)
        self.assertIn("delete_byte", mutator.operators)
        self.assertIn("dictionary_insert_boundary", mutator.operators)
        self.assertIn("dictionary_replace", mutator.operators)
        self.assertIn("state_frontier_extend", mutator.operators)

        # Mutate with standard operators
        data = b"NF01|TEST|HELLO\n"
        for op in ["flip_bit", "replace_byte", "insert_byte", "delete_byte"]:
            res, used_op = mutator.mutate_with_operator(data, operator_name=op)
            self.assertEqual(used_op, op)
            self.assertIsInstance(res, bytes)

    def test_state_frontier_extend_operator(self) -> None:
        """Verify op_state_frontier_extend behaves correctly with dictionary."""
        d = Dictionary(tokens=[b"STEP=2", b"|STEP=2"])
        mutator = Mutator(dictionary=d, boundary_aware=True, rng=random.Random(42))
        data = b"NF01|DIAG|STEP=1\n"
        res, op_name = mutator.mutate_with_operator(data, operator_name="state_frontier_extend")
        self.assertEqual(op_name, "state_frontier_extend")
        self.assertIsInstance(res, bytes)
        self.assertNotEqual(res, data)


class TestSchedulersCompatibility(unittest.TestCase):
    """Verify existing schedulers still operate without changes."""

    def test_existing_schedulers_still_work(self) -> None:
        """12. Existing schedulers still work."""
        corpus = Corpus()
        corpus.add_seed(b"NF01|PING\n", name="s1")
        corpus.add_seed(b"NF01|DIAG\n", name="s2")

        r_sched = RandomScheduler(rng=random.Random(42))
        h_sched = HeuristicScheduler(rng=random.Random(42))
        b_sched = LinearBanditScheduler(rng=random.Random(42))

        for sched in [r_sched, h_sched, b_sched]:
            chosen = sched.select_seed(corpus, iteration=1)
            self.assertIn(chosen, corpus.records)
            sched.update(chosen, reward=1, iteration=1)


class TestFuzzerFrontierSmoke(unittest.TestCase):
    """Verify Fuzzer runs cleanly with enable_frontier_extension=True."""

    def test_fuzzer_frontier_smoke_run(self) -> None:
        """Run a quick deterministic fuzzing loop with state frontier extension enabled."""
        target_path = Path("targets/structured_target.exe").resolve()
        if not target_path.exists():
            self.skipTest("structured_target.exe not compiled")

        fuzzer = Fuzzer(
            target_path=target_path,
            corpus_dir="corpus_structured",
            iterations=20,
            timeout=1.0,
            seed=101,
            stats_interval=100,
            calibrate=True,
            scheduler_type="random",
            dictionary_path="dictionaries/structured_protocol.dict",
            dictionary_probability=0.3,
            boundary_aware=True,
            enable_frontier_extension=True,
            frontier_extension_rate=0.5,
            executor_type="persistent",
        )

        stats = fuzzer.run()
        self.assertEqual(stats.total_executions, 20)
        self.assertGreaterEqual(stats.total_coverage_units, 41)
        self.assertTrue(stats.enable_frontier_extension)
        self.assertIsNotNone(fuzzer.state_frontier)
        self.assertGreaterEqual(fuzzer.state_frontier.max_depth, 14)


if __name__ == "__main__":
    unittest.main()
