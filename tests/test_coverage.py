"""Unit tests for Phase 2 coverage tracking, parsing, and corpus growth."""

from pathlib import Path
import tempfile
import unittest

from fuzzer.corpus import Corpus
from fuzzer.coverage import CoverageTracker, parse_coverage_markers
from fuzzer.executor import CrashSaver, ExecutionResult


class TestCoverageParsing(unittest.TestCase):
    """Test suite for parsing coverage markers from output streams."""

    def test_parse_coverage_markers(self) -> None:
        """Verify __COV__: markers are extracted and non-marker lines preserved."""
        raw_stream = (
            b"[Target] Starting up...\n"
            b"__COV__:PATH_ENTRY\n"
            b"__COV__:PATH_HDR_FUZZ\n"
            b"[Target] In FUZZ branch\n"
            b"__COV__:PATH_FUZZ_CMD_ECHO\n"
            b"Warning: unhandled flag\n"
        )
        cov_units, cleaned = parse_coverage_markers(raw_stream)

        self.assertEqual(cov_units, {"PATH_ENTRY", "PATH_HDR_FUZZ", "PATH_FUZZ_CMD_ECHO"})
        self.assertNotIn(b"__COV__:", cleaned)
        self.assertIn(b"[Target] Starting up...\n", cleaned)
        self.assertIn(b"[Target] In FUZZ branch\n", cleaned)
        self.assertIn(b"Warning: unhandled flag\n", cleaned)

    def test_parse_empty_stream(self) -> None:
        """Verify empty stream yields empty coverage set and empty cleaned bytes."""
        cov_units, cleaned = parse_coverage_markers(b"")
        self.assertEqual(cov_units, set())
        self.assertEqual(cleaned, b"")

    def test_parse_duplicate_markers(self) -> None:
        """Verify duplicate coverage markers in stream are deduplicated in set."""
        raw = b"__COV__:PATH_LOOP\n__COV__:PATH_LOOP\n__COV__:PATH_LOOP\n"
        cov_units, cleaned = parse_coverage_markers(raw)
        self.assertEqual(cov_units, {"PATH_LOOP"})
        self.assertEqual(cleaned, b"")


class TestCoverageTracker(unittest.TestCase):
    """Test suite for CoverageTracker."""

    def setUp(self) -> None:
        self.tracker = CoverageTracker()

    def test_initial_state(self) -> None:
        """Verify tracker starts empty."""
        self.assertEqual(self.tracker.total_units, 0)
        self.assertEqual(self.tracker.total_discoveries, 0)
        self.assertEqual(len(self.tracker), 0)

    def test_first_discovery(self) -> None:
        """Verify first observation registers all units as new."""
        is_new, new_units = self.tracker.update({"PATH_A", "PATH_B"})
        self.assertTrue(is_new)
        self.assertEqual(new_units, {"PATH_A", "PATH_B"})
        self.assertEqual(self.tracker.total_units, 2)
        self.assertEqual(self.tracker.total_discoveries, 1)

    def test_duplicate_rejection(self) -> None:
        """Verify identical observed coverage is rejected as no new coverage."""
        self.tracker.update({"PATH_A", "PATH_B"})

        is_new, new_units = self.tracker.update({"PATH_A", "PATH_B"})
        self.assertFalse(is_new)
        self.assertEqual(new_units, set())
        self.assertEqual(self.tracker.total_units, 2)
        self.assertEqual(self.tracker.total_discoveries, 1)

    def test_partial_new_coverage(self) -> None:
        """Verify observation with both old and new units returns only new units."""
        self.tracker.update({"PATH_A", "PATH_B"})

        is_new, new_units = self.tracker.update({"PATH_B", "PATH_C"})
        self.assertTrue(is_new)
        self.assertEqual(new_units, {"PATH_C"})
        self.assertEqual(self.tracker.total_units, 3)
        self.assertEqual(self.tracker.total_discoveries, 2)

    def test_has_and_get_new_coverage(self) -> None:
        """Verify query methods without modifying tracker state."""
        self.tracker.update({"PATH_1"})

        self.assertTrue(self.tracker.has_new_coverage({"PATH_1", "PATH_2"}))
        self.assertFalse(self.tracker.has_new_coverage({"PATH_1"}))

        diff = self.tracker.get_new_coverage({"PATH_1", "PATH_2", "PATH_3"})
        self.assertEqual(diff, {"PATH_2", "PATH_3"})
        # Tracker global coverage was not modified by get_new_coverage
        self.assertEqual(self.tracker.total_units, 1)


class TestCoverageGuidedCorpus(unittest.TestCase):
    """Test suite for corpus growth upon coverage discovery."""

    def test_corpus_growth_on_new_coverage(self) -> None:
        """Verify interesting inputs are added to corpus with metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "seed1.txt").write_bytes(b"INIT_SEED")
            corpus = Corpus(corpus_dir=tmp_path)
            self.assertEqual(len(corpus), 1)

            # Add input discovering new coverage
            saved_file = corpus.add_interesting_input(
                data=b"DISCOVERED_INPUT",
                coverage_units={"PATH_NEW_BRANCH"},
                iteration=42,
                save_to_disk=True,
            )

            self.assertEqual(len(corpus), 2)
            self.assertIn(b"DISCOVERED_INPUT", corpus.seeds)
            self.assertIsNotNone(saved_file)
            self.assertTrue(saved_file.exists())
            self.assertEqual(saved_file.read_bytes(), b"DISCOVERED_INPUT")

            # Check metadata file
            meta_file = saved_file.with_name(saved_file.stem + "_meta.json")
            self.assertTrue(meta_file.exists())

    def test_non_interesting_input_rejected_from_corpus(self) -> None:
        """Verify workflow where redundant input is discarded (not added to corpus)."""
        tracker = CoverageTracker()
        corpus = Corpus()
        corpus.add_seed(b"SEED_1")

        # Establish baseline
        tracker.update({"BRANCH_1"})

        # Candidate input 1 discovers BRANCH_2
        cand1_cov = {"BRANCH_1", "BRANCH_2"}
        is_new1, new_units1 = tracker.update(cand1_cov)
        if is_new1:
            corpus.add_interesting_input(b"CAND_1", new_units1, iteration=1)

        self.assertEqual(len(corpus), 2)

        # Candidate input 2 triggers only BRANCH_1 and BRANCH_2 (no new units)
        cand2_cov = {"BRANCH_1", "BRANCH_2"}
        is_new2, new_units2 = tracker.update(cand2_cov)
        if is_new2:
            corpus.add_interesting_input(b"CAND_2", new_units2, iteration=2)

        # Corpus should NOT grow
        self.assertEqual(len(corpus), 2)
        self.assertNotIn(b"CAND_2", corpus.seeds)

    def test_crash_still_saved_regardless_of_coverage(self) -> None:
        """Verify crashes are saved independently to crashes/ even without new coverage."""
        with tempfile.TemporaryDirectory() as tmp_crashes:
            saver = CrashSaver(output_dir=tmp_crashes)
            crashing_input = b"CRASH_WITHOUT_NEW_COV"

            result = ExecutionResult(
                returncode=-11,
                stdout=b"",
                stderr=b"segfault",
                timeout=False,
                duration=0.01,
                is_crash=True,
                failure_type="signal",
                description="SIGSEGV",
                coverage={"BRANCH_1"},  # known coverage
            )

            is_new, saved_path = saver.save(crashing_input, result, iteration=99)
            self.assertTrue(is_new)
            self.assertIsNotNone(saved_path)
            self.assertTrue(saved_path.exists())
            self.assertEqual(saved_path.read_bytes(), crashing_input)


if __name__ == "__main__":
    unittest.main()
