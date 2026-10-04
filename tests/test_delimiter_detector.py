"""Unit tests for DelimiterDetector and delimiter generalization (Phase 7D)."""

import unittest

from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.delimiter_detector import (
    STANDARD_DELIMITER_PRIORS,
    DelimiterCandidate,
    DelimiterDetector,
)
from fuzzer.boundary import BoundaryDetector
from fuzzer.mutator import Mutator


class TestDelimiterDetector(unittest.TestCase):
    """Test automated delimiter detection and ranking."""

    def setUp(self) -> None:
        self.detector = DelimiterDetector(default_delimiter=b"|")

    def test_detect_pipe_delimiter(self) -> None:
        """Detect pipe delimiter in typical structured messages."""
        inputs = [
            b"NF01|PING",
            b"NF01|AUTH|USER=admin|ROLE=guest",
            b"NF01|DIAG|STEP=1|STEP=2|STEP=3",
            b"NF01|CALC|OP=ADD|NUM=10|DEN=2",
        ]
        best = self.detector.detect_from_bytes(inputs)
        self.assertEqual(best.delimiter, b"|")
        self.assertGreater(best.confidence, 0.80)
        self.assertEqual(best.presence_ratio, 1.0)
        self.assertGreater(best.frequency_per_seed, 1.0)

    def test_detect_semicolon_delimiter(self) -> None:
        """Detect semicolon delimiter in semicolon-separated commands."""
        inputs = [
            b"CMD=LOGIN;USER=alice;ROLE=user",
            b"CMD=STATUS;ID=100;VERBOSE=1",
            b"CMD=LOGOUT;SESSION=abcde12345",
        ]
        best = self.detector.detect_from_bytes(inputs)
        self.assertEqual(best.delimiter, b";")
        self.assertGreater(best.confidence, 0.70)
        self.assertEqual(best.presence_ratio, 1.0)

    def test_detect_comma_delimiter(self) -> None:
        """Detect comma delimiter in CSV-style records."""
        inputs = [
            b"id,name,role,department",
            b"1,alice,engineering,core",
            b"2,bob,security,infra",
            b"3,charlie,research,ai",
        ]
        best = self.detector.detect_from_bytes(inputs)
        self.assertEqual(best.delimiter, b",")
        self.assertGreater(best.confidence, 0.70)

    def test_detect_colon_delimiter(self) -> None:
        """Detect colon delimiter in colon-separated protocol."""
        inputs = [
            b"header:payload:checksum:status",
            b"user:admin:password:hash",
            b"route:v1:endpoints:list",
        ]
        best = self.detector.detect_from_bytes(inputs)
        self.assertEqual(best.delimiter, b":")
        self.assertGreater(best.confidence, 0.60)

    def test_fallback_on_unstructured_inputs(self) -> None:
        """Fallback safely to default when inputs are unstructured random bytes."""
        inputs = [
            b"HELLOWORLD",
            b"ABCDEF123456",
            b"BINARYDATA\x00\x01\x02\x03",
        ]
        best = self.detector.detect_from_bytes(inputs)
        # Should fallback to default delimiter b"|"
        self.assertEqual(best.delimiter, b"|")
        self.assertEqual(best.confidence, 0.0)
        self.assertEqual(best.presence_ratio, 0.0)

    def test_fallback_on_empty_corpus(self) -> None:
        """Empty input list returns default delimiter."""
        best = self.detector.detect_from_bytes([])
        self.assertEqual(best.delimiter, b"|")
        self.assertEqual(best.confidence, 0.0)

    def test_detect_from_corpus_object(self) -> None:
        """Detect delimiter from a populated Corpus instance."""
        corpus = Corpus()
        corpus.add_seed(b"HDR|MSG=1|CRC=0", name="s1")
        corpus.add_seed(b"HDR|MSG=2|CRC=1", name="s2")
        corpus.add_seed(b"HDR|MSG=3|CRC=2", name="s3")

        best = self.detector.detect_from_corpus(corpus)
        self.assertEqual(best.delimiter, b"|")
        self.assertGreater(best.confidence, 0.80)

    def test_rank_candidates(self) -> None:
        """Ranking returns multiple sorted candidates."""
        inputs = [
            b"KEY1=VAL1|KEY2=VAL2|KEY3=VAL3",
            b"KEY4=VAL4|KEY5=VAL5",
        ]
        ranked = self.detector.rank_candidates(inputs)
        self.assertGreater(len(ranked), 1)
        # Top candidates should include | and =
        top_delims = [c.delimiter for c in ranked[:2]]
        self.assertIn(b"|", top_delims)
        # Verify ranking order (descending confidence)
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i].confidence, ranked[i + 1].confidence)

    def test_custom_candidates_and_priors(self) -> None:
        """DelimiterDetector respects custom candidates and priors."""
        detector = DelimiterDetector(candidate_bytes=[b"#", b"$"], default_delimiter=b"#")
        inputs = [
            b"PART1#PART2#PART3",
            b"TAG1#TAG2#TAG3",
        ]
        best = detector.detect_from_bytes(inputs)
        self.assertEqual(best.delimiter, b"#")
        self.assertGreater(best.confidence, 0.60)


class TestDelimiterIntegration(unittest.TestCase):
    """Test delimiter updates propagate through boundary, splicer, and mutator."""

    def test_boundary_detector_set_delimiter(self) -> None:
        """BoundaryDetector dynamically switches delimiter."""
        detector = BoundaryDetector(delimiter=b"|")
        boundaries = detector.find_boundaries(b"A;B;C")
        self.assertEqual(boundaries, [0, 5])  # start and end

        detector.set_delimiter(b";")
        boundaries = detector.find_boundaries(b"A;B;C")
        self.assertEqual(boundaries, [0, 1, 2, 3, 4, 5])

    def test_boundary_detector_auto_detect(self) -> None:
        """BoundaryDetector auto_detect infers delimiter from inputs."""
        detector = BoundaryDetector(delimiter=b"|")
        inputs = [b"A;B;C", b"D;E;F"]
        best_delim, conf, reason = detector.auto_detect(inputs)
        self.assertEqual(best_delim, b";")
        self.assertEqual(detector.delimiter, b";")
        self.assertGreater(conf, 0.5)

    def test_mutator_set_delimiter(self) -> None:
        """Mutator.set_delimiter updates boundary, splicer, and donor selector."""
        mutator = Mutator(delimiter=b"|", enable_splicing=True)
        self.assertEqual(mutator.boundary_detector.delimiter, b"|")
        self.assertEqual(mutator.splicer.delimiter, b"|")
        self.assertEqual(mutator.donor_selector.delimiter, b"|")

        mutator.set_delimiter(b":")
        self.assertEqual(mutator.boundary_detector.delimiter, b":")
        self.assertEqual(mutator.splicer.delimiter, b":")
        self.assertEqual(mutator.donor_selector.delimiter, b":")


if __name__ == "__main__":
    unittest.main()
