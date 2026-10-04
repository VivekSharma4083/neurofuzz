"""Unit and regression tests for Phase 6B: Boundary-Aware Dictionary Mutation."""

import random
import unittest
from pathlib import Path

from fuzzer.boundary import (
    BoundaryDetector,
    find_delimiter_boundaries,
    format_boundary_insertion,
)
from fuzzer.dictionary import Dictionary
from fuzzer.mutator import (
    Mutator,
    op_dictionary_insert_boundary,
    op_dictionary_insert_random,
    op_dictionary_insert,
)


class TestBoundaryDetection(unittest.TestCase):
    """Unit tests for find_delimiter_boundaries and BoundaryDetector."""

    def test_empty_buffer(self) -> None:
        """Empty buffer should return boundary at index 0."""
        bounds = find_delimiter_boundaries(b"")
        self.assertEqual(bounds, [0])

    def test_single_field_no_delimiter(self) -> None:
        """Buffer without delimiter should return start and end boundaries."""
        bounds = find_delimiter_boundaries(b"HELLO")
        self.assertEqual(bounds, [0, 5])

    def test_single_delimiter(self) -> None:
        """Buffer with a single delimiter should return start, before delim, after delim, end."""
        # b"A|B" -> length 3; delim at index 1 -> positions: 0, 1, 2, 3
        bounds = find_delimiter_boundaries(b"A|B")
        self.assertEqual(bounds, [0, 1, 2, 3])

    def test_structured_protocol_boundaries(self) -> None:
        """Verify boundaries found in realistic protocol payload."""
        payload = b"NF01|DIAG|STEP=1"
        # indices:
        # 0: start
        # 4: before '|'
        # 5: after '|'
        # 9: before '|'
        # 10: after '|'
        # 16: end
        bounds = find_delimiter_boundaries(payload)
        self.assertEqual(bounds, [0, 4, 5, 9, 10, 16])

    def test_consecutive_delimiters(self) -> None:
        """Consecutive delimiters should be handled without duplicate indices."""
        bounds = find_delimiter_boundaries(b"A||B")
        # length 4; delim 1 at 1 (1, 2); delim 2 at 2 (2, 3); bounds: 0, 1, 2, 3, 4
        self.assertEqual(bounds, [0, 1, 2, 3, 4])

    def test_trailing_newline_and_crlf(self) -> None:
        """Trailing newlines should create insertion boundaries before the newline."""
        # Trailing \n
        bounds_lf = find_delimiter_boundaries(b"NF01|DIAG\n")
        self.assertIn(len(b"NF01|DIAG"), bounds_lf)
        self.assertIn(len(b"NF01|DIAG\n"), bounds_lf)

        # Trailing \r\n
        bounds_crlf = find_delimiter_boundaries(b"NF01|DIAG\r\n")
        self.assertIn(len(b"NF01|DIAG"), bounds_crlf)
        self.assertIn(len(b"NF01|DIAG\r\n"), bounds_crlf)

    def test_binary_non_ascii_bytes(self) -> None:
        """Verify boundary detection functions on binary payloads containing pipe delimiters."""
        payload = bytes([0x00, 0xFF, ord('|'), 0x80, 0x7F])
        bounds = find_delimiter_boundaries(payload)
        self.assertEqual(bounds, [0, 2, 3, 5])

    def test_boundary_detector_class(self) -> None:
        """Verify BoundaryDetector wrapper class methods."""
        detector = BoundaryDetector(delimiter=b"|")
        data = b"NF01|DIAG"
        self.assertTrue(detector.has_internal_boundaries(data))
        self.assertFalse(detector.has_internal_boundaries(b"NO_DELIMITER_HERE"))

        bounds = detector.find_boundaries(data)
        self.assertEqual(bounds, [0, 4, 5, 9])

        rng = random.Random(42)
        selected = detector.select_boundary(data, rng)
        self.assertIn(selected, bounds)


class TestBoundaryFormatting(unittest.TestCase):
    """Unit tests for format_boundary_insertion syntax safety."""

    def test_token_with_leading_delimiter_after_delimiter(self) -> None:
        """Inserting '|STEP=2' right after an existing '|' should strip leading '|' to prevent '||'."""
        buf = b"NF01|DIAG|"
        token = b"|STEP=2"
        pos = len(buf)  # At end, immediately after '|'
        clamped_pos, formatted_token = format_boundary_insertion(buf, token, pos)
        self.assertEqual(clamped_pos, len(buf))
        self.assertEqual(formatted_token, b"STEP=2")

    def test_token_without_delimiter_appended_at_end(self) -> None:
        """Appending 'STEP=2' to 'NF01|DIAG|STEP=1' should prepend '|' to form valid field."""
        buf = b"NF01|DIAG|STEP=1"
        token = b"STEP=2"
        pos = len(buf)
        clamped_pos, formatted_token = format_boundary_insertion(buf, token, pos)
        self.assertEqual(clamped_pos, len(buf))
        self.assertEqual(formatted_token, b"|STEP=2")

    def test_token_with_leading_delimiter_at_end(self) -> None:
        """Appending '|STEP=2' to 'NF01|DIAG|STEP=1' should keep '|STEP=2' unchanged."""
        buf = b"NF01|DIAG|STEP=1"
        token = b"|STEP=2"
        pos = len(buf)
        clamped_pos, formatted_token = format_boundary_insertion(buf, token, pos)
        self.assertEqual(clamped_pos, len(buf))
        self.assertEqual(formatted_token, b"|STEP=2")

    def test_middle_insertion_after_delimiter(self) -> None:
        """Inserting bare token after delimiter in the middle appends delimiter."""
        buf = b"NF01|DATA"
        token = b"DIAG"
        pos = 5  # Right after '|'
        clamped_pos, formatted_token = format_boundary_insertion(buf, token, pos)
        self.assertEqual(clamped_pos, 5)
        self.assertEqual(formatted_token, b"DIAG|")

    def test_empty_buffer_or_token(self) -> None:
        """Empty buffer or token returns safely."""
        self.assertEqual(format_boundary_insertion(b"", b"TOKEN", 0), (0, b"TOKEN"))
        self.assertEqual(format_boundary_insertion(b"HELLO", b"", 2), (2, b""))


class TestBoundaryMutationOperator(unittest.TestCase):
    """Unit tests for op_dictionary_insert_boundary and boundary-aware Mutator."""

    def setUp(self) -> None:
        self.dict_obj = Dictionary()
        self.dict_obj.add_token(b"|STEP=2")
        self.dict_obj.add_token(b"|STEP=3")
        self.dict_obj.add_token(b"STEP=2")
        self.dict_obj.add_token(b"|DIAG")

    def test_boundary_insert_produces_valid_structure(self) -> None:
        """Verify boundary insertion snaps to delimiter positions."""
        rng = random.Random(12345)
        buf = bytearray(b"NF01|DIAG|STEP=1")
        mutated = op_dictionary_insert_boundary(buf, rng, self.dict_obj)
        self.assertNotEqual(bytes(mutated), b"NF01|DIAG|STEP=1")
        # Check that no double delimiters were accidentally created
        self.assertNotIn(b"||", bytes(mutated))

    def test_deterministic_regression_insertion(self) -> None:
        """Deterministic test: inserting |STEP=2 at the end of NF01|DIAG|STEP=1."""
        detector = BoundaryDetector(delimiter=b"|")
        buf = bytearray(b"NF01|DIAG|STEP=1")
        # Boundary at end (pos=16)
        pos = 16
        pos, tok = detector.format_insertion(buf, b"|STEP=2", pos)
        buf[pos:pos] = tok
        self.assertEqual(bytes(buf), b"NF01|DIAG|STEP=1|STEP=2")

    def test_empty_dictionary_fallback(self) -> None:
        """Operator handles empty dictionary gracefully without crashing, falling back to byte insert."""
        empty_dict = Dictionary()
        buf = bytearray(b"TEST|DATA")
        res = op_dictionary_insert_boundary(buf, random.Random(42), empty_dict)
        self.assertEqual(len(res), len(b"TEST|DATA") + 1)

    def test_mutator_boundary_aware_flag(self) -> None:
        """Mutator should use boundary-aware operator when boundary_aware=True."""
        mutator_random = Mutator(
            rng=random.Random(1),
            dictionary=self.dict_obj,
            dictionary_probability=1.0,
            boundary_aware=False,
        )
        mutator_boundary = Mutator(
            rng=random.Random(1),
            dictionary=self.dict_obj,
            dictionary_probability=1.0,
            boundary_aware=True,
        )

        self.assertFalse(mutator_random.boundary_aware)
        self.assertTrue(mutator_boundary.boundary_aware)

        # Test operator dispatch in mutate_with_operator
        _, op_rand = mutator_random.mutate_with_operator(b"NF01|DIAG", operator_name="dictionary_insert")
        self.assertEqual(op_rand, "dictionary_insert_random")

        _, op_bound = mutator_boundary.mutate_with_operator(b"NF01|DIAG", operator_name="dictionary_insert")
        self.assertEqual(op_bound, "dictionary_insert_boundary")

    def test_op_dictionary_insert_dispatcher(self) -> None:
        """Test op_dictionary_insert forwards to boundary or random mode."""
        buf = bytearray(b"NF01|DIAG")
        rng = random.Random(42)

        res_bound = op_dictionary_insert(buf, rng, self.dict_obj, boundary_aware=True)
        self.assertIsInstance(res_bound, bytearray)

        res_rand = op_dictionary_insert(buf, rng, self.dict_obj, boundary_aware=False)
        self.assertIsInstance(res_rand, bytearray)


if __name__ == "__main__":
    unittest.main()
