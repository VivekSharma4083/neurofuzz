"""Unit tests for the mutation engine."""

import random
import unittest

from fuzzer.mutator import (
    Mutator,
    op_delete_byte,
    op_flip_bit,
    op_insert_byte,
    op_replace_byte,
)


class TestMutator(unittest.TestCase):
    """Test suite for mutation operators and the Mutator class."""

    def setUp(self) -> None:
        self.rng = random.Random(42)
        self.mutator = Mutator(rng=self.rng)

    def test_op_flip_bit(self) -> None:
        """Verify flip_bit inverts exactly one bit in a non-empty buffer."""
        original = bytearray(b"ABCD")
        buf = bytearray(original)
        result = op_flip_bit(buf, self.rng)

        # Buffer was modified in place
        self.assertIs(result, buf)
        self.assertEqual(len(result), len(original))
        # Exactly one byte should differ, and by a power of 2
        diffs = [i for i in range(len(original)) if original[i] != result[i]]
        self.assertEqual(len(diffs), 1)
        diff_val = original[diffs[0]] ^ result[diffs[0]]
        self.assertTrue((diff_val & (diff_val - 1)) == 0, "Diff should be a single bit")

    def test_op_flip_bit_empty(self) -> None:
        """Verify flip_bit handles empty input safely by inserting a byte."""
        buf = bytearray()
        result = op_flip_bit(buf, self.rng)
        self.assertEqual(len(result), 1)

    def test_op_replace_byte(self) -> None:
        """Verify replace_byte modifies a single byte."""
        original = bytearray(b"HELLO WORLD")
        buf = bytearray(original)
        result = op_replace_byte(buf, self.rng)

        self.assertEqual(len(result), len(original))
        diffs = [i for i in range(len(original)) if original[i] != result[i]]
        self.assertEqual(len(diffs), 1)

    def test_op_replace_byte_empty(self) -> None:
        """Verify replace_byte handles empty input safely."""
        buf = bytearray()
        result = op_replace_byte(buf, self.rng)
        self.assertEqual(len(result), 1)

    def test_op_insert_byte(self) -> None:
        """Verify insert_byte increases buffer size by 1."""
        original = bytearray(b"INSERTION")
        buf = bytearray(original)
        result = op_insert_byte(buf, self.rng)

        self.assertEqual(len(result), len(original) + 1)

    def test_op_delete_byte(self) -> None:
        """Verify delete_byte decreases buffer size by 1."""
        original = bytearray(b"ABCDE")
        buf = bytearray(original)
        result = op_delete_byte(buf, self.rng)

        self.assertEqual(len(result), len(original) - 1)
        # Verify remaining bytes are a subsequence
        self.assertTrue(all(b in original for b in result))

    def test_op_delete_byte_empty(self) -> None:
        """Verify delete_byte does not fail on empty input."""
        buf = bytearray()
        result = op_delete_byte(buf, self.rng)
        self.assertEqual(len(result), 0)

    def test_mutator_does_not_modify_original(self) -> None:
        """Verify mutate() operates on a copy and leaves the original bytes intact."""
        original = b"IMMUTABLE_SEED_DATA"
        original_copy = bytes(original)

        for _ in range(50):
            mutated = self.mutator.mutate(original)
            self.assertEqual(original, original_copy, "Original bytes must never be modified")
            self.assertIsInstance(mutated, bytes)

    def test_mutator_specific_operator(self) -> None:
        """Verify specific operator execution by name."""
        data = b"TEST"
        mutated = self.mutator.mutate(data, operator_name="insert_byte")
        self.assertEqual(len(mutated), len(data) + 1)

        with self.assertRaises(KeyError):
            self.mutator.mutate(data, operator_name="nonexistent_op")

    def test_custom_operator_registration(self) -> None:
        """Verify modularity: registering and running a new mutation operator."""
        def custom_reverse(buf: bytearray, _rng: random.Random) -> bytearray:
            buf.reverse()
            return buf

        self.mutator.register_operator("reverse", custom_reverse)
        self.assertIn("reverse", self.mutator.operators)

        result = self.mutator.mutate(b"ABCDE", operator_name="reverse")
        self.assertEqual(result, b"EDCBA")

    def test_mutate_chain(self) -> None:
        """Verify chained mutations."""
        data = b"CHAIN"
        result = self.mutator.mutate_chain(data, count=5)
        self.assertIsInstance(result, bytes)


if __name__ == "__main__":
    unittest.main()
