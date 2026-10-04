"""Unit tests for Phase 6A Protocol-Aware Dictionary Mutation.

Tests:
1. Dictionary loading, file parsing, deduplication, comments, quotes, missing files.
2. Dictionary mutation operators (insert, replace) on empty/non-empty buffers.
3. Fallback behaviors when dictionary is empty or None.
4. Determinism with fixed RNG seed.
5. Mutator integration: dictionary_probability=0 vs 1, operator telemetry.
6. Immutability: original seed bytes are never modified in-place.
"""

from pathlib import Path
import random
import tempfile
import unittest

from fuzzer.dictionary import Dictionary
from fuzzer.mutator import (
    Mutator,
    op_dictionary_insert,
    op_dictionary_replace,
    op_flip_bit,
    op_insert_byte,
)


class TestDictionaryLoading(unittest.TestCase):
    """Test suite for Dictionary file loading and token parsing."""

    def test_load_valid_dictionary(self):
        """Verify tokens are correctly parsed from a valid dictionary file."""
        content = (
            "# Protocol header\n"
            "NF01\n"
            "NF02\n"
            "\n"
            "# Commands\n"
            "PING\n"
            "\"USER=admin\"\n"
            "'PASS=secret_123'\n"
            "|STEP=2\n"
        )
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write(content)
            temp_path = f.name

        try:
            d = Dictionary(filepath=temp_path)
            self.assertEqual(len(d), 6)
            self.assertIn(b"NF01", d)
            self.assertIn(b"NF02", d)
            self.assertIn(b"PING", d)
            self.assertIn(b"USER=admin", d)
            self.assertIn(b"PASS=secret_123", d)
            self.assertIn(b"|STEP=2", d)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_empty_dictionary(self):
        """Verify loading an empty file produces a dictionary of length 0."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("# Only comments\n\n   \n")
            temp_path = f.name

        try:
            d = Dictionary(filepath=temp_path)
            self.assertEqual(len(d), 0)
            self.assertEqual(d.tokens, [])
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_duplicate_tokens_deduplicated(self):
        """Verify duplicate entries in dictionary are deduplicated preserving order."""
        tokens = ["TOKEN1", "TOKEN2", "TOKEN1", "TOKEN3", "TOKEN2"]
        d = Dictionary(tokens=tokens)
        self.assertEqual(len(d), 3)
        self.assertEqual(d.tokens, [b"TOKEN1", b"TOKEN2", b"TOKEN3"])

    def test_whitespace_handling(self):
        """Verify leading and trailing whitespace is stripped."""
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
            f.write("   NF01   \n\tSTEP=1\t\n  \n")
            temp_path = f.name

        try:
            d = Dictionary(filepath=temp_path)
            self.assertEqual(d.tokens, [b"NF01", b"STEP=1"])
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_missing_file_raises_filenotfound(self):
        """Verify loading from a nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Dictionary(filepath="nonexistent_dict_path_12345.dict")

    def test_structured_protocol_dict_exists_and_loads(self):
        """Verify the project's official structured_protocol.dict loads properly."""
        dict_path = Path("dictionaries/structured_protocol.dict")
        self.assertTrue(dict_path.exists(), "structured_protocol.dict should exist")
        d = Dictionary(filepath=dict_path)
        self.assertGreater(len(d), 20)
        self.assertIn(b"NF01", d)
        self.assertIn(b"PING", d)
        self.assertIn(b"USER=admin", d)
        self.assertIn(b"STEP=1", d)
        self.assertIn(b"STEP=2", d)
        self.assertIn(b"|STEP=2", d)
        self.assertIn(b"EXEC=CRASH", d)

    def test_get_random_token_empty_raises(self):
        """Verify get_random_token on empty dictionary raises ValueError."""
        d = Dictionary()
        rng = random.Random(42)
        with self.assertRaises(ValueError):
            d.get_random_token(rng)


class TestDictionaryMutationOperators(unittest.TestCase):
    """Test suite for dictionary_insert and dictionary_replace mutation operators."""

    def setUp(self):
        self.rng = random.Random(1337)
        self.dictionary = Dictionary(
            tokens=["NF01", "PING", "USER=guest", "USER=admin", "STEP=1", "STEP=2", "|STEP=2", "EXEC=CRASH"]
        )

    def test_op_dictionary_insert_produces_valid_bytes(self):
        """Verify dictionary_insert inserts an intact token into the buffer."""
        original = bytearray(b"NF01|DIAG")
        buf = bytearray(original)
        res = op_dictionary_insert(buf, self.rng, self.dictionary)
        self.assertGreater(len(res), len(original))
        # Ensure at least one token from the dictionary is present in the result
        self.assertTrue(any(tok in res for tok in self.dictionary.tokens))

    def test_op_dictionary_insert_empty_buffer(self):
        """Verify dictionary_insert handles empty buffer safely."""
        buf = bytearray()
        res = op_dictionary_insert(buf, self.rng, self.dictionary)
        self.assertGreater(len(res), 0)
        self.assertIn(res, self.dictionary.tokens)

    def test_op_dictionary_insert_empty_dictionary_fallback(self):
        """Verify dictionary_insert falls back to insert_byte when dictionary is empty."""
        empty_d = Dictionary()
        buf = bytearray(b"TEST")
        res = op_dictionary_insert(buf, self.rng, empty_d)
        self.assertEqual(len(res), 5)  # single byte inserted

    def test_op_dictionary_replace_substitutes_known_token(self):
        """Verify dictionary_replace replaces an existing token with another."""
        # Force a token that exists in dictionary
        buf = bytearray(b"NF01|AUTH|USER=guest")
        res = op_dictionary_replace(buf, self.rng, self.dictionary)
        self.assertIsInstance(res, bytearray)
        # Result should differ from original
        self.assertNotEqual(res, bytearray(b"NF01|AUTH|USER=guest"))

    def test_op_dictionary_replace_empty_buffer(self):
        """Verify dictionary_replace handles empty buffer safely."""
        buf = bytearray()
        res = op_dictionary_replace(buf, self.rng, self.dictionary)
        self.assertGreater(len(res), 0)
        self.assertIn(res, self.dictionary.tokens)

    def test_op_dictionary_replace_empty_dictionary_fallback(self):
        """Verify dictionary_replace falls back to replace_byte when dictionary is empty."""
        empty_d = Dictionary()
        buf = bytearray(b"HELLO")
        res = op_dictionary_replace(buf, self.rng, empty_d)
        self.assertEqual(len(res), 5)
        # Exactly one byte modified
        diffs = [i for i in range(5) if buf[i] != b"HELLO"[i]]
        self.assertEqual(len(diffs), 1)

    def test_determinism_with_fixed_rng(self):
        """Verify identical RNG seeds produce identical mutation outputs."""
        seed_data = b"NF01|DIAG|STEP=1"

        rng1 = random.Random(999)
        buf1 = bytearray(seed_data)
        res1 = op_dictionary_insert(buf1, rng1, self.dictionary)

        rng2 = random.Random(999)
        buf2 = bytearray(seed_data)
        res2 = op_dictionary_insert(buf2, rng2, self.dictionary)

        self.assertEqual(res1, res2)


class TestMutatorDictionaryIntegration(unittest.TestCase):
    """Test suite for Mutator dictionary probability and operator selection."""

    def setUp(self):
        self.rng = random.Random(42)
        self.dictionary = Dictionary(tokens=["PING", "INFO", "CALC", "DIAG", "|STEP=2"])

    def test_probability_zero_disables_dictionary_mutations(self):
        """Verify dictionary_probability=0 never selects dictionary operators."""
        mutator = Mutator(rng=self.rng, dictionary=self.dictionary, dictionary_probability=0.0)
        data = b"NF01|TEST"

        for _ in range(100):
            _, op_used = mutator.mutate_with_operator(data)
            self.assertIn(op_used, mutator.byte_operators)
            self.assertNotIn(op_used, mutator.dictionary_operators)

    def test_probability_one_enables_dictionary_mutations(self):
        """Verify dictionary_probability=1.0 always selects dictionary operators when dictionary is present."""
        mutator = Mutator(rng=self.rng, dictionary=self.dictionary, dictionary_probability=1.0)
        data = b"NF01|TEST"

        for _ in range(50):
            _, op_used = mutator.mutate_with_operator(data)
            self.assertIn(op_used, mutator.dictionary_operators)

    def test_immutability_of_original_seed(self):
        """Verify original bytes object is never modified in-place."""
        original = b"PRISTINE_INPUT_BYTES"
        original_copy = bytes(original)
        mutator = Mutator(rng=self.rng, dictionary=self.dictionary, dictionary_probability=0.5)

        for _ in range(50):
            mutated = mutator.mutate(original)
            self.assertEqual(original, original_copy)
            self.assertIsInstance(mutated, bytes)

    def test_invalid_probability_raises_valueerror(self):
        """Verify probability outside [0.0, 1.0] raises ValueError."""
        with self.assertRaises(ValueError):
            Mutator(dictionary_probability=-0.1)
        with self.assertRaises(ValueError):
            Mutator(dictionary_probability=1.5)


if __name__ == "__main__":
    unittest.main()
