"""Unit tests for corpus management."""

from pathlib import Path
import random
import tempfile
import unittest

from fuzzer.corpus import Corpus


class TestCorpus(unittest.TestCase):
    """Test suite for the Corpus class."""

    def test_load_from_directory(self) -> None:
        """Verify seeds are loaded from directory files as raw bytes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "seed_a.txt").write_bytes(b"SEED_AAA")
            (tmp_path / "seed_b.bin").write_bytes(b"\x00\xFF\x42")
            (tmp_path / ".hidden").write_bytes(b"IGNORE_ME")

            corpus = Corpus(corpus_dir=tmp_path)
            self.assertEqual(len(corpus), 2)
            self.assertIn(b"SEED_AAA", corpus.seeds)
            self.assertIn(b"\x00\xFF\x42", corpus.seeds)
            self.assertNotIn(b"IGNORE_ME", corpus.seeds)

    def test_select_seed(self) -> None:
        """Verify random seed selection returns elements from the corpus."""
        rng = random.Random(123)
        corpus = Corpus(rng=rng)
        corpus.add_seed(b"ONE")
        corpus.add_seed(b"TWO")
        corpus.add_seed(b"THREE")

        selected = [corpus.select_seed() for _ in range(30)]
        self.assertTrue(all(s in [b"ONE", b"TWO", b"THREE"] for s in selected))
        # Ensure all items get selected at least once over 30 picks
        self.assertEqual(set(selected), {b"ONE", b"TWO", b"THREE"})

    def test_empty_corpus_raises(self) -> None:
        """Verify selecting from empty corpus raises ValueError."""
        corpus = Corpus()
        with self.assertRaises(ValueError):
            corpus.select_seed()

    def test_nonexistent_directory_raises(self) -> None:
        """Verify loading from nonexistent directory raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Corpus(corpus_dir="path/that/does/not/exist")

    def test_empty_directory_raises(self) -> None:
        """Verify loading from directory with no seeds raises ValueError."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(ValueError):
                Corpus(corpus_dir=empty_dir)

    def test_invalid_seed_type_raises(self) -> None:
        """Verify adding non-bytes raises TypeError."""
        corpus = Corpus()
        with self.assertRaises(TypeError):
            corpus.add_seed("string_instead_of_bytes")  # type: ignore


if __name__ == "__main__":
    unittest.main()
