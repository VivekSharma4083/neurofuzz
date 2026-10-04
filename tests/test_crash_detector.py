"""Unit tests for crash detection and crash file persistence."""

import json
from pathlib import Path
import tempfile
import unittest

from fuzzer.executor import CrashDetector, CrashSaver, ExecutionResult


class TestCrashDetector(unittest.TestCase):
    """Test suite for CrashDetector classification logic."""

    def test_classify_normal_exit(self) -> None:
        """Verify returncode 0 is classified as normal execution."""
        is_crash, failure_type, _ = CrashDetector.classify(returncode=0, timeout=False)
        self.assertFalse(is_crash)
        self.assertEqual(failure_type, "ok")

    def test_classify_timeout(self) -> None:
        """Verify timeout is classified as an interesting failure."""
        is_crash, failure_type, _ = CrashDetector.classify(returncode=None, timeout=True)
        self.assertTrue(is_crash)
        self.assertEqual(failure_type, "timeout")

    def test_classify_posix_signal(self) -> None:
        """Verify negative return codes (POSIX signals like SIGSEGV) are detected."""
        is_crash, failure_type, desc = CrashDetector.classify(returncode=-11, timeout=False)
        self.assertTrue(is_crash)
        self.assertEqual(failure_type, "signal")
        self.assertIn("SIGSEGV", desc)

    def test_classify_windows_access_violation(self) -> None:
        """Verify Windows STATUS_ACCESS_VIOLATION codes are classified as crash."""
        is_crash, failure_type, desc = CrashDetector.classify(
            returncode=-1073741819,  # 0xC0000005 as signed 32-bit
            timeout=False,
        )
        self.assertTrue(is_crash)
        self.assertEqual(failure_type, "crash")
        self.assertIn("STATUS_ACCESS_VIOLATION", desc)

    def test_classify_abort(self) -> None:
        """Verify return code 3 (abort) is classified as crash."""
        is_crash, failure_type, desc = CrashDetector.classify(returncode=3, timeout=False)
        self.assertTrue(is_crash)
        self.assertEqual(failure_type, "crash")
        self.assertIn("abort()", desc)


class TestCrashSaver(unittest.TestCase):
    """Test suite for saving crash inputs and reproduction metadata."""

    def test_save_crash_and_metadata(self) -> None:
        """Verify crash input and JSON metadata are correctly written to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = CrashSaver(output_dir=tmp_dir)
            crashing_input = b"FUZZ\x00\xFFCRASH_PAYLOAD"

            result = ExecutionResult(
                returncode=-11,
                stdout=b"out",
                stderr=b"segfault err",
                timeout=False,
                duration=0.015,
                is_crash=True,
                failure_type="signal",
                description="Terminated by signal SIGSEGV",
            )

            is_new, saved_path = saver.save(crashing_input, result, iteration=42)
            self.assertTrue(is_new)
            self.assertIsNotNone(saved_path)
            self.assertTrue(saved_path.exists())

            # Verify saved payload matches exactly byte-for-byte
            saved_bytes = saved_path.read_bytes()
            self.assertEqual(saved_bytes, crashing_input)

            # Verify metadata JSON
            meta_path = saved_path.with_name(saved_path.stem + "_meta.json")
            self.assertTrue(meta_path.exists())

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(meta["iteration"], 42)
            self.assertEqual(meta["failure_type"], "signal")
            self.assertEqual(meta["returncode"], -11)
            self.assertEqual(meta["size_bytes"], len(crashing_input))
            self.assertEqual(meta["stderr"], "segfault err")

    def test_crash_deduplication(self) -> None:
        """Verify saving identical crashing inputs returns is_new=False and avoids duplication."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = CrashSaver(output_dir=tmp_dir)
            data = b"IDENTICAL_CRASH"
            result = ExecutionResult(
                returncode=139,
                stdout=b"",
                stderr=b"",
                timeout=False,
                duration=0.01,
                is_crash=True,
                failure_type="nonzero_exit",
                description="Crash",
            )

            is_new1, path1 = saver.save(data, result, iteration=1)
            is_new2, path2 = saver.save(data, result, iteration=2)

            self.assertTrue(is_new1)
            self.assertIsNotNone(path1)
            self.assertFalse(is_new2)
            self.assertIsNone(path2)


if __name__ == "__main__":
    unittest.main()
