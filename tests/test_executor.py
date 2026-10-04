"""Unit tests for the target executor."""

from pathlib import Path
import sys
import unittest

from fuzzer.executor import ExecutionResult, Executor


class TestExecutor(unittest.TestCase):
    """Test suite for the Executor class."""

    def test_successful_execution(self) -> None:
        """Verify normal process execution, capturing stdout and zero exit code."""
        # Using sys.executable to run a quick script reading stdin and echoing it
        executor = Executor(
            target_path=sys.executable,
            timeout=2.0,
            extra_args=["-c", "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data)"],
        )
        test_payload = b"NEUROFUZZ_TEST_INPUT"
        result: ExecutionResult = executor.run(test_payload)

        self.assertFalse(result.timeout)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, test_payload)
        self.assertFalse(result.is_crash)
        self.assertEqual(result.failure_type, "ok")
        self.assertGreater(result.duration, 0.0)

    def test_nonzero_exit_code(self) -> None:
        """Verify detection of non-zero exit code as an interesting failure."""
        executor = Executor(
            target_path=sys.executable,
            timeout=2.0,
            extra_args=["-c", "import sys; sys.exit(42)"],
        )
        result = executor.run(b"test")

        self.assertFalse(result.timeout)
        self.assertEqual(result.returncode, 42)
        self.assertTrue(result.is_crash)
        self.assertEqual(result.failure_type, "nonzero_exit")

    def test_timeout_enforcement(self) -> None:
        """Verify executor detects timeout and does not hang indefinitely."""
        executor = Executor(
            target_path=sys.executable,
            timeout=0.2,
            extra_args=["-c", "import time; time.sleep(5)"],
        )
        result = executor.run(b"spin")

        self.assertTrue(result.timeout)
        self.assertIsNone(result.returncode)
        self.assertTrue(result.is_crash)
        self.assertEqual(result.failure_type, "timeout")
        self.assertIn("timed out", result.description)

    def test_nonexistent_target_raises(self) -> None:
        """Verify initializing with nonexistent target raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Executor(target_path="nonexistent_binary_xyz_123")


if __name__ == "__main__":
    unittest.main()
