"""Unit tests for NeuroFuzz Phase 7A - Persistent Execution Harness.

Verifies:
1. Process lifecycle: start, stop, restart, is_alive, context manager.
2. Initial handshake validation.
3. Accurate coverage parsing and depth reporting.
4. ExecutionResult property aliases (return_code, timed_out, crashed, execution_time, depth).
5. State isolation across multiple sequential executions (no leftover state).
6. Crash detection and transparent auto-recovery on divide-by-zero, segfault, and abort.
7. Timeout detection and recovery on hangs.
8. Binary safety with null bytes, Ctrl-Z (0x1A), empty inputs, and oversized inputs (>2048 bytes).
9. Subprocess vs. Persistent equivalence across coverage sets, depths, and crash classifications.
"""

from pathlib import Path
import subprocess
import sys
import unittest

from fuzzer.executor import CrashDetector, ExecutionResult, Executor, SubprocessExecutor
from fuzzer.persistent_executor import PersistentExecutor


class TestPersistentExecutor(unittest.TestCase):
    """Test suite for PersistentExecutor harness and process management."""

    @classmethod
    def setUpClass(cls):
        cls.target_path = Path("targets/structured_target.exe").resolve()
        if not cls.target_path.exists():
            # If not compiled with .exe extension on POSIX
            posix_path = Path("targets/structured_target")
            if posix_path.exists():
                cls.target_path = posix_path
            else:
                raise FileNotFoundError(f"Target binary not found at {cls.target_path}")

    def setUp(self):
        self.executor = PersistentExecutor(self.target_path, timeout=1.0)

    def tearDown(self):
        if self.executor is not None:
            self.executor.close()

    def test_01_startup_handshake_and_is_alive(self):
        """PersistentExecutor starts process and confirms handshake."""
        self.assertTrue(self.executor.is_alive())
        self.assertIsNotNone(self.executor.proc)
        self.assertIsNone(self.executor.proc.poll())

    def test_02_invalid_target_path_raises(self):
        """Attempting to instantiate with non-existent binary raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            PersistentExecutor("targets/non_existent_binary_xyz.exe")

    def test_03_single_execution_normal_result(self):
        """Single valid input produces expected result fields and coverage."""
        result = self.executor.execute(b"NF01|PING")
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.crashed)
        self.assertFalse(result.timed_out)
        self.assertEqual(result.failure_type, "ok")
        self.assertIn("PATH_ENTRY", result.coverage)
        self.assertIn("PATH_HDR_VALID", result.coverage)
        self.assertIn("PATH_CMD_PING", result.coverage)
        self.assertGreaterEqual(result.depth, 6)

    def test_04_execution_result_property_aliases(self):
        """ExecutionResult property aliases behave identically to primary attributes."""
        result = self.executor.execute(b"NF01|INFO")
        self.assertEqual(result.return_code, result.returncode)
        self.assertEqual(result.timed_out, result.timeout)
        self.assertEqual(result.crashed, result.is_crash)
        self.assertEqual(result.execution_time, result.duration)
        self.assertGreater(result.depth, 0)

    def test_05_multiple_sequential_executions_same_process(self):
        """Multiple sequential executions reuse the same persistent process PID."""
        initial_pid = self.executor.proc.pid
        for _ in range(5):
            res = self.executor.execute(b"NF01|PING")
            self.assertEqual(res.returncode, 0)
            self.assertFalse(res.crashed)
            self.assertEqual(self.executor.proc.pid, initial_pid)

    def test_06_state_isolation_sequence(self):
        """Repeated inputs A -> B -> A verify strict target state isolation."""
        res_a1 = self.executor.execute(b"NF01|PING")
        res_b = self.executor.execute(b"NF01|DIAG|STEP=1")
        res_a2 = self.executor.execute(b"NF01|PING")

        self.assertEqual(res_a1.coverage, res_a2.coverage)
        self.assertEqual(res_a1.depth, res_a2.depth)
        self.assertNotEqual(res_a1.depth, res_b.depth)
        self.assertIn("PATH_CMD_DIAG", res_b.coverage)
        self.assertNotIn("PATH_CMD_DIAG", res_a2.coverage)

    def test_07_crash_recovery_div_zero(self):
        """Target division by zero crash is detected and process is transparently restarted."""
        crash_input = b"NF01|CALC|OP=DIV|NUM=100|DEN=0"
        result = self.executor.execute(crash_input)

        self.assertTrue(result.crashed)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.failure_type, "crash")
        self.assertIn("BUG_DIV_ZERO", result.coverage)

        # Process must be restarted and alive for next run
        self.assertTrue(self.executor.is_alive())
        res_after = self.executor.execute(b"NF01|PING")
        self.assertFalse(res_after.crashed)
        self.assertEqual(res_after.returncode, 0)

    def test_08_crash_recovery_null_deref(self):
        """Target null pointer dereference is caught and recovered."""
        crash_input = b"NF01|DIAG|NEST=L3|FLAG=TRIGGER_NULL"
        result = self.executor.execute(crash_input)

        self.assertTrue(result.crashed)
        self.assertIn("BUG_NULL_DEREF", result.coverage)
        self.assertTrue(self.executor.is_alive())

        res_after = self.executor.execute(b"NF01|PING")
        self.assertFalse(res_after.crashed)

    def test_09_crash_recovery_abort(self):
        """Deepest state abort() assertion crash is caught and recovered."""
        crash_input = b"NF01|DIAG|STEP=1|STEP=2|STEP=3|STEP=4|STEP=5|EXEC=CRASH"
        result = self.executor.execute(crash_input)

        self.assertTrue(result.crashed)
        self.assertIn("BUG_DEEP_ABORT", result.coverage)
        self.assertEqual(result.depth, 19)
        self.assertTrue(self.executor.is_alive())

        res_after = self.executor.execute(b"NF01|PING")
        self.assertFalse(res_after.crashed)

    def test_10_empty_input_handling(self):
        """Empty input (0 bytes) is handled without desynchronization."""
        result = self.executor.execute(b"")
        self.assertFalse(result.crashed)
        self.assertIn("PATH_EMPTY", result.coverage)
        self.assertEqual(result.depth, 1)

        # Subsequent execution succeeds
        res_after = self.executor.execute(b"NF01|PING")
        self.assertFalse(res_after.crashed)

    def test_11_null_bytes_in_input(self):
        """Input containing embedded binary null bytes is processed without truncation."""
        payload = b"NF01|\x00\x00\x00|PING"
        result = self.executor.execute(payload)
        self.assertFalse(result.crashed)
        self.assertIn("PATH_HDR_VALID", result.coverage)

    def test_12_ctrl_z_eof_byte_in_input(self):
        """Input containing 0x1A (Windows Ctrl-Z) is treated as raw binary."""
        payload = b"NF01|\x1a\x1a\x1a|PING"
        result = self.executor.execute(payload)
        self.assertFalse(result.crashed)
        self.assertIn("PATH_HDR_VALID", result.coverage)

    def test_13_large_oversized_input_drain(self):
        """Inputs exceeding MAX_INPUT_SIZE drain remaining bytes to stay in sync."""
        large_payload = b"NF01|PING|" + b"A" * 3000
        result = self.executor.execute(large_payload)
        self.assertFalse(result.crashed)

        # Subsequent execution must still be aligned and succeed
        res_after = self.executor.execute(b"NF01|PING")
        self.assertFalse(res_after.crashed)
        self.assertEqual(res_after.depth, 6)

    def test_14_stop_and_restart(self):
        """stop() terminates the child and restart() provisions a fresh one."""
        self.assertTrue(self.executor.is_alive())
        self.executor.stop()
        self.assertFalse(self.executor.is_alive())

        self.executor.restart()
        self.assertTrue(self.executor.is_alive())
        res = self.executor.execute(b"NF01|PING")
        self.assertFalse(res.crashed)

    def test_15_start_idempotency(self):
        """Calling start() when already running does not spawn duplicate processes."""
        pid1 = self.executor.proc.pid
        self.executor.start()
        pid2 = self.executor.proc.pid
        self.assertEqual(pid1, pid2)

    def test_16_context_manager_support(self):
        """PersistentExecutor cleans up process when exiting context manager."""
        with PersistentExecutor(self.target_path) as pe:
            self.assertTrue(pe.is_alive())
            res = pe.execute(b"NF01|PING")
            self.assertFalse(res.crashed)
            proc_ref = pe.proc

        # After block exit, process must be terminated
        self.assertFalse(pe.is_alive())
        self.assertIsNotNone(proc_ref.poll())

    def test_17_subprocess_vs_persistent_equivalence(self):
        """PersistentExecutor and Subprocess Executor produce identical coverage and depth."""
        sub_exec = Executor(self.target_path)
        test_inputs = [
            b"",
            b"X",
            b"NF",
            b"NF01",
            b"NF01|PING",
            b"NF01|INFO|VERBOSE",
            b"NF01|CALC|OP=ADD|NUM=10|DEN=5",
            b"NF01|DIAG|STEP=1",
            b"NF01|DIAG|STEP=1|STEP=2",
            b"NF01|DIAG|STEP=1|STEP=2|STEP=3",
            b"NF01|INVALID_COMMAND",
        ]

        for data in test_inputs:
            res_sub = sub_exec.run(data)
            res_pers = self.executor.execute(data)

            self.assertEqual(
                res_sub.coverage,
                res_pers.coverage,
                f"Coverage mismatch for input {data!r}: sub={res_sub.coverage} vs pers={res_pers.coverage}",
            )
            self.assertEqual(
                res_sub.depth,
                res_pers.depth,
                f"Depth mismatch for input {data!r}: sub={res_sub.depth} vs pers={res_pers.depth}",
            )
            self.assertEqual(res_sub.crashed, res_pers.crashed)

    def test_18_subprocess_executor_alias(self):
        """SubprocessExecutor is an alias for Executor."""
        self.assertIs(SubprocessExecutor, Executor)
        sub = SubprocessExecutor(self.target_path)
        res = sub.run(b"NF01|PING")
        self.assertFalse(res.crashed)
        self.assertEqual(res.depth, 6)

    def test_19_timeout_recovery(self):
        """Target hang triggers timeout, cleanly terminates, and restarts target."""
        # Using a very low timeout executor
        short_pe = PersistentExecutor(self.target_path, timeout=0.15)
        try:
            # We simulate a hang by intercepting stdin write to not send anything,
            # or by sending incomplete frame. Let's send length saying 100 bytes, but send 0 bytes.
            short_pe.proc.stdin.write(b"100\n")
            short_pe.proc.stdin.flush()

            # Now call execute which will wait for response and timeout
            res = short_pe.execute(b"")
            self.assertTrue(res.timed_out)
            self.assertTrue(res.crashed)
            self.assertEqual(res.failure_type, "timeout")

            # Must have cleanly restarted
            self.assertTrue(short_pe.is_alive())
            res_after = short_pe.execute(b"NF01|PING")
            self.assertFalse(res_after.crashed)
        finally:
            short_pe.close()

    def test_20_run_alias_on_persistent_executor(self):
        """PersistentExecutor.run() delegates seamlessly to execute()."""
        res = self.executor.run(b"NF01|PING")
        self.assertFalse(res.crashed)
        self.assertEqual(res.returncode, 0)
        self.assertGreater(len(res.coverage), 0)


if __name__ == "__main__":
    unittest.main()
