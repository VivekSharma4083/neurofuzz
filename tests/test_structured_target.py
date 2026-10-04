"""Comprehensive unit and benchmark validation tests for structured_target (Phase 5).

Verifies the 9 benchmark criteria:
1. Minimal input reaches shallow coverage.
2. Valid structured input reaches deeper coverage.
3. Different commands produce different execution paths.
4. Authentication branches and privilege tiers are distinguishable.
5. Deep sequential state machine states are actually reachable.
6. Each of the 6 intentional bugs is reachable with known triggering inputs.
7. Coverage markers are deterministic across repeated runs.
8. Invalid inputs do not accidentally reach deep states.
9. Target does not crash on ordinary initial seeds.
"""

from pathlib import Path
import unittest

from fuzzer.coverage import CoverageTracker, extract_coverage_depth
from fuzzer.executor import ExecutionResult, Executor


class TestStructuredTargetBenchmark(unittest.TestCase):
    """Validation test suite for targets/structured_target.exe."""

    @classmethod
    def setUpClass(cls):
        cls.target_exe = Path("targets/structured_target.exe").resolve()
        if not cls.target_exe.exists():
            raise FileNotFoundError(
                f"Benchmark target binary not found at {cls.target_exe}. Please compile targets/structured_target.c first."
            )
        cls.executor = Executor(target_path=cls.target_exe, timeout=1.0)

    def test_1_minimal_input_reaches_shallow_coverage(self):
        """Criterion 1: Minimal input reaches only shallow coverage."""
        res = self.executor.run(b"NF")
        self.assertFalse(res.is_crash)
        self.assertIn("PATH_ENTRY", res.coverage)
        self.assertIn("PATH_HDR_N", res.coverage)
        self.assertIn("PATH_HDR_VALID", res.coverage)
        # Should not reach command or deep parsing stages
        self.assertNotIn("PATH_FIELDS_PARSED", res.coverage)
        self.assertNotIn("PATH_CMD_PING", res.coverage)
        self.assertLessEqual(extract_coverage_depth(res.coverage), 2)

    def test_2_structured_input_reaches_deeper_coverage(self):
        """Criterion 2: Valid structured input reaches deeper parser stages."""
        res = self.executor.run(b"NF01|PING\n")
        self.assertFalse(res.is_crash)
        self.assertIn("PATH_HDR_VALID", res.coverage)
        self.assertIn("PATH_VER_1", res.coverage)
        self.assertIn("PATH_DELIM_1_VALID", res.coverage)
        self.assertIn("PATH_FIELDS_PARSED", res.coverage)
        self.assertIn("PATH_CMD_PING", res.coverage)
        self.assertGreaterEqual(extract_coverage_depth(res.coverage), 6)

    def test_3_different_commands_produce_different_paths(self):
        """Criterion 3: Different commands produce distinct coverage paths."""
        res_ping = self.executor.run(b"NF01|PING\n")
        res_info = self.executor.run(b"NF01|INFO|VERBOSE\n")

        self.assertIn("PATH_CMD_PING", res_ping.coverage)
        self.assertNotIn("PATH_CMD_INFO", res_ping.coverage)

        self.assertIn("PATH_CMD_INFO", res_info.coverage)
        self.assertIn("PATH_INFO_VERBOSE", res_info.coverage)
        self.assertNotIn("PATH_CMD_PING", res_info.coverage)

    def test_4_authentication_branches_distinguishable(self):
        """Criterion 4: Authentication tiers and credentials branch cleanly."""
        res_guest = self.executor.run(b"NF01|AUTH|USER=guest\n")
        res_dev = self.executor.run(b"NF01|AUTH|USER=dev|TOKEN=1234\n")
        res_admin = self.executor.run(b"NF02|AUTH|USER=admin|TOKEN=adm_ok|MODE=NORMAL\n")
        res_root = self.executor.run(b"NF02|AUTH|USER=root|TOKEN=!#ROOT#!|MODE=NORMAL\n")

        # Guest tier
        self.assertIn("PATH_USER_GUEST", res_guest.coverage)
        self.assertNotIn("PATH_USER_DEV", res_guest.coverage)

        # Dev tier
        self.assertIn("PATH_USER_DEV", res_dev.coverage)
        self.assertIn("PATH_AUTH_DEV_OK", res_dev.coverage)

        # Admin tier
        self.assertIn("PATH_USER_ADMIN", res_admin.coverage)
        self.assertIn("PATH_AUTH_ADMIN_OK", res_admin.coverage)
        self.assertIn("PATH_ADMIN_AUTHORIZED", res_admin.coverage)

        # Root tier
        self.assertIn("PATH_USER_ROOT", res_root.coverage)
        self.assertIn("PATH_AUTH_ROOT_OK", res_root.coverage)
        self.assertIn("PATH_ADMIN_AUTHORIZED", res_root.coverage)

    def test_5_deep_state_machine_reachability(self):
        """Criterion 5: Deep state sequence (STEP 1 through 5) is fully reachable."""
        deep_payload = b"NF01|DIAG|STEP=1|STEP=2|STEP=3|STEP=4|STEP=5\n"
        res = self.executor.run(deep_payload)

        self.assertFalse(res.is_crash)
        for s in range(1, 6):
            self.assertIn(f"PATH_DEEP_STATE_{s}", res.coverage)
        self.assertIn("PATH_DEEP_STATE_MAX", res.coverage)
        self.assertIn("DEPTH_19", res.coverage)
        self.assertEqual(extract_coverage_depth(res.coverage), 19)

    def test_6_intentional_bug_1_div_zero(self):
        """Criterion 6a: Bug 1 - Division by zero in math calculator."""
        payload = b"NF01|CALC|OP=DIV|NUM=100|DEN=0\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "Div-zero should trigger crash")
        self.assertIn("BUG_DIV_ZERO", res.coverage)

    def test_6_intentional_bug_2_int_overflow(self):
        """Criterion 6b: Bug 2 - Integer truncation in write size."""
        payload = b"NF01|WRITE|SIZE=65535|DATA=" + b"A" * 64 + b"\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "Integer truncation should trigger crash")
        self.assertIn("BUG_INT_OVERFLOW", res.coverage)

    def test_6_intentional_bug_3_oob_read(self):
        """Criterion 6c: Bug 3 - Out-of-bounds memory read in admin debug memory dump."""
        payload = b"NF02|AUTH|USER=admin|TOKEN=adm_sec|MODE=DEBUG|DUMP=MEM|OFF=9999\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "OOB read should trigger crash")
        self.assertIn("BUG_OOB_READ", res.coverage)

    def test_6_intentional_bug_4_null_deref(self):
        """Criterion 6d: Bug 4 - Null pointer dereference in nested pipeline L3."""
        payload = b"NF01|DIAG|NEST=L3|FLAG=TRIGGER_NULL\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "Null dereference should trigger crash")
        self.assertIn("BUG_NULL_DEREF", res.coverage)

    def test_6_intentional_bug_5_stack_overflow(self):
        """Criterion 6e: Bug 5 - Stack buffer overflow in L5 RLE unpacker."""
        payload = b"NF01|DIAG|NEST=L5|ENC=RLE|PAY=R250\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "Stack overflow should trigger crash")
        self.assertIn("BUG_STACK_OVERFLOW", res.coverage)

    def test_6_intentional_bug_6_deep_abort(self):
        """Criterion 6f: Bug 6 - Assertion abort in final DEEP_STATE_MAX handler."""
        payload = b"NF01|DIAG|STEP=1|STEP=2|STEP=3|STEP=4|STEP=5|EXEC=CRASH\n"
        res = self.executor.run(payload)
        self.assertTrue(res.is_crash, "Deep state abort should trigger crash")
        self.assertIn("BUG_DEEP_ABORT", res.coverage)

    def test_7_coverage_markers_are_deterministic(self):
        """Criterion 7: Coverage markers are identical across repeated executions."""
        payload = b"NF02|AUTH|USER=admin|TOKEN=adm_123|MODE=DEBUG|DUMP=LOG\n"
        res1 = self.executor.run(payload)
        res2 = self.executor.run(payload)
        self.assertEqual(res1.coverage, res2.coverage)
        self.assertGreater(len(res1.coverage), 10)

    def test_8_invalid_inputs_do_not_reach_deep_states(self):
        """Criterion 8: Random/garbage inputs are stopped early."""
        garbage = b"\x00\xFF\xAA\x55GARBAGE_PAYLOAD"
        res = self.executor.run(garbage)
        self.assertFalse(res.is_crash)
        self.assertIn("PATH_HDR_INVALID", res.coverage)
        self.assertNotIn("PATH_FIELDS_PARSED", res.coverage)
        self.assertNotIn("PATH_DEEP_STATE_1", res.coverage)
        self.assertEqual(extract_coverage_depth(res.coverage), 1)

    def test_9_initial_seeds_do_not_immediately_crash(self):
        """Criterion 9: None of the standard initial benchmark seeds crash."""
        corpus_dir = Path("corpus_structured")
        seed_files = list(corpus_dir.glob("seed*.txt"))
        self.assertGreaterEqual(len(seed_files), 5)

        for seed_file in seed_files:
            data = seed_file.read_bytes()
            res = self.executor.run(data)
            self.assertFalse(
                res.is_crash,
                f"Seed file '{seed_file.name}' should not cause a crash upon initial load",
            )
            self.assertGreater(len(res.coverage), 3)


if __name__ == "__main__":
    unittest.main()
