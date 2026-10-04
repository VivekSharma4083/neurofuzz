"""Target execution and crash detection module for NeuroFuzz.

Executes a target binary with test inputs using subprocess, enforces timeouts,
captures output, and detects crashes, signals, and abnormal terminations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Set, Tuple, Union

from fuzzer.coverage import parse_coverage_markers


# Common Windows NTSTATUS exception codes
WINDOWS_FAULT_CODES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION (Segfault)",
    3221225477: "STATUS_ACCESS_VIOLATION (Segfault)",
    -1073741819: "STATUS_ACCESS_VIOLATION (Segfault)",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    3221225620: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    -1073741676: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
    3221226505: "STATUS_STACK_BUFFER_OVERRUN",
    -1073740791: "STATUS_STACK_BUFFER_OVERRUN",
    0xC0000028: "STATUS_BAD_STACK",
    0x80000003: "STATUS_BREAKPOINT",
    3: "SIGABRT / abort()",
}

# Common POSIX signal names mapped from negative return codes
POSIX_SIGNALS = {
    -4: "SIGILL (Illegal Instruction)",
    -6: "SIGABRT (Abort)",
    -8: "SIGFPE (Floating Point Exception)",
    -9: "SIGKILL (Killed)",
    -11: "SIGSEGV (Segmentation Fault)",
    -14: "SIGALRM (Alarm Clock)",
}


@dataclass
class ExecutionResult:
    """Encapsulates the result of a single target execution."""

    returncode: Optional[int]
    stdout: bytes
    stderr: bytes
    timeout: bool
    duration: float
    is_crash: bool
    failure_type: str
    description: str
    coverage: Set[str] = field(default_factory=set)
    depth: int = 0

    def __post_init__(self) -> None:
        if self.depth == 0 and self.coverage:
            for marker in self.coverage:
                if marker.startswith("DEPTH_"):
                    try:
                        d = int(marker.split("_")[1])
                        if d > self.depth:
                            self.depth = d
                    except (ValueError, IndexError):
                        pass

    @property
    def return_code(self) -> Optional[int]:
        """Alias for returncode."""
        return self.returncode

    @property
    def timed_out(self) -> bool:
        """Alias for timeout."""
        return self.timeout

    @property
    def crashed(self) -> bool:
        """Alias for is_crash."""
        return self.is_crash

    @property
    def execution_time(self) -> float:
        """Alias for duration."""
        return self.duration


class CrashDetector:
    """Analyzes execution outcome to determine if an interesting failure occurred."""

    @staticmethod
    def classify(
        returncode: Optional[int],
        timeout: bool,
    ) -> Tuple[bool, str, str]:
        """Classify whether the execution represents a crash, timeout, or normal run.

        Returns:
            Tuple of (is_crash, failure_type, description)
        """
        if timeout:
            return True, "timeout", "Execution timed out (potential hang or infinite loop)"

        if returncode is None:
            return True, "unknown", "Process terminated with unknown status"

        if returncode == 0:
            return False, "ok", "Execution completed normally (exit code 0)"

        # Check POSIX signals (negative return codes)
        if returncode in POSIX_SIGNALS:
            sig_name = POSIX_SIGNALS[returncode]
            return True, "signal", f"Terminated by signal {sig_name} (code {returncode})"

        # Check Windows exception codes
        if returncode in WINDOWS_FAULT_CODES:
            name = WINDOWS_FAULT_CODES[returncode]
            return True, "crash", f"Crashed with fault {name} (code {returncode})"

        # Other abnormal termination / non-zero exit codes
        return True, "nonzero_exit", f"Non-zero exit code: {returncode}"


class BaseExecutor:
    """Abstract base class for target execution harnesses."""

    def execute(self, data: bytes) -> ExecutionResult:
        """Execute target with given input bytes.

        Args:
            data: Raw input bytes.

        Returns:
            ExecutionResult containing execution metrics.
        """
        raise NotImplementedError

    def run(self, data: bytes) -> ExecutionResult:
        """Alias for execute() to ensure backward compatibility."""
        return self.execute(data)

    def close(self) -> None:
        """Clean up process resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Executor(BaseExecutor):
    """Executes target binary with generated inputs via subprocess."""


    def __init__(
        self,
        target_path: Union[str, Path],
        timeout: float = 1.0,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        """Initialize target executor.

        Args:
            target_path: Path to the target executable.
            timeout: Maximum execution time in seconds before terminating.
            extra_args: Optional list of additional command line arguments.
        """
        self.target_path = Path(target_path).resolve()
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target executable not found: {self.target_path}")

        self.timeout = timeout
        self.extra_args = extra_args or []

        # On Windows, suppress the system crash/WerFault modal dialog
        # so child processes terminate immediately with their crash NTSTATUS.
        if sys.platform.startswith("win"):
            try:
                import ctypes
                # SEM_FAILCRITICALERRORS (0x0001) | SEM_NOGPFAULTERRORBOX (0x0002)
                ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
            except Exception:
                pass

    def execute(self, data: bytes) -> ExecutionResult:
        """Run the target process passing data via stdin.

        Args:
            data: Raw input bytes to feed into stdin.

        Returns:
            ExecutionResult containing execution metrics and crash classification.
        """
        cmd = [str(self.target_path)] + self.extra_args
        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                input=data,
                capture_output=True,
                timeout=self.timeout,
            )
            duration = time.perf_counter() - start_time
            cov_units, cleaned_stderr = parse_coverage_markers(proc.stderr)
            is_crash, fail_type, desc = CrashDetector.classify(proc.returncode, timeout=False)
            return ExecutionResult(
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=cleaned_stderr,
                timeout=False,
                duration=duration,
                is_crash=is_crash,
                failure_type=fail_type,
                description=desc,
                coverage=cov_units,
            )

        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start_time
            raw_stdout = exc.stdout or b""
            raw_stderr = exc.stderr or b""
            raw_stderr_bytes = raw_stderr if isinstance(raw_stderr, bytes) else raw_stderr.encode("latin1", "replace")
            cov_units, cleaned_stderr = parse_coverage_markers(raw_stderr_bytes)
            is_crash, fail_type, desc = CrashDetector.classify(None, timeout=True)
            return ExecutionResult(
                returncode=None,
                stdout=raw_stdout if isinstance(raw_stdout, bytes) else raw_stdout.encode("latin1", "replace"),
                stderr=cleaned_stderr,
                timeout=True,
                duration=duration,
                is_crash=is_crash,
                failure_type=fail_type,
                description=desc,
                coverage=cov_units,
            )


class CrashSaver:
    """Manages saving crashing inputs and reproduction metadata to disk."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """Initialize crash saver with output directory.

        Args:
            output_dir: Path to crashes directory.
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.saved_hashes: Set[str] = set()

    def save(
        self,
        data: bytes,
        result: ExecutionResult,
        iteration: int,
    ) -> Tuple[bool, Optional[Path]]:
        """Save a crashing input and its metadata to the crashes directory.

        Deduplicates inputs based on their SHA-256 hash.

        Args:
            data: Mutated input bytes that triggered the crash.
            result: The ExecutionResult detailing the crash.
            iteration: The fuzzing iteration number.

        Returns:
            Tuple of (is_new_crash, path_to_saved_input_file)
        """
        data_hash = hashlib.sha256(data).hexdigest()
        if data_hash in self.saved_hashes:
            return False, None

        self.saved_hashes.add(data_hash)
        short_hash = data_hash[:10]
        safe_type = result.failure_type.replace(" ", "_")

        base_name = f"crash_iter{iteration:06d}_{safe_type}_{short_hash}"
        input_path = self.output_dir / f"{base_name}.bin"
        meta_path = self.output_dir / f"{base_name}_meta.json"

        # Write exact byte payload
        input_path.write_bytes(data)

        # Write metadata JSON
        meta = {
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failure_type": result.failure_type,
            "description": result.description,
            "returncode": result.returncode,
            "timeout": result.timeout,
            "duration_seconds": round(result.duration, 5),
            "sha256": data_hash,
            "size_bytes": len(data),
            "stdout": result.stdout.decode("latin1", errors="replace"),
            "stderr": result.stderr.decode("latin1", errors="replace"),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return True, input_path


SubprocessExecutor = Executor
