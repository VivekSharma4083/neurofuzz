"""Persistent target execution harness for NeuroFuzz Phase 7A.

Maintains a long-lived persistent process across fuzzing iterations to eliminate
per-execution process-spawn overhead while providing strict state isolation,
crash recovery, and timeout protection.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Set, Tuple, Union

from fuzzer.coverage import parse_coverage_markers
from fuzzer.executor import BaseExecutor, CrashDetector, ExecutionResult


class PersistentExecutor(BaseExecutor):
    """Executes target inputs against a persistent child process via framed pipe IPC."""

    def __init__(
        self,
        target_path: Union[str, Path],
        timeout: float = 1.0,
        extra_args: Optional[List[str]] = None,
        auto_start: bool = True,
    ) -> None:
        """Initialize persistent target executor.

        Args:
            target_path: Path to the target executable.
            timeout: Maximum execution time in seconds before terminating.
            extra_args: Optional list of additional command line arguments.
            auto_start: If True, immediately start the persistent process.
        """
        self.target_path = Path(target_path).resolve()
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target executable not found: {self.target_path}")

        self.timeout = float(timeout)
        self.extra_args = extra_args or []
        self.proc: Optional[subprocess.Popen] = None
        self._pool: Optional[ThreadPoolExecutor] = None

        # On Windows, suppress the system crash/WerFault modal dialog
        if sys.platform.startswith("win"):
            try:
                import ctypes

                ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
            except Exception:
                pass

        if auto_start:
            self.start()

    def is_alive(self) -> bool:
        """Check if the persistent child process is currently running."""
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        """Launch the persistent target process and await initial handshake."""
        if self.is_alive():
            return

        self._pool = ThreadPoolExecutor(max_workers=1)

        cmd = [str(self.target_path), "--persistent"] + self.extra_args
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        def read_handshake() -> bytes:
            assert self.proc is not None and self.proc.stdout is not None
            return self.proc.stdout.readline()

        future = self._pool.submit(read_handshake)
        try:
            line = future.result(timeout=self.timeout)
            if line.strip() != b"READY":
                self.stop()
                raise RuntimeError(f"Unexpected initial handshake: {line!r}")
        except (FutureTimeoutError, Exception) as exc:
            self.stop()
            raise RuntimeError(f"Failed to establish persistent handshake with target: {exc}") from exc

    def stop(self) -> None:
        """Terminate the persistent process and clean up resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

        if self.proc is not None:
            try:
                if self.proc.stdin is not None and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception:
                pass

            if self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=0.2)
                except Exception:
                    try:
                        self.proc.kill()
                        self.proc.wait(timeout=0.5)
                    except Exception:
                        pass

            try:
                if self.proc.stdout is not None and not self.proc.stdout.closed:
                    self.proc.stdout.close()
                if self.proc.stderr is not None and not self.proc.stderr.closed:
                    self.proc.stderr.close()
            except Exception:
                pass

            self.proc = None

    def restart(self) -> None:
        """Restart the persistent target process."""
        self.stop()
        self.start()

    def close(self) -> None:
        """Clean up resources on shutdown."""
        self.stop()

    def execute(self, data: bytes) -> ExecutionResult:
        """Execute target with input data over persistent framed pipe."""
        if not self.is_alive():
            self.start()

        start_time = time.perf_counter()

        # Send input framed as length\n + payload
        frame = f"{len(data)}\n".encode("ascii") + data
        try:
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write(frame)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            duration = time.perf_counter() - start_time
            return_code = self.proc.poll() if self.proc else None
            is_crash, fail_type, desc = CrashDetector.classify(return_code, timeout=False)
            self.restart()
            return ExecutionResult(
                returncode=return_code,
                stdout=b"",
                stderr=b"Broken pipe on write",
                timeout=False,
                duration=duration,
                is_crash=is_crash,
                failure_type=fail_type,
                description=desc,
                coverage=set(),
                depth=0,
            )

        def read_response() -> Tuple[List[bytes], bool, int]:
            assert self.proc is not None and self.proc.stdout is not None
            cov_lines: List[bytes] = []
            parsed_depth = 0
            saw_ready = False
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                if line == b"READY\n":
                    saw_ready = True
                    break
                if line.startswith(b"RESULT "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            parsed_depth = int(parts[3])
                        except ValueError:
                            pass
                else:
                    cov_lines.append(line)
            return cov_lines, saw_ready, parsed_depth

        assert self._pool is not None
        future = self._pool.submit(read_response)

        try:
            cov_lines, saw_ready, parsed_depth = future.result(timeout=self.timeout)
            duration = time.perf_counter() - start_time
            raw_output = b"".join(cov_lines)
            cov_units, _ = parse_coverage_markers(raw_output)

            # Check if process terminated abnormally during execution
            if not saw_ready or not self.is_alive():
                proc_exit = None
                if self.proc is not None:
                    try:
                        proc_exit = self.proc.wait(timeout=0.5)
                    except Exception:
                        proc_exit = self.proc.poll()
                is_crash, fail_type, desc = CrashDetector.classify(proc_exit, timeout=False)
                self.restart()
                return ExecutionResult(
                    returncode=proc_exit,
                    stdout=raw_output,
                    stderr=b"",
                    timeout=False,
                    duration=duration,
                    is_crash=is_crash,
                    failure_type=fail_type,
                    description=desc,
                    coverage=cov_units,
                    depth=parsed_depth,
                )

            # Normal successful execution
            return ExecutionResult(
                returncode=0,
                stdout=raw_output,
                stderr=b"",
                timeout=False,
                duration=duration,
                is_crash=False,
                failure_type="ok",
                description="Execution completed normally (exit code 0)",
                coverage=cov_units,
                depth=parsed_depth,
            )

        except FutureTimeoutError:
            duration = time.perf_counter() - start_time
            self.stop()
            self.start()
            is_crash, fail_type, desc = CrashDetector.classify(None, timeout=True)
            return ExecutionResult(
                returncode=None,
                stdout=b"",
                stderr=b"Target timed out",
                timeout=True,
                duration=duration,
                is_crash=is_crash,
                failure_type=fail_type,
                description=desc,
                coverage=set(),
                depth=0,
            )
