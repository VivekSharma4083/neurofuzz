"""Coverage tracking and parsing module for NeuroFuzz.

Maintains global coverage state, per-execution coverage observations,
and computes newly discovered coverage frontiers.
"""

from typing import FrozenSet, Iterable, List, Optional, Set, Tuple, Union

CoverageUnit = Union[int, str]
COV_PREFIX = "__COV__:"


def parse_coverage_markers(stream: bytes) -> Tuple[Set[str], bytes]:
    """Extract coverage markers from an output stream (e.g. stderr).

    Lines matching '__COV__:<identifier>' are parsed as coverage units.
    All other lines are retained in the returned cleaned byte stream.

    Args:
        stream: Raw bytes captured from target output.

    Returns:
        Tuple of (set_of_coverage_units, cleaned_stream_bytes).
    """
    coverage_units: Set[str] = set()
    cleaned_lines: List[bytes] = []

    for line in stream.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(b"__COV__:"):
            marker = stripped[len(b"__COV__:") :].decode("utf-8", errors="replace").strip()
            if marker:
                coverage_units.add(marker)
        else:
            cleaned_lines.append(line)

    return coverage_units, b"".join(cleaned_lines)


class CoverageTracker:
    """Tracks global and per-run coverage units discovered during fuzzing."""

    def __init__(self) -> None:
        """Initialize an empty coverage tracker."""
        self._global_coverage: Set[CoverageUnit] = set()
        self._total_discoveries: int = 0
        self._history: List[Set[CoverageUnit]] = []

    @property
    def global_coverage(self) -> FrozenSet[CoverageUnit]:
        """Return an immutable view of all unique coverage units seen so far."""
        return frozenset(self._global_coverage)

    @property
    def total_units(self) -> int:
        """Return total count of unique coverage units discovered."""
        return len(self._global_coverage)

    @property
    def total_discoveries(self) -> int:
        """Return total count of interesting inputs that discovered new coverage."""
        return self._total_discoveries

    def get_new_coverage(self, observed: Iterable[CoverageUnit]) -> Set[CoverageUnit]:
        """Compute coverage units in `observed` that are not yet in global coverage.

        Args:
            observed: Coverage units observed during a target execution.

        Returns:
            Set of newly discovered coverage units.
        """
        return set(observed) - self._global_coverage

    def has_new_coverage(self, observed: Iterable[CoverageUnit]) -> bool:
        """Check if `observed` contains any coverage units not yet in global coverage.

        Args:
            observed: Coverage units observed during a target execution.

        Returns:
            True if any new coverage is present, False otherwise.
        """
        for unit in observed:
            if unit not in self._global_coverage:
                return True
        return False

    def update(self, observed: Iterable[CoverageUnit]) -> Tuple[bool, Set[CoverageUnit]]:
        """Update global coverage with observed units from an execution.

        Args:
            observed: Coverage units observed during a target execution.

        Returns:
            Tuple of (is_new_coverage, set_of_newly_discovered_units).
        """
        observed_set = set(observed)
        new_units = observed_set - self._global_coverage

        if new_units:
            self._global_coverage.update(new_units)
            self._total_discoveries += 1
            self._history.append(new_units)
            return True, new_units

        return False, set()

    @property
    def max_depth(self) -> int:
        """Return the maximum logical depth reached across all recorded coverage."""
        return extract_coverage_depth(self._global_coverage)

    def reset(self) -> None:
        """Reset all tracked coverage data."""
        self._global_coverage.clear()
        self._total_discoveries = 0
        self._history.clear()

    def __len__(self) -> int:
        """Return number of unique coverage units recorded."""
        return len(self._global_coverage)

    def __repr__(self) -> str:
        return f"<CoverageTracker units={len(self._global_coverage)} discoveries={self._total_discoveries} max_depth={self.max_depth}>"


def extract_coverage_depth(coverage_units: Iterable[CoverageUnit]) -> int:
    """Extract maximum numerical depth reached from DEPTH_XX coverage markers.

    Args:
        coverage_units: Collection of observed coverage markers.

    Returns:
        Integer depth (e.g. 19 for DEPTH_19), or 0 if no depth markers found.
    """
    max_d = 0
    for unit in coverage_units:
        if isinstance(unit, str) and unit.startswith("DEPTH_"):
            try:
                val = int(unit.split("_")[1])
                if val > max_d:
                    max_d = val
            except (IndexError, ValueError):
                pass
    return max_d
