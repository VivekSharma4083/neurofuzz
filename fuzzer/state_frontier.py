"""State-Sequence-Aware Frontier Mutation module for NeuroFuzz (Phase 7C).

Maintains a structured state frontier tracking the deepest discovered program
states and systematically enumerates boundary-aware dictionary extensions
without hard-coding target-specific sequences.
"""

from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import random
import time
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from fuzzer.boundary import BoundaryDetector, find_delimiter_boundaries, format_boundary_insertion
from fuzzer.corpus import SeedRecord
from fuzzer.coverage import CoverageUnit, extract_coverage_depth
from fuzzer.dictionary import Dictionary


@dataclass
class FrontierCandidate:
    """Represents a systematically generated mutation candidate for a frontier seed."""

    seed_id: int
    seed_name: str
    seed_hash: str
    parent_depth: int
    boundary_pos: int
    token: bytes
    mutated_data: bytes
    mutation_type: str = "boundary_insert"
    candidate_key: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_key:
            token_repr = self.token.hex()
            self.candidate_key = (
                f"{self.seed_hash}:{self.boundary_pos}:{token_repr}:{self.mutation_type}"
            )


@dataclass
class FrontierSeed:
    """Metadata and exploration state for a representative frontier seed."""

    record: SeedRecord
    depth: int
    discovery_iteration: int
    candidates_generated: int = 0
    candidates_attempted: int = 0
    exhausted: bool = False
    attempted_candidate_hashes: Set[str] = field(default_factory=set)


@dataclass
class FrontierTelemetry:
    """Observability telemetry for state frontier exploration (Phase 7C)."""

    frontier_extension_attempts: int = 0
    unique_frontier_candidates: int = 0
    frontier_candidates_exhausted: int = 0
    frontier_discoveries: int = 0
    frontier_coverage_discovered: int = 0
    frontier_depth_discoveries: int = 0
    frontier_seed_selected_count: int = 0
    deepest_frontier_reached: int = 0
    candidate_generation_cost_ms: float = 0.0
    total_candidate_generation_time_s: float = 0.0
    candidate_generation_calls: int = 0
    depth_transitions: Dict[str, int] = field(default_factory=dict)
    rate_mode: str = "fixed"
    current_frontier_rate: float = 0.20
    starting_frontier_rate: float = 0.20
    min_frontier_rate: float = 0.02
    max_frontier_rate: float = 0.30
    rate_changes_count: int = 0
    stagnant_frontier_attempts: int = 0
    iterations_since_last_depth_advance: int = 0
    last_rate_change_iteration: int = 0
    last_rate_change_reason: str = "init"

    @property
    def frontier_mutation_share(self) -> float:
        """Fraction of total mutations allocated to frontier extension."""
        return 0.0  # Computed dynamically against total fuzzer executions

    @property
    def successful_frontier_mutation_rate(self) -> float:
        """Rate at which frontier mutations yield new coverage."""
        return self.frontier_discoveries / max(1, self.frontier_extension_attempts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to a serializable dictionary."""
        return {
            "frontier_extension_attempts": self.frontier_extension_attempts,
            "unique_frontier_candidates": self.unique_frontier_candidates,
            "frontier_candidates_exhausted": self.frontier_candidates_exhausted,
            "frontier_discoveries": self.frontier_discoveries,
            "frontier_coverage_discovered": self.frontier_coverage_discovered,
            "frontier_depth_discoveries": self.frontier_depth_discoveries,
            "frontier_seed_selected_count": self.frontier_seed_selected_count,
            "deepest_frontier_reached": self.deepest_frontier_reached,
            "candidate_generation_cost_ms": round(self.candidate_generation_cost_ms, 4),
            "successful_frontier_mutation_rate": round(self.successful_frontier_mutation_rate, 4),
            "depth_transitions": dict(self.depth_transitions),
            "rate_mode": self.rate_mode,
            "current_frontier_rate": round(self.current_frontier_rate, 4),
            "starting_frontier_rate": round(self.starting_frontier_rate, 4),
            "min_frontier_rate": round(self.min_frontier_rate, 4),
            "max_frontier_rate": round(self.max_frontier_rate, 4),
            "rate_changes_count": self.rate_changes_count,
            "stagnant_frontier_attempts": self.stagnant_frontier_attempts,
            "iterations_since_last_depth_advance": self.iterations_since_last_depth_advance,
            "last_rate_change_iteration": self.last_rate_change_iteration,
            "last_rate_change_reason": self.last_rate_change_reason,
        }


class StateFrontier:
    """State-Sequence-Aware Frontier Search Engine.

    Tracks the deepest reachable states in the corpus and systematically enumerates
    valid structural boundary extensions using the protocol dictionary.
    """

    def __init__(
        self,
        dictionary: Optional[Dictionary] = None,
        delimiter: bytes = b"|",
        boundary_detector: Optional[BoundaryDetector] = None,
        rng: Optional[random.Random] = None,
        max_candidates_per_seed: int = 1500,
        priority_burst_budget: int = 250,
        rate_mode: str = "fixed",
        frontier_rate: float = 0.20,
        min_frontier_rate: float = 0.02,
        max_frontier_rate: float = 0.30,
        decay_factor: float = 0.85,
        stagnation_threshold: int = 25,
    ) -> None:
        """Initialize the StateFrontier.

        Args:
            dictionary: Protocol dictionary containing domain tokens.
            delimiter: Protocol field delimiter byte (default: b"|").
            boundary_detector: Optional BoundaryDetector instance.
            rng: Optional random.Random instance for tie-breaking/ordering.
            max_candidates_per_seed: Maximum candidates to generate per frontier seed.
            priority_burst_budget: Number of consecutive frontier iterations to prioritize
                immediately upon discovering a new deepest state.
            rate_mode: Frontier allocation mode ('fixed' or 'adaptive', default: 'fixed').
            frontier_rate: Nominal/starting frontier allocation rate (default: 0.20).
            min_frontier_rate: Minimum frontier allocation floor in adaptive mode (default: 0.02).
            max_frontier_rate: Maximum frontier allocation ceiling in adaptive mode (default: 0.30).
            decay_factor: Multiplicative decay factor applied when frontier stagnates (default: 0.85).
            stagnation_threshold: Number of consecutive unproductive frontier attempts before decay (default: 25).
        """
        self.dictionary = dictionary
        self.delimiter = bytes(delimiter)
        self.boundary_detector = boundary_detector or BoundaryDetector(delimiter=self.delimiter)
        self.rng = rng or random.Random()
        self.max_candidates_per_seed = max_candidates_per_seed
        self.priority_burst_budget = priority_burst_budget

        self.rate_mode = rate_mode.lower()
        self.frontier_rate = float(frontier_rate)
        self.min_frontier_rate = float(min_frontier_rate)
        self.max_frontier_rate = float(max_frontier_rate)
        self.decay_factor = float(decay_factor)
        self.stagnation_threshold = int(stagnation_threshold)

        self._current_rate: float = self.frontier_rate
        self._stagnant_attempts: int = 0
        self._last_depth_advance_iteration: int = 0
        self._last_rate_change_iteration: int = 0
        self._last_rate_change_reason: str = "init"
        self._rate_changes_count: int = 0

        self._frontier_by_depth: Dict[int, FrontierSeed] = {}
        self._max_depth: int = 0
        self._deepest_frontier_seed: Optional[FrontierSeed] = None
        self._candidate_queue: Deque[FrontierCandidate] = deque()
        self._seen_candidate_keys: Set[str] = set()
        self._seen_payload_hashes: Set[str] = set()
        self._priority_burst_remaining: int = 0

        self.telemetry = FrontierTelemetry(
            rate_mode=self.rate_mode,
            current_frontier_rate=self._current_rate,
            starting_frontier_rate=self.frontier_rate,
            min_frontier_rate=self.min_frontier_rate,
            max_frontier_rate=self.max_frontier_rate,
        )

    def set_delimiter(self, delimiter: Union[bytes, bytearray, str]) -> None:
        """Update the structural delimiter used for frontier boundary generation."""
        if isinstance(delimiter, str):
            self.delimiter = delimiter.encode("latin1")
        else:
            self.delimiter = bytes(delimiter)
        self.boundary_detector.set_delimiter(self.delimiter)

    def get_current_rate(self) -> float:
        """Return the current frontier execution allocation probability."""
        if self.rate_mode == "adaptive":
            return self._current_rate
        return self.frontier_rate

    def on_depth_advanced(self, new_depth: int, iteration: int = 0) -> None:
        """Signal that target has discovered a deeper sequential state level."""
        if self.rate_mode == "adaptive":
            old_rate = self._current_rate
            self._current_rate = self.max_frontier_rate
            self._stagnant_attempts = 0
            self._last_depth_advance_iteration = iteration
            self._last_rate_change_iteration = iteration
            self._last_rate_change_reason = f"depth_advanced_to_{new_depth}"
            if abs(self._current_rate - old_rate) >= 0.005:
                self._rate_changes_count += 1
            self.telemetry.current_frontier_rate = self._current_rate
            self.telemetry.rate_changes_count = self._rate_changes_count
            self.telemetry.last_rate_change_iteration = iteration
            self.telemetry.last_rate_change_reason = self._last_rate_change_reason

    @property
    def max_depth(self) -> int:
        """Return the maximum depth recorded in the state frontier."""
        return self._max_depth

    @property
    def deepest_seed(self) -> Optional[SeedRecord]:
        """Return the SeedRecord representing the deepest frontier state."""
        return self._deepest_frontier_seed.record if self._deepest_frontier_seed else None

    @property
    def frontier_by_depth(self) -> Dict[int, FrontierSeed]:
        """Return the mapping of depth levels to representative frontier seeds."""
        return dict(self._frontier_by_depth)

    @property
    def queue_length(self) -> int:
        """Return number of untried candidates currently in the frontier queue."""
        return len(self._candidate_queue)

    def is_priority_active(self) -> bool:
        """Check if an immediate priority burst is currently active."""
        return self._priority_burst_remaining > 0 and len(self._candidate_queue) > 0

    def has_candidate(self) -> bool:
        """Return True if there are untried frontier candidates available."""
        return len(self._candidate_queue) > 0

    def register_seed(
        self,
        record: SeedRecord,
        depth: Optional[int] = None,
        iteration: int = 0,
    ) -> bool:
        """Register a corpus seed into the frontier if it represents a frontier state.

        Args:
            record: SeedRecord to inspect.
            depth: Optional known depth level. If None, extracted from record coverage.
            iteration: Current fuzzing iteration.

        Returns:
            True if the seed established a new maximum depth frontier or updated an
            existing frontier with a cleaner representative; False otherwise.
        """
        if depth is None:
            depth = extract_coverage_depth(record.coverage_units)

        if depth <= 0:
            return False

        is_new_max = depth > self._max_depth
        is_depth_first_seen = depth not in self._frontier_by_depth

        # Prefer shorter payloads for the same depth level
        is_cleaner_seed = False
        if depth in self._frontier_by_depth:
            existing = self._frontier_by_depth[depth]
            if len(record.data) < len(existing.record.data):
                is_cleaner_seed = True

        if is_new_max or is_depth_first_seen or is_cleaner_seed:
            f_seed = FrontierSeed(
                record=record,
                depth=depth,
                discovery_iteration=iteration,
            )
            self._frontier_by_depth[depth] = f_seed

            if depth >= self._max_depth:
                self._max_depth = depth
                self._deepest_frontier_seed = f_seed
                self.telemetry.deepest_frontier_reached = depth

                # Generate candidates for the newly discovered deepest frontier
                new_candidates = self.generate_candidates(f_seed)

                # Prioritize new deeper frontier candidates at the front of the queue
                self._candidate_queue.clear()
                self._candidate_queue.extend(new_candidates)

                # Trigger immediate priority exploration burst for the new frontier
                self._priority_burst_remaining = min(
                    self.priority_burst_budget, len(self._candidate_queue)
                )
                if iteration > 0:
                    self.on_depth_advanced(depth, iteration)

            return True

        return False

    def generate_candidates(self, frontier_seed: FrontierSeed) -> List[FrontierCandidate]:
        """Systematically generate boundary-aware candidate extensions for a frontier seed.

        Enumerates insertion positions prioritizing trailing boundaries (end of sequence),
        followed by right-to-left internal boundaries, across all dictionary tokens.
        Ensures strict deduplication without target-specific hard-coding.

        Args:
            frontier_seed: The FrontierSeed to extend.

        Returns:
            List of unique, untried FrontierCandidate instances.
        """
        t_start = time.perf_counter()
        data = bytes(frontier_seed.record.data)
        parent_hash = hashlib.sha256(data).hexdigest()[:12]

        if self.dictionary is None or len(self.dictionary) == 0:
            return []

        boundaries = self.boundary_detector.find_boundaries(data)
        if not boundaries:
            boundaries = [0, len(data)]

        # Determine trailing boundary position (before trailing newline or at end of buffer)
        trailing_pos = len(data)
        if data.endswith(b"\r\n") and len(data) >= 2:
            trailing_pos = len(data) - 2
        elif data.endswith(b"\n") and len(data) >= 1:
            trailing_pos = len(data) - 1

        # Priority ordering: trailing boundary first (extending sequence),
        # followed by internal boundaries ordered from right to left (tail-first)
        sorted_boundaries = sorted(
            boundaries,
            key=lambda p: (0 if p == trailing_pos or p == len(data) else 1, -p),
        )

        candidates: List[FrontierCandidate] = []
        tokens = list(self.dictionary.tokens)

        # 1. Systematic Boundary Insertions
        for pos in sorted_boundaries:
            for token in tokens:
                clamped_pos, formatted_tok = self.boundary_detector.format_insertion(
                    data, token, pos
                )
                mutated = data[:clamped_pos] + formatted_tok + data[clamped_pos:]

                if mutated == data:
                    continue

                payload_hash = hashlib.sha256(mutated).hexdigest()
                if payload_hash in self._seen_payload_hashes:
                    continue

                cand_key = f"{parent_hash}:{clamped_pos}:{token.hex()}:insert"
                if cand_key in self._seen_candidate_keys:
                    continue

                self._seen_candidate_keys.add(cand_key)
                self._seen_payload_hashes.add(payload_hash)
                frontier_seed.attempted_candidate_hashes.add(cand_key)

                candidate = FrontierCandidate(
                    seed_id=frontier_seed.record.id,
                    seed_name=frontier_seed.record.name,
                    seed_hash=parent_hash,
                    parent_depth=frontier_seed.depth,
                    boundary_pos=clamped_pos,
                    token=token,
                    mutated_data=mutated,
                    mutation_type="boundary_insert",
                    candidate_key=cand_key,
                )
                candidates.append(candidate)

                if len(candidates) >= self.max_candidates_per_seed:
                    break
            if len(candidates) >= self.max_candidates_per_seed:
                break

        # 2. If insertion candidates are few or exhausted, add systematic boundary replacements
        if len(candidates) < min(100, self.max_candidates_per_seed):
            known_tokens = [tok for tok in tokens if len(tok) >= 2 and tok in data]
            for target_tok in known_tokens:
                start_idx = 0
                while True:
                    idx = data.find(target_tok, start_idx)
                    if idx == -1:
                        break
                    start_idx = idx + 1
                    for repl_tok in tokens:
                        if repl_tok == target_tok:
                            continue
                        mutated = data[:idx] + repl_tok + data[idx + len(target_tok) :]
                        payload_hash = hashlib.sha256(mutated).hexdigest()
                        if payload_hash in self._seen_payload_hashes:
                            continue
                        cand_key = f"{parent_hash}:{idx}:{repl_tok.hex()}:replace"
                        if cand_key in self._seen_candidate_keys:
                            continue

                        self._seen_candidate_keys.add(cand_key)
                        self._seen_payload_hashes.add(payload_hash)
                        frontier_seed.attempted_candidate_hashes.add(cand_key)

                        candidate = FrontierCandidate(
                            seed_id=frontier_seed.record.id,
                            seed_name=frontier_seed.record.name,
                            seed_hash=parent_hash,
                            parent_depth=frontier_seed.depth,
                            boundary_pos=idx,
                            token=repl_tok,
                            mutated_data=mutated,
                            mutation_type="boundary_replace",
                            candidate_key=cand_key,
                        )
                        candidates.append(candidate)
                        if len(candidates) >= self.max_candidates_per_seed:
                            break
                    if len(candidates) >= self.max_candidates_per_seed:
                        break
                if len(candidates) >= self.max_candidates_per_seed:
                    break

        elapsed = time.perf_counter() - t_start
        self.telemetry.total_candidate_generation_time_s += elapsed
        self.telemetry.candidate_generation_calls += 1
        self.telemetry.candidate_generation_cost_ms = (
            self.telemetry.total_candidate_generation_time_s
            / max(1, self.telemetry.candidate_generation_calls)
            * 1000.0
        )
        self.telemetry.unique_frontier_candidates += len(candidates)
        frontier_seed.candidates_generated += len(candidates)

        return candidates

    def next_candidate(
        self,
    ) -> Optional[Tuple[SeedRecord, bytes, str, FrontierCandidate]]:
        """Retrieve and pop the next systematically scheduled frontier candidate.

        Returns:
            Tuple of (parent_record, mutated_payload, operator_name, candidate_obj)
            or None if no candidates are currently available.
        """
        if not self._candidate_queue:
            if self._deepest_frontier_seed and not self._deepest_frontier_seed.exhausted:
                self._deepest_frontier_seed.exhausted = True
                self.telemetry.frontier_candidates_exhausted += 1
            return None

        candidate = self._candidate_queue.popleft()
        frontier_seed = self._frontier_by_depth.get(candidate.parent_depth)

        if frontier_seed is None:
            frontier_seed = self._deepest_frontier_seed

        if frontier_seed is not None:
            frontier_seed.candidates_attempted += 1
            if len(self._candidate_queue) == 0:
                frontier_seed.exhausted = True
                self.telemetry.frontier_candidates_exhausted += 1

        self.telemetry.frontier_extension_attempts += 1
        self.telemetry.frontier_seed_selected_count += 1
        if self._priority_burst_remaining > 0:
            self._priority_burst_remaining -= 1

        parent_record = frontier_seed.record if frontier_seed else SeedRecord(
            id=candidate.seed_id,
            data=candidate.mutated_data,
            name=candidate.seed_name,
        )

        return (
            parent_record,
            candidate.mutated_data,
            "state_frontier_extend",
            candidate,
        )

    def on_execution_result(
        self,
        candidate: FrontierCandidate,
        result_coverage: Iterable[CoverageUnit],
        is_new_cov: bool,
        new_units: Set[CoverageUnit],
        iteration: int,
    ) -> None:
        """Process execution feedback for a frontier mutation attempt.

        Args:
            candidate: The FrontierCandidate that was executed.
            result_coverage: The raw coverage markers emitted by the target.
            is_new_cov: Whether this execution discovered novel coverage units.
            new_units: The newly discovered coverage units.
            iteration: Current fuzzing iteration.
        """
        child_depth = extract_coverage_depth(result_coverage)

        if child_depth > candidate.parent_depth:
            self.telemetry.frontier_depth_discoveries += 1
            trans_key = f"{candidate.parent_depth}->{child_depth}"
            self.telemetry.depth_transitions[trans_key] = (
                self.telemetry.depth_transitions.get(trans_key, 0) + 1
            )
            self.on_depth_advanced(child_depth, iteration)

        if is_new_cov:
            self.telemetry.frontier_discoveries += 1
            self.telemetry.frontier_coverage_discovered += len(new_units)
            if self.rate_mode == "adaptive":
                self._stagnant_attempts = 0
                if self._current_rate < self.frontier_rate:
                    old_rate = self._current_rate
                    self._current_rate = min(self.frontier_rate, self._current_rate + 0.02)
                    self._rate_changes_count += 1
                    self._last_rate_change_iteration = iteration
                    self._last_rate_change_reason = "coverage_discovery_nudge"
        else:
            if self.rate_mode == "adaptive":
                self._stagnant_attempts += 1
                if self._stagnant_attempts >= self.stagnation_threshold:
                    old_rate = self._current_rate
                    new_rate = max(self.min_frontier_rate, self._current_rate * self.decay_factor)
                    if new_rate < old_rate:
                        self._current_rate = new_rate
                        self._rate_changes_count += 1
                        self._last_rate_change_iteration = iteration
                        self._last_rate_change_reason = f"stagnation_decay_{self.stagnation_threshold}_attempts"
                    self._stagnant_attempts = 0

        # Update telemetry snapshot
        if self.rate_mode == "adaptive":
            self.telemetry.current_frontier_rate = round(self._current_rate, 4)
            self.telemetry.rate_changes_count = self._rate_changes_count
            self.telemetry.stagnant_frontier_attempts = self._stagnant_attempts
            self.telemetry.iterations_since_last_depth_advance = (
                max(0, iteration - self._last_depth_advance_iteration) if self._last_depth_advance_iteration > 0 else iteration
            )
            self.telemetry.last_rate_change_iteration = self._last_rate_change_iteration
            self.telemetry.last_rate_change_reason = self._last_rate_change_reason

    def reset(self) -> None:
        """Reset all internal state and candidate queues for isolation between runs."""
        self._frontier_by_depth.clear()
        self._max_depth = 0
        self._deepest_frontier_seed = None
        self._candidate_queue.clear()
        self._seen_candidate_keys.clear()
        self._seen_payload_hashes.clear()
        self._priority_burst_remaining = 0
        self._current_rate = self.frontier_rate
        self._stagnant_attempts = 0
        self._last_depth_advance_iteration = 0
        self._last_rate_change_iteration = 0
        self._last_rate_change_reason = "reset"
        self._rate_changes_count = 0
        self.telemetry = FrontierTelemetry(
            rate_mode=self.rate_mode,
            current_frontier_rate=self._current_rate,
            starting_frontier_rate=self.frontier_rate,
            min_frontier_rate=self.min_frontier_rate,
            max_frontier_rate=self.max_frontier_rate,
        )
