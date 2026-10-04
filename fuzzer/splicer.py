"""Structured crossover and field splicing engine for NeuroFuzz (Phase 6E).

Provides generic delimiter-based field parsing, longest common prefix detection,
duplicate-preventing field recombination, and multi-strategy splicing across
corpus seeds.
"""

from dataclasses import dataclass, field
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class SplicerTelemetry:
    """Telemetry data captured during structured field splicing."""

    splice_attempts: int = 0
    successful_splices: int = 0
    rejected_splices: int = 0
    duplicate_prevention_events: int = 0
    strategy_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "prefix_suffix": 0,
            "single_field_append": 0,
            "field_segment": 0,
        }
    )
    total_parent_a_fields: int = 0
    total_parent_b_fields: int = 0
    total_result_fields: int = 0
    coverage_discoveries: int = 0
    max_depth_reached: int = 0
    transition_14_15_count: int = 0

    @property
    def avg_parent_a_fields(self) -> float:
        """Return average field count of parent A."""
        return self.total_parent_a_fields / max(1, self.successful_splices)

    @property
    def avg_parent_b_fields(self) -> float:
        """Return average field count of parent B."""
        return self.total_parent_b_fields / max(1, self.successful_splices)

    @property
    def avg_result_fields(self) -> float:
        """Return average field count of successful candidates."""
        return self.total_result_fields / max(1, self.successful_splices)

    @property
    def success_rate(self) -> float:
        """Return proportion of attempts that successfully produced valid new candidates."""
        return self.successful_splices / max(1, self.splice_attempts)


class FieldSplicer:
    """Generic delimiter-aware field parser and crossover recombiner."""

    STRATEGIES: List[str] = [
        "prefix_suffix",
        "single_field_append",
        "field_segment",
    ]

    def __init__(
        self,
        delimiter: bytes = b"|",
        max_splice_size: int = 512,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize the FieldSplicer.

        Args:
            delimiter: Byte delimiter separating protocol fields (default: b"|").
            max_splice_size: Maximum allowed byte length of spliced candidates (default: 512).
            rng: Optional seeded random.Random instance for deterministic execution.
        """
        if not delimiter:
            raise ValueError("delimiter cannot be empty")
        if max_splice_size <= 0:
            raise ValueError(f"max_splice_size must be positive, got {max_splice_size}")

        self.delimiter: bytes = delimiter
        self.max_splice_size: int = int(max_splice_size)
        self.rng: random.Random = rng if rng is not None else random.Random()
        self.telemetry = SplicerTelemetry()

    def split_fields(self, data: bytes) -> List[bytes]:
        """Parse raw byte payload into individual field segments.

        Extracts non-empty field segments separated by the delimiter.
        Works generically for arbitrary opcodes (PING, INFO, CALC, WRITE, AUTH, DIAG)
        and arbitrary subfields.

        Args:
            data: Raw input bytes.

        Returns:
            List of non-empty byte fields.
        """
        if not data:
            return []
        raw_parts = data.split(self.delimiter)
        # Filter out empty tokens to avoid accidental empty fields or delimiter repetition
        return [part for part in raw_parts if len(part) > 0]

    def join_fields(self, fields: Sequence[bytes]) -> bytes:
        """Serialize a sequence of fields using the delimiter.

        Args:
            fields: Sequence of field bytes.

        Returns:
            Serialized byte string.
        """
        if not fields:
            return b""
        return self.delimiter.join(fields)

    def find_common_prefix(
        self,
        fields_a: Sequence[bytes],
        fields_b: Sequence[bytes],
    ) -> List[bytes]:
        """Find the longest common prefix between two field sequences.

        Args:
            fields_a: Field sequence of parent A.
            fields_b: Field sequence of parent B.

        Returns:
            List of matching prefix fields in order.
        """
        prefix: List[bytes] = []
        for fa, fb in zip(fields_a, fields_b):
            if fa == fb:
                prefix.append(fa)
            else:
                break
        return prefix

    def deduplicate_adjacent(self, fields: Sequence[bytes]) -> List[bytes]:
        """Remove immediately adjacent duplicate fields while preserving field order.

        Example:
            [A, B, B, C] -> [A, B, C]

        Args:
            fields: Sequence of fields.

        Returns:
            Deduplicated list of fields.
        """
        deduped: List[bytes] = []
        for f in fields:
            if not deduped or f != deduped[-1]:
                deduped.append(f)
            else:
                self.telemetry.duplicate_prevention_events += 1
        return deduped

    def select_donor(
        self,
        corpus_seeds: Sequence[Union[bytes, Any]],
        parent_a: bytes,
    ) -> Optional[bytes]:
        """Select a random donor seed (Parent B) distinct from Parent A.

        Args:
            corpus_seeds: Sequence of available seed byte payloads or SeedRecords.
            parent_a: Payload of Parent A.

        Returns:
            Donor bytes if a distinct seed exists, else None.
        """
        if not corpus_seeds:
            return None

        candidates: List[bytes] = []
        for s in corpus_seeds:
            data = bytes(s.data) if hasattr(s, "data") else bytes(s)
            if data != parent_a and len(data) > 0:
                candidates.append(data)

        if not candidates:
            return None

        return self.rng.choice(candidates)

    def validate_candidate(
        self,
        candidate_bytes: bytes,
        parent_a: bytes,
    ) -> bool:
        """Validate candidate payload against structural constraints.

        Constraints:
        - Must not be empty.
        - Must not exceed max_splice_size.
        - Must differ from Parent A.
        - Must contain at least one valid field.

        Args:
            candidate_bytes: Newly synthesized candidate bytes.
            parent_a: Original Parent A bytes.

        Returns:
            True if candidate is valid, False otherwise.
        """
        if not candidate_bytes:
            return False
        if len(candidate_bytes) > self.max_splice_size:
            return False
        if candidate_bytes == parent_a:
            return False
        fields = self.split_fields(candidate_bytes)
        if not fields:
            return False
        return True

    def splice_prefix_suffix(
        self,
        fields_a: Sequence[bytes],
        fields_b: Sequence[bytes],
    ) -> Optional[List[bytes]]:
        """Strategy A: Longest Common Prefix + Donor Suffix.

        Identifies common prefix P of A and B, then appends a sub-slice
        of donor fields from B that follow P.
        """
        prefix = self.find_common_prefix(fields_a, fields_b)
        len_p = len(prefix)

        # Donor suffix contains fields in B following the common prefix
        donor_suffix = fields_b[len_p:]
        if not donor_suffix:
            # If donor B has no additional fields beyond prefix,
            # check if B has fields that can be appended
            return None

        # Choose length of donor suffix slice: k in [1, len(donor_suffix)]
        k = self.rng.randint(1, len(donor_suffix))
        chosen_suffix = donor_suffix[:k]

        combined = list(prefix) + list(chosen_suffix)
        return self.deduplicate_adjacent(combined)

    def splice_single_field_append(
        self,
        fields_a: Sequence[bytes],
        fields_b: Sequence[bytes],
    ) -> Optional[List[bytes]]:
        """Strategy B: Single Field Append.

        Selects a single field from donor B and appends it to parent A,
        preventing immediate adjacent duplication.
        """
        if not fields_b:
            return None

        candidate = list(fields_a)
        last_field = candidate[-1] if candidate else None

        # Pick candidate donor fields that don't immediately repeat last field
        eligible = [f for f in fields_b if f != last_field]
        if not eligible:
            # If all fields in B match last field of A, cannot append without immediate duplicate
            return None

        chosen_field = self.rng.choice(eligible)
        candidate.append(chosen_field)
        return self.deduplicate_adjacent(candidate)

    def splice_field_segment(
        self,
        fields_a: Sequence[bytes],
        fields_b: Sequence[bytes],
    ) -> Optional[List[bytes]]:
        """Strategy C: Field Segment Splice.

        Extracts a contiguous field segment from donor B and inserts it
        into parent A after the common prefix.
        """
        if not fields_b:
            return None

        prefix = self.find_common_prefix(fields_a, fields_b)
        insert_idx = len(prefix)

        # Choose a contiguous segment in B
        # Prefer segments following prefix if available
        if len(fields_b) > insert_idx:
            sub_b = fields_b[insert_idx:]
            start = self.rng.randint(0, len(sub_b) - 1)
            end = self.rng.randint(start + 1, len(sub_b))
            segment = sub_b[start:end]
        else:
            start = self.rng.randint(0, len(fields_b) - 1)
            end = self.rng.randint(start + 1, len(fields_b))
            segment = fields_b[start:end]

        candidate = list(fields_a[:insert_idx]) + list(segment) + list(fields_a[insert_idx:])
        return self.deduplicate_adjacent(candidate)

    def splice(
        self,
        parent_a: bytes,
        parent_b: bytes,
        strategy: Optional[str] = None,
    ) -> Optional[bytes]:
        """Recombine parent_a and parent_b into a new candidate payload.

        Args:
            parent_a: Primary seed bytes.
            parent_b: Donor seed bytes.
            strategy: Optional explicit strategy ("prefix_suffix", "single_field_append",
                or "field_segment"). If None, chosen randomly.

        Returns:
            Mutated bytes if successful and valid, or None if rejected.
        """
        self.telemetry.splice_attempts += 1

        fields_a = self.split_fields(parent_a)
        fields_b = self.split_fields(parent_b)

        if not fields_a or not fields_b or parent_a == parent_b:
            self.telemetry.rejected_splices += 1
            return None

        # Choose strategy
        strat = strategy if strategy is not None else self.rng.choice(self.STRATEGIES)
        if strat not in self.STRATEGIES:
            strat = self.rng.choice(self.STRATEGIES)

        result_fields: Optional[List[bytes]] = None

        if strat == "prefix_suffix":
            result_fields = self.splice_prefix_suffix(fields_a, fields_b)
        elif strat == "single_field_append":
            result_fields = self.splice_single_field_append(fields_a, fields_b)
        elif strat == "field_segment":
            result_fields = self.splice_field_segment(fields_a, fields_b)

        # Fallback among alternative strategies if primary produced None
        if result_fields is None:
            remaining = [s for s in self.STRATEGIES if s != strat]
            for fallback_strat in remaining:
                if fallback_strat == "prefix_suffix":
                    result_fields = self.splice_prefix_suffix(fields_a, fields_b)
                elif fallback_strat == "single_field_append":
                    result_fields = self.splice_single_field_append(fields_a, fields_b)
                elif fallback_strat == "field_segment":
                    result_fields = self.splice_field_segment(fields_a, fields_b)
                if result_fields is not None:
                    strat = fallback_strat
                    break

        if result_fields is None:
            self.telemetry.rejected_splices += 1
            return None

        candidate = self.join_fields(result_fields)

        # Apply structural validity filter
        if not self.validate_candidate(candidate, parent_a):
            self.telemetry.rejected_splices += 1
            return None

        # Record success telemetry
        self.telemetry.successful_splices += 1
        self.telemetry.strategy_counts[strat] = self.telemetry.strategy_counts.get(strat, 0) + 1
        self.telemetry.total_parent_a_fields += len(fields_a)
        self.telemetry.total_parent_b_fields += len(fields_b)
        self.telemetry.total_result_fields += len(result_fields)

        return candidate

    def get_telemetry(self) -> Dict[str, Any]:
        """Return comprehensive telemetry dictionary for inspection and logging."""
        return {
            "splice_attempts": self.telemetry.splice_attempts,
            "successful_splices": self.telemetry.successful_splices,
            "rejected_splices": self.telemetry.rejected_splices,
            "success_rate": self.telemetry.success_rate,
            "duplicate_prevention_events": self.telemetry.duplicate_prevention_events,
            "strategy_counts": dict(self.telemetry.strategy_counts),
            "avg_parent_a_fields": round(self.telemetry.avg_parent_a_fields, 2),
            "avg_parent_b_fields": round(self.telemetry.avg_parent_b_fields, 2),
            "avg_result_fields": round(self.telemetry.avg_result_fields, 2),
            "coverage_discoveries": self.telemetry.coverage_discoveries,
            "max_depth_reached": self.telemetry.max_depth_reached,
            "transition_14_15_count": self.telemetry.transition_14_15_count,
        }
