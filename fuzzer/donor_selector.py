"""Compatibility-aware donor parent selector for structured crossover (Phase 6F).

Provides generic structural feature extraction, heuristic compatibility scoring,
and compatibility-aware donor selection for multi-seed field splicing.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from fuzzer.coverage import extract_coverage_depth


@dataclass
class DonorCompatibility:
    """Calculated structural compatibility metrics between Parent A and candidate Parent B."""

    common_prefix_fields: int = 0
    same_protocol_prefix: int = 0
    same_command: int = 0
    field_count_difference: int = 0
    depth_difference: int = 0
    donor_extra_fields: int = 0
    compatibility_score: float = 0.0


@dataclass
class DonorSelectionRecord:
    """Telemetry record for a single donor selection event."""

    parent_a: bytes
    selected_parent_b: bytes
    donor_mode: str
    compatibility_score: float
    common_prefix_fields: int
    same_command: bool
    same_protocol_prefix: bool
    donor_extra_fields: int
    has_useful_structure: bool
    result_field_count: int = 0
    new_coverage: int = 0


@dataclass
class DonorSelectorTelemetry:
    """Aggregated telemetry data for donor selection."""

    total_attempts: int = 0
    successful_selections: int = 0
    single_seed_fallbacks: int = 0
    random_selections: int = 0
    compatible_selections: int = 0
    total_selections: int = 0
    useful_donor_selections: int = 0
    same_command_pairings: int = 0
    same_protocol_pairings: int = 0
    total_compatibility_score: float = 0.0
    total_common_prefix_fields: int = 0
    total_donor_extra_fields: int = 0
    compatible_new_coverage: int = 0
    diag_diag_pairings: int = 0
    records: List[DonorSelectionRecord] = field(default_factory=list)

    @property
    def avg_compatibility_score(self) -> float:
        """Return average compatibility score of selected donors."""
        return self.total_compatibility_score / max(1, self.total_selections)

    @property
    def avg_common_prefix(self) -> float:
        """Return average common prefix length (in fields) with selected donors."""
        return self.total_common_prefix_fields / max(1, self.total_selections)

    @property
    def avg_donor_extra_fields(self) -> float:
        """Return average extra fields contributed by selected donors."""
        return self.total_donor_extra_fields / max(1, self.total_selections)

    @property
    def useful_donor_rate(self) -> float:
        """Return proportion of selected donors with donor_extra_fields > 0."""
        return self.useful_donor_selections / max(1, self.total_selections)

    @property
    def same_command_rate(self) -> float:
        """Return proportion of donor pairings with identical command field."""
        return self.same_command_pairings / max(1, self.total_selections)

    @property
    def same_protocol_rate(self) -> float:
        """Return proportion of donor pairings with identical protocol prefix."""
        return self.same_protocol_pairings / max(1, self.total_selections)


class DonorSelector:
    """Selects structurally compatible donor seeds (Parent B) for field splicing."""

    POLICIES: List[str] = ["random", "compatible"]

    def __init__(
        self,
        policy: Optional[str] = None,
        mode: Optional[str] = None,
        delimiter: bytes = b"|",
        donor_prefix_weight: Optional[float] = None,
        prefix_weight: Optional[float] = None,
        donor_command_weight: Optional[float] = None,
        command_weight: Optional[float] = None,
        donor_protocol_weight: Optional[float] = None,
        protocol_weight: Optional[float] = None,
        donor_extra_field_weight: Optional[float] = None,
        extra_field_weight: Optional[float] = None,
        donor_field_diff_penalty: Optional[float] = None,
        field_diff_penalty: Optional[float] = None,
        donor_depth_diff_penalty: Optional[float] = None,
        depth_diff_penalty: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize the DonorSelector.

        Args:
            policy: Donor selection strategy ('random' or 'compatible'). Default: 'random'.
            mode: Alias for policy.
            delimiter: Byte delimiter separating protocol fields (default: b"|").
            donor_prefix_weight: Score weight for each common prefix field (default: 3.0).
            prefix_weight: Alias for donor_prefix_weight.
            donor_command_weight: Score weight for matching command field (default: 2.0).
            command_weight: Alias for donor_command_weight.
            donor_protocol_weight: Score weight for matching protocol prefix (default: 1.0).
            protocol_weight: Alias for donor_protocol_weight.
            donor_extra_field_weight: Score weight for donor fields beyond common prefix (default: 1.0).
            extra_field_weight: Alias for donor_extra_field_weight.
            donor_field_diff_penalty: Penalty weight per field count difference (default: 0.5).
            field_diff_penalty: Alias for donor_field_diff_penalty.
            donor_depth_diff_penalty: Penalty weight per depth difference (default: 0.25).
            depth_diff_penalty: Alias for donor_depth_diff_penalty.
            rng: Optional seeded random.Random instance for reproducible tie-breaking.
        """
        resolved_policy = mode if mode is not None else (policy if policy is not None else "random")
        if resolved_policy not in self.POLICIES:
            raise ValueError(f"Unknown donor policy: '{resolved_policy}'. Expected one of {self.POLICIES}.")
        if not delimiter:
            raise ValueError("delimiter cannot be empty.")

        self.policy: str = resolved_policy
        self.delimiter: bytes = delimiter
        self.donor_prefix_weight: float = float(
            prefix_weight if prefix_weight is not None else (donor_prefix_weight if donor_prefix_weight is not None else 3.0)
        )
        self.donor_command_weight: float = float(
            command_weight if command_weight is not None else (donor_command_weight if donor_command_weight is not None else 2.0)
        )
        self.donor_protocol_weight: float = float(
            protocol_weight if protocol_weight is not None else (donor_protocol_weight if donor_protocol_weight is not None else 1.0)
        )
        self.donor_extra_field_weight: float = float(
            extra_field_weight if extra_field_weight is not None else (donor_extra_field_weight if donor_extra_field_weight is not None else 1.0)
        )
        self.donor_field_diff_penalty: float = float(
            field_diff_penalty if field_diff_penalty is not None else (donor_field_diff_penalty if donor_field_diff_penalty is not None else 0.5)
        )
        self.donor_depth_diff_penalty: float = float(
            depth_diff_penalty if depth_diff_penalty is not None else (donor_depth_diff_penalty if donor_depth_diff_penalty is not None else 0.25)
        )
        self.rng: random.Random = rng if rng is not None else random.Random()
        self.telemetry = DonorSelectorTelemetry()
        self.last_record: Optional[DonorSelectionRecord] = None

    @property
    def mode(self) -> str:
        """Return the active policy mode."""
        return self.policy

    def split_fields(self, data: bytes) -> List[bytes]:
        """Deconstruct raw bytes into non-empty field segments separated by delimiter.

        Args:
            data: Raw input bytes.

        Returns:
            List of non-empty byte fields.
        """
        if not data:
            return []
        raw_parts = data.split(self.delimiter)
        return [part for part in raw_parts if len(part) > 0]

    def compute_compatibility(
        self,
        parent_a_data: Union[bytes, Any],
        parent_b_data: Union[bytes, Any],
        parent_a_depth: Optional[int] = None,
        parent_b_depth: Optional[int] = None,
        candidate_b_depth: Optional[int] = None,
    ) -> DonorCompatibility:
        """Compute structural compatibility metrics between Parent A and candidate Parent B.

        Features:
        1. common_prefix_fields: Number of identical fields from beginning.
        2. same_protocol_prefix: 1 if field[0] is identical, else 0.
        3. same_command: 1 if field[1] is identical, else 0.
        4. field_count_difference: abs(len(fields_a) - len(fields_b)).
        5. depth_difference: abs(depth_a - depth_b).
        6. donor_extra_fields: max(0, len(fields_b) - common_prefix_fields).

        Formula:
            score = (
                prefix_weight * common_prefix_fields
                + command_weight * same_command
                + protocol_weight * same_protocol_prefix
                + extra_field_weight * donor_extra_fields
                - field_diff_penalty * field_count_difference
                - depth_diff_penalty * depth_difference
            )

        Args:
            parent_a_data: Byte payload or SeedRecord of primary parent A.
            parent_b_data: Byte payload or SeedRecord of candidate donor B.
            parent_a_depth: Known protocol state depth of parent A.
            parent_b_depth: Known protocol state depth of candidate B.
            candidate_b_depth: Alias for parent_b_depth.

        Returns:
            DonorCompatibility dataclass containing features and score.
        """
        # Resolve parent A bytes and depth
        if hasattr(parent_a_data, "data"):
            p_a_bytes = bytes(parent_a_data.data)
            depth_a = (
                parent_a_depth
                if parent_a_depth is not None
                else extract_coverage_depth(getattr(parent_a_data, "coverage_units", set()))
            )
        else:
            p_a_bytes = bytes(parent_a_data)
            depth_a = parent_a_depth if parent_a_depth is not None else 0

        # Resolve candidate B bytes and depth
        b_depth_arg = candidate_b_depth if candidate_b_depth is not None else parent_b_depth
        if hasattr(parent_b_data, "data"):
            p_b_bytes = bytes(parent_b_data.data)
            depth_b = (
                b_depth_arg
                if b_depth_arg is not None
                else extract_coverage_depth(getattr(parent_b_data, "coverage_units", set()))
            )
        else:
            p_b_bytes = bytes(parent_b_data)
            depth_b = b_depth_arg if b_depth_arg is not None else 0

        fields_a = self.split_fields(p_a_bytes)
        fields_b = self.split_fields(p_b_bytes)

        # 1. common_prefix_fields
        common_prefix = 0
        for fa, fb in zip(fields_a, fields_b):
            if fa == fb:
                common_prefix += 1
            else:
                break

        # 2. same_protocol_prefix (field 0)
        same_proto = (
            1
            if (len(fields_a) > 0 and len(fields_b) > 0 and fields_a[0] == fields_b[0])
            else 0
        )

        # 3. same_command (field 1)
        same_cmd = (
            1
            if (len(fields_a) > 1 and len(fields_b) > 1 and fields_a[1] == fields_b[1])
            else 0
        )

        # 4. field_count_difference
        field_diff = abs(len(fields_a) - len(fields_b))

        # 5. depth_difference
        depth_diff = abs(int(depth_a) - int(depth_b))

        # 6. donor_extra_fields
        extra_fields = max(0, len(fields_b) - common_prefix)

        # Score calculation
        score = (
            self.donor_prefix_weight * common_prefix
            + self.donor_command_weight * same_cmd
            + self.donor_protocol_weight * same_proto
            + self.donor_extra_field_weight * extra_fields
            - self.donor_field_diff_penalty * field_diff
            - self.donor_depth_diff_penalty * depth_diff
        )

        # Guard against NaN or infinite values
        if math.isnan(score) or math.isinf(score):
            score = 0.0

        return DonorCompatibility(
            common_prefix_fields=common_prefix,
            same_protocol_prefix=same_proto,
            same_command=same_cmd,
            field_count_difference=field_diff,
            depth_difference=depth_diff,
            donor_extra_fields=extra_fields,
            compatibility_score=score,
        )

    def evaluate_compatibility(
        self,
        parent_a: Union[bytes, Any],
        candidate_b: Union[bytes, Any],
        parent_a_depth: Optional[int] = None,
        candidate_b_depth: Optional[int] = None,
    ) -> DonorCompatibility:
        """Alias for compute_compatibility."""
        return self.compute_compatibility(
            parent_a, candidate_b, parent_a_depth=parent_a_depth, candidate_b_depth=candidate_b_depth
        )

    def select_donor(
        self,
        parent_a: Union[bytes, Any],
        corpus: Sequence[Union[bytes, Any]],
        parent_a_depth: Optional[int] = None,
    ) -> Optional[bytes]:
        """Select a donor Parent B from the corpus according to the configured policy.

        Args:
            parent_a: Primary seed (bytes or SeedRecord).
            corpus: Collection of candidate seeds (bytes, SeedRecords, or Corpus).
            parent_a_depth: Optional known depth of Parent A.

        Returns:
            Donor byte string, or None if no valid donor is available.
        """
        self.telemetry.total_attempts += 1
        if not corpus:
            self.telemetry.single_seed_fallbacks += 1
            return None

        # Resolve parent_a bytes and depth
        if hasattr(parent_a, "data"):
            parent_a_bytes = bytes(parent_a.data)
            p_depth = (
                parent_a_depth
                if parent_a_depth is not None
                else extract_coverage_depth(getattr(parent_a, "coverage_units", set()))
            )
        else:
            parent_a_bytes = bytes(parent_a)
            p_depth = parent_a_depth if parent_a_depth is not None else 0

        # Extract normalized candidate tuples: (bytes, depth)
        candidates: List[Tuple[bytes, int]] = []
        for item in corpus:
            if hasattr(item, "data"):
                c_bytes = bytes(item.data)
                c_depth = extract_coverage_depth(getattr(item, "coverage_units", set()))
            else:
                c_bytes = bytes(item)
                c_depth = 0

            # Section 7: Parent B must not equal Parent A when distinct seeds exist
            if c_bytes != parent_a_bytes and len(c_bytes) > 0:
                candidates.append((c_bytes, c_depth))

        if not candidates:
            # Single-seed or no distinct seeds available: safe fallback
            self.telemetry.single_seed_fallbacks += 1
            return None

        selected_bytes: bytes
        compat_info: DonorCompatibility

        if self.policy == "random":
            # Baseline: uniform random selection
            selected_bytes, sel_depth = self.rng.choice(candidates)
            compat_info = self.compute_compatibility(
                parent_a_bytes, selected_bytes, p_depth, sel_depth
            )
            self.telemetry.random_selections += 1
        else:
            # Phase 6F: Compatibility-aware selection
            scored_candidates: List[Tuple[bytes, int, DonorCompatibility]] = []
            for c_bytes, c_depth in candidates:
                compat = self.compute_compatibility(
                    parent_a_bytes, c_bytes, p_depth, c_depth
                )
                scored_candidates.append((c_bytes, c_depth, compat))

            # Section 8: Prefer donors where donor_extra_fields > 0 when available
            useful_candidates = [
                entry for entry in scored_candidates if entry[2].donor_extra_fields > 0
            ]
            candidate_pool = (
                useful_candidates if useful_candidates else scored_candidates
            )

            # Find candidates with the maximum compatibility score
            max_score = max(entry[2].compatibility_score for entry in candidate_pool)
            best_candidates = [
                entry
                for entry in candidate_pool
                if entry[2].compatibility_score == max_score
            ]

            # Section 6: Break ties deterministically via seeded RNG
            chosen_entry = self.rng.choice(best_candidates)
            selected_bytes = chosen_entry[0]
            compat_info = chosen_entry[2]
            self.telemetry.compatible_selections += 1

        # Check for DIAG+DIAG pairing telemetry specifically
        fields_a = self.split_fields(parent_a_bytes)
        fields_b = self.split_fields(selected_bytes)
        if (
            len(fields_a) > 1
            and len(fields_b) > 1
            and fields_a[1] == b"DIAG"
            and fields_b[1] == b"DIAG"
        ):
            self.telemetry.diag_diag_pairings += 1

        # Record telemetry
        self.telemetry.successful_selections += 1
        self.telemetry.total_selections += 1
        self.telemetry.total_compatibility_score += compat_info.compatibility_score
        self.telemetry.total_common_prefix_fields += compat_info.common_prefix_fields
        self.telemetry.total_donor_extra_fields += compat_info.donor_extra_fields
        if compat_info.donor_extra_fields > 0:
            self.telemetry.useful_donor_selections += 1
        if compat_info.same_command:
            self.telemetry.same_command_pairings += 1
        if compat_info.same_protocol_prefix:
            self.telemetry.same_protocol_pairings += 1

        self.last_record = DonorSelectionRecord(
            parent_a=parent_a_bytes,
            selected_parent_b=selected_bytes,
            donor_mode=self.policy,
            compatibility_score=compat_info.compatibility_score,
            common_prefix_fields=compat_info.common_prefix_fields,
            same_command=bool(compat_info.same_command),
            same_protocol_prefix=bool(compat_info.same_protocol_prefix),
            donor_extra_fields=compat_info.donor_extra_fields,
            has_useful_structure=compat_info.donor_extra_fields > 0,
            result_field_count=len(fields_b),
        )

        return selected_bytes

    def get_telemetry(self) -> Dict[str, Any]:
        """Return aggregated donor selector telemetry dictionary."""
        return {
            "policy": self.policy,
            "policy_mode": self.policy,
            "total_attempts": self.telemetry.total_attempts,
            "total_selections": self.telemetry.total_selections,
            "successful_selections": self.telemetry.successful_selections,
            "single_seed_fallbacks": self.telemetry.single_seed_fallbacks,
            "random_selections": self.telemetry.random_selections,
            "compatible_selections": self.telemetry.compatible_selections,
            "avg_compatibility_score": round(self.telemetry.avg_compatibility_score, 4),
            "average_compatibility_score": round(self.telemetry.avg_compatibility_score, 4),
            "avg_common_prefix": round(self.telemetry.avg_common_prefix, 2),
            "average_common_prefix_length": round(self.telemetry.avg_common_prefix, 2),
            "avg_donor_extra_fields": round(self.telemetry.avg_donor_extra_fields, 2),
            "average_donor_extra_fields": round(self.telemetry.avg_donor_extra_fields, 2),
            "useful_donor_rate": round(self.telemetry.useful_donor_rate, 4),
            "same_command_rate": round(self.telemetry.same_command_rate, 4),
            "same_command_pairing_rate": round(self.telemetry.same_command_rate, 4),
            "same_protocol_rate": round(self.telemetry.same_protocol_rate, 4),
            "same_protocol_pairing_rate": round(self.telemetry.same_protocol_rate, 4),
            "diag_diag_pairings": self.telemetry.diag_diag_pairings,
            "compatible_new_coverage": self.telemetry.compatible_new_coverage,
        }
