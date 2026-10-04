"""Feature extraction module for NeuroFuzz contextual mutation policies (Phase 6D).

Extracts a compact, normalized, interpretable numerical feature vector from a candidate
SeedRecord prior to mutation, describing its syntactic and coverage context.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np

from fuzzer.corpus import SeedRecord
from fuzzer.coverage import extract_coverage_depth


class MutationFeatureExtractor:
    """Extracts normalized context feature vectors from candidate SeedRecords for mutation arm selection.

    Feature Vector Definition (dimension = 6):
    ------------------------------------------------------------------------------------------------
    Index | Feature Name            | Formula / Normalization                   | Interpretation
    ------------------------------------------------------------------------------------------------
      0   | bias                    | 1.0                                       | Arm baseline / intercept
      1   | seed_depth              | min(1.0, depth / 19.0)                    | Protocol coverage depth level
      2   | command_type            | mapped value in [0.0, 1.0]                | High-level command category
      3   | delimiter_density       | count(b"|") / max(1, len(seed))           | Protocol boundary density
      4   | normalized_seed_length  | ln(1 + len(seed)) / ln(1 + 256)           | Payload length proxy
      5   | prior_yield             | min(1.0, cov_found / max(1, picks) / 5.0) | Historical discovery density
    ------------------------------------------------------------------------------------------------
    """

    FEATURE_NAMES: List[str] = [
        "bias",
        "seed_depth",
        "command_type",
        "delimiter_density",
        "normalized_seed_length",
        "prior_yield",
    ]

    COMMAND_ENCODING: Dict[bytes, float] = {
        b"PING": 0.1,
        b"INFO": 0.2,
        b"CALC": 0.3,
        b"WRITE": 0.4,
        b"AUTH": 0.5,
        b"DIAG": 0.6,
    }

    def __init__(self, max_depth: float = 19.0, max_expected_length: int = 256) -> None:
        """Initialize MutationFeatureExtractor.

        Args:
            max_depth: Maximum expected depth in benchmark target (default: 19.0).
            max_expected_length: Scale factor for normalizing seed byte length (default: 256).
        """
        self.max_depth = max(1.0, float(max_depth))
        self.max_expected_length = max(1, int(max_expected_length))
        self._log_max_len = math.log1p(self.max_expected_length)

    @property
    def feature_dim(self) -> int:
        """Return the dimension of the feature vector."""
        return len(self.FEATURE_NAMES)

    def _extract_command_type(self, data: bytes) -> float:
        """Map protocol command token to normalized numeric value."""
        if not data:
            return 0.0
        # Check command keywords in seed payload
        for cmd_bytes, val in self.COMMAND_ENCODING.items():
            if cmd_bytes in data:
                return val
        return 0.0

    def extract(self, seed: SeedRecord) -> np.ndarray:
        """Extract normalized 1D numpy context vector from candidate seed.

        Must be called STRICTLY BEFORE mutation to prevent feature leakage.

        Args:
            seed: Candidate SeedRecord.

        Returns:
            np.ndarray of shape (6,) with dtype float64, bounded in [0.0, 1.0] (bias=1.0).
        """
        # Feature 0: Bias term (arm baseline)
        f_bias = 1.0

        # Feature 1: Protocol coverage depth of parent seed
        depth = extract_coverage_depth(seed.coverage_units)
        f_depth = min(1.0, max(0.0, float(depth) / self.max_depth))

        # Feature 2: High-level command category
        data = seed.data if seed.data is not None else b""
        f_cmd = self._extract_command_type(data)

        # Feature 3: Delimiter density (density of field boundaries)
        data_len = len(data)
        delim_count = data.count(b"|")
        f_delim_density = min(1.0, float(delim_count) / max(1.0, float(data_len)))

        # Feature 4: Normalized seed byte length (log-scaled)
        f_len = min(1.0, math.log1p(data_len) / self._log_max_len)

        # Feature 5: Normalized historical yield
        denom = max(1, seed.times_selected)
        raw_yield = float(seed.total_coverage_discovered) / float(denom)
        f_prior_yield = min(1.0, raw_yield / 5.0)

        vector = np.array(
            [
                f_bias,
                f_depth,
                f_cmd,
                f_delim_density,
                f_len,
                f_prior_yield,
            ],
            dtype=np.float64,
        )

        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)

    def extract_dict(self, seed: SeedRecord) -> Dict[str, float]:
        """Extract features as an interpretable key-value dictionary for inspection and logging."""
        vec = self.extract(seed)
        return {name: float(val) for name, val in zip(self.FEATURE_NAMES, vec)}
