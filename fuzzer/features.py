"""Feature extraction module for NeuroFuzz contextual bandit policies.

Extracts an interpretable, normalized numerical feature vector from a candidate
SeedRecord at a given fuzzing iteration.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np

from fuzzer.corpus import SeedRecord


class FeatureExtractor:
    """Extracts normalized feature vectors from candidate SeedRecords.

    Feature Vector Definition (dimension = 8):
    -------------------------------------------------------------------------
    Index | Feature Name         | Formula / Normalization               | Interpretation
    -------------------------------------------------------------------------
      0   | bias                 | 1.0                                   | Intercept / base expected reward
      1   | log_size             | ln(1 + size_bytes) / 10.0             | Input length (log-scaled, bounded)
      2   | coverage_density     | min(1.0, len(cov_units) / 25.0)       | Fraction of target branches reached
      3   | log_selections       | ln(1 + times_selected) / 10.0         | Selection frequency / exhaustion proxy
      4   | yield_ratio          | total_cov_found / max(1, times_sel)   | Empirical new coverage discovered/pick
      5   | success_rate         | times_prod_cov / max(1, times_sel)    | Probability of mutation producing new cov
      6   | recency              | (iteration - last_iter) / max(1, iter)| Normalized staleness / idle duration
      7   | log_total_coverage   | ln(1 + total_cov_found) / 5.0         | Magnitude of historical path discovery
    -------------------------------------------------------------------------
    """

    FEATURE_NAMES: List[str] = [
        "bias",
        "log_size",
        "coverage_density",
        "log_selections",
        "yield_ratio",
        "success_rate",
        "recency",
        "log_total_coverage",
    ]

    def __init__(self, max_expected_branches: float = 25.0) -> None:
        """Initialize FeatureExtractor.

        Args:
            max_expected_branches: Scale factor for normalizing coverage density.
        """
        self.max_expected_branches = max_expected_branches

    @property
    def feature_dim(self) -> int:
        """Return the dimension of the feature vector."""
        return len(self.FEATURE_NAMES)

    def extract(self, seed: SeedRecord, current_iteration: int) -> np.ndarray:
        """Extract a normalized 1D numpy feature vector for a candidate seed.

        Args:
            seed: Candidate SeedRecord.
            current_iteration: The current fuzzing loop iteration number.

        Returns:
            np.ndarray of shape (feature_dim,) with dtype float64.
        """
        # Feature 0: Bias term
        f_bias = 1.0

        # Feature 1: Log-scaled seed payload size (normalized roughly to [0, 1])
        f_size = math.log1p(max(0, seed.size_bytes)) / 10.0

        # Feature 2: Coverage density reached by this seed
        cov_count = len(seed.coverage_units)
        f_cov_density = min(1.0, cov_count / self.max_expected_branches)

        # Feature 3: Log-scaled times selected (measures fatigue/exhaustion)
        f_log_sel = math.log1p(seed.times_selected) / 10.0

        # Feature 4: Empirical yield ratio (new coverage discovered per selection)
        denom = max(1, seed.times_selected)
        f_yield = float(seed.total_coverage_discovered) / float(denom)

        # Feature 5: Hit rate / success rate (frequency of producing new coverage)
        f_success_rate = float(seed.times_produced_coverage) / float(denom)

        # Feature 6: Recency / staleness: how long ago was this seed selected?
        # A brand-new seed has last_iteration_selected=0 -> recency = 1.0 (fresh/unexplored)
        iter_denom = max(1, current_iteration)
        idle_iters = max(0, current_iteration - seed.last_iteration_selected)
        f_recency = min(1.0, float(idle_iters) / float(iter_denom))

        # Feature 7: Magnitude of total coverage discovered by offspring
        f_log_total_cov = math.log1p(seed.total_coverage_discovered) / 5.0

        vector = np.array(
            [
                f_bias,
                f_size,
                f_cov_density,
                f_log_sel,
                f_yield,
                f_success_rate,
                f_recency,
                f_log_total_cov,
            ],
            dtype=np.float64,
        )

        # Guard against any numerical anomalies
        return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=0.0)

    def extract_dict(self, seed: SeedRecord, current_iteration: int) -> Dict[str, float]:
        """Extract features as an interpretable key-value dictionary for inspection and logging."""
        vec = self.extract(seed, current_iteration)
        return {name: float(val) for name, val in zip(self.FEATURE_NAMES, vec)}
