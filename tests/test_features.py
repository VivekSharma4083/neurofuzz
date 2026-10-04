"""Unit tests for FeatureExtractor in NeuroFuzz Phase 4."""

import unittest
import numpy as np

from fuzzer.corpus import SeedRecord
from fuzzer.features import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Tests for feature extraction, normalization, and bounds."""

    def setUp(self):
        self.extractor = FeatureExtractor(max_expected_branches=25.0)

    def test_feature_dimension_matches(self):
        """Verify feature vector length matches FEATURE_NAMES count."""
        seed = SeedRecord(id=0, data=b"TEST", name="test")
        vec = self.extractor.extract(seed, current_iteration=1)
        self.assertEqual(len(vec), self.extractor.feature_dim)
        self.assertEqual(len(vec), 8)

    def test_fresh_seed_features(self):
        """Verify initial values for a brand-new seed."""
        seed = SeedRecord(id=0, data=b"ABCDEF", name="fresh_seed")
        vec = self.extractor.extract(seed, current_iteration=10)
        feat_dict = self.extractor.extract_dict(seed, current_iteration=10)

        # Feature 0: bias
        self.assertEqual(feat_dict["bias"], 1.0)
        # Feature 1: size_bytes = 6
        self.assertGreater(feat_dict["log_size"], 0.0)
        # Feature 2: coverage_units is empty
        self.assertEqual(feat_dict["coverage_density"], 0.0)
        # Feature 3: times_selected = 0
        self.assertEqual(feat_dict["log_selections"], 0.0)
        # Feature 4: yield_ratio = 0 / 1
        self.assertEqual(feat_dict["yield_ratio"], 0.0)
        # Feature 5: success_rate = 0 / 1
        self.assertEqual(feat_dict["success_rate"], 0.0)
        # Feature 6: recency = 10 / 10 = 1.0 (never selected before)
        self.assertEqual(feat_dict["recency"], 1.0)
        # Feature 7: log_total_coverage = 0
        self.assertEqual(feat_dict["log_total_coverage"], 0.0)

    def test_active_seed_feature_updates(self):
        """Verify features change appropriately as seed accumulates metrics."""
        seed = SeedRecord(
            id=1,
            data=b"DATA",
            name="active_seed",
            times_selected=10,
            times_mutated=10,
            times_produced_coverage=4,
            total_coverage_discovered=8,
            last_iteration_selected=45,
            coverage_units={"BRANCH_A", "BRANCH_B", "BRANCH_C"},
        )
        current_iter = 50
        feat_dict = self.extractor.extract_dict(seed, current_iteration=current_iter)

        self.assertEqual(feat_dict["bias"], 1.0)
        # Coverage density: 3 / 25 = 0.12
        self.assertAlmostEqual(feat_dict["coverage_density"], 3.0 / 25.0)
        # Yield ratio: 8 / 10 = 0.8
        self.assertAlmostEqual(feat_dict["yield_ratio"], 0.8)
        # Success rate: 4 / 10 = 0.4
        self.assertAlmostEqual(feat_dict["success_rate"], 0.4)
        # Recency: (50 - 45) / 50 = 5 / 50 = 0.1
        self.assertAlmostEqual(feat_dict["recency"], 0.1)
        # Total coverage discovered is positive
        self.assertGreater(feat_dict["log_total_coverage"], 0.0)

    def test_no_nan_or_infinities_on_extreme_values(self):
        """Verify feature extractor handles edge cases without NaNs or Infs."""
        empty_seed = SeedRecord(id=2, data=b"", name="empty")
        vec_empty = self.extractor.extract(empty_seed, current_iteration=0)
        self.assertFalse(np.isnan(vec_empty).any())
        self.assertFalse(np.isinf(vec_empty).any())

        giant_seed = SeedRecord(
            id=3,
            data=b"X" * 1_000_000,
            name="giant",
            times_selected=1_000_000,
            total_coverage_discovered=500_000,
            last_iteration_selected=1_000_000,
        )
        vec_giant = self.extractor.extract(giant_seed, current_iteration=1_000_000)
        self.assertFalse(np.isnan(vec_giant).any())
        self.assertFalse(np.isinf(vec_giant).any())


if __name__ == "__main__":
    unittest.main()
