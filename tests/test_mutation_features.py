"""Unit tests for MutationFeatureExtractor (Phase 6D)."""

import unittest
import numpy as np

from fuzzer.corpus import SeedRecord
from fuzzer.mutation_features import MutationFeatureExtractor


class TestMutationFeatureExtractor(unittest.TestCase):
    """Test suite for MutationFeatureExtractor."""

    def setUp(self) -> None:
        self.extractor = MutationFeatureExtractor(max_depth=19.0, max_expected_length=256)

    def test_feature_dimension_and_names(self) -> None:
        """Verify feature dimension matches FEATURE_NAMES count."""
        self.assertEqual(self.extractor.feature_dim, 6)
        self.assertEqual(
            self.extractor.FEATURE_NAMES,
            [
                "bias",
                "seed_depth",
                "command_type",
                "delimiter_density",
                "normalized_seed_length",
                "prior_yield",
            ],
        )

    def test_bias_feature_is_always_one(self) -> None:
        """Feature 0 must always be 1.0 (intercept)."""
        seed = SeedRecord(id=1, data=b"NF01|PING", name="dummy.bin")
        vec = self.extractor.extract(seed)
        self.assertEqual(vec[0], 1.0)

    def test_depth_normalization(self) -> None:
        """Depth feature must be normalized by max_depth and bounded in [0, 1]."""
        seed_no_cov = SeedRecord(id=1, data=b"raw", name="s0.bin")
        vec0 = self.extractor.extract(seed_no_cov)
        self.assertEqual(vec0[1], 0.0)

        seed_depth_5 = SeedRecord(id=2, data=b"raw", name="s5.bin", coverage_units={"DEPTH_05"})
        vec5 = self.extractor.extract(seed_depth_5)
        self.assertAlmostEqual(vec5[1], 5.0 / 19.0, places=5)

        seed_depth_19 = SeedRecord(
            id=3,
            data=b"raw",
            name="s19.bin",
            coverage_units={f"DEPTH_{i:02d}" for i in range(1, 20)},
        )
        vec19 = self.extractor.extract(seed_depth_19)
        self.assertEqual(vec19[1], 1.0)

    def test_command_type_encoding(self) -> None:
        """Known protocol commands should map to their respective normalized float values."""
        commands = {
            b"NF01|PING|HELLO": 0.1,
            b"NF01|INFO|SYS": 0.2,
            b"NF01|CALC|OP=ADD": 0.3,
            b"NF01|WRITE|DATA=1": 0.4,
            b"NF01|AUTH|ROOT": 0.5,
            b"NF01|DIAG|STEP=1": 0.6,
            b"NF01|UNKNOWN": 0.0,
            b"RANDOM_BYTES": 0.0,
            b"": 0.0,
        }
        for payload, expected_val in commands.items():
            seed = SeedRecord(id=1, data=payload, name="cmd.bin")
            vec = self.extractor.extract(seed)
            self.assertAlmostEqual(vec[2], expected_val, places=4, msg=f"Failed for {payload}")

    def test_delimiter_density(self) -> None:
        """Delimiter density must count '|' characters relative to length and stay in [0, 1]."""
        # No delimiters
        s1 = SeedRecord(id=1, data=b"ABCDEFGH", name="s1.bin")
        self.assertEqual(self.extractor.extract(s1)[3], 0.0)

        # 4 delimiters in 10 bytes -> 0.4
        s2 = SeedRecord(id=2, data=b"A|B|C|D|EF", name="s2.bin")
        self.assertAlmostEqual(self.extractor.extract(s2)[3], 4.0 / 10.0, places=4)

        # Empty data
        s3 = SeedRecord(id=3, data=b"", name="s3.bin")
        self.assertEqual(self.extractor.extract(s3)[3], 0.0)

    def test_normalized_seed_length(self) -> None:
        """Seed byte length should be log-scaled and bounded in [0, 1]."""
        s_empty = SeedRecord(id=1, data=b"", name="empty.bin")
        self.assertEqual(self.extractor.extract(s_empty)[4], 0.0)

        s_256 = SeedRecord(id=2, data=b"X" * 256, name="256.bin")
        self.assertAlmostEqual(self.extractor.extract(s_256)[4], 1.0, places=4)

        # Longer than max should clamp to 1.0
        s_1000 = SeedRecord(id=3, data=b"X" * 1000, name="1000.bin")
        self.assertEqual(self.extractor.extract(s_1000)[4], 1.0)

    def test_prior_yield_feature(self) -> None:
        """Prior yield feature should reflect discovered coverage per selection."""
        seed = SeedRecord(
            id=1,
            data=b"seed",
            name="s.bin",
            times_selected=4,
            total_coverage_discovered=10,
        )
        # raw yield = 10 / 4 = 2.5 -> normalized = 2.5 / 5.0 = 0.5
        vec = self.extractor.extract(seed)
        self.assertAlmostEqual(vec[5], 0.5, places=4)

    def test_extract_dict_consistency(self) -> None:
        """extract_dict should produce a dictionary matching extract() array values."""
        seed = SeedRecord(
            id=1,
            data=b"NF01|DIAG|STEP=1",
            name="s.bin",
            coverage_units={"DEPTH_06"},
            times_selected=2,
            total_coverage_discovered=3,
        )
        vec = self.extractor.extract(seed)
        d = self.extractor.extract_dict(seed)

        self.assertEqual(len(d), 6)
        for i, name in enumerate(self.extractor.FEATURE_NAMES):
            self.assertEqual(d[name], vec[i])

    def test_robustness_to_none_and_empty(self) -> None:
        """Extractor should handle malformed or empty SeedRecords without exceptions or NaNs."""
        seed = SeedRecord(id=1, data=b"", name="none.bin")
        vec = self.extractor.extract(seed)
        self.assertFalse(np.isnan(vec).any())
        self.assertFalse(np.isinf(vec).any())
        self.assertEqual(len(vec), 6)


if __name__ == "__main__":
    unittest.main()
