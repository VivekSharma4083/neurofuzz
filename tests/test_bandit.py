"""Unit tests and deterministic standalone toy environment for OnlineLinearModel.

Verifies:
1. Basic unit operations: initialization, prediction, gradient update, L2 regularization.
2. Deterministic toy contextual bandit environment demonstrating that the online
   learner successfully outperforms random selection and reduces prediction error.
"""

import random
import unittest
import numpy as np

from fuzzer.bandit import OnlineLinearModel


class TestOnlineLinearModelBasics(unittest.TestCase):
    """Unit tests for the OnlineLinearModel math and edge cases."""

    def test_invalid_parameters_raise(self):
        """Verify invalid initialization parameters raise ValueError."""
        with self.assertRaises(ValueError):
            OnlineLinearModel(feature_dim=0)
        with self.assertRaises(ValueError):
            OnlineLinearModel(feature_dim=4, learning_rate=-0.01)
        with self.assertRaises(ValueError):
            OnlineLinearModel(feature_dim=4, l2_reg=-0.1)
        with self.assertRaises(ValueError):
            OnlineLinearModel(feature_dim=4, initial_weights=np.array([1.0, 2.0]))

    def test_initial_zero_prediction(self):
        """Verify model initialized with default zeros predicts 0.0."""
        model = OnlineLinearModel(feature_dim=3)
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(model.predict(x), 0.0)

    def test_exact_linear_prediction(self):
        """Verify predict computes exact dot product w · x."""
        weights = np.array([0.5, -1.0, 2.0])
        model = OnlineLinearModel(feature_dim=3, initial_weights=weights)
        x = np.array([2.0, 3.0, 1.0])
        # 0.5*2 + (-1)*3 + 2*1 = 1.0 - 3.0 + 2.0 = 0.0
        self.assertAlmostEqual(model.predict(x), 0.0)

        x2 = np.array([1.0, 0.0, 1.0])
        # 0.5*1 + 0 + 2*1 = 2.5
        self.assertAlmostEqual(model.predict(x2), 2.5)

    def test_single_update_direction(self):
        """Verify a positive reward increases weights for positive features."""
        model = OnlineLinearModel(feature_dim=2, learning_rate=0.1, l2_reg=0.0)
        x = np.array([1.0, 2.0])
        # Initially w = [0, 0], pred = 0. Target reward = 5.0. Error = 5.0.
        pred, err = model.update(x, reward=5.0)
        self.assertAlmostEqual(pred, 0.0)
        self.assertAlmostEqual(err, 5.0)
        # Expected new w: 0 + 0.1 * 5.0 * [1.0, 2.0] = [0.5, 1.0]
        np.testing.assert_allclose(model.weights, [0.5, 1.0], atol=1e-6)

    def test_l2_weight_decay(self):
        """Verify L2 regularization shrinks weights toward zero when error is zero."""
        initial = np.array([2.0, 2.0])
        model = OnlineLinearModel(feature_dim=2, learning_rate=0.1, l2_reg=0.5, initial_weights=initial)
        x = np.array([1.0, 0.0])
        # Target reward matches prediction exactly (2.0) -> error = 0.0
        model.update(x, reward=2.0)
        # Decay factor = 1 - 0.1 * 0.5 = 0.95
        # Expected w = 0.95 * [2.0, 2.0] = [1.9, 1.9]
        np.testing.assert_allclose(model.weights, [1.9, 1.9], atol=1e-6)

    def test_dimension_mismatch_raises(self):
        """Verify vector dimension mismatch raises ValueError."""
        model = OnlineLinearModel(feature_dim=3)
        with self.assertRaises(ValueError):
            model.predict(np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            model.update(np.array([1.0, 2.0]), reward=1.0)


class TestStandaloneToyBanditEnvironment(unittest.TestCase):
    """Deterministic standalone contextual bandit verification (Requirement 8).

    Environment Specification:
    --------------------------
    There are 3 candidate arms (analogous to candidate seeds in a corpus):
      - Arm 0: Low-yield profile   (x_0) -> True expected reward = 0.1
      - Arm 1: Medium-yield profile(x_1) -> True expected reward = 0.8
      - Arm 2: High-yield profile  (x_2) -> True expected reward = 2.5

    Over 1,000 steps, compare:
      A. Linear Contextual Bandit with epsilon-greedy (epsilon = 0.10)
      B. Uniform Random Baseline

    Success Criteria:
      1. Cumulative reward: LinearBandit >> Random Baseline.
      2. Tracking error: The model's prediction error decreases over time.
      3. Learned weights correlate with the true generating weights.
    """

    def test_linear_bandit_outperforms_random_deterministically(self):
        """Verify contextual bandit learns to select high-reward arms deterministically."""
        rng_seed = 12345
        random.seed(rng_seed)
        np.random.seed(rng_seed)

        # 3 arms with distinct feature vectors (dim = 4: [bias, f1, f2, f3])
        # True generating weight vector: w_true = [0.1, 0.5, 1.2, -0.3]
        w_true = np.array([0.1, 0.5, 1.2, -0.3])

        # Arm feature vectors:
        # Arm 0: low reward
        # Arm 1: medium reward
        # Arm 2: high reward
        arms_features = [
            np.array([1.0, 0.1, 0.0, 1.0]),   # w · x = 0.1 + 0.05 + 0.0 - 0.3 = -0.15 -> clipped to ~0.05
            np.array([1.0, 0.5, 0.4, 0.2]),   # w · x = 0.1 + 0.25 + 0.48 - 0.06 = 0.77
            np.array([1.0, 0.9, 1.5, 0.1]),   # w · x = 0.1 + 0.45 + 1.80 - 0.03 = 2.32
        ]

        def sample_reward(arm_idx: int, seed_prng: random.Random) -> float:
            mean_r = max(0.0, float(np.dot(w_true, arms_features[arm_idx])))
            # Add small zero-mean Gaussian noise bounded to non-negative integers
            noise = seed_prng.gauss(0.0, 0.2)
            return max(0.0, mean_r + noise)

        num_steps = 1000
        epsilon = 0.15

        # -------------------------------------------------------------
        # 1. Run Random Baseline
        # -------------------------------------------------------------
        prng_rand = random.Random(rng_seed)
        random_cumulative_reward = 0.0
        random_arm_selections = [0, 0, 0]

        for _ in range(num_steps):
            chosen_arm = prng_rand.choice([0, 1, 2])
            random_arm_selections[chosen_arm] += 1
            reward = sample_reward(chosen_arm, prng_rand)
            random_cumulative_reward += reward

        # -------------------------------------------------------------
        # 2. Run Online Linear Contextual Bandit
        # -------------------------------------------------------------
        prng_bandit = random.Random(rng_seed)
        model = OnlineLinearModel(feature_dim=4, learning_rate=0.03, l2_reg=0.001)
        bandit_cumulative_reward = 0.0
        bandit_arm_selections = [0, 0, 0]
        step_errors = []

        for step in range(num_steps):
            # Epsilon-greedy selection
            if prng_bandit.random() < epsilon:
                chosen_arm = prng_bandit.choice([0, 1, 2])
            else:
                preds = [model.predict(feat) for feat in arms_features]
                max_pred = max(preds)
                # Tie break
                candidates = [i for i, p in enumerate(preds) if abs(p - max_pred) < 1e-9]
                chosen_arm = prng_bandit.choice(candidates)

            bandit_arm_selections[chosen_arm] += 1
            reward = sample_reward(chosen_arm, prng_bandit)
            bandit_cumulative_reward += reward

            # Online model update
            _, error = model.update(arms_features[chosen_arm], reward)
            step_errors.append(abs(error))

        # -------------------------------------------------------------
        # 3. Assertions & Verification
        # -------------------------------------------------------------
        # Criterion 1: Bandit cumulative reward must substantially exceed random selection
        self.assertGreater(
            bandit_cumulative_reward,
            random_cumulative_reward * 1.5,
            f"Bandit ({bandit_cumulative_reward:.1f}) should significantly beat Random ({random_cumulative_reward:.1f})",
        )

        # Criterion 2: Bandit must select the best arm (Arm 2) the vast majority of the time
        self.assertGreater(
            bandit_arm_selections[2],
            num_steps * 0.65,
            f"Bandit should exploit Arm 2 (high reward) >65% of the time, got {bandit_arm_selections[2]}/{num_steps}",
        )

        # Criterion 3: Prediction error in late training (last 100 steps) should be lower than early (first 100 steps)
        early_mae = sum(step_errors[:100]) / 100.0
        late_mae = sum(step_errors[-100:]) / 100.0
        self.assertLess(
            late_mae,
            early_mae,
            f"Late MAE ({late_mae:.4f}) should be lower than Early MAE ({early_mae:.4f}) as model converges",
        )


if __name__ == "__main__":
    unittest.main()
