"""Contextual Multi-Armed Bandit for online mutation operator selection (Phase 6D).

Maintains a distinct OnlineLinearModel for each mutation operator arm.
Given the seed's context feature vector x_t, predicts the expected coverage yield
for each operator, selects an arm via epsilon-greedy exploration, and updates
only the chosen arm via online SGD with L2 regularization.
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from fuzzer.bandit import OnlineLinearModel


class ContextualMutationBandit:
    """Online linear contextual bandit for learned mutation operator selection."""

    def __init__(
        self,
        operators: Sequence[str],
        feature_dim: int = 6,
        epsilon: float = 0.20,
        learning_rate: float = 0.05,
        l2_reg: float = 0.001,
        max_grad_norm: float = 5.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize the contextual mutation bandit.

        Args:
            operators: Sequence of mutation operator names (the arms of the bandit).
            feature_dim: Length of the context feature vector (default: 6).
            epsilon: Probability of choosing a random operator arm (default: 0.20).
            learning_rate: Step size eta for online gradient updates (default: 0.05).
            l2_reg: L2 regularization penalty lambda (default: 0.001).
            max_grad_norm: Gradient clipping threshold.
            rng: Optional seeded random.Random instance for deterministic execution.
        """
        if not operators:
            raise ValueError("operators list must contain at least one operator.")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0.0, 1.0], got {epsilon}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg must be non-negative, got {l2_reg}")

        self._operators: List[str] = list(operators)
        self.feature_dim: int = feature_dim
        self.epsilon: float = float(epsilon)
        self.learning_rate: float = float(learning_rate)
        self.l2_reg: float = float(l2_reg)
        self.max_grad_norm: float = float(max_grad_norm)
        self.rng: random.Random = rng if rng is not None else random.Random()

        # Instantiate an independent OnlineLinearModel for each operator arm
        self.models: Dict[str, OnlineLinearModel] = {
            op: OnlineLinearModel(
                feature_dim=self.feature_dim,
                learning_rate=self.learning_rate,
                l2_reg=self.l2_reg,
                max_grad_norm=self.max_grad_norm,
            )
            for op in self._operators
        }

        # Arm tracking statistics
        self.counts: Dict[str, int] = {op: 0 for op in self._operators}
        self.total_rewards: Dict[str, float] = {op: 0.0 for op in self._operators}
        self.nonzero_rewards: Dict[str, int] = {op: 0 for op in self._operators}
        self.coverage_discoveries: Dict[str, int] = {op: 0 for op in self._operators}

        self.total_selections: int = 0
        self.explorations: int = 0
        self.exploitations: int = 0
        self.history: List[Tuple[str, float, float]] = []  # (operator, prediction, reward)

    @property
    def operators(self) -> List[str]:
        """Return a copy of the available operator arm names."""
        return list(self._operators)

    def get_mean_reward(self, operator: str) -> float:
        """Return empirical mean reward for operator arm."""
        if operator not in self.counts:
            raise KeyError(f"Unknown operator: '{operator}'")
        n = self.counts[operator]
        if n == 0:
            return 0.0
        return self.total_rewards[operator] / n

    def predict(self, operator: str, context: np.ndarray) -> float:
        """Compute the predicted expected reward for a specific operator arm."""
        if operator not in self.models:
            raise KeyError(f"Unknown operator: '{operator}'")
        return self.models[operator].predict(context)

    def predict_all(self, context: np.ndarray) -> Dict[str, float]:
        """Compute predicted expected rewards across all available operator arms."""
        return {op: self.models[op].predict(context) for op in self._operators}

    def select_operator(self, context: np.ndarray) -> str:
        """Select a mutation operator arm using epsilon-greedy linear policy.

        Args:
            context: 1D numpy array of shape (feature_dim,) extracted from the candidate seed.

        Returns:
            The selected operator arm name.
        """
        if len(context) != self.feature_dim:
            raise ValueError(f"Expected context of dimension {self.feature_dim}, got {len(context)}")

        self.total_selections += 1

        # Epsilon-greedy: Exploration step
        if self.rng.random() < self.epsilon:
            chosen = self.rng.choice(self._operators)
            self.explorations += 1
            return chosen

        # Exploitation step: Select arm with maximum predicted yield w_a^T x
        predictions = self.predict_all(context)
        max_pred = max(predictions.values())

        # Collect ties within numerical tolerance
        best_ops = [
            op for op, pred in predictions.items()
            if math.isclose(pred, max_pred, rel_tol=1e-9, abs_tol=1e-9)
        ]

        if len(best_ops) == 1:
            chosen = best_ops[0]
        else:
            chosen = self.rng.choice(best_ops)

        self.exploitations += 1
        return chosen

    def update(self, operator: str, context: np.ndarray, reward: float) -> Tuple[float, float]:
        """Perform an online SGD update for the selected operator arm only.

        Args:
            operator: Name of the mutation operator arm that was selected and executed.
            context: 1D numpy array of context features describing the mutated seed.
            reward: Observed non-negative scalar reward (new coverage units discovered).

        Returns:
            Tuple of (prediction, prediction_error) for the updated arm.
        """
        if operator not in self.models:
            raise KeyError(f"Unknown operator: '{operator}'")
        if len(context) != self.feature_dim:
            raise ValueError(f"Expected context of dimension {self.feature_dim}, got {len(context)}")

        r = float(reward)
        self.counts[operator] += 1
        self.total_rewards[operator] += r

        if r > 0.0:
            self.nonzero_rewards[operator] += 1
            self.coverage_discoveries[operator] += int(r)

        # Update ONLY the model of the pulled arm
        prediction, error = self.models[operator].update(context, r)

        if len(self.history) < 2000:
            self.history.append((operator, prediction, r))

        return prediction, error

    def reset(self) -> None:
        """Reset all bandit statistics and model weights to initial state."""
        self.models = {
            op: OnlineLinearModel(
                feature_dim=self.feature_dim,
                learning_rate=self.learning_rate,
                l2_reg=self.l2_reg,
                max_grad_norm=self.max_grad_norm,
            )
            for op in self._operators
        }
        for op in self._operators:
            self.counts[op] = 0
            self.total_rewards[op] = 0.0
            self.nonzero_rewards[op] = 0
            self.coverage_discoveries[op] = 0
        self.total_selections = 0
        self.explorations = 0
        self.exploitations = 0
        self.history.clear()

    def get_weights_dict(self, feature_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Return learned weights per arm mapped to feature names."""
        return {
            op: self.models[op].get_weights_dict(feature_names=feature_names)
            for op in self._operators
        }

    def get_statistics(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return comprehensive telemetry dictionary for all arms and models."""
        ops_dict: Dict[str, Any] = {}
        for op in self._operators:
            cnt = self.counts[op]
            mean_r = self.get_mean_reward(op)
            model = self.models[op]
            ops_dict[op] = {
                "selections": cnt,
                "selection_rate": cnt / max(1, self.total_selections),
                "total_reward": self.total_rewards[op],
                "mean_reward": mean_r,
                "nonzero_rewards": self.nonzero_rewards[op],
                "coverage_discoveries": self.coverage_discoveries[op],
                "model_updates": model.total_updates,
                "model_mse": model.mean_squared_error,
                "model_mae": model.mean_absolute_error,
                "weights": model.get_weights_dict(feature_names=feature_names),
            }

        return {
            "strategy": "contextual_mutation_bandit",
            "feature_dim": self.feature_dim,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "total_selections": self.total_selections,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
            "empirical_exploration_rate": self.explorations / max(1, self.total_selections),
            "operators": ops_dict,
        }
