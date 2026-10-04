"""Online linear contextual bandit model for NeuroFuzz Phase 4.

Implements lightweight online linear regression trained via Online Stochastic
Gradient Descent (SGD) with L2 regularization to predict expected coverage yield.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class OnlineLinearModel:
    """Lightweight online linear regression model.

    Model Equation:
        prediction = w · x = sum(w_i * x_i)

    Objective Function (Ridge / L2 regularized squared error at step t):
        L_t(w) = 0.5 * (r_t - w · x_t)^2 + 0.5 * lambda * ||w||^2

    Gradient:
        grad = -(r_t - w · x_t) * x_t + lambda * w = -e_t * x_t + lambda * w

    Online SGD Update Rule:
        w <- w - eta * grad
          = (1 - eta * lambda) * w + eta * e_t * x_t
        where:
          eta    = learning rate (step size)
          lambda = L2 regularization coefficient (weight decay)
          e_t    = r_t - pred_t (residual error)
    """

    def __init__(
        self,
        feature_dim: int,
        learning_rate: float = 0.05,
        l2_reg: float = 0.001,
        max_grad_norm: float = 5.0,
        initial_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the online linear model.

        Args:
            feature_dim: Length of the input feature vector.
            learning_rate: Step size eta for gradient updates (default: 0.05).
            l2_reg: L2 weight decay coefficient lambda (default: 0.001).
            max_grad_norm: Gradient clipping threshold for numerical stability.
            initial_weights: Optional starting weights vector of shape (feature_dim,).
        """
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if l2_reg < 0.0:
            raise ValueError(f"l2_reg must be non-negative, got {l2_reg}")

        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.max_grad_norm = max_grad_norm

        if initial_weights is not None:
            if len(initial_weights) != feature_dim:
                raise ValueError(
                    f"initial_weights length {len(initial_weights)} does not match feature_dim {feature_dim}"
                )
            self.weights = np.array(initial_weights, dtype=np.float64)
        else:
            # Initialize weights to zeros (neutral prior)
            self.weights = np.zeros(feature_dim, dtype=np.float64)

        # Telemetry & performance statistics
        self.total_updates: int = 0
        self.cumulative_loss: float = 0.0
        self.cumulative_abs_error: float = 0.0
        self.last_prediction: float = 0.0
        self.last_error: float = 0.0

    @property
    def mean_squared_error(self) -> float:
        """Return cumulative mean squared error (MSE) across all updates."""
        if self.total_updates == 0:
            return 0.0
        return (2.0 * self.cumulative_loss) / self.total_updates

    @property
    def mean_absolute_error(self) -> float:
        """Return cumulative mean absolute error (MAE) across all updates."""
        if self.total_updates == 0:
            return 0.0
        return self.cumulative_abs_error / self.total_updates

    def predict(self, x: np.ndarray) -> float:
        """Compute the predicted expected reward for feature vector x.

        prediction = w · x

        Args:
            x: 1D numpy array of shape (feature_dim,).

        Returns:
            Scalar float prediction.
        """
        if len(x) != self.feature_dim:
            raise ValueError(f"Expected feature vector of length {self.feature_dim}, got {len(x)}")

        pred = float(np.dot(self.weights, x))
        # Prevent wild negative predictions in early iterations
        return pred

    def update(self, x: np.ndarray, reward: float) -> Tuple[float, float]:
        """Perform an online gradient update given context x and observed reward.

        Args:
            x: 1D numpy array of shape (feature_dim,).
            reward: Observed scalar reward (e.g., new coverage units discovered).

        Returns:
            Tuple of (prediction, prediction_error).
        """
        if len(x) != self.feature_dim:
            raise ValueError(f"Expected feature vector of length {self.feature_dim}, got {len(x)}")

        prediction = float(np.dot(self.weights, x))
        error = float(reward) - prediction

        # Update telemetry
        loss = 0.5 * (error ** 2)
        self.cumulative_loss += loss
        self.cumulative_abs_error += abs(error)
        self.total_updates += 1
        self.last_prediction = prediction
        self.last_error = error

        # Clip error to avoid numerical explosion on outliers
        clipped_error = np.clip(error, -self.max_grad_norm, self.max_grad_norm)

        # Online SGD Step with L2 weight decay:
        # w <- (1 - eta * lambda) * w + eta * error * x
        decay_factor = max(0.0, 1.0 - self.learning_rate * self.l2_reg)
        gradient_step = self.learning_rate * clipped_error * x

        self.weights = decay_factor * self.weights + gradient_step

        return prediction, error

    def get_weights_dict(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Return learned weights mapped to feature names for transparency."""
        names = feature_names or [f"w_{i}" for i in range(self.feature_dim)]
        return {name: round(float(w), 6) for name, w in zip(names, self.weights)}
