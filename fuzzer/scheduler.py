"""Seed scheduling algorithms for NeuroFuzz.

Provides an extensible SeedScheduler abstraction and concrete implementations:
- RandomScheduler: Uniform random selection baseline.
- HeuristicScheduler: Non-ML heuristic scheduler based on empirical coverage yield
  with epsilon-greedy exploration to prevent seed starvation.
- LinearBanditScheduler: Online contextual bandit policy with online SGD learning.
"""

from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from fuzzer.bandit import OnlineLinearModel
from fuzzer.corpus import Corpus, SeedRecord
from fuzzer.features import FeatureExtractor


class SeedScheduler(ABC):
    """Abstract base class for seed scheduling policies."""

    def __init__(self, name: str, rng: Optional[random.Random] = None) -> None:
        """Initialize base scheduler.

        Args:
            name: Human-readable name for the scheduler strategy.
            rng: Optional random.Random instance for deterministic behavior.
        """
        self.name = name
        self.rng = rng or random.Random()
        self.total_selections: int = 0
        self.total_reward: int = 0

    @property
    def average_reward(self) -> float:
        """Return average reward (new coverage units discovered per execution)."""
        if self.total_selections == 0:
            return 0.0
        return self.total_reward / self.total_selections

    @abstractmethod
    def select_seed(self, corpus: Corpus, iteration: int) -> SeedRecord:
        """Select a seed record from the corpus.

        Args:
            corpus: Active corpus containing available seed records.
            iteration: Current fuzzing iteration number.

        Returns:
            The selected SeedRecord.
        """
        pass

    @abstractmethod
    def update(self, seed: SeedRecord, reward: int, iteration: int) -> None:
        """Update scheduler statistics following execution of a mutated seed.

        Args:
            seed: The SeedRecord that was selected as parent for the mutation.
            reward: Number of new coverage units discovered (>= 0).
            iteration: Current fuzzing iteration number.
        """
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Return a dictionary of scheduler-specific telemetry."""
        pass


class RandomScheduler(SeedScheduler):
    """Baseline scheduler that selects seeds uniformly at random.

    Preserves Phase 1 & 2 baseline behavior for experimental comparison.
    """

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        """Initialize random scheduler."""
        super().__init__(name="random", rng=rng)

    def select_seed(self, corpus: Corpus, iteration: int) -> SeedRecord:
        """Select a seed from the corpus uniformly at random.

        Args:
            corpus: Active corpus.
            iteration: Current iteration number.

        Returns:
            A randomly chosen SeedRecord.

        Raises:
            ValueError: If the corpus is empty.
        """
        if not corpus.records:
            raise ValueError("Corpus is empty. Please add or load at least one seed.")

        selected = self.rng.choice(corpus.records)
        selected.record_selection(iteration)
        self.total_selections += 1
        return selected

    def update(self, seed: SeedRecord, reward: int, iteration: int) -> None:
        """Update seed yield metrics."""
        self.total_reward += reward
        seed.record_reward(reward)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return random scheduler telemetry."""
        return {
            "strategy": self.name,
            "total_selections": self.total_selections,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "exploration_rate": 1.0,
        }


class HeuristicScheduler(SeedScheduler):
    """Interpretable, non-ML heuristic scheduler with epsilon-greedy exploration.

    Scoring Rationale:
    Each seed is scored by its empirical coverage yield:
        Score(s) = (1 + total_coverage_discovered(s)) / (1 + times_selected(s))

    Why this formula?
    1. Direct usefulness: Seeds that historically uncovered more new coverage receive higher priority.
    2. Diminishing returns: Every time a seed is selected without uncovering new coverage,
       the denominator increases, steadily reducing its score and preventing hotspotting.
    3. Laplace smoothing (+1): Brand-new seeds with 0 selections start with a neutral
       score of 1.0, ensuring fair initial evaluation.
    4. Zero arbitrary hyperparameter weights.

    Starvation Prevention:
    Epsilon-greedy exploration ensures seeds with lower current scores are not starved:
    - (1 - epsilon) probability: Exploit the highest-scoring seed(s).
    - epsilon probability: Explore a randomly selected seed from the corpus.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize heuristic scheduler.

        Args:
            epsilon: Exploration probability in [0.0, 1.0] (default: 0.2 = 20% exploration).
            rng: Optional random.Random instance for deterministic behavior.
        """
        super().__init__(name="heuristic", rng=rng)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"Epsilon must be between 0.0 and 1.0, got {epsilon}")
        self.epsilon = epsilon
        self.explorations: int = 0
        self.exploitations: int = 0

    @staticmethod
    def compute_score(seed: SeedRecord) -> float:
        """Compute the empirical yield score for a seed.

        Score(s) = (1 + total_coverage_discovered) / (1 + times_selected)
        """
        numerator = 1.0 + float(seed.total_coverage_discovered)
        denominator = 1.0 + float(seed.times_selected)
        return numerator / denominator

    def select_seed(self, corpus: Corpus, iteration: int) -> SeedRecord:
        """Select a seed using epsilon-greedy heuristic scoring.

        Args:
            corpus: Active corpus.
            iteration: Current iteration number.

        Returns:
            The selected SeedRecord.

        Raises:
            ValueError: If the corpus is empty.
        """
        if not corpus.records:
            raise ValueError("Corpus is empty. Please add or load at least one seed.")

        # Epsilon-greedy decision
        if len(corpus.records) > 1 and self.rng.random() < self.epsilon:
            # Exploration: uniformly pick any seed
            chosen = self.rng.choice(corpus.records)
            self.explorations += 1
        else:
            # Exploitation: score all seeds and pick the highest scorer
            best_score = -1.0
            best_candidates: List[SeedRecord] = []

            for record in corpus.records:
                score = self.compute_score(record)
                if score > best_score:
                    best_score = score
                    best_candidates = [record]
                elif abs(score - best_score) < 1e-9:
                    best_candidates.append(record)

            # Tie-break randomly among top candidates
            chosen = self.rng.choice(best_candidates)
            self.exploitations += 1

        chosen.record_selection(iteration)
        self.total_selections += 1
        return chosen

    def update(self, seed: SeedRecord, reward: int, iteration: int) -> None:
        """Update seed yield metrics and scheduler stats."""
        self.total_reward += reward
        seed.record_reward(reward)

    def get_telemetry(self) -> Dict[str, Any]:
        """Return heuristic scheduler telemetry."""
        exploration_rate = (
            self.explorations / self.total_selections if self.total_selections > 0 else self.epsilon
        )
        return {
            "strategy": self.name,
            "epsilon": self.epsilon,
            "total_selections": self.total_selections,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
            "empirical_exploration_rate": round(exploration_rate, 4),
        }


class LinearBanditScheduler(SeedScheduler):
    """Contextual bandit scheduler with online linear regression.

    Formulation:
    - Context x(s, t): Normalized feature vector extracted from candidate seed s.
    - Action a_t: Select one seed from corpus.records.
    - Prediction: pred(s) = w · x(s, t) (expected future new coverage).
    - Exploration / Exploitation: Epsilon-greedy.
      - With prob epsilon: select seed uniformly at random.
      - With prob (1 - epsilon): select seed with highest predicted reward.
    - Online Update: Observed reward r_t (new coverage units discovered) updates
      w via online SGD with L2 regularization:
      w <- (1 - eta * lambda) * w + eta * (r_t - pred) * x(a_t)
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        learning_rate: float = 0.05,
        l2_reg: float = 0.001,
        max_expected_branches: float = 25.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize linear contextual bandit scheduler.

        Args:
            epsilon: Exploration probability in [0.0, 1.0] (default: 0.2).
            learning_rate: Step size eta for online SGD (default: 0.05).
            l2_reg: L2 regularization coefficient (default: 0.001).
            max_expected_branches: Scale factor for feature normalizer.
            rng: Optional random.Random instance for deterministic behavior.
        """
        super().__init__(name="linear_bandit", rng=rng)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"Epsilon must be between 0.0 and 1.0, got {epsilon}")

        self.epsilon = epsilon
        self.feature_extractor = FeatureExtractor(max_expected_branches=max_expected_branches)
        self.model = OnlineLinearModel(
            feature_dim=self.feature_extractor.feature_dim,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
        )

        self.explorations: int = 0
        self.exploitations: int = 0
        # Cache the most recent selection decision: (SeedRecord, feature_vector, prediction)
        self._cached_selection: Optional[Tuple[SeedRecord, np.ndarray, float]] = None
        self.last_prediction: float = 0.0
        self.last_error: float = 0.0

    def select_seed(self, corpus: Corpus, iteration: int) -> SeedRecord:
        """Select a seed using epsilon-greedy linear contextual bandit policy.

        Args:
            corpus: Active corpus containing seed records.
            iteration: Current fuzzing loop iteration number.

        Returns:
            The chosen SeedRecord.

        Raises:
            ValueError: If corpus is empty.
        """
        if not corpus.records:
            raise ValueError("Corpus is empty. Please add or load at least one seed.")

        # Epsilon-greedy decision
        if len(corpus.records) > 1 and self.rng.random() < self.epsilon:
            # Exploration: sample any seed uniformly
            chosen = self.rng.choice(corpus.records)
            feat = self.feature_extractor.extract(chosen, iteration)
            pred = self.model.predict(feat)
            self.explorations += 1
        else:
            # Exploitation: score all candidate seeds and select argmax
            best_pred = -float("inf")
            best_candidates: List[Tuple[SeedRecord, np.ndarray, float]] = []

            for record in corpus.records:
                feat = self.feature_extractor.extract(record, iteration)
                pred = self.model.predict(feat)

                if pred > best_pred:
                    best_pred = pred
                    best_candidates = [(record, feat, pred)]
                elif abs(pred - best_pred) < 1e-9:
                    best_candidates.append((record, feat, pred))

            # Tie-break randomly among top-scoring seeds
            chosen_tuple = self.rng.choice(best_candidates)
            chosen, feat, pred = chosen_tuple
            self.exploitations += 1

        self.last_prediction = pred
        self._cached_selection = (chosen, feat, pred)
        chosen.record_selection(iteration)
        self.total_selections += 1
        return chosen

    def update(self, seed: SeedRecord, reward: int, iteration: int) -> None:
        """Update the online linear model and seed metadata with observed reward.

        Args:
            seed: The SeedRecord that was selected for mutation.
            reward: Number of new coverage units discovered (>= 0).
            iteration: Current fuzzing loop iteration number.
        """
        self.total_reward += reward
        seed.record_reward(reward)

        # Retrieve feature vector from decision time if possible
        if self._cached_selection is not None and self._cached_selection[0].id == seed.id:
            _, feat, _ = self._cached_selection
        else:
            feat = self.feature_extractor.extract(seed, iteration)

        # Perform online SGD update
        pred, error = self.model.update(feat, float(reward))
        self.last_prediction = pred
        self.last_error = error

    def get_telemetry(self) -> Dict[str, Any]:
        """Return comprehensive contextual bandit telemetry."""
        exploration_rate = (
            self.explorations / self.total_selections if self.total_selections > 0 else self.epsilon
        )
        return {
            "strategy": self.name,
            "epsilon": self.epsilon,
            "total_selections": self.total_selections,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
            "empirical_exploration_rate": round(exploration_rate, 4),
            "model_updates": self.model.total_updates,
            "model_mse": round(self.model.mean_squared_error, 6),
            "model_mae": round(self.model.mean_absolute_error, 6),
            "last_prediction": round(self.last_prediction, 4),
            "last_error": round(self.last_error, 4),
            "learned_weights": self.model.get_weights_dict(self.feature_extractor.FEATURE_NAMES),
        }

