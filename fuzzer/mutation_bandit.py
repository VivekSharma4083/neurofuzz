"""UCB1 Multi-Armed Bandit for online mutation operator selection (Phase 6C).

Implements the upper confidence bound (UCB1) algorithm to dynamically learn
which mutation operator produces the highest coverage discovery yield during
a fuzzing campaign.
"""

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def get_default_mutation_arms(
    has_dictionary: bool = False,
    boundary_aware: bool = False,
    include_splicing: bool = False,
) -> List[str]:
    """Return list of candidate mutation operator arm names.

    Args:
        has_dictionary: True if a non-empty protocol dictionary is loaded.
        boundary_aware: True if boundary-aware dictionary insertion is enabled.
        include_splicing: True if structured field splicing is enabled (Phase 6E).

    Returns:
        List of operator names (4, 6, or 7 arms).
    """
    byte_arms = ["flip_bit", "replace_byte", "insert_byte", "delete_byte"]
    arms = list(byte_arms)
    if has_dictionary:
        insert_arm = "dictionary_insert_boundary" if boundary_aware else "dictionary_insert_random"
        arms.extend([insert_arm, "dictionary_replace"])
    if include_splicing:
        arms.append("field_splice")
    return arms


class MutationBandit:
    """UCB1 Multi-Armed Bandit for learning productive mutation operators online."""

    def __init__(
        self,
        operators: Sequence[str],
        exploration_constant: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initialize the UCB1 mutation bandit.

        Args:
            operators: Sequence of mutation operator names (the arms of the bandit).
            exploration_constant: Exploration parameter c >= 0 (default: 1.0).
            rng: Optional seeded random.Random instance for deterministic tie-breaking.
        """
        if not operators:
            raise ValueError("operators list must contain at least one operator.")
        if exploration_constant < 0.0:
            raise ValueError(f"exploration_constant must be >= 0.0, got {exploration_constant}")

        self._operators: List[str] = list(operators)
        self.c: float = float(exploration_constant)
        self.rng: random.Random = rng if rng is not None else random.Random()

        # Arm statistics
        self.counts: Dict[str, int] = {op: 0 for op in self._operators}
        self.total_rewards: Dict[str, float] = {op: 0.0 for op in self._operators}
        self.nonzero_rewards: Dict[str, int] = {op: 0 for op in self._operators}
        self.coverage_discoveries: Dict[str, int] = {op: 0 for op in self._operators}

        self.total_selections: int = 0
        self.explorations: int = 0
        self.exploitations: int = 0
        self.history: List[Tuple[str, float]] = []

    @property
    def operators(self) -> List[str]:
        """Return a copy of the available operator arm names."""
        return list(self._operators)

    def get_mean_reward(self, operator: str) -> float:
        """Return empirical mean reward Q_i for operator i."""
        if operator not in self.counts:
            raise KeyError(f"Unknown operator: '{operator}'")
        n = self.counts[operator]
        if n == 0:
            return 0.0
        return self.total_rewards[operator] / n

    def get_ucb_score(self, operator: str) -> float:
        """Calculate UCB score for operator i: Q_i + c * sqrt(ln(N) / n_i).

        For unvisited operators (n_i == 0) or uninitialized bandit (N == 0),
        returns +infinity so that each arm is guaranteed to be visited.
        """
        if operator not in self.counts:
            raise KeyError(f"Unknown operator: '{operator}'")
        n = self.counts[operator]
        if n == 0 or self.total_selections == 0:
            return float("inf")

        mean_reward = self.total_rewards[operator] / n
        bonus = self.c * math.sqrt(math.log(self.total_selections) / n)
        return mean_reward + bonus

    def select_operator(self) -> str:
        """Select the next mutation operator using the UCB1 policy.

        Returns:
            The selected operator arm name.
        """
        # Step 1: Initial exploration of all unvisited arms
        unvisited = [op for op in self._operators if self.counts[op] == 0]
        if unvisited:
            chosen = unvisited[0]
            self.explorations += 1
            return chosen

        # Step 2: Compute UCB1 score for each arm
        best_score = -float("inf")
        best_operators: List[str] = []

        for op in self._operators:
            score = self.get_ucb_score(op)
            if score > best_score:
                best_score = score
                best_operators = [op]
            elif math.isclose(score, best_score, rel_tol=1e-9, abs_tol=1e-9):
                best_operators.append(op)

        # Break ties deterministically if single, or via seeded RNG
        if len(best_operators) == 1:
            chosen = best_operators[0]
        else:
            chosen = self.rng.choice(best_operators)

        # Telemetry: Classify as exploitation vs exploration
        # Exploitation occurs if chosen arm has maximal empirical mean reward
        max_mean = max(self.get_mean_reward(op) for op in self._operators)
        if math.isclose(self.get_mean_reward(chosen), max_mean, rel_tol=1e-9, abs_tol=1e-9):
            self.exploitations += 1
        else:
            self.explorations += 1

        return chosen

    def update(self, operator: str, reward: float) -> None:
        """Update empirical statistics for the executed operator with observed reward.

        Args:
            operator: Name of the mutation operator that was applied.
            reward: Non-negative reward (typically number of new coverage units).
        """
        if operator not in self.counts:
            raise KeyError(f"Unknown operator: '{operator}'")

        r = float(reward)
        self.counts[operator] += 1
        self.total_rewards[operator] += r
        self.total_selections += 1

        if r > 0.0:
            self.nonzero_rewards[operator] += 1
            self.coverage_discoveries[operator] += int(r)

        # Keep lightweight history of recent selections (capped to conserve memory)
        if len(self.history) < 2000:
            self.history.append((operator, r))

    def reset(self) -> None:
        """Reset all bandit statistics to their initial unvisited state."""
        for op in self._operators:
            self.counts[op] = 0
            self.total_rewards[op] = 0.0
            self.nonzero_rewards[op] = 0
            self.coverage_discoveries[op] = 0
        self.total_selections = 0
        self.explorations = 0
        self.exploitations = 0
        self.history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive telemetry dictionary for all arms."""
        stats: Dict[str, Any] = {
            "total_selections": self.total_selections,
            "exploration_constant": self.c,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
            "operators": {},
        }
        for op in self._operators:
            n = self.counts[op]
            mean_r = self.get_mean_reward(op)
            ucb = self.get_ucb_score(op)
            stats["operators"][op] = {
                "selections": n,
                "selection_rate": (n / max(1, self.total_selections)),
                "total_reward": self.total_rewards[op],
                "mean_reward": mean_r,
                "ucb_score": ucb,
                "nonzero_rewards": self.nonzero_rewards[op],
                "coverage_discoveries": self.coverage_discoveries[op],
            }
        return stats

    def snapshot(self) -> Dict[str, Any]:
        """Serialize state for checkpointing and analysis."""
        return {
            "operators": list(self._operators),
            "c": self.c,
            "counts": dict(self.counts),
            "total_rewards": dict(self.total_rewards),
            "nonzero_rewards": dict(self.nonzero_rewards),
            "coverage_discoveries": dict(self.coverage_discoveries),
            "total_selections": self.total_selections,
            "explorations": self.explorations,
            "exploitations": self.exploitations,
        }
