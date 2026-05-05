"""UCB1 contextual bandit for RL-guided ad optimisation.  numpy only.

Each *dimension* of the ad-creation pipeline (LLM model, keyword set,
tone, hook, CTA, image style, voice) is treated as an independent
multi-armed bandit.  Arms within a dimension share the same scalar
reward signal from the composite reward function.

The UCB1 selection rule balances exploitation (highest Q-value) with
exploration (bonus for under-sampled arms), governed by the constant
``C`` (default √2 ≈ 1.414).  An optional ε-greedy override allows
forced exploration during early runs.

Reconciles TRD §4.2.2 pseudo-code for incremental mean updates.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class UCB1ContextualBandit:
    """Per-dimension UCB1 bandit with ε-greedy fallback.

    Parameters
    ----------
    dim_sizes:
        Mapping of dimension names to the number of arms.
        e.g. ``{"keyword_set_idx": 5, "llm_model_idx": 4, ...}``
    C:
        UCB1 exploration constant.  Default ``√2 ≈ 1.414``.
    """

    def __init__(self, dim_sizes: Dict[str, int], C: float = 1.414) -> None:
        self.C: float = C
        self.dim_sizes: Dict[str, int] = dict(dim_sizes)
        self.N: int = 0  # total runs across all dimensions

        # Per-dimension, per-arm Q-values (running mean reward)
        self.Q: Dict[str, np.ndarray] = {
            dim: np.zeros(size, dtype=np.float64)
            for dim, size in dim_sizes.items()
        }
        # Per-dimension, per-arm pull counts
        self.n: Dict[str, np.ndarray] = {
            dim: np.zeros(size, dtype=np.float64)
            for dim, size in dim_sizes.items()
        }

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    def get_action_vector(self, epsilon: float = 0.0) -> Dict[str, int]:
        """Select one arm per dimension via UCB1 (+ optional ε-greedy).

        Parameters
        ----------
        epsilon:
            Probability of choosing a uniformly random arm instead of
            the UCB1-maximising arm.  ``0.0`` = pure UCB1.

        Returns
        -------
        dict
            Action vector mapping each dimension name to the chosen
            arm index (int).
        """
        action: Dict[str, int] = {}

        for dim, size in self.dim_sizes.items():
            if np.random.random() < epsilon:
                # ε-greedy exploration
                action[dim] = int(np.random.randint(size))
            else:
                # UCB1 exploitation + exploration bonus
                N_safe = max(self.N, 1)
                ucb_scores = self.Q[dim] + self.C * np.sqrt(
                    np.log(N_safe) / np.maximum(self.n[dim], 1)
                )
                action[dim] = int(np.argmax(ucb_scores))

        return action

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def update(self, action_vec: Dict[str, int], reward: float) -> None:
        """Update Q-values for every arm pulled in *action_vec*.

        Uses the incremental mean formula (TRD §4.2.2)::

            Q_new = (Q_old × n + reward) / (n + 1)

        Parameters
        ----------
        action_vec:
            The action vector that produced the observed reward.
        reward:
            Scalar composite reward for this run.
        """
        for dim, arm_idx in action_vec.items():
            if dim not in self.Q:
                continue
            n_prev = self.n[dim][arm_idx]
            self.Q[dim][arm_idx] = (
                self.Q[dim][arm_idx] * n_prev + reward
            ) / (n_prev + 1)
            self.n[dim][arm_idx] += 1

        self.N += 1

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def get_policy_dict(self) -> Dict[str, Any]:
        """Export the current policy state as a JSON-serialisable dict.

        Returns
        -------
        dict
            Contains ``q_values``, ``n_values``, and ``total_runs``.
        """
        return {
            "q_values": {d: self.Q[d].tolist() for d in self.Q},
            "n_values": {d: self.n[d].tolist() for d in self.n},
            "total_runs": self.N,
        }

    def load_from_policy_dict(self, policy: Dict[str, Any]) -> None:
        """Restore Q, n, N from a previously saved policy dict.

        Called on startup to resume learning from
        ``rl_memory/policy.json``.

        Parameters
        ----------
        policy:
            Dict produced by ``get_policy_dict()``.  Missing or
            extra dimensions are handled gracefully.
        """
        self.N = int(policy.get("total_runs", 0))

        q_values = policy.get("q_values", {})
        n_values = policy.get("n_values", {})

        for dim, size in self.dim_sizes.items():
            if dim in q_values:
                arr = np.array(q_values[dim], dtype=np.float64)
                # Handle size mismatches (action space may have changed)
                if len(arr) >= size:
                    self.Q[dim] = arr[:size]
                else:
                    self.Q[dim][:len(arr)] = arr

            if dim in n_values:
                arr = np.array(n_values[dim], dtype=np.float64)
                if len(arr) >= size:
                    self.n[dim] = arr[:size]
                else:
                    self.n[dim][:len(arr)] = arr
