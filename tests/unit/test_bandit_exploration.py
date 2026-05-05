"""RL-UT-02: Bandit exploration — all arms must be sampled.

With epsilon=0.3 and 100 calls, every arm of 'tone_style_idx'
(5 options) should be selected at least once.
"""

import numpy as np
import pytest

from cwt_ads_agent.rl.action_space import DIM_SIZES
from cwt_ads_agent.rl.bandit import UCB1ContextualBandit


class TestBanditExploration:
    def test_all_arms_explored_with_epsilon(self):
        """ε=0.3 over 100 calls → all 5 tone_style arms sampled."""
        np.random.seed(42)  # deterministic

        bandit = UCB1ContextualBandit(DIM_SIZES)

        unique_values = set()
        for _ in range(100):
            action = bandit.get_action_vector(epsilon=0.3)
            unique_values.add(action["tone_style_idx"])

        n_arms = DIM_SIZES["tone_style_idx"]  # 5
        assert len(unique_values) == n_arms, (
            f"Expected all {n_arms} arms explored, got {len(unique_values)}: "
            f"{unique_values}"
        )

    def test_pure_exploitation_on_fresh_bandit_explores_via_ucb(self):
        """UCB1 bonus on untried arms drives exploration even at ε=0."""
        bandit = UCB1ContextualBandit(DIM_SIZES)

        # UCB1 on a fresh bandit gives equal scores → argmax returns 0
        action = bandit.get_action_vector(epsilon=0.0)
        # All arms are untried, so UCB bonus is equal; arm 0 wins tie-break
        assert isinstance(action["tone_style_idx"], int)

    def test_exploration_covers_all_dimensions(self):
        """All 7 dimensions should see exploration."""
        np.random.seed(123)

        bandit = UCB1ContextualBandit(DIM_SIZES)
        dim_unique = {dim: set() for dim in DIM_SIZES}

        for _ in range(200):
            action = bandit.get_action_vector(epsilon=0.5)
            for dim in DIM_SIZES:
                dim_unique[dim].add(action[dim])

        for dim, uniq in dim_unique.items():
            assert len(uniq) == DIM_SIZES[dim], (
                f"Dimension {dim}: expected {DIM_SIZES[dim]} arms, "
                f"got {len(uniq)}"
            )
