"""Discrete action space for the RL bandit.  No external dependencies.

Each dimension of the action vector maps to one of the lists below.
The RL agent selects an *index* into each dimension; downstream code
calls ``resolve_action()`` to map indices to concrete values.

Public API
----------
ACTION_SPACE : dict
    Catalogue of arms keyed by dimension name.
DIMS : list[str]
    Ordered list of dimension-index names used in action vectors.
DIM_SIZES : dict[str, int]
    Cardinality of each dimension (convenience for bandit init).
random_action() -> dict
    Sample a uniformly random valid action vector.
resolve_action(action_vec) -> dict
    Map an index-based action vector to human-readable parameters.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

# ------------------------------------------------------------------ #
# Action space catalogue
# ------------------------------------------------------------------ #

ACTION_SPACE: Dict[str, List[Any]] = {
    "keyword_sets": [
        "trading signals,stock alerts",
        "day trading,options trading",
        "crypto signals,bitcoin alerts",
        "crowdwisdomtrading,cwt signals",
        "stock market,investment tips",
    ],
    "llm_models": [
        "mistralai/mistral-7b-instruct:free",
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
    ],
    "tone_styles": [
        "urgent and fear-based",
        "conversational and friendly",
        "authoritative and expert",
        "inspirational and aspirational",
        "social proof and community",
    ],
    "hook_types": [
        "fear_of_missing_out",
        "curiosity_gap",
        "shocking_statistic",
    ],
    "cta_levels": [
        "soft",
        "medium",
        "hard",
    ],
    "image_styles": [
        "cinematic financial trading floor",
        "minimalist chart and data visualization",
        "lifestyle wealthy trader home office",
        "dramatic dark moody finance",
        "bright optimistic success achievement",
    ],
    "voice_ids": [
        {"idx": 0, "id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel"},
        {"idx": 1, "id": "pNInz6obpgDQGcFmaJgB", "name": "Adam"},
        {"idx": 2, "id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli"},
    ],
}

# ------------------------------------------------------------------ #
# Dimension metadata
# ------------------------------------------------------------------ #

DIMS: List[str] = [
    "keyword_set_idx",
    "llm_model_idx",
    "tone_style_idx",
    "hook_type_idx",
    "cta_aggression_idx",
    "image_style_idx",
    "voice_id_idx",
]

DIM_SIZES: Dict[str, int] = {
    "keyword_set_idx": len(ACTION_SPACE["keyword_sets"]),     # 5
    "llm_model_idx": len(ACTION_SPACE["llm_models"]),         # 4
    "tone_style_idx": len(ACTION_SPACE["tone_styles"]),       # 5
    "hook_type_idx": len(ACTION_SPACE["hook_types"]),          # 3
    "cta_aggression_idx": len(ACTION_SPACE["cta_levels"]),    # 3
    "image_style_idx": len(ACTION_SPACE["image_styles"]),     # 5
    "voice_id_idx": len(ACTION_SPACE["voice_ids"]),           # 3
}

# Internal mapping from dim-index name → ACTION_SPACE key
_DIM_TO_KEY: Dict[str, str] = {
    "keyword_set_idx": "keyword_sets",
    "llm_model_idx": "llm_models",
    "tone_style_idx": "tone_styles",
    "hook_type_idx": "hook_types",
    "cta_aggression_idx": "cta_levels",
    "image_style_idx": "image_styles",
    "voice_id_idx": "voice_ids",
}


# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #

def random_action() -> Dict[str, int]:
    """Return a uniformly random valid action vector.

    Returns
    -------
    dict
        Mapping of each dimension name (from ``DIMS``) to a random
        integer index within that dimension's valid range.
    """
    return {dim: random.randint(0, DIM_SIZES[dim] - 1) for dim in DIMS}


def resolve_action(action_vec: Dict[str, int]) -> Dict[str, Any]:
    """Map an index-based action vector to human-readable parameters.

    Parameters
    ----------
    action_vec:
        Dict mapping dimension names (``DIMS``) to integer indices.
        Missing keys are silently skipped.

    Returns
    -------
    dict
        Human-readable parameters keyed by the ACTION_SPACE list name.
        For ``voice_ids`` the full dict entry is returned.

    Example
    -------
    >>> resolve_action({"llm_model_idx": 2, "tone_style_idx": 0})
    {'llm_models': 'google/gemma-2-9b-it:free',
     'tone_styles': 'urgent and fear-based'}
    """
    resolved: Dict[str, Any] = {}
    for dim, idx in action_vec.items():
        key = _DIM_TO_KEY.get(dim)
        if key is None:
            continue
        entries = ACTION_SPACE[key]
        resolved[key] = entries[int(idx) % len(entries)]
    return resolved
