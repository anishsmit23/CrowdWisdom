"""Centralised, singleton configuration for cwt_ads_agent.

Loads every .env variable via python-dotenv and exposes a mutable
``Config`` dataclass.  The RL subsystem calls ``inject_rl_params``
before agents start; downstream code reads resolved properties
(``model_name``, ``get_keyword_set()``) which transparently apply
bandit-selected overrides when RL is active.

TRD anomaly A-04 — MODEL_NAME resolution
-----------------------------------------
If RL has been active for >= 1 run *and* the bandit selected an
``llm_model_idx``, the model string is pulled from
``ACTION_SPACE.llm_models[idx]`` rather than the .env value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from cwt_ads_agent.rl.action_space import ACTION_SPACE

# ---------------------------------------------------------------------------
# Bootstrap: load .env from the project root (two levels up from this file,
# or CWD — whichever contains a .env first).
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_project_root / ".env", override=False)


def _env(key: str, default: str = "") -> str:
    """Read an env-var with a fallback default."""
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Parse a boolean env-var (accepts true/1/yes, case-insensitive)."""
    return _env(key, str(default)).strip().lower() in ("true", "1", "yes")


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Application-wide configuration — instantiated once at module level.

    Attributes are populated from environment variables with sensible
    defaults so the app can start even without a .env file (useful in CI).
    """

    # --- API keys / tokens ---
    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY"))
    apify_api_token: str = field(default_factory=lambda: _env("APIFY_API_TOKEN"))
    elevenlabs_api_key: str = field(default_factory=lambda: _env("ELEVENLABS_API_KEY"))

    # --- Google Drive ---
    gdrive_file_id: str = field(default_factory=lambda: _env("GDRIVE_FILE_ID"))
    gdrive_credentials_path: str = field(
        default_factory=lambda: _env("GDRIVE_CREDENTIALS_PATH", "credentials.json")
    )

    # --- LLM ---
    _model_name_env: str = field(
        default_factory=lambda: _env(
            "MODEL_NAME", "openrouter/google/gemini-2.0-flash-001"
        )
    )

    # --- Ad / content ---
    ad_keywords: str = field(
        default_factory=lambda: _env(
            "AD_KEYWORDS",
            "crowd wisdom, prediction markets, collective intelligence",
        )
    )

    # --- RL ---
    rl_enabled: bool = field(default_factory=lambda: _env_bool("RL_ENABLED", True))
    rl_algorithm: str = field(
        default_factory=lambda: _env("RL_ALGORITHM", "epsilon_greedy")
    )
    rl_exploration_epsilon: float = field(
        default_factory=lambda: _env_float("RL_EXPLORATION_EPSILON", 0.15)
    )
    rl_ucb1_c: float = field(
        default_factory=lambda: _env_float("RL_UCB1_C", 2.0)
    )
    rl_memory_dir: str = field(
        default_factory=lambda: _env("RL_MEMORY_DIR", "rl_memory")
    )

    # --- Paths ---
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "output"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # --- Runtime state (set by inject_rl_params) ---
    rl_params: Dict[str, Any] = field(default_factory=dict)
    total_runs: int = 0

    # ------------------------------------------------------------------ #
    # RL injection (called by crew.py before agents start)
    # ------------------------------------------------------------------ #
    def inject_rl_params(self, action_vec: dict) -> None:
        """Merge bandit-selected parameters into the live config.

        Parameters
        ----------
        action_vec : dict
            Mapping of dimension names to *indices* into the
            corresponding ``ACTION_SPACE`` lists.  Example::

                {"llm_model_idx": 1, "keyword_set_idx": 0,
                 "voice_style_idx": 2, "ad_length_idx": 1}

        If ``rl_enabled`` is ``False`` this method is a no-op.
        """
        if not self.rl_enabled:
            return
        self.rl_params = {**action_vec}

    # ------------------------------------------------------------------ #
    # Resolved properties
    # ------------------------------------------------------------------ #
    @property
    def model_name(self) -> str:
        """Return the active LLM model identifier.

        Resolution order (TRD anomaly A-04):
        1. If RL is enabled, ``total_runs >= 1``, and ``rl_params``
           contains ``llm_model_idx`` → resolve from ACTION_SPACE.
        2. Otherwise → fall back to the .env ``MODEL_NAME``.
        """
        if (
            self.rl_enabled
            and self.total_runs >= 1
            and "llm_model_idx" in self.rl_params
        ):
            idx: int = int(self.rl_params["llm_model_idx"])
            return ACTION_SPACE["llm_models"][idx % len(ACTION_SPACE["llm_models"])]
        return self._model_name_env

    def get_active_model(self) -> str:
        """Convenience alias for the ``model_name`` property.

        Called by agent factories (RL-FR-02) to obtain the
        RL-resolved or .env model identifier.
        """
        return self.model_name

    def get_keyword_set(self) -> str:
        """Return the resolved keyword string for the current run.

        If the RL bandit selected a ``keyword_set_idx``, map it to the
        concrete keyword string from ``ACTION_SPACE.keyword_sets``.
        Otherwise fall back to the ``AD_KEYWORDS`` env-var.
        """
        if self.rl_enabled and "keyword_set_idx" in self.rl_params:
            idx = int(self.rl_params["keyword_set_idx"])
            return ACTION_SPACE["keyword_sets"][idx % len(ACTION_SPACE["keyword_sets"])]
        return self.ad_keywords

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    @property
    def rl_memory_path(self) -> Path:
        """Resolved absolute path to the RL memory directory."""
        return (_project_root / self.rl_memory_dir).resolve()

    @property
    def output_path(self) -> Path:
        """Resolved absolute path to the output directory."""
        return (_project_root / self.output_dir).resolve()

    def __repr__(self) -> str:  # pragma: no cover
        """Redact secrets in repr for safe logging."""
        return (
            f"Config(model_name={self.model_name!r}, "
            f"rl_enabled={self.rl_enabled}, "
            f"rl_algorithm={self.rl_algorithm!r}, "
            f"total_runs={self.total_runs}, "
            f"log_level={self.log_level!r})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
config: Config = Config()
