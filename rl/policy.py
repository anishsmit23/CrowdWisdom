"""Atomic policy persistence for the RL bandit.

Reads and writes ``policy.json`` in the RL memory directory.  Writes
use an atomic temp-file-then-rename pattern so a crash mid-write can
never leave a corrupt policy file (satisfies RL-UT-05).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from cwt_ads_agent.models import PolicyJSON
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)


class PolicyManager:
    """Read / write ``policy.json`` with atomic writes and retry.

    Parameters
    ----------
    policy_path:
        Absolute path to the policy JSON file
        (typically ``rl_memory/policy.json``).
    """

    def __init__(self, policy_path: Path) -> None:
        self.policy_path = Path(policy_path)
        self.tmp_path = self.policy_path.with_suffix(".json.tmp")

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def read(self) -> Optional[PolicyJSON]:
        """Load the persisted policy, or ``None`` on cold start / error.

        Returns
        -------
        PolicyJSON or None
            Parsed policy model, or ``None`` if the file does not
            exist or cannot be parsed.
        """
        if not self.policy_path.exists():
            _log.info("No policy.json found — cold start")
            return None

        try:
            raw = self.policy_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            policy = PolicyJSON(**data)
            _log.info(
                "Loaded policy.json — total_runs=%d, best_reward=%.3f",
                policy.total_runs,
                policy.best_reward,
            )
            return policy
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            _log.warning("Failed to parse policy.json: %s — treating as cold start", exc)
            return None
        except OSError as exc:
            _log.warning("Failed to read policy.json: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Write (atomic, RL-UT-05)
    # ------------------------------------------------------------------ #

    def write(self, policy_data: dict) -> None:
        """Atomically persist *policy_data* to ``policy.json``.

        1. Serialise to a ``.json.tmp`` file.
        2. ``os.replace`` (atomic rename) to the final path.
        3. Verify the final file exists.

        Retries up to 3 times with backoff ``[0.1, 0.5, 1.0]``.
        On total failure the error is logged but **not** raised —
        the pipeline must continue.

        Parameters
        ----------
        policy_data:
            Dict-serialisable policy state (typically from
            ``build_policy_json``).
        """
        backoff = [0.1, 0.5, 1.0]
        max_retries = 3

        for attempt in range(1 + max_retries):
            try:
                self._atomic_write(policy_data)
                _log.info("policy.json written successfully")
                return
            except OSError as exc:
                _log.warning(
                    "Write attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff[attempt])
            finally:
                # Always clean up temp file
                try:
                    self.tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

        _log.error(
            "All %d write attempts failed — policy.json NOT updated",
            1 + max_retries,
        )

    def _atomic_write(self, policy_data: dict) -> None:
        """Write-to-temp then atomic-rename."""
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Write to temp file
        with open(self.tmp_path, "w", encoding="utf-8") as f:
            json.dump(policy_data, f, indent=2, ensure_ascii=False)

        # 2. Atomic rename
        os.replace(str(self.tmp_path), str(self.policy_path))

        # 3. Verify
        if not self.policy_path.exists():
            raise OSError(f"Verification failed: {self.policy_path} missing after replace")

    # ------------------------------------------------------------------ #
    # Builder
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_policy_json(
        bandit: Any,
        action_vec: Dict[str, int],
        run_id: str,
        best_reward: float,
    ) -> Dict[str, Any]:
        """Construct a full ``PolicyJSON``-compatible dict from bandit state.

        Parameters
        ----------
        bandit:
            ``UCB1ContextualBandit`` instance.
        action_vec:
            The action vector selected for this run.
        run_id:
            Pipeline run identifier.
        best_reward:
            Highest reward observed across all runs.

        Returns
        -------
        dict
            Serialisable dict conforming to the ``PolicyJSON`` schema.
        """
        policy_dict = bandit.get_policy_dict()

        return {
            "run_id_generated": run_id,
            "total_runs": bandit.N,
            "best_reward": round(best_reward, 4),
            "action_vector": {k: int(v) for k, v in action_vec.items()},
            "q_values": policy_dict["q_values"],
            "exploration_phase": bandit.N < 10,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
