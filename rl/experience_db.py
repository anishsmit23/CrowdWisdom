"""SQLite-backed experience database for RL run records.

Stores every pipeline run as a row in the ``runs`` table, enabling
the bandit to learn from history and the ``--rl-report`` CLI to
display aggregate statistics.

Fault-tolerance (SRD §4.2): if the database file is corrupt on
startup, it is deleted and recreated with a WARNING.  Insert failures
are retried but never propagated — the pipeline must always continue.

Schema follows TRD §4.5.1.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cwt_ads_agent.models import RLRunRecord
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# SQL constants
# ------------------------------------------------------------------ #

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS runs (
    run_id                  TEXT PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    keyword_set_idx         INTEGER NOT NULL,
    llm_model_idx           INTEGER NOT NULL,
    tone_style_idx          INTEGER NOT NULL,
    hook_type_idx           INTEGER NOT NULL,
    cta_aggression_idx      INTEGER NOT NULL,
    image_style_idx         INTEGER NOT NULL,
    voice_id_idx            INTEGER NOT NULL,
    reward                  REAL NOT NULL,
    script_quality          REAL NOT NULL,
    visual_coherence        REAL NOT NULL,
    audio_clarity           REAL NOT NULL,
    production_completeness REAL NOT NULL,
    diversity_bonus         REAL NOT NULL,
    human_override          INTEGER NOT NULL DEFAULT 0,
    pipeline_duration_s     REAL NOT NULL
)
"""

_INSERT = """\
INSERT OR REPLACE INTO runs (
    run_id, timestamp,
    keyword_set_idx, llm_model_idx, tone_style_idx,
    hook_type_idx, cta_aggression_idx, image_style_idx, voice_id_idx,
    reward, script_quality, visual_coherence,
    audio_clarity, production_completeness, diversity_bonus,
    human_override, pipeline_duration_s
) VALUES (
    :run_id, :timestamp,
    :keyword_set_idx, :llm_model_idx, :tone_style_idx,
    :hook_type_idx, :cta_aggression_idx, :image_style_idx, :voice_id_idx,
    :reward, :script_quality, :visual_coherence,
    :audio_clarity, :production_completeness, :diversity_bonus,
    :human_override, :pipeline_duration_s
)
"""

_SELECT_LAST = """\
SELECT * FROM runs ORDER BY rowid DESC LIMIT 1
"""

_SELECT_ALL = """\
SELECT * FROM runs ORDER BY rowid ASC
"""

_STATS = """\
SELECT
    COUNT(*)        AS total_runs,
    MAX(reward)     AS best_reward,
    AVG(reward)     AS avg_reward
FROM runs
"""

# Dimension columns used to reconstruct an action vector
_ACTION_DIMS = [
    "keyword_set_idx",
    "llm_model_idx",
    "tone_style_idx",
    "hook_type_idx",
    "cta_aggression_idx",
    "image_style_idx",
    "voice_id_idx",
]


# ------------------------------------------------------------------ #
# ExperienceDB
# ------------------------------------------------------------------ #

class ExperienceDB:
    """Thin SQLite wrapper for RL run records.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Parent directories are
        created automatically.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        """Create the ``runs`` table, recreating the file on corruption."""
        try:
            self._execute(_CREATE_TABLE)
        except sqlite3.DatabaseError as exc:
            _log.warning(
                "Database corrupt or incompatible (%s) — deleting and "
                "recreating (SRD §4.2)",
                exc,
            )
            self.db_path.unlink(missing_ok=True)
            self._execute(_CREATE_TABLE)

    def _execute(self, sql: str, params: dict | None = None) -> None:
        """Execute a write statement, always closing the connection."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(sql, params or {})
            conn.commit()
        finally:
            conn.close()

    def _query(self, sql: str) -> list:
        """Execute a read query, return rows, always close."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchall()
            return rows
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def insert(self, record: RLRunRecord) -> None:
        """Persist a run record with retry.

        Parameters
        ----------
        record:
            Validated ``RLRunRecord`` Pydantic model.

        On total failure after 3 retries the error is logged but
        **not** raised — the pipeline must continue (SRD §4.2).
        """
        params = record.model_dump()
        backoff = [0.1, 0.5, 1.0]
        max_retries = 3

        for attempt in range(1 + max_retries):
            try:
                self._execute(_INSERT, params)
                _log.info("Inserted run_id=%s into experience DB", record.run_id)
                return
            except sqlite3.Error as exc:
                _log.warning(
                    "Insert attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff[attempt])

        _log.error(
            "All %d insert attempts failed for run_id=%s — "
            "pipeline continues without persisting this record",
            1 + max_retries,
            record.run_id,
        )

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def get_last_action_vector(self) -> Optional[Dict[str, int]]:
        """Return the action vector from the most recent run.

        Returns
        -------
        dict or None
            ``{dim_name: arm_index}`` for all 7 dimensions, or
            ``None`` if the database is empty.
        """
        try:
            rows = self._query(_SELECT_LAST)
        except sqlite3.Error as exc:
            _log.warning("get_last_action_vector failed: %s", exc)
            return None

        if not rows:
            return None

        row = rows[0]
        return {dim: row[dim] for dim in _ACTION_DIMS}

    def get_all(self) -> List[Dict[str, Any]]:
        """Return every run as a list of dicts (for ``--rl-report``)."""
        try:
            rows = self._query(_SELECT_ALL)
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            _log.warning("get_all failed: %s", exc)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics.

        Returns
        -------
        dict
            Keys: ``total_runs`` (int), ``best_reward`` (float),
            ``avg_reward`` (float).  All zero when the DB is empty.
        """
        try:
            rows = self._query(_STATS)
            row = rows[0]
            return {
                "total_runs": row["total_runs"] or 0,
                "best_reward": round(row["best_reward"] or 0.0, 4),
                "avg_reward": round(row["avg_reward"] or 0.0, 4),
            }
        except sqlite3.Error as exc:
            _log.warning("get_stats failed: %s", exc)
            return {"total_runs": 0, "best_reward": 0.0, "avg_reward": 0.0}

