"""Subtitle timing and checkpoint utilities.

Small helpers used by the video pipeline and the checkpoint-skip
logic (FR-05, FR-12, FR-27).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from cwt_ads_agent.models import ScriptSection
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)


def compute_subtitle_timings(
    sections: List[ScriptSection],
) -> List[Dict[str, Any]]:
    """Derive subtitle timing entries from script sections.

    Each section's narration is mapped to a timed subtitle entry
    spanning ``[start_s, end_s]``.  The final entry's ``end_s`` is
    clamped to ``60.0`` to match the fixed ad duration.

    Parameters
    ----------
    sections:
        List of ``ScriptSection`` models (must have ``start_s``,
        ``end_s``, and ``narration``).

    Returns
    -------
    list[dict]
        Each dict contains ``text``, ``start_s``, and ``end_s``.
    """
    timings: List[Dict[str, Any]] = []

    for i, sec in enumerate(sections):
        entry = {
            "text": sec.narration,
            "start_s": sec.start_s,
            "end_s": sec.end_s,
        }
        # Clamp the last section to exactly 60.0
        if i == len(sections) - 1:
            entry["end_s"] = 60.0

        timings.append(entry)

    return timings


def check_checkpoint(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON checkpoint file (FR-05, FR-12, FR-27).

    Parameters
    ----------
    checkpoint_path:
        Path to the checkpoint ``.json`` file.

    Returns
    -------
    dict or None
        Parsed checkpoint data, or ``None`` if the file is missing
        or contains invalid JSON.
    """
    try:
        raw = checkpoint_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        _log.info("Loaded checkpoint from %s", checkpoint_path.name)
        return data
    except FileNotFoundError:
        _log.info("No checkpoint file at %s — starting fresh", checkpoint_path)
        return None
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning(
            "Invalid checkpoint at %s: %s — skipping",
            checkpoint_path,
            exc,
        )
        return None
