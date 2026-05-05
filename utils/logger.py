"""Structured logging setup for cwt_ads_agent.

Configures a root ``cwt_ads`` logger with two handlers:

1. **StreamHandler** → ``stdout`` at the level from ``LOG_LEVEL`` env-var.
2. **FileHandler** → ``logs/pipeline.log`` at ``DEBUG`` (captures everything).

All pipeline modules call ``get_logger(__name__)`` which returns a
child of the root logger, inheriting both handlers automatically.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_ROOT_LOGGER_NAME = "cwt_ads"
_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Resolve paths relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"


# ------------------------------------------------------------------ #
# One-time root logger setup
# ------------------------------------------------------------------ #

def _setup_root_logger() -> logging.Logger:
    """Configure the ``cwt_ads`` root logger (idempotent).

    Called once on first import.  Subsequent calls return the
    already-configured logger without adding duplicate handlers.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    # Guard against duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # let handlers decide their own level
    logger.propagate = False        # don't bubble to the stdlib root logger

    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    # --- 1. StreamHandler (stdout) ---
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    stream_level = getattr(logging, log_level_str, logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # --- 2. FileHandler (logs/pipeline.log, DEBUG) ---
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            str(_LOG_FILE), encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # If we can't create the log directory / file (e.g. read-only FS),
        # fall back to stream-only logging silently.
        pass

    return logger


# Initialise on import
_root = _setup_root_logger()


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #

def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``cwt_ads`` namespace.

    Parameters
    ----------
    name:
        Typically ``__name__`` from the calling module.  The returned
        logger is named ``cwt_ads.<name>``, inheriting both the
        stream and file handlers from the root.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
