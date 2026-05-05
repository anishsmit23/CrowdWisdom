"""Generic retry-with-backoff helper for tool calls.

Supports two usage patterns:

**1. Decorator factory** (new, per spec)::

    @retry_with_backoff(max_retries=3, backoff=[2, 4, 8])
    def fetch_data():
        ...

**2. Direct call** (existing tools — backward compatible)::

    result = retry_with_backoff(
        fn=some_callable,
        max_retries=2,
        backoff=[5, 10],
        error_context={"tool": "ApifyMetaAdsTool"},
        keyword_arg="value",
    )

If a ``requests.Response`` object is available on the exception and
contains a ``Retry-After`` header (HTTP 429), that value is honoured
instead of the backoff schedule.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)

T = TypeVar("T")

_DEFAULT_BACKOFF = [2, 4, 8]


# ------------------------------------------------------------------ #
# AgentError
# ------------------------------------------------------------------ #

class AgentError(RuntimeError):
    """Raised when a tool exhausts all retry attempts.

    Carries a ``context`` dict for structured error reporting
    back to the CrewAI agent loop.  Logs at ERROR level on creation.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.context: Dict[str, Any] = context or {}
        _log.error("AgentError: %s | context=%s", message, self.context)


# ------------------------------------------------------------------ #
# Core retry engine (shared by both decorator and direct-call paths)
# ------------------------------------------------------------------ #

def _run_with_retries(
    fn: Callable[..., T],
    args: tuple,
    kwargs: dict,
    *,
    max_retries: int,
    backoff: List[int],
    exceptions: Tuple[type, ...],
    error_context: Optional[Dict[str, Any]],
) -> T:
    """Execute *fn* with retry logic.

    Parameters
    ----------
    fn:
        Callable to invoke.
    args, kwargs:
        Positional and keyword arguments forwarded to *fn*.
    max_retries:
        Number of *retries* (not total attempts).
    backoff:
        Sleep durations (seconds) between retries.  Cycled if
        shorter than *max_retries*.
    exceptions:
        Exception types that trigger a retry.
    error_context:
        Extra context dict attached to the ``AgentError`` on
        total failure.

    Returns
    -------
    T
        Whatever *fn* returns on a successful call.

    Raises
    ------
    AgentError
        If *fn* raises on every attempt.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(1 + max_retries):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:
            last_exc = exc
            _log.warning(
                "Attempt %d/%d failed: %s [%s]",
                attempt + 1,
                1 + max_retries,
                exc,
                fn.__name__ if hasattr(fn, "__name__") else str(fn),
            )
            if attempt < max_retries:
                sleep_s = _get_sleep_duration(exc, backoff, attempt)
                _log.info("Backing off %ds before retry …", sleep_s)
                time.sleep(sleep_s)

    ctx = {
        **(error_context or {}),
        "last_exception": str(last_exc),
        "max_retries": max_retries,
    }
    raise AgentError(
        f"All {1 + max_retries} attempts failed: {last_exc}",
        context=ctx,
    )


def _get_sleep_duration(
    exc: Exception,
    backoff: List[int],
    attempt: int,
) -> float:
    """Return the sleep time, honouring ``Retry-After`` if present."""
    # Check for requests.Response Retry-After header (HTTP 429)
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    return backoff[attempt % len(backoff)]


# ------------------------------------------------------------------ #
# Public API — dual-mode (decorator factory OR direct call)
# ------------------------------------------------------------------ #

def retry_with_backoff(
    fn: Optional[Callable[..., T]] = None,
    *,
    max_retries: int = 3,
    backoff: Optional[List[int]] = None,
    exceptions: Tuple[type, ...] = (Exception,),
    error_context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Retry-with-backoff — usable as both decorator and direct call.

    **Decorator factory** (no ``fn``)::

        @retry_with_backoff(max_retries=3, backoff=[2, 4, 8])
        def my_func():
            ...

    **Direct call** (``fn`` provided — backward compatible)::

        result = retry_with_backoff(
            fn=my_callable,
            max_retries=2,
            backoff=[5, 10],
            error_context={"tool": "name"},
            some_kwarg="value",          # forwarded to fn
        )

    Parameters
    ----------
    fn:
        Callable to invoke directly (direct-call mode) or ``None``
        (decorator-factory mode).
    max_retries:
        Number of *retries* (not total attempts).  ``0`` means
        call once with no retry.
    backoff:
        Sleep durations between retries.  Defaults to ``[2, 4, 8]``.
        Cycled if shorter than ``max_retries``.
    exceptions:
        Exception types that trigger a retry.
    error_context:
        Extra context dict attached to ``AgentError`` on failure.
    **kwargs:
        In direct-call mode these are forwarded to *fn*.

    Returns
    -------
    In decorator mode: a decorated function.
    In direct-call mode: whatever *fn* returns.
    """
    _backoff = backoff or _DEFAULT_BACKOFF

    # --- Direct-call mode (backward compatible) ---
    if fn is not None:
        return _run_with_retries(
            fn, (), kwargs,
            max_retries=max_retries,
            backoff=_backoff,
            exceptions=exceptions,
            error_context=error_context,
        )

    # --- Decorator-factory mode ---
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kw: Any) -> T:
            return _run_with_retries(
                func, args, kw,
                max_retries=max_retries,
                backoff=_backoff,
                exceptions=exceptions,
                error_context=error_context,
            )
        return wrapper

    return decorator
