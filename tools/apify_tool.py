"""CrewAI custom tool — scrapes Meta Ads Library via Apify.

Fetches active ads in the trading / prediction-market niche,
scores them by estimated engagement, and returns the top-10
as a serialised ``List[AdResearch]`` JSON string.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.models import AdResearch
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError, retry_with_backoff

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Tool input schema (Pydantic v2, used by CrewAI for validation)
# ------------------------------------------------------------------ #

class _ApifyInput(BaseModel):
    """Input schema for ApifyMetaAdsTool."""

    keywords: str = Field(
        default="",
        description=(
            "Comma-separated keyword string.  Leave empty to use the "
            "RL-resolved or .env keyword set automatically."
        ),
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #

class ApifyMetaAdsTool(BaseTool):
    """Scrapes Meta Ads Library via Apify for trading-niche ads."""

    name: str = "Meta Ads Scraper"
    description: str = (
        "Scrapes Meta Ads Library via Apify for trading niche ads. "
        "Returns a JSON list of the top-10 highest-engagement ads "
        "from the last 30 days, scored by reach × run-days."
    )
    args_schema: Type[BaseModel] = _ApifyInput

    # ------------------------------------------------------------------ #
    # Core logic
    # ------------------------------------------------------------------ #

    def _run(self, keywords: str = "") -> str:
        """Execute the scrape-filter-score pipeline.

        Parameters
        ----------
        keywords:
            Comma-separated search terms.  Falls back to
            ``config.get_keyword_set()`` when empty.

        Returns
        -------
        str
            JSON-serialised ``List[AdResearch]`` (top 10 by score).

        Raises
        ------
        AgentError
            After exhausting retries, with a context dict describing
            the failure.
        """
        resolved_keywords = keywords.strip() if keywords else config.get_keyword_set()
        _log.info("ApifyMetaAdsTool invoked — keywords=%r", resolved_keywords)

        return retry_with_backoff(
            fn=self._scrape_and_score,
            max_retries=2,
            backoff=[5, 10],
            error_context={
                "tool": self.name,
                "keywords": resolved_keywords,
            },
            keywords=resolved_keywords,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _scrape_and_score(self, *, keywords: str) -> str:
        """Run the Apify actor, filter, score, and serialise."""
        # Late import so the module can be loaded without apify installed
        # (e.g. during tests with mocks).
        try:
            from apify_client import ApifyClient
        except ImportError as exc:
            raise AgentError(
                "apify-client package is not installed",
                context={"tool": self.name, "dependency": "apify-client"},
            ) from exc

        token = config.apify_api_token
        if not token:
            raise AgentError(
                "APIFY_API_TOKEN is not set",
                context={"tool": self.name},
            )

        client = ApifyClient(token=token)

        actor_input: Dict[str, Any] = {
            "searchTerms": [kw.strip() for kw in keywords.split(",") if kw.strip()],
            "country": "US",
            "adActiveStatus": "ACTIVE",
            "limit": 50,
        }

        _log.info("Starting Apify actor 'apify/meta-ads-scraper' …")
        run = client.actor("apify/meta-ads-scraper").call(run_input=actor_input)
        items: List[Dict[str, Any]] = list(
            client.dataset(run["defaultDatasetId"]).iterate_items()
        )
        _log.info("Apify returned %d raw items", len(items))

        # --- filter: only ads started within the last 30 days ----------
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        filtered = self._filter_recent(items, cutoff)
        _log.info("%d items remain after 30-day filter", len(filtered))

        # --- score & rank ----------------------------------------------
        scored = self._score_items(filtered)
        scored.sort(key=lambda t: t[1], reverse=True)
        top_10 = scored[:10]

        # --- map to AdResearch models ----------------------------------
        results: List[AdResearch] = []
        for item, score in top_10:
            results.append(
                AdResearch(
                    ad_id=str(item.get("id", item.get("adArchiveID", ""))),
                    advertiser=item.get("pageName", item.get("advertiser", "unknown")),
                    headline=item.get("title", item.get("headline", "")),
                    body_text=item.get("body", item.get("bodyText", "")),
                    engagement_score=round(score, 4),
                )
            )

        serialised = json.dumps(
            [r.model_dump() for r in results],
            ensure_ascii=False,
            indent=2,
        )
        _log.info("Returning %d scored ads", len(results))
        return serialised

    # ------------------------------------------------------------------ #

    @staticmethod
    def _filter_recent(
        items: List[Dict[str, Any]],
        cutoff: datetime,
    ) -> List[Dict[str, Any]]:
        """Keep only ads whose start_date >= *cutoff*."""
        kept: List[Dict[str, Any]] = []
        for item in items:
            raw_date = item.get("startDate") or item.get("start_date") or ""
            if not raw_date:
                continue
            try:
                # Apify typically returns ISO-8601 strings.
                start_dt = datetime.fromisoformat(
                    raw_date.replace("Z", "+00:00")
                )
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                if start_dt >= cutoff:
                    # Stash parsed date for scoring.
                    item["_start_dt"] = start_dt
                    kept.append(item)
            except (ValueError, TypeError):
                continue
        return kept

    @staticmethod
    def _score_items(
        items: List[Dict[str, Any]],
    ) -> List[tuple[Dict[str, Any], float]]:
        """Score each ad: ``run_days × ln(1 + reach_midpoint)``.

        Falls back to ``run_days × 1.0`` when reach data is missing.
        """
        now = datetime.now(timezone.utc)
        scored: List[tuple[Dict[str, Any], float]] = []

        for item in items:
            start_dt: datetime = item.get("_start_dt", now)
            run_days = max((now - start_dt).total_seconds() / 86_400, 0.1)

            reach_lo = item.get("reach_lower_bound") or item.get("reachLowerBound")
            reach_hi = item.get("reach_upper_bound") or item.get("reachUpperBound")

            if reach_lo is not None and reach_hi is not None:
                try:
                    midpoint = (float(reach_lo) + float(reach_hi)) / 2.0
                    score = run_days * math.log(1 + midpoint)
                except (ValueError, TypeError):
                    score = run_days * 1.0
            else:
                score = run_days * 1.0

            scored.append((item, score))

        return scored
