"""Insights agent — extracts marketing psychology from scraped Meta ads.

This module exposes two factory functions consumed by ``crew.py``:

* ``build_insights_agent()`` → ``crewai.Agent``
* ``build_insights_task(agent, ads_json, run_id)`` → ``crewai.Task``

RL integration (RL-FR-02): the LLM model string is resolved via
``config.get_active_model()``, which transparently honours any
bandit-selected ``llm_model_idx``.
"""

from __future__ import annotations

from textwrap import dedent

from crewai import Agent, Task

from cwt_ads_agent.config import config
from cwt_ads_agent.models import MarketingInsights
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# Prompt constants
# ------------------------------------------------------------------ #

_BACKSTORY = dedent("""\
    You are a conversion copywriting expert trained on $500M+ in ad spend.
    You identify the precise psychological triggers in top-performing ads.
    You output ONLY valid JSON. No markdown. No explanations outside JSON fields.\
""")

_TASK_DESCRIPTION_TEMPLATE = dedent("""\
    Analyse these top-performing Meta ads for the trading/investing niche:
    {ads_json}

    Extract and return a JSON object with EXACTLY these keys:

    - pain_points: array of exactly 3 strings, each a distinct emotional pain
      point targeted by these ads (e.g., fear of missing gains, feeling left out,
      stress about financial future)
    - hook_formulas: array of opening hook patterns used (classify each as
      fear_of_missing_out, curiosity_gap, or shocking_statistic)
    - cta_types: array of CTA styles used (classify as soft/medium/hard)
    - concept_brief: exactly 150-word summary of the winning marketing angle.
      Must include: primary emotion targeted, proof elements used, urgency triggers.

    Conform to MarketingInsights Pydantic schema. run_id: {run_id}\
""")

_EXPECTED_OUTPUT = dedent("""\
    A single, valid JSON object conforming to the MarketingInsights schema
    with keys: pain_points (exactly 3), hook_formulas, cta_types,
    concept_brief (150 words), and run_id.\
""")


# ------------------------------------------------------------------ #
# Factory: Agent
# ------------------------------------------------------------------ #

def build_insights_agent() -> Agent:
    """Construct the Marketing Insights agent.

    The LLM is resolved at call-time via ``config.get_active_model()``
    so RL-injected model overrides take effect (RL-FR-02).

    Returns
    -------
    crewai.Agent
    """
    model = config.get_active_model()
    _log.info("Building InsightsAgent with model=%s", model)

    return Agent(
        role="Marketing Insights Analyst",
        goal=(
            "Distil the top-performing Meta ad creatives into actionable "
            "psychological insights — pain points, hook formulas, CTA styles, "
            "and a tight concept brief — formatted as valid JSON."
        ),
        backstory=_BACKSTORY,
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# ------------------------------------------------------------------ #
# Factory: Task
# ------------------------------------------------------------------ #

def build_insights_task(
    agent: Agent,
    ads_json: str,
    run_id: str,
) -> Task:
    """Construct the insights-extraction task.

    Parameters
    ----------
    agent:
        The ``Agent`` instance returned by ``build_insights_agent()``.
    ads_json:
        Serialised ``List[AdResearch]`` JSON string from the research stage.
    run_id:
        Pipeline run identifier (UUID4 hex).

    Returns
    -------
    crewai.Task
    """
    description = _TASK_DESCRIPTION_TEMPLATE.format(
        ads_json=ads_json,
        run_id=run_id,
    )

    _log.info("Building InsightsTask for run_id=%s", run_id)

    return Task(
        description=description,
        expected_output=_EXPECTED_OUTPUT,
        agent=agent,
        output_json=MarketingInsights,
    )
