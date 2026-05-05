"""Scriptwriter agent — produces a 60-second video ad script.

Exposes:

* ``build_rl_prefix(rl_params)``       → RL tone directive string
* ``build_scriptwriter_agent()``       → ``crewai.Agent``
* ``build_scriptwriter_task(...)``     → ``crewai.Task``
* ``validate_word_count(result, ...)`` → retry helper for 130-165 range

RL integration:
  - LLM model resolved via ``config.get_active_model()`` (RL-FR-02).
  - Tone, hook, and CTA style injected into the prompt prefix from the
    bandit-selected ``rl_params`` dict (tone_style_idx, hook_type_idx,
    cta_aggression_idx).
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict, Optional

from crewai import Agent, Task

from cwt_ads_agent.config import config
from cwt_ads_agent.models import AdScript
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# RL action space for tone / hook / CTA (scriptwriter-local)
# ------------------------------------------------------------------ #

_SCRIPT_ACTION_SPACE: Dict[str, list] = {
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
        "soft: Learn more",
        "medium: Join thousands of traders",
        "hard: Sign up now - limited spots",
    ],
}


# ------------------------------------------------------------------ #
# RL prefix builder
# ------------------------------------------------------------------ #

def build_rl_prefix(rl_params: dict) -> str:
    """Build the TONE DIRECTIVE prefix from bandit-selected indices.

    Parameters
    ----------
    rl_params:
        Dict with optional keys ``tone_style_idx``, ``hook_type_idx``,
        ``cta_aggression_idx`` (all ``int``).  Missing keys fall back
        to safe defaults.

    Returns
    -------
    str
        Multi-line directive prepended to the task description.
    """
    tone = _SCRIPT_ACTION_SPACE["tone_styles"][
        rl_params.get("tone_style_idx", 0)
    ]
    hook = _SCRIPT_ACTION_SPACE["hook_types"][
        rl_params.get("hook_type_idx", 0)
    ]
    cta = _SCRIPT_ACTION_SPACE["cta_levels"][
        rl_params.get("cta_aggression_idx", 1)
    ]
    return (
        f"TONE DIRECTIVE: Write in a [{tone}] tone.\n"
        f"Opening hook style: [{hook}].\n"
        f"CTA aggression: [{cta}].\n"
    )


# ------------------------------------------------------------------ #
# Prompt constants
# ------------------------------------------------------------------ #

_BACKSTORY = dedent("""\
    You are an elite direct-response copywriter who has generated over
    $200M in tracked revenue from video ad scripts.  You specialise in
    financial / trading products and write scripts that convert viewers
    into paying members within 60 seconds.
    You output ONLY valid JSON. No markdown. No explanations outside
    JSON fields.\
""")

_TASK_DESCRIPTION_TEMPLATE = dedent("""\
    {rl_prefix}
    CONTEXT - Winning Marketing Insights: {marketing_insights_json}
    CWT PRODUCT DATA (from Google Drive): {drive_content}

    TASK: Write a 60-second video ad script for CrowdWisdomTrading.

    STRICT REQUIREMENTS:
    - Exactly 5 sections: Hook (0-10s), Problem (10-25s), Solution (25-40s),
      Proof (40-50s), CTA (50-60s)
    - Each section has: section_name, start_s, end_s, narration, visual_description
    - total word count between 130-165 words (count ONLY narration text)
    - Include at least 2 specific CWT data points from the drive document
      (exact stats, prices, or features - not generic claims)
    - Apply the TONE DIRECTIVE above throughout ALL sections
    - CTA section must use: {cta_level}

    Output ONLY valid JSON conforming to AdScript Pydantic schema.
    run_id: {run_id}\
""")

_EXPECTED_OUTPUT = dedent("""\
    A single, valid JSON object conforming to the AdScript schema with
    exactly 5 sections (Hook, Problem, Solution, Proof, CTA), narration
    word count between 130-165, and at least 2 CWT data points.\
""")

_WORD_COUNT_CORRECTION = (
    "\n\nYour previous response had {n} words. "
    "Adjust narration length. Target: 148 words (midpoint of 130-165 range)."
)


# ------------------------------------------------------------------ #
# Factory: Agent
# ------------------------------------------------------------------ #

def build_scriptwriter_agent() -> Agent:
    """Construct the Scriptwriter agent.

    LLM resolved at call-time via ``config.get_active_model()``
    (RL-FR-02).

    Returns
    -------
    crewai.Agent
    """
    model = config.get_active_model()
    _log.info("Building ScriptwriterAgent with model=%s", model)

    return Agent(
        role="Video Ad Scriptwriter",
        goal=(
            "Write a high-converting 60-second video ad script for "
            "CrowdWisdomTrading that satisfies every constraint in the "
            "task prompt — section structure, word count, data points, "
            "and the RL-injected tone directive."
        ),
        backstory=_BACKSTORY,
        llm=model,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )


# ------------------------------------------------------------------ #
# Factory: Task
# ------------------------------------------------------------------ #

def build_scriptwriter_task(
    agent: Agent,
    marketing_insights_json: str,
    drive_content: str,
    run_id: str,
    rl_params: Optional[Dict[str, Any]] = None,
) -> Task:
    """Construct the script-writing task with RL prefix injection.

    Parameters
    ----------
    agent:
        The ``Agent`` from ``build_scriptwriter_agent()``.
    marketing_insights_json:
        Serialised ``MarketingInsights`` JSON from the insights stage.
    drive_content:
        Plain-text CWT product data from ``GDriveTool``.
    run_id:
        Pipeline run identifier.
    rl_params:
        Bandit-selected action vector.  Falls back to
        ``config.rl_params`` when ``None``.

    Returns
    -------
    crewai.Task
    """
    params = rl_params if rl_params is not None else config.rl_params
    rl_prefix = build_rl_prefix(params)
    cta_level = _SCRIPT_ACTION_SPACE["cta_levels"][
        params.get("cta_aggression_idx", 1)
    ]

    description = _TASK_DESCRIPTION_TEMPLATE.format(
        rl_prefix=rl_prefix,
        marketing_insights_json=marketing_insights_json,
        drive_content=drive_content,
        cta_level=cta_level,
        run_id=run_id,
    )

    _log.info("Building ScriptwriterTask for run_id=%s", run_id)

    return Task(
        description=description,
        expected_output=_EXPECTED_OUTPUT,
        agent=agent,
        output_json=AdScript,
    )


# ------------------------------------------------------------------ #
# Word-count validation / retry
# ------------------------------------------------------------------ #

def validate_word_count(
    result: str,
    agent: Agent,
    task: Task,
    *,
    max_retries: int = 2,
) -> str:
    """Re-run the task if the narration word count is outside [130, 165].

    Parameters
    ----------
    result:
        Raw JSON string returned by the first CrewAI execution.
    agent:
        Scriptwriter agent instance.
    task:
        Original task instance (description will be extended on retry).
    max_retries:
        Maximum number of correction attempts.

    Returns
    -------
    str
        The (possibly corrected) JSON string that passes validation.
    """
    for attempt in range(1 + max_retries):
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            word_count = _count_narration_words(parsed)
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            _log.warning(
                "Attempt %d: failed to parse script JSON — %s", attempt + 1, exc
            )
            break  # can't fix a structurally invalid response here

        if 130 <= word_count <= 165:
            _log.info(
                "Word count %d is within [130, 165] — accepted (attempt %d)",
                word_count,
                attempt + 1,
            )
            return result if isinstance(result, str) else json.dumps(parsed)

        _log.warning(
            "Attempt %d: word count %d outside [130, 165]",
            attempt + 1,
            word_count,
        )

        if attempt < max_retries:
            correction = _WORD_COUNT_CORRECTION.format(n=word_count)
            task.description += correction
            _log.info("Appending correction prompt and re-running task …")
            crew_output = task.execute_sync()
            result = (
                crew_output.raw
                if hasattr(crew_output, "raw")
                else str(crew_output)
            )

    return result if isinstance(result, str) else json.dumps(result)


def _count_narration_words(parsed: dict) -> int:
    """Sum narration word counts across all sections."""
    sections = parsed.get("sections", [])
    total = 0
    for sec in sections:
        narration = sec.get("narration", "")
        total += len(narration.split())
    return total
