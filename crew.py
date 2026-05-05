"""Main entry point — orchestrates the full ad-creation pipeline.

Wires together all agents, tools, and the RL subsystem into a
sequential CrewAI workflow.  Provides CLI flags for operational
control (``--reset-rl``, ``--rl-report``, ``--dry-run``).

RL integration hooks
--------------------
* **PRE-RUN (RL-FR-20):**  The bandit selects an action vector,
  which is injected into ``config`` before any agent is instantiated.
* **POST-RUN (RL-FR-19):**  The reward computer scores the output
  artefacts and the bandit is updated with the reward.  Policy is
  written atomically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cwt_ads_agent.config import config
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)


# ------------------------------------------------------------------ #
# RL subsystem lazy imports (avoids circular / missing-dep issues)
# ------------------------------------------------------------------ #

def _init_rl():
    """Initialise bandit, policy manager, experience DB, and reward computer."""
    from cwt_ads_agent.rl.action_space import DIM_SIZES
    from cwt_ads_agent.rl.bandit import UCB1ContextualBandit
    from cwt_ads_agent.rl.experience_db import ExperienceDB
    from cwt_ads_agent.rl.policy import PolicyManager
    from cwt_ads_agent.rl.reward import RewardComputer

    policy_path = config.rl_memory_path / "policy.json"
    db_path = config.rl_memory_path / "experience.db"

    bandit = UCB1ContextualBandit(
        dim_sizes=DIM_SIZES,
        C=config.rl_ucb1_c,
    )
    policy_mgr = PolicyManager(policy_path)
    xp_db = ExperienceDB(db_path)
    reward_computer = RewardComputer()

    # Restore bandit from persisted policy (warm start)
    existing_policy = policy_mgr.read()
    if existing_policy is not None:
        bandit.load_from_policy_dict({
            "q_values": existing_policy.q_values,
            "n_values": {},  # n_values are not in PolicyJSON — cold n is safe
            "total_runs": existing_policy.total_runs,
        })
        config.total_runs = existing_policy.total_runs
        _log.info("Bandit restored — N=%d", bandit.N)

    return bandit, policy_mgr, xp_db, reward_computer


def _build_context(run_id: str, bandit) -> dict:
    """Build the context dict passed to the bandit for action selection."""
    return {
        "run_id": run_id,
        "day_of_week": datetime.now(timezone.utc).weekday(),
        "num_prior_runs": bandit.N,
    }


# ------------------------------------------------------------------ #
# Agent & Task factories
# ------------------------------------------------------------------ #

def _get_research_agent():
    """Return the research agent (stub-safe)."""
    try:
        from cwt_ads_agent.agents.research import build_research_agent
        return build_research_agent()
    except (ImportError, AttributeError):
        from crewai import Agent
        _log.warning("research.py stub — using placeholder agent")
        return Agent(
            role="Ad Researcher",
            goal="Scrape and score Meta ads for the trading niche",
            backstory="You are a world-class competitive ad researcher.",
            llm=config.get_active_model(),
            verbose=True,
            allow_delegation=False,
        )


def _get_insights_agent():
    """Return the insights agent."""
    from cwt_ads_agent.agents.insights import build_insights_agent
    return build_insights_agent()


def _get_script_agent():
    """Return the scriptwriter agent."""
    from cwt_ads_agent.agents.scriptwriter import build_scriptwriter_agent
    return build_scriptwriter_agent()


def _get_video_agent():
    """Return the video producer agent (stub-safe)."""
    try:
        from cwt_ads_agent.agents.videoproducer import build_video_agent
        return build_video_agent()
    except (ImportError, AttributeError):
        from crewai import Agent
        _log.warning("videoproducer.py stub — using placeholder agent")
        return Agent(
            role="Video Producer",
            goal="Assemble images, audio, and subtitles into a 60-second ad video",
            backstory="You are a senior video producer specialising in short-form ads.",
            llm=config.get_active_model(),
            verbose=True,
            allow_delegation=False,
        )


def _build_tasks(run_id, research_agent, insights_agent, script_agent, video_agent, cwt_knowledge: str):
    """Construct the ordered task list for the sequential crew."""
    from crewai import Task

    from cwt_ads_agent.tools.apify_tool import ApifyMetaAdsTool
    from cwt_ads_agent.tools.cwt_scraper_tool import CWTScraperTool

    # --- Task 1: Research ---
    research_task = Task(
        description=(
            f"Scrape active Meta ads for keywords: {config.get_keyword_set()}. "
            "Score by engagement, filter top 10, and return JSON. "
            f"run_id: {run_id}"
        ),
        expected_output="JSON array of top-10 AdResearch objects",
        agent=research_agent,
        tools=[ApifyMetaAdsTool()],
    )

    # --- Task 2: Insights ---
    insights_task = Task(
        description=(
            "Analyse the research output from the previous task. "
            "Extract pain_points (exactly 3), hook_formulas, cta_types, "
            "and a 150-word concept_brief. Return valid JSON conforming "
            f"to MarketingInsights schema. run_id: {run_id}"
        ),
        expected_output=(
            "A single JSON object with keys: pain_points, hook_formulas, "
            "cta_types, concept_brief, run_id"
        ),
        agent=insights_agent,
        context=[research_task],
    )

    # --- Task 3: Scriptwriting ---
    from cwt_ads_agent.agents.scriptwriter import build_rl_prefix
    from cwt_ads_agent.models import AdScript
    
    rl_prefix = build_rl_prefix(config.rl_params) if config.rl_enabled else ""

    cwt_tool = CWTScraperTool()
    script_task = Task(
        description=(
            f"{rl_prefix}"
            "Using the marketing insights from the previous task and CWT "
            "product data (scraped from crowdwisdomtrading.com), write a 60-second ad script "
            "with exactly 5 sections (130-165 words total narration). "
            f"CWT product content (excerpt): {cwt_knowledge[:1600]}\n\nReturn valid JSON conforming to AdScript schema. run_id: {run_id}"
        ),
        expected_output=(
            "A single JSON object with keys: sections (5), full_script, "
            "word_count (130-165), brand_data_points (>=2), run_id"
        ),
        agent=script_agent,
        tools=[cwt_tool],
        context=[insights_task],
        output_json=AdScript,  # Enforce schema validation
    )

    # --- Task 4: Video Production ---
    from cwt_ads_agent.agents.videoproducer import build_video_task
    video_task = build_video_task(video_agent, [script_task])

    return [research_task, insights_task, script_task, video_task]


# ------------------------------------------------------------------ #
# RL post-run hook
# ------------------------------------------------------------------ #

def _rl_post_run(
    bandit,
    policy_mgr,
    xp_db,
    reward_computer,
    action_vec,
    run_id,
    start_time,
) -> float:
    """Compute reward, update bandit, persist policy + experience (RL-FR-19)."""
    from cwt_ads_agent.models import RLRunRecord

    output_dir = config.output_path
    prev_action = xp_db.get_last_action_vector()

    reward_info = reward_computer.compute(
        output_dir=output_dir,
        action_vec=action_vec,
        prev_action_vec=prev_action,
    )
    reward = reward_info["reward"]

    # Update bandit Q-values
    bandit.update(action_vec, reward)

    # Persist policy (atomic)
    stats = xp_db.get_stats()
    best_reward = max(stats.get("best_reward", 0.0), reward)
    policy_data = policy_mgr.build_policy_json(
        bandit=bandit,
        action_vec=action_vec,
        run_id=run_id,
        best_reward=best_reward,
    )
    policy_mgr.write(policy_data)

    # Record to experience DB
    duration_s = round(time.time() - start_time, 2)
    record = RLRunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        keyword_set_idx=action_vec.get("keyword_set_idx", 0),
        llm_model_idx=action_vec.get("llm_model_idx", 0),
        tone_style_idx=action_vec.get("tone_style_idx", 0),
        hook_type_idx=action_vec.get("hook_type_idx", 0),
        cta_aggression_idx=action_vec.get("cta_aggression_idx", 0),
        image_style_idx=action_vec.get("image_style_idx", 0),
        voice_id_idx=action_vec.get("voice_id_idx", 0),
        reward=reward,
        script_quality=reward_info["script_quality"],
        visual_coherence=reward_info["visual_coherence"],
        audio_clarity=reward_info["audio_clarity"],
        production_completeness=reward_info["production_completeness"],
        diversity_bonus=reward_info["diversity_bonus"],
        human_override=int(reward_info["human_override"]),
        pipeline_duration_s=duration_s,
    )
    xp_db.insert(record)

    # Write to learning_curve.jsonl
    learning_curve_path = config.rl_memory_path / "learning_curve.jsonl"
    with open(learning_curve_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    return reward


# ------------------------------------------------------------------ #
# CLI helpers
# ------------------------------------------------------------------ #

def _handle_reset_rl() -> None:
    """Delete experience.db and policy.json (RL-FR-18)."""
    db_path = config.rl_memory_path / "experience.db"
    policy_path = config.rl_memory_path / "policy.json"

    for p in (db_path, policy_path):
        if p.exists():
            p.unlink()
            _log.info("Deleted %s", p)

    _log.info("RL state reset complete")


def _handle_rl_report() -> None:
    """Print reward table from experience.db (SRD §4.5)."""
    from cwt_ads_agent.rl.experience_db import ExperienceDB

    db_path = config.rl_memory_path / "experience.db"
    if not db_path.exists():
        print("No experience.db found — no runs recorded yet.")
        return

    xp_db = ExperienceDB(db_path)
    stats = xp_db.get_stats()

    print("\n╔══════════════════════════════════════════════╗")
    print("║           CWT Ads Agent — RL Report          ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Total runs:   {stats['total_runs']:<29}║")
    print(f"║  Best reward:  {stats['best_reward']:<29.4f}║")
    print(f"║  Avg reward:   {stats['avg_reward']:<29.4f}║")
    print("╠══════════════════════════════════════════════╣")

    rows = xp_db.get_all()
    if rows:
        print("║ run_id       │ reward │ script │ visual │ dur_s ║")
        print("╟──────────────┼────────┼────────┼────────┼───────╢")
        for r in rows[-20:]:  # last 20 rows
            print(
                f"║ {r['run_id']:<12} │ {r['reward']:6.3f} │ "
                f"{r['script_quality']:6.3f} │ {r['visual_coherence']:6.3f} │ "
                f"{r['pipeline_duration_s']:5.1f} ║"
            )
    print("╚══════════════════════════════════════════════╝\n")


def _handle_dry_run() -> None:
    """Validate config and exit."""
    _log.info("Dry run — validating configuration")
    _log.info("Config: %r", config)
    _log.info("RL enabled: %s", config.rl_enabled)
    _log.info("Active model: %s", config.get_active_model())
    _log.info("Keywords: %s", config.get_keyword_set())
    _log.info("Output path: %s", config.output_path)
    _log.info("RL memory path: %s", config.rl_memory_path)
    print("\n✓ Configuration valid. Exiting dry-run.\n")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    """Run the full CWT ads pipeline."""
    parser = argparse.ArgumentParser(
        description="CWT Ads Agent — AI-powered ad creation pipeline"
    )
    parser.add_argument(
        "--reset-rl",
        action="store_true",
        help="Delete experience.db and policy.json, then run (RL-FR-18)",
    )
    parser.add_argument(
        "--rl-report",
        action="store_true",
        help="Print reward table from experience.db, then exit (SRD §4.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit without running agents",
    )
    args = parser.parse_args()

    # --- CLI-only modes ---
    if args.rl_report:
        _handle_rl_report()
        return

    if args.dry_run:
        _handle_dry_run()
        return

    if args.reset_rl:
        _handle_reset_rl()

    # --- Pipeline run ---
    start_time = time.time()
    run_id = f"run_{uuid4().hex[:8]}"
    action_vec: dict = {}

    _log.info("Pipeline start | run_id=%s", run_id)

    # RL PRE-RUN HOOK (RL-FR-20)
    bandit = policy_mgr = xp_db = reward_computer = None
    if config.rl_enabled:
        bandit, policy_mgr, xp_db, reward_computer = _init_rl()
        action_vec = bandit.get_action_vector(
            epsilon=config.rl_exploration_epsilon,
        )
        config.inject_rl_params(action_vec)
        _log.info("RL action_vec: %s", action_vec)

    # Instantiate agents with RL-injected config
    research_agent = _get_research_agent()
    insights_agent = _get_insights_agent()
    script_agent = _get_script_agent()
    video_agent = _get_video_agent()

    # Build tasks
    # Fetch CWT product knowledge from website (replaces Google Drive)
    from cwt_ads_agent.tools.cwt_scraper_tool import CWTScraperTool
    _log.info("Fetching CWT product knowledge from website...")
    try:
        cwt_knowledge = CWTScraperTool()._run()
        _log.info("CWT knowledge ready: %d characters", len(cwt_knowledge))
    except Exception as exc:
        _log.warning("CWT scraper failed: %s — continuing with empty knowledge", exc)
        cwt_knowledge = ""

    tasks = _build_tasks(
        run_id, research_agent, insights_agent, script_agent, video_agent, cwt_knowledge,
    )

    # Assemble crew (without video task - we'll do that directly after)
    from crewai import Crew, Process

    crew = Crew(
        agents=[research_agent, insights_agent, script_agent],
        tasks=tasks[:3],  # Only research, insights, and script tasks
        process=Process.sequential,
        verbose=True,
    )

    result = None
    try:
        result = crew.kickoff()
        _log.info("Pipeline completed successfully")
        
        # --- Direct video production (after script generation) ---
        _log.info("Starting direct video production...")
        try:
            from cwt_ads_agent.tools.video_orchestrator_tool import VideoProductionOrchestratorTool
            
            _log.info("Script result type: %s", type(result))
            
            # Call the orchestrator directly with result (handles dict, string, markdown, etc.)
            orchestrator = VideoProductionOrchestratorTool()
            video_path = orchestrator._run(script_json=result)
            _log.info("Video production completed: %s", video_path)
            
        except Exception as e:
            _log.error("Direct video production failed: %s", e, exc_info=True)
            
    except Exception as exc:
        _log.error("Pipeline failed: %s", exc, exc_info=True)
    finally:
        # RL POST-RUN HOOK — always runs (RL-FR-19)
        if config.rl_enabled and bandit is not None:
            try:
                reward = _rl_post_run(
                    bandit, policy_mgr, xp_db, reward_computer,
                    action_vec, run_id, start_time,
                )
                _log.info("RL reward=%.4f | run_id=%s", reward, run_id)
            except Exception as exc:
                _log.error("RL post-run failed: %s", exc, exc_info=True)

    elapsed = round(time.time() - start_time, 1)
    _log.info("Pipeline finished in %.1fs | run_id=%s", elapsed, run_id)


if __name__ == "__main__":
    main()
