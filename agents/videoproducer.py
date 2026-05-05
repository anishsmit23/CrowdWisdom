"""Video producer agent — orchestrates image generation, TTS, and video rendering.

Exposes:
* ``build_video_agent()``  → ``crewai.Agent``
* ``build_video_task(...)`` → ``crewai.Task`` with Video Production Orchestrator tool

RL integration:
  - Image style selected via ``image_style_idx`` from RL bandit (RL-FR-05).
  - Voice ID selected via ``voice_id_idx`` from RL bandit (RL-FR-06).
  - LLM model resolved via ``config.get_active_model()`` (RL-FR-02).

Implementation:
  The task is configured with the VideoProductionOrchestratorTool, which
  the agent will use to process the script JSON and generate the final video.
"""

from __future__ import annotations

import json
from textwrap import dedent

from crewai import Agent, Task

from cwt_ads_agent.config import config
from cwt_ads_agent.tools.video_orchestrator_tool import VideoProductionOrchestratorTool
from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)


# ------------------------------------------------------------------ #
# Agent & Task builder
# ------------------------------------------------------------------ #

def build_video_agent() -> Agent:
    """Return the video producer agent."""
    return Agent(
        role="Video Producer",
        goal=(
            "Transform the ad script into a polished, on-brand 60-second video "
            "with generated scene images, synthesised narration, and final MP4 output."
        ),
        backstory=dedent("""\
            You are an expert video producer who specialises in transforming
            ad scripts into professional short-form videos. You understand
            timing, visual composition, audio synchronisation, and have
            successfully produced hundreds of high-converting video ads.
            
            Your workflow: receive the script JSON, use the Video Production
            Orchestrator tool to automatically generate all images and audio,
            and deliver the final MP4 file path.
        """),
        llm=config.get_active_model(),
        verbose=True,
        allow_delegation=False,
    )


def build_video_task(video_agent: Agent, context_tasks: list) -> Task:
    """Construct the video production task.

    Parameters
    ----------
    video_agent : Agent
        The video producer agent.
    context_tasks : list
        List of upstream Task objects (script_task, etc.) to use as context.

    Returns
    -------
    Task
        The task for video production with the orchestrator tool.
    """
    video_task = Task(
        description=(
            "CRITICAL INSTRUCTIONS:\n"
            "1. The previous task (Scriptwriter) has provided an AdScript JSON object\n"
            "2. You MUST extract the script JSON from the previous task context\n"
            "3. Convert it to a JSON string using json.dumps() if it's a dict\n"
            "4. Call the Video Production Orchestrator tool with this JSON string\n"
            "\n"
            "The Orchestrator will automatically:\n"
            "- Generate scene images (1 per section) via Pollinations.ai\n"
            "- Synthesise voice-over audio via ElevenLabs TTS\n"
            "- Build timing and subtitles\n"
            "- Render the final MP4 video via Remotion\n"
            "\n"
            "Your output should be the file path to the final_ad.mp4 file path "
            "(absolute path, e.g., C:\\projects\\CrowdWisdom-agent\\output\\final_ad.mp4)"
        ),
        expected_output=(
            "Absolute file path to final_ad.mp4. "
            "Example: C:\\projects\\CrowdWisdom-agent\\output\\final_ad.mp4"
        ),
        agent=video_agent,
        tools=[VideoProductionOrchestratorTool()],
        context=context_tasks,
    )

    return video_task
