"""High-level orchestrator tool for complete video production workflow.

Handles:
1. Parsing the AdScript from JSON
2. Generating images for each section
3. Generating voice-over audio
4. Rendering the final video
5. Validation and artifact collection

This tool abstracts all the complexity away from the agent, allowing it to
simply call one tool with the script and get back a path to final_ad.mp4.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.models import AdScript
from cwt_ads_agent.tools.elevenlabs_tool import ElevenLabsTTSTool
from cwt_ads_agent.tools.image_tool import PollinationsImageTool
from cwt_ads_agent.tools.remotion_tool import RemotionRenderTool
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError

_log = get_logger(__name__)

# Video parameters
_FPS = 30  # frames per second
_TOTAL_DURATION_S = 60  # seconds
_TOTAL_FRAMES = _FPS * _TOTAL_DURATION_S  # 1800 frames


# ------------------------------------------------------------------ #
# Tool input schema
# ------------------------------------------------------------------ #

class _VideoOrchestrationInput(BaseModel):
    """Input schema for VideoProductionOrchestratorTool."""

    script_json: Any = Field(
        ...,
        description=(
            "Complete ad script as a JSON string conforming to AdScript schema. "
            "Must contain sections (5), full_script, word_count, brand_data_points."
        ),
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #

class VideoProductionOrchestratorTool(BaseTool):
    """Orchestrates complete video production workflow (images → audio → render)."""

    name: str = "Video Production Orchestrator"
    description: str = (
        "End-to-end orchestrator for video ad production. "
        "Accepts the ad script as JSON, generates all images and audio, "
        "and renders the final 60-second video via Remotion. "
        "Returns the file path to final_ad.mp4."
    )
    args_schema: Type[BaseModel] = _VideoOrchestrationInput

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _run(self, script_json: Any) -> str:
        """Execute complete video production workflow.

        Parameters
        ----------
        script_json:
            Ad script as JSON string conforming to AdScript schema.

        Returns
        -------
        str
            Absolute path to the rendered final_ad.mp4 file.

        Raises
        ------
        AgentError
            On any step failure (image generation, TTS, or rendering).
        """
        _log.info("VideoProductionOrchestratorTool starting")

        # Parse script
        script = self._parse_script(script_json)
        _log.info("Parsed script: %d sections, %d words", len(script.sections), script.word_count)

        # Get RL parameters (image style, voice ID)
        image_style_idx = config.rl_params.get("image_style_idx", 0)
        voice_id_idx = config.rl_params.get("voice_id_idx", 0)
        _log.info("RL params: image_style_idx=%d, voice_id_idx=%d", image_style_idx, voice_id_idx)

        # Step 1: Generate images for each section
        scene_paths = self._generate_images(script, image_style_idx)

        # Step 2: Generate voice-over audio
        audio_path = self._generate_audio(script.full_script, voice_id_idx)

        # Step 3: Build subtitles from sections
        subtitles = self._build_subtitles(script.sections)

        # Step 4: Build scene objects with timing
        scenes = self._build_scenes(scene_paths, script.sections)

        # Step 5: Render video
        composition_props = {
            "scenes": scenes,
            "audioPath": str(audio_path) if audio_path else "",
            "subtitles": subtitles,
        }
        video_path = self._render_video(composition_props)

        _log.info("Video production complete: %s", video_path)
        return str(video_path)

    # ------------------------------------------------------------------ #
    # Step helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_script(script_json: Any) -> AdScript:
        """Parse and validate the AdScript JSON.

        Parameters
        ----------
        script_json : str
            JSON string conforming to AdScript schema.
            May be wrapped in markdown code blocks.

        Returns
        -------
        AdScript
            Parsed and validated AdScript object.

        Raises
        ------
        AgentError
            On JSON parse error or validation failure.
        """
        try:
            if isinstance(script_json, AdScript):
                return script_json

            # Handle different input formats
            if isinstance(script_json, dict):
                # If it's a dict with a 'raw' key, extract that
                if "pydantic" in script_json and script_json["pydantic"] is not None:
                    pydantic_obj = script_json["pydantic"]
                    if isinstance(pydantic_obj, AdScript):
                        return pydantic_obj
                    if isinstance(pydantic_obj, BaseModel):
                        return AdScript(**pydantic_obj.model_dump())
                if "json_dict" in script_json and isinstance(script_json["json_dict"], dict):
                    return AdScript(**script_json["json_dict"])
                if "raw" in script_json:
                    script_json_str = script_json["raw"]
                else:
                    # Direct dict - validate as AdScript
                    script = AdScript(**script_json)
                    return script
            elif hasattr(script_json, "pydantic") and getattr(script_json, "pydantic") is not None:
                pydantic_obj = getattr(script_json, "pydantic")
                if isinstance(pydantic_obj, AdScript):
                    return pydantic_obj
                if isinstance(pydantic_obj, BaseModel):
                    return AdScript(**pydantic_obj.model_dump())
            elif hasattr(script_json, "json_dict") and isinstance(getattr(script_json, "json_dict"), dict):
                return AdScript(**getattr(script_json, "json_dict"))
            elif hasattr(script_json, "raw"):
                script_json_str = str(getattr(script_json, "raw"))
            else:
                # Assume it's a JSON string
                script_json_str = str(script_json)
            
            # Clean up markdown code blocks if present
            script_json_str = script_json_str.strip()
            if script_json_str.startswith("```json"):
                script_json_str = script_json_str[7:]  # Remove ```json
            elif script_json_str.startswith("```"):
                script_json_str = script_json_str[3:]  # Remove ```
            
            if script_json_str.endswith("```"):
                script_json_str = script_json_str[:-3]  # Remove trailing ```
            
            script_json_str = script_json_str.strip()
            
            try:
                data = json.loads(script_json_str)
            except json.JSONDecodeError:
                # CrewOutput string repr can look like a Python dict with single quotes.
                data = ast.literal_eval(script_json_str)
            script = AdScript(**data)
            return script
        except json.JSONDecodeError as exc:
            raise AgentError(
                f"Failed to parse script JSON: {exc}",
                context={"tool": "VideoProductionOrchestratorTool", "step": "parse_script"},
            ) from exc
        except Exception as exc:
            raise AgentError(
                f"Failed to validate AdScript: {exc}",
                context={"tool": "VideoProductionOrchestratorTool", "step": "validate_script"},
            ) from exc

    @staticmethod
    def _generate_images(script: AdScript, image_style_idx: int) -> List[str]:
        """Generate images for each section.

        Parameters
        ----------
        script : AdScript
            Parsed ad script with 5 sections.
        image_style_idx : int
            Index into IMAGE_STYLES (RL-selected).

        Returns
        -------
        List[str]
            List of absolute paths to generated PNG images.

        Raises
        ------
        AgentError
            On image generation failure.
        """
        image_tool = PollinationsImageTool()
        scene_paths = []

        for i, section in enumerate(script.sections, start=1):
            try:
                path = image_tool._run(
                    section_name=section.section_name,
                    visual_description=section.visual_description,
                    image_style_idx=image_style_idx,
                    scene_num=i,
                )
                scene_paths.append(path)
                _log.info("Generated image %d/%d: %s", i, len(script.sections), path)
            except Exception as exc:
                raise AgentError(
                    f"Failed to generate image for section {i} ({section.section_name}): {exc}",
                    context={
                        "tool": "VideoProductionOrchestratorTool",
                        "step": "generate_images",
                        "section_num": i,
                    },
                ) from exc

        return scene_paths

    @staticmethod
    def _generate_audio(full_script: str, voice_id_idx: int) -> Path | None:
        """Generate voice-over audio from full script.

        Parameters
        ----------
        full_script : str
            Concatenated narration text (130-165 words).
        voice_id_idx : int
            Index into VOICE_MAP (RL-selected).

        Returns
        -------
        Path | None
            Absolute path to generated MP3 file, or None if TTS fails.

        Notes
        -----
        If TTS fails due provider limits, returns None and video rendering
        continues without audio.
        """
        tts_tool = ElevenLabsTTSTool()
        try:
            audio_path_str = tts_tool._run(
                full_script=full_script,
                voice_id_idx=voice_id_idx,
            )
            audio_path = Path(audio_path_str)
            _log.info("Generated audio: %s", audio_path)
            return audio_path
        except Exception as exc:
            _log.warning("Voice-over generation failed, continuing without audio: %s", exc)
            return None

    @staticmethod
    def _build_subtitles(sections: list) -> List[Dict[str, Any]]:
        """Build subtitles array from script sections.

        Parameters
        ----------
        sections : list
            List of ScriptSection objects with start_s, end_s, narration.

        Returns
        -------
        List[Dict[str, Any]]
            List of subtitle objects {text, startFrame, endFrame}.
        """
        subtitles = []
        for section in sections:
            start_frame = int(section.start_s * _FPS)
            end_frame = int(section.end_s * _FPS)
            # Split narration into chunks for subtitle display (approx 10 words per line)
            words = section.narration.split()
            chunk_size = 10
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                # Distribute chunk time evenly across its words
                chunk_start = start_frame + int((i / len(words)) * (end_frame - start_frame))
                chunk_end = start_frame + int(((i + chunk_size) / len(words)) * (end_frame - start_frame))
                chunk_end = min(chunk_end, end_frame)
                subtitles.append({
                    "text": chunk,
                    "startFrame": chunk_start,
                    "endFrame": chunk_end,
                })
        _log.info("Built %d subtitle chunks", len(subtitles))
        return subtitles

    @staticmethod
    def _build_scenes(scene_paths: List[str], sections: list) -> List[Dict[str, Any]]:
        """Build scenes array with timing information.

        Parameters
        ----------
        scene_paths : List[str]
            Absolute paths to generated PNG images (5 images).
        sections : list
            List of ScriptSection objects with start_s, end_s.

        Returns
        -------
        List[Dict[str, Any]]
            List of scene objects {imagePath, narration, startFrame, durationFrames}.
        """
        scenes = []
        for i, (path, section) in enumerate(zip(scene_paths, sections)):
            start_frame = int(section.start_s * _FPS)
            end_frame = int(section.end_s * _FPS)
            duration_frames = max(1, end_frame - start_frame)
            scenes.append({
                "imagePath": path,
                "narration": section.narration,
                "startFrame": start_frame,
                "durationFrames": duration_frames,
            })
        _log.info("Built %d scenes", len(scenes))
        return scenes

    @staticmethod
    def _render_video(composition_props: Dict[str, Any]) -> Path:
        """Render final video via Remotion.

        Parameters
        ----------
        composition_props : Dict[str, Any]
            Props dict with scenes, audioPath, subtitles.

        Returns
        -------
        Path
            Absolute path to rendered final_ad.mp4.

        Raises
        ------
        AgentError
            On Remotion render failure.
        """
        render_tool = RemotionRenderTool()
        try:
            output_path_str = render_tool._run(composition_props=composition_props)
            output_path = Path(output_path_str)
            _log.info("Rendered video: %s", output_path)
            return output_path
        except Exception as exc:
            raise AgentError(
                f"Failed to render video: {exc}",
                context={"tool": "VideoProductionOrchestratorTool", "step": "render_video"},
            ) from exc
