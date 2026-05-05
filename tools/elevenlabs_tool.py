"""CrewAI custom tool — synthesises voice-over audio via ElevenLabs.

Uses the ElevenLabs Python SDK to convert the full ad script into a
spoken MP3 file.  The voice is selected by the RL bandit via
``voice_id_idx`` into ``VOICE_MAP``.

Duration is validated post-hoc with ``mutagen`` but never retried
(script word-count is the upstream lever for timing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError, retry_with_backoff

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# RL-selectable voice map
# ------------------------------------------------------------------ #

VOICE_MAP: Dict[int, Dict[str, str]] = {
    0: {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "style": "warm female"},
    1: {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "style": "authoritative male"},
    2: {"id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli", "style": "energetic female"},
}

_MAX_CHARS = 9_500  # stay within ElevenLabs 10K char/month free tier
_EXPECTED_DURATION_RANGE = (55.0, 65.0)  # seconds


# ------------------------------------------------------------------ #
# Tool input schema
# ------------------------------------------------------------------ #

class _TTSInput(BaseModel):
    """Input schema for ElevenLabsTTSTool."""

    full_script: str = Field(
        ...,
        description="Complete narration text to synthesise (from AdScript.full_script).",
    )
    voice_id_idx: int = Field(
        default=0,
        description="Index into VOICE_MAP selected by the RL bandit.",
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #

class ElevenLabsTTSTool(BaseTool):
    """Synthesises voice-over audio from ad script via ElevenLabs."""

    name: str = "ElevenLabs TTS"
    description: str = "Synthesises voice-over audio from ad script"
    args_schema: Type[BaseModel] = _TTSInput

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _run(self, full_script: str, voice_id_idx: int = 0) -> str:
        """Convert *full_script* to speech and save as MP3.

        Parameters
        ----------
        full_script:
            Narration text from ``AdScript.full_script``.
        voice_id_idx:
            Key into ``VOICE_MAP`` (RL-selected).

        Returns
        -------
        str
            Absolute path to the saved MP3 file.
        """
        voice = VOICE_MAP.get(
            voice_id_idx % len(VOICE_MAP),
            VOICE_MAP[0],
        )

        # --- character-count guard (req 6) ---
        script_text = full_script
        if len(script_text) > _MAX_CHARS:
            _log.warning(
                "Script length %d exceeds %d-char guard — truncating",
                len(script_text),
                _MAX_CHARS,
            )
            script_text = script_text[:_MAX_CHARS]

        audio_dir = config.output_path / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        out_path = audio_dir / "voiceover.mp3"

        _log.info(
            "ElevenLabsTTSTool — voice=%s (%s), chars=%d",
            voice["name"],
            voice["style"],
            len(script_text),
        )

        retry_with_backoff(
            fn=self._synthesise,
            max_retries=3,
            backoff=[2, 4, 8],
            error_context={
                "tool": self.name,
                "voice": voice["name"],
                "char_count": len(script_text),
            },
            script_text=script_text,
            voice=voice,
            out_path=out_path,
        )

        # --- duration check (req 4) ---
        self._check_duration(out_path)

        return str(out_path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthesise(
        *,
        script_text: str,
        voice: Dict[str, str],
        out_path: Path,
    ) -> None:
        """Call the ElevenLabs SDK and write the audio file."""
        try:
            from elevenlabs import ElevenLabs
            from elevenlabs.types import VoiceSettings
        except ImportError as exc:
            raise AgentError(
                "elevenlabs package is not installed",
                context={"dependency": "elevenlabs"},
            ) from exc

        api_key = config.elevenlabs_api_key
        if not api_key:
            raise AgentError(
                "ELEVENLABS_API_KEY is not set",
                context={"tool": "ElevenLabs TTS"},
            )

        client = ElevenLabs(api_key=api_key)

        audio_iter = client.text_to_speech.convert(
            voice_id=voice["id"],
            text=script_text,
            model_id="eleven_monolingual_v1",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.8,
            ),
        )

        # The SDK returns an iterator of bytes chunks.
        with open(out_path, "wb") as f:
            if isinstance(audio_iter, bytes):
                f.write(audio_iter)
            else:
                for chunk in audio_iter:
                    f.write(chunk)

        size_kb = out_path.stat().st_size // 1024
        _log.info("Audio saved to %s (%d KB)", out_path.name, size_kb)

    @staticmethod
    def _check_duration(out_path: Path) -> None:
        """Validate MP3 duration is within the expected 55-65 s window."""
        try:
            from mutagen.mp3 import MP3
        except ImportError:
            _log.warning(
                "mutagen not installed — skipping duration validation"
            )
            return

        if not out_path.exists():
            return

        try:
            audio = MP3(str(out_path))
            duration = audio.info.length
        except Exception as exc:  # noqa: BLE001
            _log.warning("Could not read MP3 metadata: %s", exc)
            return

        lo, hi = _EXPECTED_DURATION_RANGE
        if lo <= duration <= hi:
            _log.info("Audio duration %.1fs is within [%.0f, %.0f]s", duration, lo, hi)
        else:
            _log.warning(
                "Audio duration %.1fs is OUTSIDE expected [%.0f, %.0f]s — "
                "script word count is the upstream lever for this",
                duration,
                lo,
                hi,
            )
