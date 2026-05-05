"""CrewAI custom tool — renders the final video via Remotion CLI.

Writes composition props to disk, invokes ``npx remotion render``
as a subprocess, validates the output MP4, and returns the file path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError

_log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REMOTION_DIR = _PROJECT_ROOT / "remotion_project"
_PROPS_PATH = _REMOTION_DIR / "src" / "props.json"
_RENDER_TIMEOUT = 300  # 5 minutes
_MIN_OUTPUT_BYTES = 5 * 1024 * 1024  # 5 MB


# ------------------------------------------------------------------ #
# Tool input schema
# ------------------------------------------------------------------ #

class _RemotionInput(BaseModel):
    """Input schema for RemotionRenderTool."""

    composition_props: Dict[str, Any] = Field(
        ...,
        description=(
            "Props dict passed to the CWTAd Remotion composition. "
            "Must contain scenes, audioPath, and subtitles."
        ),
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #

class RemotionRenderTool(BaseTool):
    """Renders the final ad video via Remotion CLI."""

    name: str = "Remotion Video Renderer"
    description: str = (
        "Renders the CWTAd Remotion composition into a final MP4 video "
        "at 1080×1920 @ 30 fps."
    )
    args_schema: Type[BaseModel] = _RemotionInput

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _run(self, composition_props: dict) -> str:
        """Render the video and return the output path.

        Parameters
        ----------
        composition_props:
            Dict containing ``scenes``, ``audioPath``, and ``subtitles``
            as expected by the ``CWTAd`` Remotion composition.

        Returns
        -------
        str
            Absolute path to the rendered MP4 file.

        Raises
        ------
        AgentError
            On subprocess failure or invalid output file.
        """
        out_dir = config.output_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "final_ad.mp4"

        # 1. Write props JSON
        self._write_props(composition_props)

        # 2. Run Remotion render
        self._render(out_path)

        # 5. Verify output
        self._verify_output(out_path)

        _log.info("Render complete — %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_props(composition_props: dict) -> None:
        """Serialise composition props to ``remotion_project/src/props.json``."""
        _PROPS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(_PROPS_PATH, "w", encoding="utf-8") as f:
            json.dump(composition_props, f, ensure_ascii=False, indent=2)

        _log.info("Wrote props.json (%d bytes)", _PROPS_PATH.stat().st_size)

    @staticmethod
    def _render(out_path: Path) -> None:
        """Invoke ``npx remotion render`` as a subprocess."""
        cmd = [
            "npx", "remotion", "render",
            "CWTAd",
            str(out_path),
            f"--props={_PROPS_PATH}",
            "--width=1080",
            "--height=1920",
            "--fps=30",
        ]

        _log.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(_REMOTION_DIR),
                capture_output=True,
                text=True,
                timeout=_RENDER_TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(
                f"Remotion render timed out after {_RENDER_TIMEOUT}s",
                context={
                    "tool": "Remotion Video Renderer",
                    "timeout_s": _RENDER_TIMEOUT,
                    "stdout": (exc.stdout or "")[-500:] if exc.stdout else "",
                    "stderr": (exc.stderr or "")[-500:] if exc.stderr else "",
                },
            ) from exc

        if result.returncode != 0:
            _log.error("Remotion stderr:\n%s", result.stderr[-1000:])
            raise AgentError(
                f"Remotion render failed (exit code {result.returncode})",
                context={
                    "tool": "Remotion Video Renderer",
                    "exit_code": result.returncode,
                    "stderr": result.stderr[-1000:],
                    "stdout": result.stdout[-500:],
                },
            )

        _log.info("Remotion render finished (exit 0)")

    @staticmethod
    def _verify_output(out_path: Path) -> None:
        """Check that the rendered MP4 exists and is > 5 MB."""
        if not out_path.exists():
            raise AgentError(
                f"Output file not found after render: {out_path}",
                context={"tool": "Remotion Video Renderer", "path": str(out_path)},
            )

        size = out_path.stat().st_size
        if size < _MIN_OUTPUT_BYTES:
            raise AgentError(
                f"Output file too small ({size} bytes < {_MIN_OUTPUT_BYTES})",
                context={
                    "tool": "Remotion Video Renderer",
                    "path": str(out_path),
                    "size_bytes": size,
                    "min_bytes": _MIN_OUTPUT_BYTES,
                },
            )

        _log.info("Output verified — %s (%.1f MB)", out_path.name, size / 1e6)
