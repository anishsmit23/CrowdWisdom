"""CrewAI custom tool — generates scene images via Pollinations.ai.

Pollinations is a free, no-auth image generation API.  The tool builds
a cinematic prompt from the script section metadata and the RL-selected
image style, downloads the result, and validates file size.

If all retries fail, a solid-colour placeholder PNG (#1B3A5C) is
generated locally via Pillow so the pipeline never stalls on visuals.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import List, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from cwt_ads_agent.config import config
from cwt_ads_agent.utils.logger import get_logger
from cwt_ads_agent.utils.retry import AgentError, retry_with_backoff

_log = get_logger(__name__)

# ------------------------------------------------------------------ #
# RL-selectable image styles
# ------------------------------------------------------------------ #

IMAGE_STYLES: List[str] = [
    "cinematic financial trading floor",
    "minimalist chart and data visualization",
    "lifestyle wealthy trader home office",
    "dramatic dark moody finance",
    "bright optimistic success achievement",
]

_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
_MIN_FILE_BYTES = 10 * 1024  # 10 KB
_FALLBACK_COLOUR = (0x1B, 0x3A, 0x5C)  # dark navy


# ------------------------------------------------------------------ #
# Tool input schema
# ------------------------------------------------------------------ #

class _ImageInput(BaseModel):
    """Input schema for PollinationsImageTool."""

    section_name: str = Field(
        ..., description="Script section name (e.g. 'Hook', 'CTA')."
    )
    visual_description: str = Field(
        ..., description="Visual treatment description from ScriptSection."
    )
    image_style_idx: int = Field(
        default=0,
        description="Index into IMAGE_STYLES selected by the RL bandit.",
    )
    scene_num: int = Field(
        ..., description="Scene number used for the output filename."
    )


# ------------------------------------------------------------------ #
# Tool implementation
# ------------------------------------------------------------------ #

class PollinationsImageTool(BaseTool):
    """Generates scene images via Pollinations.ai (free, no auth)."""

    name: str = "Pollinations Image Generator"
    description: str = (
        "Generates scene images via Pollinations.ai (free, no auth). "
        "Returns the file path to the saved PNG."
    )
    args_schema: Type[BaseModel] = _ImageInput

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def _run(
        self,
        section_name: str,
        visual_description: str,
        image_style_idx: int = 0,
        scene_num: int = 1,
    ) -> str:
        """Generate and save a scene image.

        Parameters
        ----------
        section_name:
            Script section label (e.g. ``"Hook"``).
        visual_description:
            Free-text visual brief from the ``ScriptSection``.
        image_style_idx:
            Index into ``IMAGE_STYLES`` (RL-selected).
        scene_num:
            Ordinal used in the output filename.

        Returns
        -------
        str
            Absolute path to the saved PNG file.
        """
        style = IMAGE_STYLES[image_style_idx % len(IMAGE_STYLES)]
        prompt = (
            f"{style} scene, {section_name} concept, {visual_description}, "
            "cinematic lighting, modern, clean background, no text overlay, "
            "4K quality"
        )

        images_dir = config.output_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        out_path = images_dir / f"scene_{scene_num}.png"

        _log.info(
            "PollinationsImageTool — scene %d, style=%r", scene_num, style
        )

        try:
            retry_with_backoff(
                fn=self._download_image,
                max_retries=3,
                backoff=[1, 2, 4],
                error_context={
                    "tool": self.name,
                    "section": section_name,
                    "scene_num": scene_num,
                },
                prompt=prompt,
                out_path=out_path,
            )
        except AgentError:
            _log.warning(
                "All Pollinations retries failed for scene %d — "
                "generating fallback placeholder",
                scene_num,
            )
            self._generate_fallback(out_path)

        return str(out_path)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _download_image(*, prompt: str, out_path: Path) -> None:
        """GET the image from Pollinations and validate file size."""
        import requests  # lazy — not needed at import time

        encoded = urllib.parse.quote(prompt, safe="")
        url = (
            f"{_POLLINATIONS_BASE}/{encoded}"
            "?width=1080&height=1920&nologo=true"
        )

        _log.info("GET %s", url[:120] + "…")

        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()

        # Stream to disk
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size = out_path.stat().st_size
        if size < _MIN_FILE_BYTES:
            out_path.unlink(missing_ok=True)
            raise AgentError(
                f"Downloaded image too small ({size} bytes < {_MIN_FILE_BYTES})",
                context={"file": str(out_path), "size_bytes": size},
            )

        _log.info("Saved %s (%d KB)", out_path.name, size // 1024)

    @staticmethod
    def _generate_fallback(out_path: Path) -> None:
        """Create a solid-colour 1080×1920 PNG as a last-resort placeholder."""
        try:
            from PIL import Image

            img = Image.new("RGB", (1080, 1920), _FALLBACK_COLOUR)
            img.save(out_path, format="PNG")
            _log.info(
                "Fallback placeholder saved to %s (%d KB)",
                out_path.name,
                out_path.stat().st_size // 1024,
            )
        except ImportError:
            # Pillow not installed — write a minimal valid 1×1 PNG manually.
            _log.warning("Pillow not available — writing minimal PNG stub")
            _write_minimal_png(out_path)


def _write_minimal_png(out_path: Path) -> None:
    """Write a valid 1×1 pixel PNG without any external dependency."""
    import struct
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        raw = chunk_type + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + bytes(_FALLBACK_COLOUR)
    idat_data = zlib.compress(raw_row)

    with open(out_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr_data))
        f.write(_chunk(b"IDAT", idat_data))
        f.write(_chunk(b"IEND", b""))
