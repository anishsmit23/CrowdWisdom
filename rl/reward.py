"""Composite reward function for the RL bandit.

Scores each pipeline run across five sub-components and produces a
single scalar reward.  An optional human-feedback override file
(``human_feedback.json``) can replace the computed reward.

Sub-component weights (must sum to 1.0)::

    R = 0.25·script + 0.20·visual + 0.20·audio
      + 0.25·production + 0.10·diversity

Anomaly A-03 fix:  ``_production_completeness`` attempts ``ffprobe``
for duration validation but falls back gracefully to a size-only
heuristic when ffprobe is unavailable.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from cwt_ads_agent.utils.logger import get_logger

_log = get_logger(__name__)

# Sub-component weights
_W_SCRIPT = 0.25
_W_VISUAL = 0.20
_W_AUDIO = 0.20
_W_PRODUCTION = 0.25
_W_DIVERSITY = 0.10


class RewardComputer:
    """Computes the composite reward for a single pipeline run."""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute(
        self,
        output_dir: Path,
        action_vec: Dict[str, int],
        prev_action_vec: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Score all sub-components and return the reward dict.

        Parameters
        ----------
        output_dir:
            Directory containing run artefacts (``ad_script.json``,
            ``images/``, ``audio/``, ``final_ad.mp4``).
        action_vec:
            The bandit-selected action vector for *this* run.
        prev_action_vec:
            Action vector from the *previous* run (``None`` on the
            very first run).

        Returns
        -------
        dict
            Keys: ``reward``, ``script_quality``, ``visual_coherence``,
            ``audio_clarity``, ``production_completeness``,
            ``diversity_bonus``, ``human_override``.
        """
        output_dir = Path(output_dir)

        script_quality = self._script_quality(output_dir)
        visual_coherence = self._visual_coherence(output_dir)
        audio_clarity = self._audio_clarity(output_dir)
        production_completeness = self._production_completeness(output_dir)
        diversity_bonus = self._diversity_bonus(action_vec, prev_action_vec)

        reward = (
            _W_SCRIPT * script_quality
            + _W_VISUAL * visual_coherence
            + _W_AUDIO * audio_clarity
            + _W_PRODUCTION * production_completeness
            + _W_DIVERSITY * diversity_bonus
        )

        human_override = False

        # --- Human feedback override ---
        feedback_path = output_dir / "human_feedback.json"
        if feedback_path.exists():
            try:
                fb = json.loads(feedback_path.read_text(encoding="utf-8"))
                score = float(fb.get("score", -1))
                if 0.0 <= score <= 1.0:
                    _log.info(
                        "Human override applied: %.3f → %.3f (notes: %s)",
                        reward,
                        score,
                        fb.get("notes", ""),
                    )
                    reward = score
                    human_override = True
                else:
                    _log.warning(
                        "human_feedback.json has invalid score %.3f — ignoring",
                        score,
                    )
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                _log.warning("Failed to parse human_feedback.json: %s", exc)
            finally:
                feedback_path.unlink(missing_ok=True)

        _log.info(
            "Reward=%.3f  script=%.2f  visual=%.2f  audio=%.2f  "
            "prod=%.2f  diversity=%.2f  human=%s",
            reward,
            script_quality,
            visual_coherence,
            audio_clarity,
            production_completeness,
            diversity_bonus,
            human_override,
        )

        return {
            "reward": round(reward, 4),
            "script_quality": round(script_quality, 4),
            "visual_coherence": round(visual_coherence, 4),
            "audio_clarity": round(audio_clarity, 4),
            "production_completeness": round(production_completeness, 4),
            "diversity_bonus": round(diversity_bonus, 4),
            "human_override": human_override,
        }

    # ------------------------------------------------------------------ #
    # Sub-component scorers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _script_quality(output_dir: Path) -> float:
        """Score the ad script: word count + readability + brand data points."""
        script_path = output_dir / "ad_script.json"
        if not script_path.exists():
            _log.warning("ad_script.json not found — script_quality = 0")
            return 0.0

        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            _log.warning("Cannot parse ad_script.json: %s", exc)
            return 0.0

        # Word count (0.3)
        wc = script.get("word_count", 0)
        wc_score = 0.3 if 130 <= wc <= 165 else 0.0

        # Readability (0.4)
        try:
            import textstat

            flesch = textstat.flesch_reading_ease(script.get("full_script", ""))
            readability_score = 0.4 * min(flesch / 100.0, 1.0)
        except ImportError:
            _log.warning("textstat not installed — readability_score = 0")
            readability_score = 0.0

        # Brand data points (0.3)
        brand_score = (
            0.3 if len(script.get("brand_data_points", [])) >= 2 else 0.0
        )

        return wc_score + readability_score + brand_score

    @staticmethod
    def _visual_coherence(output_dir: Path) -> float:
        """Score generated images: existence count + mean file size."""
        images_dir = output_dir / "images"
        images = sorted(images_dir.glob("scene_*.png")) if images_dir.is_dir() else []

        if not images:
            _log.warning("No scene images found — visual_coherence = 0")
            return 0.0

        # Existence (0.5) — proportional if < 5 images
        exist_score = 0.5 if len(images) >= 5 else len(images) / 5 * 0.5

        # Mean size (0.5) — full marks at ≥ 200 KB average
        sizes = [p.stat().st_size / 1024 for p in images]
        mean_kb = sum(sizes) / len(sizes)
        size_score = 0.5 * min(mean_kb / 200.0, 1.0)

        return exist_score + size_score

    @staticmethod
    def _audio_clarity(output_dir: Path) -> float:
        """Score TTS audio: duration window + minimum file size."""
        mp3 = output_dir / "audio" / "voiceover.mp3"
        if not mp3.exists():
            _log.warning("voiceover.mp3 not found — audio_clarity = 0")
            return 0.0

        # Duration (0.5) — must be 55-65 s
        dur_score = 0.0
        try:
            from mutagen.mp3 import MP3

            duration = MP3(str(mp3)).info.length
            dur_score = 0.5 if 55 <= duration <= 65 else 0.0
        except ImportError:
            _log.warning("mutagen not installed — dur_score = 0")
        except Exception as exc:  # noqa: BLE001
            _log.warning("MP3 metadata read failed: %s", exc)

        # File size (0.5) — at least 500 KB
        size_kb = mp3.stat().st_size / 1024
        size_score = 0.5 if size_kb >= 500 else 0.0

        return dur_score + size_score

    @staticmethod
    def _production_completeness(output_dir: Path) -> float:
        """Score final video: existence, size, and optional ffprobe duration.

        Anomaly A-03 fix: ffprobe is attempted for accurate duration
        validation but is *not* required — falls back to a size-only
        heuristic.
        """
        mp4 = output_dir / "final_ad.mp4"
        if not mp4.exists():
            _log.warning("final_ad.mp4 not found — production_completeness = 0")
            return 0.0

        size_mb = mp4.stat().st_size / (1024 * 1024)

        # Try ffprobe for duration (A-03 fix)
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=nw=1:nk=1",
                    str(mp4),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            duration = float(result.stdout.strip())
            if 58 <= duration <= 62 and size_mb >= 5:
                return 1.0
            elif size_mb >= 2:
                return 0.5
            else:
                return 0.2
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            # ffprobe not available — size-only fallback
            if size_mb >= 5:
                return 1.0
            elif size_mb >= 2:
                return 0.5
            else:
                return 0.0

    @staticmethod
    def _diversity_bonus(
        action_vec: Dict[str, int],
        prev_action_vec: Optional[Dict[str, int]],
    ) -> float:
        """Reward exploration: bonus if ≥ 2 dimensions changed vs. previous run."""
        if prev_action_vec is None:
            return 1.0  # first run always gets full bonus

        hamming = sum(
            1
            for k in action_vec
            if action_vec[k] != prev_action_vec.get(k)
        )
        return 1.0 if hamming >= 2 else 0.0
