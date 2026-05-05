"""RL-UT-03: RewardComputer with fully mocked output directory.

Creates a realistic output dir with 150-word script, 5 images,
audio, and video. Mocks mutagen and ffprobe. Asserts reward >= 0.7.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cwt_ads_agent.rl.reward import RewardComputer


def _build_output_dir(tmp_path: Path) -> Path:
    """Create a fully populated output directory."""
    out = tmp_path / "output"
    out.mkdir()

    # --- ad_script.json (150 words, high readability) ---
    base = (
        "Are you tired of missing out on big trades? "
        "You watch the market go up and wonder why you are not in. "
        "It is time to stop guessing and start winning. "
        "Our AI finds the best trades for you every single day. "
        "Join over twelve thousand traders who trust our signals. "
    )
    words = base.split()
    full_script = " ".join((words * 5)[:150])
    script = {
        "word_count": 150,
        "full_script": full_script,
        "brand_data_points": ["87% accuracy", "12000+ traders"],
        "sections": [],
    }
    (out / "ad_script.json").write_text(json.dumps(script), encoding="utf-8")

    # --- images (5 × 60KB) ---
    img_dir = out / "images"
    img_dir.mkdir()
    for i in range(1, 6):
        (img_dir / f"scene_{i}.png").write_bytes(b"\x89PNG" + b"\x00" * (60 * 1024))

    # --- audio (600KB) ---
    audio_dir = out / "audio"
    audio_dir.mkdir()
    (audio_dir / "voiceover.mp3").write_bytes(b"\xff\xfb" + b"\x00" * (600 * 1024))

    # --- final video (6MB) ---
    (out / "final_ad.mp4").write_bytes(b"\x00" * (6 * 1024 * 1024))

    return out


class TestRewardComputer:
    def test_high_quality_output_scores_above_07(self, tmp_path, mocker):
        """All sub-components present and correct → reward >= 0.7."""
        out = _build_output_dir(tmp_path)

        # Mock mutagen to return 60s duration
        mock_mp3 = MagicMock()
        mock_mp3.return_value.info.length = 60.0
        mocker.patch.dict(
            "sys.modules",
            {"mutagen": MagicMock(), "mutagen.mp3": MagicMock(MP3=mock_mp3)},
        )

        # Mock ffprobe to return 60.0 seconds
        mock_ffprobe = MagicMock()
        mock_ffprobe.stdout = "60.0\n"
        mocker.patch("subprocess.run", return_value=mock_ffprobe)

        rc = RewardComputer()
        result = rc.compute(
            output_dir=out,
            action_vec={"tone_style_idx": 0, "hook_type_idx": 1},
            prev_action_vec=None,
        )

        assert result["reward"] >= 0.7, (
            f"Expected reward >= 0.7, got {result['reward']}\n"
            f"Sub-scores: script={result['script_quality']}, "
            f"visual={result['visual_coherence']}, "
            f"audio={result['audio_clarity']}, "
            f"prod={result['production_completeness']}, "
            f"diversity={result['diversity_bonus']}"
        )

    def test_missing_artefacts_give_low_reward(self, tmp_path):
        """Empty output dir → reward near 0."""
        out = tmp_path / "empty_output"
        out.mkdir()

        rc = RewardComputer()
        result = rc.compute(
            output_dir=out,
            action_vec={"a": 0},
            prev_action_vec=None,
        )

        # Only diversity_bonus (first run) contributes
        assert result["reward"] <= 0.2

    def test_diversity_bonus_first_run(self, tmp_path):
        """First run (prev=None) gets full diversity bonus."""
        out = tmp_path / "output"
        out.mkdir()

        rc = RewardComputer()
        result = rc.compute(out, action_vec={"a": 0}, prev_action_vec=None)
        assert result["diversity_bonus"] == 1.0

    def test_diversity_bonus_same_action(self, tmp_path):
        """Same action vector as previous → diversity_bonus == 0."""
        out = tmp_path / "output"
        out.mkdir()

        rc = RewardComputer()
        vec = {"a": 0, "b": 1}
        result = rc.compute(out, action_vec=vec, prev_action_vec=vec)
        assert result["diversity_bonus"] == 0.0
