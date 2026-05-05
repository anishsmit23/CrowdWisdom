"""RL-UT-06: Human feedback override in RewardComputer.

Creates a LOW-quality output dir (auto-reward < 0.5) then
places a human_feedback.json with score=0.95.  Asserts the
override takes effect and the file is deleted.
"""

import json
from pathlib import Path

import pytest

from cwt_ads_agent.rl.reward import RewardComputer


def _build_low_quality_output(tmp_path: Path) -> Path:
    """Output dir with minimal artefacts → low auto-reward."""
    out = tmp_path / "output"
    out.mkdir()

    # Bare-minimum script with bad word count (200 → outside 130-165)
    script = {
        "word_count": 200,
        "full_script": " ".join(["word"] * 200),
        "brand_data_points": [],  # no brand data
        "sections": [],
    }
    (out / "ad_script.json").write_text(json.dumps(script), encoding="utf-8")

    # No images, no audio, no video → all sub-scores = 0
    return out


class TestHumanFeedbackOverride:
    def test_override_replaces_auto_reward(self, tmp_path):
        """human_feedback.json score=0.95 overrides low auto-reward."""
        out = _build_low_quality_output(tmp_path)

        # Write human feedback
        fb = {"run_id": "test-run", "score": 0.95, "notes": "excellent creative"}
        (out / "human_feedback.json").write_text(
            json.dumps(fb), encoding="utf-8"
        )

        rc = RewardComputer()
        result = rc.compute(
            output_dir=out,
            action_vec={"a": 0},
            prev_action_vec=None,
        )

        assert result["reward"] == 0.95
        assert result["human_override"] is True

    def test_feedback_file_deleted_after_read(self, tmp_path):
        """human_feedback.json is consumed (deleted) after compute."""
        out = _build_low_quality_output(tmp_path)
        fb_path = out / "human_feedback.json"

        fb = {"score": 0.80, "notes": "ok"}
        fb_path.write_text(json.dumps(fb), encoding="utf-8")

        rc = RewardComputer()
        rc.compute(out, action_vec={"a": 0}, prev_action_vec=None)

        assert not fb_path.exists(), (
            "human_feedback.json should be deleted after compute()"
        )

    def test_auto_reward_without_override(self, tmp_path):
        """Without human_feedback.json, auto-reward is < 0.5 for bad output."""
        out = _build_low_quality_output(tmp_path)

        rc = RewardComputer()
        result = rc.compute(
            output_dir=out,
            action_vec={"a": 0},
            prev_action_vec=None,
        )

        assert result["reward"] < 0.5
        assert result["human_override"] is False

    def test_invalid_score_ignored(self, tmp_path):
        """Score outside [0,1] → override is ignored."""
        out = _build_low_quality_output(tmp_path)

        fb = {"score": 1.5, "notes": "invalid"}
        (out / "human_feedback.json").write_text(
            json.dumps(fb), encoding="utf-8"
        )

        rc = RewardComputer()
        result = rc.compute(out, action_vec={"a": 0}, prev_action_vec=None)

        # Override should be ignored — auto-reward used
        assert result["human_override"] is False
        assert result["reward"] < 0.5

    def test_corrupt_feedback_json_handled(self, tmp_path):
        """Corrupt human_feedback.json → graceful fallback to auto-reward."""
        out = _build_low_quality_output(tmp_path)

        (out / "human_feedback.json").write_text(
            "{{{INVALID JSON", encoding="utf-8"
        )

        rc = RewardComputer()
        result = rc.compute(out, action_vec={"a": 0}, prev_action_vec=None)

        assert result["human_override"] is False
