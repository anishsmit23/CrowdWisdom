"""RL-UT-04: ExperienceDB — insert, query, corrupt-DB recovery.

All file I/O uses the ``tmp_path`` fixture.
"""

import pytest

from cwt_ads_agent.models import RLRunRecord
from cwt_ads_agent.rl.experience_db import ExperienceDB


def _make_record(run_id: str, reward: float = 0.75, **overrides) -> RLRunRecord:
    defaults = dict(
        run_id=run_id,
        timestamp="2026-05-05T00:00:00Z",
        keyword_set_idx=0,
        llm_model_idx=1,
        tone_style_idx=2,
        hook_type_idx=0,
        cta_aggression_idx=1,
        image_style_idx=3,
        voice_id_idx=0,
        reward=reward,
        script_quality=0.8,
        visual_coherence=0.7,
        audio_clarity=0.6,
        production_completeness=0.9,
        diversity_bonus=1.0,
        human_override=0,
        pipeline_duration_s=45.2,
    )
    defaults.update(overrides)
    return RLRunRecord(**defaults)


class TestExperienceDB:
    def test_insert_and_get_all_returns_3(self, tmp_path):
        """Insert 3 records → get_all() returns list of length 3."""
        db = ExperienceDB(tmp_path / "test.db")

        db.insert(_make_record("run-1", reward=0.5))
        db.insert(_make_record("run-2", reward=0.7))
        db.insert(_make_record("run-3", reward=0.9))

        rows = db.get_all()
        assert len(rows) == 3
        assert rows[0]["run_id"] == "run-1"
        assert rows[2]["run_id"] == "run-3"

    def test_get_stats(self, tmp_path):
        """Stats reflect inserted records."""
        db = ExperienceDB(tmp_path / "test.db")
        db.insert(_make_record("r1", reward=0.4))
        db.insert(_make_record("r2", reward=0.8))
        db.insert(_make_record("r3", reward=0.6))

        stats = db.get_stats()
        assert stats["total_runs"] == 3
        assert stats["best_reward"] == 0.8
        assert abs(stats["avg_reward"] - 0.6) < 0.01

    def test_get_last_action_vector(self, tmp_path):
        """Last inserted record's action vector is returned."""
        db = ExperienceDB(tmp_path / "test.db")
        db.insert(_make_record("r1", keyword_set_idx=0))
        db.insert(_make_record("r2", keyword_set_idx=4))

        av = db.get_last_action_vector()
        assert av is not None
        assert av["keyword_set_idx"] == 4

    def test_empty_db_returns_none(self, tmp_path):
        """Empty DB → get_last_action_vector() returns None."""
        db = ExperienceDB(tmp_path / "test.db")
        assert db.get_last_action_vector() is None

    def test_corrupt_db_recovery(self, tmp_path):
        """Corrupt DB file is deleted and recreated (SRD §4.2)."""
        db_path = tmp_path / "corrupt.db"
        db_path.write_text("THIS IS NOT A SQLITE FILE")

        # Should not raise — DB is recreated
        db = ExperienceDB(db_path)

        # Verify it works after recovery
        db.insert(_make_record("after-recovery"))
        assert db.get_stats()["total_runs"] == 1

    def test_insert_or_replace(self, tmp_path):
        """Duplicate run_id → record is updated, not duplicated."""
        db = ExperienceDB(tmp_path / "test.db")
        db.insert(_make_record("dup", reward=0.3))
        db.insert(_make_record("dup", reward=0.9))

        assert db.get_stats()["total_runs"] == 1
        rows = db.get_all()
        assert rows[0]["reward"] == 0.9
