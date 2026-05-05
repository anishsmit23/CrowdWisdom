"""RL-UT-05: PolicyManager — atomic write, read, tmp cleanup.

All file I/O uses the ``tmp_path`` fixture.
"""

import json

import pytest

from cwt_ads_agent.rl.policy import PolicyManager


@pytest.fixture
def policy_data():
    """Minimal valid PolicyJSON dict."""
    return {
        "run_id_generated": "run-001",
        "total_runs": 1,
        "best_reward": 0.5,
        "action_vector": {"tone_style_idx": 2},
        "q_values": {"tone_style_idx": [0.0, 0.0, 0.5, 0.0, 0.0]},
        "exploration_phase": True,
        "updated_at": "2026-05-05T00:00:00Z",
    }


class TestPolicyManager:
    def test_write_creates_valid_json(self, tmp_path, policy_data):
        """write() creates a valid JSON file at policy_path."""
        pm = PolicyManager(tmp_path / "policy.json")
        pm.write(policy_data)

        assert pm.policy_path.exists()
        raw = pm.policy_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert parsed["total_runs"] == 1
        assert parsed["best_reward"] == 0.5

    def test_tmp_file_cleaned_up(self, tmp_path, policy_data):
        """After write(), the .json.tmp file must not exist."""
        pm = PolicyManager(tmp_path / "policy.json")
        pm.write(policy_data)

        assert not pm.tmp_path.exists(), (
            f"Temp file {pm.tmp_path} should be cleaned up"
        )

    def test_read_returns_policy_json(self, tmp_path, policy_data):
        """read() returns a PolicyJSON model with correct fields."""
        pm = PolicyManager(tmp_path / "policy.json")
        pm.write(policy_data)

        loaded = pm.read()
        assert loaded is not None
        assert loaded.total_runs == 1
        assert loaded.best_reward == 0.5
        assert loaded.exploration_phase is True
        assert loaded.action_vector["tone_style_idx"] == 2

    def test_read_cold_start(self, tmp_path):
        """read() returns None when no file exists."""
        pm = PolicyManager(tmp_path / "nonexistent.json")
        assert pm.read() is None

    def test_read_corrupt_json(self, tmp_path):
        """read() returns None on corrupt JSON."""
        p = tmp_path / "policy.json"
        p.write_text("{{{NOT VALID", encoding="utf-8")
        pm = PolicyManager(p)
        assert pm.read() is None

    def test_write_failure_does_not_raise(self, tmp_path, policy_data, mocker):
        """On all retries failing, write() logs error but does NOT raise."""
        pm = PolicyManager(tmp_path / "policy.json")
        mocker.patch.object(
            pm, "_atomic_write", side_effect=OSError("disk full")
        )

        # Must not raise
        pm.write(policy_data)

        # File should not exist since write failed
        assert not pm.policy_path.exists()

    def test_round_trip_multiple_writes(self, tmp_path, policy_data):
        """Multiple writes overwrite correctly, last one wins."""
        pm = PolicyManager(tmp_path / "policy.json")

        pm.write(policy_data)
        policy_data["total_runs"] = 10
        policy_data["best_reward"] = 0.95
        pm.write(policy_data)

        loaded = pm.read()
        assert loaded.total_runs == 10
        assert loaded.best_reward == 0.95
