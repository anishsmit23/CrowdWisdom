"""UT-05: Checkpoint reader tests.

Tests:
- Valid JSON checkpoint returns parsed data.
- Missing file returns None.
- Invalid JSON returns None and logs WARNING.
"""

import json
import logging
from pathlib import Path

import pytest

from cwt_ads_agent.utils.helpers import check_checkpoint


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def checkpoint_data():
    return {
        "run_id": "run-abc",
        "stage": "insights",
        "output": {"pain_points": ["a", "b", "c"]},
    }


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestCheckCheckpoint:
    def test_valid_json_returns_data(self, tmp_path, checkpoint_data):
        """Valid JSON file → parsed dict returned."""
        cp_file = tmp_path / "checkpoint.json"
        cp_file.write_text(json.dumps(checkpoint_data), encoding="utf-8")

        result = check_checkpoint(cp_file)

        assert result is not None
        assert result["run_id"] == "run-abc"
        assert result["stage"] == "insights"
        assert len(result["output"]["pain_points"]) == 3

    def test_missing_file_returns_none(self, tmp_path):
        """FileNotFoundError → returns None."""
        cp_file = tmp_path / "nonexistent.json"

        result = check_checkpoint(cp_file)

        assert result is None

    def test_invalid_json_returns_none_and_warns(self, tmp_path, caplog):
        """Corrupt JSON → returns None and logs WARNING."""
        cp_file = tmp_path / "bad.json"
        cp_file.write_text("{{{NOT VALID JSON", encoding="utf-8")

        # Temporarily enable propagation so caplog captures
        import logging
        logger = logging.getLogger("cwt_ads")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="cwt_ads"):
                result = check_checkpoint(cp_file)
        finally:
            logger.propagate = old_propagate

        assert result is None
        assert any("Invalid checkpoint" in msg for msg in caplog.messages)

    def test_empty_file_returns_none(self, tmp_path):
        """Empty file → JSONDecodeError → None."""
        cp_file = tmp_path / "empty.json"
        cp_file.write_text("", encoding="utf-8")

        result = check_checkpoint(cp_file)

        assert result is None

    def test_mocker_builtins_open(self, mocker, checkpoint_data):
        """Mock Path.read_text using mocker (pytest-mock)."""
        mock_path = mocker.MagicMock(spec=Path)
        mock_path.read_text.return_value = json.dumps(checkpoint_data)
        mock_path.name = "mocked.json"

        result = check_checkpoint(mock_path)

        assert result is not None
        assert result["run_id"] == "run-abc"
        mock_path.read_text.assert_called_once_with(encoding="utf-8")

    def test_mocker_file_not_found(self, mocker):
        """Mock FileNotFoundError via mocker."""
        mock_path = mocker.MagicMock(spec=Path)
        mock_path.read_text.side_effect = FileNotFoundError("no such file")
        mock_path.name = "missing.json"

        result = check_checkpoint(mock_path)

        assert result is None

    def test_mocker_invalid_json(self, mocker, caplog):
        """Mock invalid JSON via mocker and verify WARNING log."""
        mock_path = mocker.MagicMock(spec=Path)
        mock_path.read_text.return_value = "NOT JSON"
        mock_path.name = "corrupt.json"

        logger = logging.getLogger("cwt_ads")
        old_propagate = logger.propagate
        logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger="cwt_ads"):
                result = check_checkpoint(mock_path)
        finally:
            logger.propagate = old_propagate

        assert result is None
        assert any("Invalid checkpoint" in msg for msg in caplog.messages)
