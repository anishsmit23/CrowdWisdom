"""UT-02: Scriptwriter word-count validation and retry tests.

Tests:
- 148-word narration passes validate_word_count.
- 200-word narration triggers correction (outside [130, 165]).
- Retry logic attempts reformulation up to 2 times.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from cwt_ads_agent.agents.scriptwriter import validate_word_count, _count_narration_words


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_script_json(word_count: int) -> str:
    """Build a valid-looking AdScript JSON string with N words."""
    words_per_section = word_count // 5
    remainder = word_count % 5
    sections = []
    for i in range(5):
        wc = words_per_section + (1 if i < remainder else 0)
        sections.append({
            "section_name": f"section_{i}",
            "start_s": i * 12.0,
            "end_s": (i + 1) * 12.0,
            "narration": " ".join(["word"] * wc),
            "visual_description": f"scene {i}",
        })
    return json.dumps({"sections": sections})


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestWordCountValidator:
    def test_148_words_accepted(self):
        """148 words is within [130, 165] → accepted on first attempt."""
        result = _make_script_json(148)
        agent = MagicMock()
        task = MagicMock()

        validated = validate_word_count(result, agent, task, max_retries=2)
        parsed = json.loads(validated)
        assert _count_narration_words(parsed) == 148

    def test_200_words_triggers_correction(self):
        """200 words is outside [130, 165] → triggers retry."""
        bad_result = _make_script_json(200)
        good_result = _make_script_json(148)

        agent = MagicMock()
        task = MagicMock()

        # Simulate task.execute_sync returning a corrected result
        mock_output = MagicMock()
        mock_output.raw = good_result
        task.execute_sync.return_value = mock_output

        validated = validate_word_count(bad_result, agent, task, max_retries=2)
        parsed = json.loads(validated)
        assert _count_narration_words(parsed) == 148
        # Task was re-executed at least once
        assert task.execute_sync.call_count >= 1

    def test_retry_attempts_up_to_max(self):
        """If every attempt returns bad word count, retries max_retries times."""
        bad_result = _make_script_json(200)
        agent = MagicMock()
        task = MagicMock()

        # Every retry also returns 200 words
        mock_output = MagicMock()
        mock_output.raw = _make_script_json(200)
        task.execute_sync.return_value = mock_output

        _ = validate_word_count(bad_result, agent, task, max_retries=2)

        # Should have called execute_sync exactly 2 times (max_retries)
        assert task.execute_sync.call_count == 2


class TestCountNarrationWords:
    def test_counts_correctly(self):
        parsed = json.loads(_make_script_json(150))
        assert _count_narration_words(parsed) == 150

    def test_empty_sections(self):
        assert _count_narration_words({"sections": []}) == 0

    def test_missing_sections_key(self):
        assert _count_narration_words({}) == 0
