"""UT-04: Subtitle timing computation tests.

Tests:
- 5 ScriptSections produce 5 timing entries with start_s < end_s.
- Final entry's end_s is clamped to 60.0.
"""

import pytest

from cwt_ads_agent.models import ScriptSection
from cwt_ads_agent.utils.helpers import compute_subtitle_timings


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def five_sections():
    """Create 5 valid ScriptSection objects covering 0-58s."""
    specs = [
        ("hook", 0.0, 10.0),
        ("problem", 10.0, 22.0),
        ("solution", 22.0, 36.0),
        ("proof", 36.0, 48.0),
        ("cta", 48.0, 58.0),  # intentionally < 60 to test clamping
    ]
    return [
        ScriptSection(
            section_name=name,
            start_s=start,
            end_s=end,
            narration=f"Narration for {name} section",
            visual_description=f"Visual for {name}",
        )
        for name, start, end in specs
    ]


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestComputeSubtitleTimings:
    def test_produces_5_entries(self, five_sections):
        timings = compute_subtitle_timings(five_sections)
        assert len(timings) == 5

    def test_all_start_before_end(self, five_sections):
        timings = compute_subtitle_timings(five_sections)
        for entry in timings:
            assert entry["start_s"] < entry["end_s"], (
                f"start_s={entry['start_s']} >= end_s={entry['end_s']}"
            )

    def test_final_end_equals_60(self, five_sections):
        timings = compute_subtitle_timings(five_sections)
        assert timings[-1]["end_s"] == 60.0

    def test_narration_text_preserved(self, five_sections):
        timings = compute_subtitle_timings(five_sections)
        for i, entry in enumerate(timings):
            assert entry["text"] == five_sections[i].narration

    def test_non_final_entries_keep_original_end(self, five_sections):
        timings = compute_subtitle_timings(five_sections)
        # Entries 0-3 should keep their original end_s
        for i in range(4):
            assert timings[i]["end_s"] == five_sections[i].end_s

    def test_empty_sections(self):
        timings = compute_subtitle_timings([])
        assert timings == []

    def test_single_section_clamped_to_60(self):
        sec = ScriptSection(
            section_name="solo",
            start_s=0.0,
            end_s=45.0,
            narration="Solo narration",
            visual_description="Solo visual",
        )
        timings = compute_subtitle_timings([sec])
        assert len(timings) == 1
        assert timings[0]["end_s"] == 60.0
