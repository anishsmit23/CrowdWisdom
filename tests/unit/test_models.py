"""UT-01: Pydantic model validation tests.

Tests:
- AdResearch accepts valid data.
- AdResearch rejects missing ad_id.
- AdScript rejects word_count below 130.
- ScriptSection rejects start_s >= end_s.
"""

import pytest
from pydantic import ValidationError

from cwt_ads_agent.models import AdResearch, AdScript, ScriptSection


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def valid_ad_research_data():
    return {
        "ad_id": "ad-001",
        "advertiser": "TradingCo",
        "headline": "Start Trading Today",
        "body_text": "Join thousands of successful traders.",
        "engagement_score": 42.5,
    }


@pytest.fixture
def valid_section():
    return {
        "section_name": "hook",
        "start_s": 0.0,
        "end_s": 12.0,
        "narration": " ".join(["word"] * 30),
        "visual_description": "Trader at desk",
    }


def _make_sections(word_counts=(30, 30, 30, 30, 28)):
    """Create 5 valid ScriptSection dicts with given narration lengths."""
    starts = [0.0, 12.0, 24.0, 36.0, 48.0]
    ends = [12.0, 24.0, 36.0, 48.0, 60.0]
    names = ["hook", "problem", "solution", "proof", "cta"]
    return [
        {
            "section_name": names[i],
            "start_s": starts[i],
            "end_s": ends[i],
            "narration": " ".join(["word"] * wc),
            "visual_description": f"Scene {i+1}",
        }
        for i, wc in enumerate(word_counts)
    ]


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestAdResearch:
    def test_valid_data(self, valid_ad_research_data):
        ad = AdResearch(**valid_ad_research_data)
        assert ad.ad_id == "ad-001"
        assert ad.advertiser == "TradingCo"
        assert ad.engagement_score == 42.5

    def test_missing_ad_id_raises(self, valid_ad_research_data):
        del valid_ad_research_data["ad_id"]
        with pytest.raises(ValidationError, match="ad_id"):
            AdResearch(**valid_ad_research_data)

    def test_negative_engagement_raises(self, valid_ad_research_data):
        valid_ad_research_data["engagement_score"] = -1.0
        with pytest.raises(ValidationError, match="engagement_score"):
            AdResearch(**valid_ad_research_data)


class TestAdScript:
    def test_word_count_129_raises(self):
        """word_count=129 is below ge=130 → ValidationError."""
        sections = _make_sections(word_counts=(26, 26, 26, 26, 25))
        full_script = " ".join(
            s["narration"] for s in sections
        )
        with pytest.raises(ValidationError, match="word_count"):
            AdScript(
                sections=sections,
                full_script=full_script,
                word_count=129,
                brand_data_points=["stat1", "stat2"],
                run_id="run-test",
            )

    def test_valid_148_words(self):
        """148-word script with matching word_count passes."""
        sections = _make_sections(word_counts=(30, 30, 30, 30, 28))
        full_script = " ".join(s["narration"] for s in sections)
        ad = AdScript(
            sections=sections,
            full_script=full_script,
            word_count=148,
            brand_data_points=["stat1", "stat2"],
            run_id="run-test",
        )
        assert ad.word_count == 148

    def test_word_count_mismatch_raises(self):
        """word_count != len(full_script.split()) → ValidationError."""
        sections = _make_sections(word_counts=(30, 30, 30, 30, 28))
        full_script = " ".join(s["narration"] for s in sections)
        with pytest.raises(ValidationError, match="does not match"):
            AdScript(
                sections=sections,
                full_script=full_script,
                word_count=140,  # actual is 148
                brand_data_points=["stat1", "stat2"],
                run_id="run-test",
            )


class TestScriptSection:
    def test_start_equals_end_raises(self, valid_section):
        valid_section["end_s"] = valid_section["start_s"]
        with pytest.raises(ValidationError, match="end_s.*greater than.*start_s"):
            ScriptSection(**valid_section)

    def test_start_greater_than_end_raises(self, valid_section):
        valid_section["start_s"] = 15.0
        valid_section["end_s"] = 10.0
        with pytest.raises(ValidationError, match="end_s.*greater than.*start_s"):
            ScriptSection(**valid_section)

    def test_valid_section_passes(self, valid_section):
        sec = ScriptSection(**valid_section)
        assert sec.start_s < sec.end_s
