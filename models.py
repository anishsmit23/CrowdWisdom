"""Pydantic v2 models for every JSON artefact in the CWT Ads pipeline.

Each model maps 1-to-1 with a stage output or persistence record.
Field-level ``description`` strings double as schema docs for
CrewAI structured-output and OpenAPI consumers.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ------------------------------------------------------------------ #
# 1. Ad Research
# ------------------------------------------------------------------ #
class AdResearch(BaseModel):
    """A single scraped/enriched ad record."""

    ad_id: str = Field(
        ...,
        description="Unique identifier for the ad (e.g. Apify actor run ID).",
    )
    advertiser: str = Field(
        ...,
        description="Brand or advertiser name extracted from the ad creative.",
    )
    headline: str = Field(
        ...,
        description="Primary headline text of the ad.",
    )
    body_text: str = Field(
        ...,
        description="Full body / description copy of the ad.",
    )
    engagement_score: float = Field(
        ...,
        ge=0,
        description="Normalised engagement metric (likes + shares + comments). Must be >= 0.",
    )


# ------------------------------------------------------------------ #
# 2. Ads Research Output (batch wrapper)
# ------------------------------------------------------------------ #
class AdsResearchOutput(BaseModel):
    """Container returned by the Research agent after scraping."""

    ads: List[AdResearch] = Field(
        ...,
        description="List of ad records collected in this run.",
    )
    run_id: str = Field(
        ...,
        description="Pipeline run identifier (UUID4 hex).",
    )
    scraped_at: str = Field(
        ...,
        description="ISO-8601 timestamp of when the scrape completed.",
    )


# ------------------------------------------------------------------ #
# 3. Marketing Insights
# ------------------------------------------------------------------ #
class MarketingInsights(BaseModel):
    """Distilled insights produced by the Insights agent."""

    pain_points: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly 3 audience pain-points derived from ad research.",
    )
    hook_formulas: List[str] = Field(
        ...,
        description="Proven hook formulas (e.g. 'Problem-Agitate-Solve').",
    )
    cta_types: List[str] = Field(
        ...,
        description="Call-to-action categories ranked by predicted CTR.",
    )
    concept_brief: str = Field(
        ...,
        max_length=800,
        description="Creative brief summarising the ad concept (max 800 chars).",
    )
    run_id: str = Field(
        ...,
        description="Pipeline run identifier.",
    )


# ------------------------------------------------------------------ #
# 4. Script Section
# ------------------------------------------------------------------ #
class ScriptSection(BaseModel):
    """One timed section of the ad script (intro, hook, body, CTA, outro)."""

    section_name: str = Field(
        ...,
        description="Semantic label for this section (e.g. 'hook', 'body').",
    )
    start_s: float = Field(
        ...,
        description="Start time in seconds within the final video.",
    )
    end_s: float = Field(
        ...,
        description="End time in seconds within the final video.",
    )
    narration: str = Field(
        ...,
        description="Voice-over narration text for this section.",
    )
    visual_description: str = Field(
        ...,
        description="Description of the visual treatment / imagery.",
    )

    @field_validator("end_s")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start_s")
        if start is not None and v <= start:
            raise ValueError(
                f"end_s ({v}) must be strictly greater than start_s ({start})"
            )
        return v


# ------------------------------------------------------------------ #
# 5. Ad Script
# ------------------------------------------------------------------ #
class AdScript(BaseModel):
    """Complete ad script output from the Scriptwriter agent."""

    sections: List[ScriptSection] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 timed script sections.",
    )
    full_script: str = Field(
        ...,
        description="Concatenated narration text for TTS input.",
    )
    word_count: int = Field(
        ...,
        ge=130,
        le=165,
        description="Total word count of full_script (must be 130-165).",
    )
    brand_data_points: List[str] = Field(
        ...,
        min_length=2,
        description="At least 2 brand-specific data points woven into the script.",
    )
    run_id: str = Field(
        ...,
        description="Pipeline run identifier.",
    )

    @model_validator(mode="after")
    def _word_count_matches_script(self) -> "AdScript":
        actual = len(self.full_script.split())
        if actual != self.word_count:
            raise ValueError(
                f"word_count ({self.word_count}) does not match "
                f"len(full_script.split()) ({actual})"
            )
        return self


# ------------------------------------------------------------------ #
# 6. RL Run Record (maps to SQLite row in experience_db)
# ------------------------------------------------------------------ #
class RLRunRecord(BaseModel):
    """One row in the RL experience database (SQLite)."""

    run_id: str = Field(..., description="Pipeline run identifier.")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the run.")

    # --- action indices ---
    keyword_set_idx: int = Field(
        ..., description="Index into ACTION_SPACE.keyword_sets."
    )
    llm_model_idx: int = Field(
        ..., description="Index into ACTION_SPACE.llm_models."
    )
    tone_style_idx: int = Field(
        ..., description="Index into tone-style arm list."
    )
    hook_type_idx: int = Field(
        ..., description="Index into hook-type arm list."
    )
    cta_aggression_idx: int = Field(
        ..., description="Index into CTA aggression arm list."
    )
    image_style_idx: int = Field(
        ..., description="Index into image-style arm list."
    )
    voice_id_idx: int = Field(
        ..., description="Index into voice-ID arm list."
    )

    # --- composite reward ---
    reward: float = Field(
        ..., description="Composite reward signal for this run."
    )

    # --- reward sub-components ---
    script_quality: float = Field(
        ..., description="Readability / coherence sub-score."
    )
    visual_coherence: float = Field(
        ..., description="Image-to-script alignment sub-score."
    )
    audio_clarity: float = Field(
        ..., description="TTS audio clarity sub-score."
    )
    production_completeness: float = Field(
        ..., description="Pipeline stage completion ratio (0-1)."
    )
    diversity_bonus: float = Field(
        ..., description="Exploration diversity bonus."
    )

    # --- metadata ---
    human_override: int = Field(
        ...,
        description="1 if a human overrode the reward, 0 otherwise.",
    )
    pipeline_duration_s: float = Field(
        ..., description="Wall-clock pipeline duration in seconds."
    )

    @field_validator("human_override")
    @classmethod
    def _binary_flag(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"human_override must be 0 or 1, got {v}")
        return v


# ------------------------------------------------------------------ #
# 7. Policy JSON (persisted to rl_memory/policy.json)
# ------------------------------------------------------------------ #
class PolicyJSON(BaseModel):
    """Serialised bandit policy snapshot."""

    run_id_generated: str = Field(
        ..., description="Run ID that produced this policy update."
    )
    total_runs: int = Field(
        ..., description="Cumulative number of runs completed."
    )
    best_reward: float = Field(
        ..., description="Highest reward observed so far."
    )
    action_vector: Dict[str, int] = Field(
        ...,
        description="Mapping of dimension names to best-known arm indices.",
    )
    q_values: Dict[str, List[float]] = Field(
        ...,
        description="Q-value estimates per dimension → per arm.",
    )
    exploration_phase: bool = Field(
        ...,
        description="True while the bandit is still in pure exploration.",
    )
    updated_at: str = Field(
        ..., description="ISO-8601 timestamp of last policy update."
    )


# ------------------------------------------------------------------ #
# 8. Human Feedback
# ------------------------------------------------------------------ #
class HumanFeedback(BaseModel):
    """Optional human-in-the-loop feedback injected after a run."""

    run_id: str = Field(..., description="Pipeline run identifier.")
    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Human quality score normalised to [0, 1].",
    )
    notes: str = Field(
        default="",
        description="Free-text reviewer notes (optional).",
    )


# ------------------------------------------------------------------ #
# 9. Final Output manifest
# ------------------------------------------------------------------ #
class FinalOutput(BaseModel):
    """Manifest of all artefacts produced by a single pipeline run."""

    run_id: str = Field(..., description="Pipeline run identifier.")
    video_path: str = Field(
        ..., description="Relative path to the rendered video file."
    )
    audio_path: str = Field(
        ..., description="Relative path to the TTS audio file."
    )
    images: List[str] = Field(
        ..., description="List of relative paths to generated image assets."
    )
    script_path: str = Field(
        ..., description="Relative path to the serialised AdScript JSON."
    )
    duration_s: float = Field(
        ..., description="Total video duration in seconds."
    )
    file_size_mb: float = Field(
        ..., description="Final video file size in megabytes."
    )
