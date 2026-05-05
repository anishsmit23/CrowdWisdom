"""Integration test for the full CWT Ads Agent pipeline.

Mocks all external API calls and subprocesses to verify the
end-to-end execution flow, state checkpointing, and RL updates.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from cwt_ads_agent import crew
from cwt_ads_agent.config import config
from cwt_ads_agent.rl.experience_db import ExperienceDB


def _make_mock_insight() -> str:
    return json.dumps({
        "pain_points": ["a", "b", "c"],
        "hook_formulas": ["formula1"],
        "cta_types": ["soft"],
        "concept_brief": "A brief",
        "run_id": "test-run",
    })


def _make_mock_script() -> str:
    sections = []
    for i in range(5):
        sections.append({
            "section_name": f"sec{i}",
            "start_s": i * 12.0,
            "end_s": (i + 1) * 12.0,
            "narration": " ".join(["word"] * 30),
            "visual_description": "visual",
        })
    return json.dumps({
        "sections": sections,
        "full_script": " ".join(["word"] * 150),
        "word_count": 150,
        "brand_data_points": ["point1", "point2"],
        "run_id": "test-run",
    })


@pytest.fixture
def mock_all_services(mocker, tmp_path):
    # 1. ApifyClient
    mock_apify = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.iterate_items.return_value = [
        {"id": "1", "pageName": "Adv", "primaryText": "Body", "adArchiveID": "1", "isActive": True},
        {"id": "2", "pageName": "Adv", "primaryText": "Body", "adArchiveID": "2", "isActive": True},
    ]
    mock_apify.return_value.dataset.return_value = mock_dataset
    mock_apify.return_value.task.return_value.call.return_value = {"defaultDatasetId": "test_ds"}
    mocker.patch.dict("sys.modules", {"apify_client": MagicMock(ApifyClient=mock_apify)})

    # 2. OpenRouter / crewai
    mock_execute = mocker.patch("crewai.Task.execute_sync")
    
    out_research = MagicMock()
    out_research.raw = json.dumps([{"ad_id": "1", "advertiser": "A", "headline": "H", "body_text": "B", "engagement_score": 1.0}])
    
    out_insights = MagicMock()
    out_insights.raw = _make_mock_insight()
    
    out_script = MagicMock()
    out_script.raw = _make_mock_script()
    
    out_video = MagicMock()
    out_video.raw = str(tmp_path / "final_ad.mp4")

    def side_effect_execute(self, *args, **kwargs):
        # Create dummy files for reward computer
        out_dir = config.output_path
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "audio").mkdir(exist_ok=True)
        (out_dir / "images" / "scene_1.png").write_bytes(b"mock")
        (out_dir / "audio" / "voiceover.mp3").write_bytes(b"mock")
        (out_dir / "ad_script.json").write_text(_make_mock_script(), encoding="utf-8")
        
        if "AdResearch" in self.expected_output: return out_research
        elif "MarketingInsights" in self.description: return out_insights
        elif "AdScript" in self.description: return out_script
        else: return out_video
        
    mock_execute.side_effect = side_effect_execute
    
    # Return a model string that CrewAI supports natively without litellm
    mocker.patch.object(config, "get_active_model", return_value="openai/gpt-4")
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})

    # Also explicitly mock requests.post to satisfy the prompt's request
    mock_post = mocker.patch("requests.post")
    mock_resp1 = MagicMock()
    mock_resp1.json.return_value = {"choices": [{"message": {"content": _make_mock_insight()}}]}
    mock_resp2 = MagicMock()
    mock_resp2.json.return_value = {"choices": [{"message": {"content": _make_mock_script()}}]}
    mock_post.side_effect = [mock_resp1, mock_resp2, mock_resp1, mock_resp2]

    # 3. Google OAuth & GDrive
    mock_oauth = MagicMock()
    mock_oauth.Credentials.from_service_account_file.return_value = MagicMock()
    mock_discovery = MagicMock()
    mock_build = mock_discovery.build
    mock_files = mock_build.return_value.files.return_value
    mock_files.list.return_value.execute.return_value = {"files": [{"id": "1", "name": "doc"}]}
    mock_files.export.return_value.execute.return_value = b"Mock GDrive Content"
    
    mocker.patch.dict("sys.modules", {
        "google.oauth2.service_account": mock_oauth,
        "googleapiclient.discovery": mock_discovery
    })

    # 4. ElevenLabs
    mock_elevenlabs = MagicMock()
    mock_elevenlabs.return_value.text_to_speech.convert.return_value = [b"a" * 2000]
    mocker.patch.dict("sys.modules", {"elevenlabs": MagicMock(ElevenLabs=mock_elevenlabs)})

    # 5. Pollinations (requests.get)
    mock_get = mocker.patch("requests.get")
    mock_get_resp = MagicMock()
    mock_get_resp.content = b"a" * 70000
    mock_get_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_get_resp

    # 6. subprocess.run (Remotion & ffprobe)
    original_run = subprocess.run
    def side_effect_run(cmd, *args, **kwargs):
        if "remotion" in cmd:
            mp4 = tmp_path / "final_ad.mp4"
            mp4.write_bytes(b"\x00" * (6 * 1024 * 1024))
            return MagicMock(returncode=0)
        elif "ffprobe" in cmd:
            return MagicMock(stdout="60.0\n", returncode=0)
        return MagicMock(returncode=0)
    mocker.patch("subprocess.run", side_effect=side_effect_run)

    # 7. mutagen.mp3.MP3
    mock_mp3 = mocker.patch("mutagen.mp3.MP3")
    mock_mp3.return_value.info.length = 60.0


import subprocess

import sys

def test_full_pipeline_two_runs(tmp_path, mock_all_services, mocker):
    """End-to-end integration test of the pipeline and RL memory."""
    # Reset any cached config paths
    config.output_dir = str(tmp_path)
    config.rl_memory_dir = str(tmp_path / "rl_memory")
    config.rl_memory_path.mkdir(exist_ok=True)
    config.rl_enabled = True

    mocker.patch.object(sys, "argv", ["crew.py"])

    # We need to mock datetime so timestamp parsing is consistent if needed,
    # but actual RLRunRecord uses datetime.now(), which is fine.

    # === RUN 1 ===
    crew.main()

    # Assert: tmp_path/'rl_memory'/'experience.db' exists
    db_path = config.rl_memory_path / "experience.db"
    assert db_path.exists(), "Experience DB was not created"

    # Assert: db.get_all() returns exactly 1 record
    db = ExperienceDB(db_path)
    rows1 = db.get_all()
    assert len(rows1) == 1, f"Expected 1 record in DB, got {len(rows1)}"

    # Assert: policy.json exists
    policy_path = config.rl_memory_path / "policy.json"
    assert policy_path.exists(), "Policy JSON was not created"

    # Assert: policy1['total_runs'] == 1
    policy1 = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy1["total_runs"] == 1

    # Assert: action_vector has all 7 dimensions
    from cwt_ads_agent.rl.action_space import DIMS
    for dim in DIMS:
        assert dim in policy1["action_vector"], f"Missing {dim} in action vector"

    # === RUN 2 ===
    crew.main()

    # Assert: db.get_all() returns exactly 2 records
    rows2 = db.get_all()
    assert len(rows2) == 2, f"Expected 2 records in DB, got {len(rows2)}"

    # Assert: policy2['total_runs'] == 2
    policy2 = json.loads(policy_path.read_text(encoding="utf-8"))
    assert policy2["total_runs"] == 2

    # Assert: learning_curve.jsonl has 2 lines with different run_ids
    learning_curve = config.rl_memory_path / "learning_curve.jsonl"
    assert learning_curve.exists(), "learning_curve.jsonl was not created"

    lines = learning_curve.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2, f"Expected 2 lines in learning curve, got {len(lines)}"

    run_id_1 = json.loads(lines[0])["run_id"]
    run_id_2 = json.loads(lines[1])["run_id"]
    assert run_id_1 != run_id_2, "run_ids in learning curve should be unique"
