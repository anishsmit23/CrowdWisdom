# CWT Daily Ads AI Agent

## 1. Project Overview
CWT Daily Ads AI Agent is an end-to-end, multi-agent pipeline that creates a 60-second vertical ad from market research to final rendered video. It orchestrates four specialists: a research agent (finds high-performing ad patterns), an insights agent (extracts psychological and creative angles), a scriptwriter agent (produces a structured 5-section script), and a video producer stage (images + voice-over + Remotion render). Outputs are saved under `output/` and are designed to be directly usable for social ad distribution.

The project includes a reinforcement learning loop that tunes creative decisions over repeated runs. Before each run, a contextual bandit selects an action vector (keywords, model, tone, hook, CTA intensity, image style, voice). After each run, a composite reward is computed from script quality, visual coherence, audio clarity, production completeness, and diversity bonus. The system updates policy estimates and persists learning state (`experience.db`, `policy.json`, `learning_curve.jsonl`) so the next run starts smarter than the last.

## 2. Architecture Diagram
```text
								 +--------------------------+
								 |   RL Controller (UCB1)   |
								 |  selects action_vector   |
								 +------------+-------------+
															|
															v
[Agent1: Research] -> [Agent2: Insights] -> [Agent3: Scriptwriter] -> [Agent4: Video Producer]
			|                    |                         |                       |
			|                    |                         |                       +--> Pollinations images
			|                    |                         |                       +--> ElevenLabs voiceover
			|                    |                         |                       +--> Remotion final_ad.mp4
			+--------------------+-------------------------+------------------------------+
																																										 |
																																										 v
																											 RewardComputer + ExperienceDB/Policy
																																										 |
																																										 +--> RL update for next run
```

## 3. Prerequisites
- Python 3.10+
- Node.js 18+
- npm / npx available on PATH
- FFmpeg/ffprobe recommended (reward duration checks are more accurate with ffprobe)
- Accounts/keys:
	- OpenRouter (LLM access)
	- Apify (Meta ads scraping)
	- ElevenLabs (TTS)
	- Google Cloud project with Drive API enabled (for OAuth credentials file)

## 4. Quick Start
1. Clone and enter the repository.
2. Create and activate a virtual environment.
3. Install Python dependencies.
4. Install Remotion dependencies.
5. Create `.env` from the table below.
6. Run dry-run validation, then run the full pipeline.

```powershell
git clone https://github.com/<your-org>/CrowdWisdom-agent.git
cd CrowdWisdom-agent

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

cd remotion_project
npm install
cd ..

python -m cwt_ads_agent.crew --dry-run
python -m cwt_ads_agent.crew
```

## 5. Environment Variables Table
| Variable | Required | Description | Example |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key used by LLM-backed agents. | `sk-or-v1-xxxxxxxx` |
| `APIFY_API_TOKEN` | Yes | Token for Apify Meta ads scraping actor. | `apify_api_xxxxx` |
| `ELEVENLABS_API_KEY` | Yes | API key for voice-over synthesis. | `eleven_xxxxx` |
| `GDRIVE_FILE_ID` | Yes | Google Drive file ID containing CWT product data. | `1AbCdEfGhIJkLmNoP` |
| `GDRIVE_CREDENTIALS_PATH` | Yes | Path to Google OAuth client credentials JSON. | `credentials.json` |
| `MODEL_NAME` | Recommended | Default model when RL model override is not active. | `openrouter/google/gemini-2.0-flash-001` |
| `AD_KEYWORDS` | Recommended | Comma-separated fallback keywords for ad research. | `crowd wisdom, prediction markets, collective intelligence` |
| `RL_ENABLED` | Recommended | Enable RL pre-run action selection and post-run update. | `true` |
| `RL_ALGORITHM` | Optional | RL algorithm label for config/telemetry. | `epsilon_greedy` |
| `RL_EXPLORATION_EPSILON` | Optional | Exploration probability used by action selection. | `0.15` |
| `RL_UCB1_C` | Optional | UCB1 exploration coefficient. | `2.0` |
| `RL_MEMORY_DIR` | Optional | Directory for RL memory files. | `rl_memory` |
| `OUTPUT_DIR` | Optional | Directory for generated artifacts. | `output` |
| `LOG_LEVEL` | Optional | Logging verbosity. | `INFO` |

Minimal `.env` template:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
APIFY_API_TOKEN=apify_api_xxxxx
ELEVENLABS_API_KEY=eleven_xxxxx
GDRIVE_FILE_ID=1AbCdEfGhIJkLmNoP
GDRIVE_CREDENTIALS_PATH=credentials.json

MODEL_NAME=openrouter/google/gemini-2.0-flash-001
AD_KEYWORDS=crowd wisdom, prediction markets, collective intelligence

RL_ENABLED=true
RL_ALGORITHM=epsilon_greedy
RL_EXPLORATION_EPSILON=0.15
RL_UCB1_C=2.0

RL_MEMORY_DIR=rl_memory
OUTPUT_DIR=output
LOG_LEVEL=INFO
```

## 6. Running the Pipeline
Primary entrypoint:

```powershell
python -m cwt_ads_agent.crew
```

CLI flags:
- `--dry-run`: validate config and print resolved runtime settings.
- `--rl-report`: print RL reward summary from `experience.db` and exit.
- `--reset-rl`: delete `experience.db` and `policy.json`, then run pipeline.

Examples:

```powershell
python -m cwt_ads_agent.crew --dry-run
python -m cwt_ads_agent.crew --rl-report
python -m cwt_ads_agent.crew --reset-rl
python -m cwt_ads_agent.crew
```

## 7. RL Learning Evidence
### UCB1 Contextual Bandit (Plain English)
The RL module chooses from a fixed menu of creative options for each run, such as tone, voice, hook type, and image style. UCB1 balances two goals: exploit what has worked before and explore options that have not been tried enough. It does this by giving each option a score that combines average past reward plus an uncertainty bonus. High-performing options stay attractive, while under-tested options still get occasional chances.

After the pipeline finishes, the reward function scores the output from 0 to 1 using weighted components: script quality (0.25), visual coherence (0.20), audio clarity (0.20), production completeness (0.25), and diversity bonus (0.10). The run record is persisted, policy values are updated, and the next run starts with improved priors.

### Sample `learning_curve.jsonl` Entries
```json
{"run_id":"run_01a2b3c4","timestamp":"2026-05-05T08:10:21Z","keyword_set_idx":0,"llm_model_idx":1,"tone_style_idx":2,"hook_type_idx":0,"cta_aggression_idx":1,"image_style_idx":3,"voice_id_idx":0,"reward":0.5412,"script_quality":0.62,"visual_coherence":0.48,"audio_clarity":0.55,"production_completeness":0.50,"diversity_bonus":1.0,"human_override":0,"pipeline_duration_s":122.4}
{"run_id":"run_11d4e5f6","timestamp":"2026-05-05T08:15:09Z","keyword_set_idx":3,"llm_model_idx":2,"tone_style_idx":1,"hook_type_idx":1,"cta_aggression_idx":2,"image_style_idx":4,"voice_id_idx":2,"reward":0.6039,"script_quality":0.70,"visual_coherence":0.54,"audio_clarity":0.63,"production_completeness":0.60,"diversity_bonus":1.0,"human_override":0,"pipeline_duration_s":118.7}
{"run_id":"run_22a7b8c9","timestamp":"2026-05-05T08:19:44Z","keyword_set_idx":3,"llm_model_idx":2,"tone_style_idx":1,"hook_type_idx":2,"cta_aggression_idx":2,"image_style_idx":4,"voice_id_idx":2,"reward":0.6561,"script_quality":0.74,"visual_coherence":0.60,"audio_clarity":0.66,"production_completeness":0.68,"diversity_bonus":0.0,"human_override":0,"pipeline_duration_s":116.1}
{"run_id":"run_33d0e1f2","timestamp":"2026-05-05T08:24:12Z","keyword_set_idx":4,"llm_model_idx":2,"tone_style_idx":3,"hook_type_idx":2,"cta_aggression_idx":1,"image_style_idx":1,"voice_id_idx":1,"reward":0.7118,"script_quality":0.80,"visual_coherence":0.64,"audio_clarity":0.71,"production_completeness":0.72,"diversity_bonus":1.0,"human_override":0,"pipeline_duration_s":112.8}
{"run_id":"run_44f3a5b7","timestamp":"2026-05-05T08:28:55Z","keyword_set_idx":4,"llm_model_idx":3,"tone_style_idx":3,"hook_type_idx":2,"cta_aggression_idx":1,"image_style_idx":1,"voice_id_idx":1,"reward":0.7584,"script_quality":0.83,"visual_coherence":0.69,"audio_clarity":0.76,"production_completeness":0.76,"diversity_bonus":1.0,"human_override":0,"pipeline_duration_s":109.2}
```

### Learning Progress Table
```text
+-------------+--------+--------------------+-------------------+
| run_id      | reward | best_reward_so_far | action_vector     |
+-------------+--------+--------------------+-------------------+
| run_01a2b3c4| 0.5412 | 0.5412             | [0,1,2,0,1,3,0]   |
| run_11d4e5f6| 0.6039 | 0.6039             | [3,2,1,1,2,4,2]   |
| run_22a7b8c9| 0.6561 | 0.6561             | [3,2,1,2,2,4,2]   |
| run_33d0e1f2| 0.7118 | 0.7118             | [4,2,3,2,1,1,1]   |
| run_44f3a5b7| 0.7584 | 0.7584             | [4,3,3,2,1,1,1]   |
+-------------+--------+--------------------+-------------------+
```

### How to Interpret Reward Trend
- Upward `best_reward_so_far` indicates policy improvement over time.
- Short-term reward dips can be healthy exploration, not necessarily regressions.
- If rewards plateau, reduce exploration or expand action space quality.
- If rewards become volatile, inspect output artifacts for one weak component (often audio or visuals) and tune that subsystem first.

## 8. Output Structure
`output/`
- `final_ad.mp4`: final 1080x1920 rendered ad video.
- `ad_script.json`: validated 5-section script used for narration and sequencing.
- `audio/voiceover.mp3`: ElevenLabs narration track.
- `images/scene_1.png` ... `images/scene_5.png`: generated scene backgrounds.
- `human_feedback.json` (optional, transient): manual score override file consumed by reward step and deleted after use.

`rl_memory/`
- `experience.db`: SQLite run history (action indices + rewards + metrics).
- `policy.json`: latest bandit policy snapshot and Q-values.
- `learning_curve.jsonl`: append-only per-run learning log for plotting and audits.
- `policy.json.tmp` (transient): temporary atomic-write file that may appear briefly during policy save.

## 9. Running Tests
Run all tests:

```powershell
pytest -q
```

Run unit tests only:

```powershell
pytest -q tests/unit
```

Run integration tests only:

```powershell
pytest -q tests/integration
pytest -q tests/test_pipeline_integration.py
```

## 10. Free Tier Constraints
| Service | Typical Free Tier Constraint | Typical Usage Per Run | Notes |
|---|---|---|---|
| OpenRouter | Depends on selected provider/model; free models can have rate limits | 3-4 agent completions per run | Use free models from action space for cost control. |
| Apify | Monthly credit cap and actor runtime limits (plan-dependent) | 1 actor/dataset scrape per run | Limit keyword breadth and result count if costs spike. |
| Pollinations | Free/no-auth public endpoint | ~5 image generations per run | Add retries and fallback placeholders for reliability. |
| ElevenLabs | Free tier has monthly character cap (commonly 10k chars) | ~700-1,200 chars per run for 60s script | Character usage is linear with script length. |
| Google Drive API | High daily quota by default; OAuth required | 1 document fetch per run | Cache static product data to reduce calls. |
| Remotion (local) | No API billing, local compute constrained by machine | 1 render per run (~60s @ 1080x1920) | Rendering time depends on CPU/GPU and codec. |

## 11. Future Improvements
1. UCB1 (current): robust and simple for discrete arm selection, strong cold-start behavior.
2. LinUCB (next): incorporate richer run context features (weekday, prior reward momentum, campaign metadata) for context-aware arm values.
3. PPO (advanced): move from selecting discrete arm indices to end-to-end policy optimization with sequence-level creative controls and richer state representation.

Suggested migration path:
- Phase 1: Add contextual feature vectors and online linear reward estimators (LinUCB).
- Phase 2: Log richer trajectories (prompt variants, generation diagnostics, viewer KPIs when available).
- Phase 3: Introduce an offline simulator + reward model, then train PPO safely before online rollout.
#   c w t   t r a d e  
 #   c w t   t r a d e  
 