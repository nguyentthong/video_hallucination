# Tracking the Truth: Object-Centric Spatio-Temporal Monitoring for Video Large Language Models

A three-stage pipeline for long-form video question answering that localises
and mitigates hallucination in monolithic vision-language models (VLMs).

## How to set up the environment

```bash
uv python install 3.10
uv sync --group openai --group gemini
```

Or use the Makefile shortcut:

```bash
make env
source .venv/bin/activate
```

Create a `.env` file at the project root with your API keys:

```bash
export GOOGLE_API_KEY="..."         # Gemini native API
export OPENROUTER_API_KEY="..."     # OpenRouter (Qwen3 text, identity linking)
export VLLM_API_KEY="EMPTY"         # Local vLLM (any non-empty value)
export VLLM_BASE_URL="http://localhost:8700/v1"
```

For experiments that use a local Qwen3-VL answerer, start a vLLM server
before running:

```bash
vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --host 0.0.0.0 --port 8700 \
  --served-model-name Qwen/Qwen3-VL-32B-Thinking
```

## Our method

The pipeline decomposes video QA into three stages, each backed by a
different model:

1. **Stage A1 — Extractor (VLM):** splits the video into 15 s chunks, runs a
   VLM with a structured-extraction prompt, and outputs per-chunk JSON states.
2. **Stage B — Filter + identity linking (text LLM):** selects
   question-relevant events from the timeline and resolves cross-chunk entity
   aliases.
3. **Stage C — Answerer (VLM):** answers yes/no questions conditioned on the
   filtered timeline and 64 raw video frames, emitting an evidence trace.

See [`docs/methodology.md`](docs/methodology.md) for the full description.

### Baseline (all-Gemini)

```bash
python method/run_pipeline.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend gemini --stage_b_model gemini-3-flash-preview \
  --state_extractor_backend gemini --state_extractor_model gemini-3-flash-preview \
  --states_cache_dir cache/pipeline_baseline \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend gemini --model_id gemini-3-flash-preview \
  --gemini_answerer_thinking_budget 0 --gemini_answerer_max_concurrency 4 \
  --vllm_n_frames 64 --frames_per_chunk 60 \
  --prompt_method filter_v3_gemini \
  --mode all --metrics accuracy \
  --max_new_tokens 4096 \
  --questions_dir benchmark
```

### A1 — Gemini extractor + Qwen filter + Qwen answerer

```bash
python method/run_pipeline.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend gemini --state_extractor_model gemini-3-flash-preview \
  --states_cache_dir cache/pipeline_a1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/Qwen/Qwen3-VL-235B-A22B \
  --vllm_base_url http://localhost:8700/v1 \
  --vllm_api_key_env VLLM_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --prompt_method filter_q3t_v3_q3vl \
  --mode all --metrics accuracy \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### D1 — All open-source (no Gemini)

```bash
python method/run_pipeline.py \
  --state_strategy filter --aggregator_backend concat \
  --stage_b_backend openrouter --stage_b_model qwen/qwen3-235b-a22b \
  --state_extractor_backend vllm --state_extractor_model Qwen/Qwen3-VL-235B-A22B \
  --state_extractor_vllm_base_url http://localhost:8200/v1 \
  --state_extractor_vllm_api_key_env VLLM_API_KEY \
  --states_cache_dir cache/pipeline_b1_d1 \
  --chunk_prompt_version v6 --answerer_prompt_version v3 \
  --enable_identity_link --aggregation_routing \
  --answerer_backend vllm --model_id vllm/Qwen/Qwen3-VL-32B-Thinking \
  --vllm_base_url http://localhost:8700/v1 \
  --vllm_api_key_env VLLM_API_KEY \
  --vllm_n_frames 64 --vllm_max_concurrency 4 \
  --frames_per_chunk 60 \
  --prompt_method filter_q3t_v3_q3vl \
  --mode all --metrics accuracy \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### Vanilla VLM baseline (no pipeline)

Run a single VLM end-to-end without the three-stage decomposition:

```bash
python method/run_baseline.py \
  --model_id gemini-3-flash-preview \
  --metrics all \
  --questions_dir benchmark
```

## Experiment matrix

| ID           | A1 Extractor | B Filter + IdLink | C Answerer | Cache dir                  |
| ------------ | ------------ | ----------------- | ---------- | -------------------------- |
| **Baseline** | `gfl`        | `gfl`             | `gfl`      | `cache/pipeline_baseline/` |
| **A1**       | `gfl`        | `q3t`             | `q3vl`     | `cache/pipeline_a1/`       |
| **B1**       | `q3vl`       | `gfl`             | `q3vl`     | `cache/pipeline_b1_d1/`    |
| **C1**       | `q3vl`       | `q3t`             | `gfl`      | `cache/pipeline_c1/`       |
| **D1**       | `q3vl`       | `q3t`             | `q3vl`     | `cache/pipeline_b1_d1/`    |

Model tags: `gfl` = Gemini 3 Flash, `q3vl` = Qwen3-VL-235B-Thinking,
`q3t` = Qwen3-235B-A22B (text-only), `q35` = Qwen3.5-27B.

## Data

The benchmark lives under `benchmark/`, organised as one JSON file per video
sample (~88 samples). Each file contains:

```json
{
  "video_name": "0016_NtTb-Cw6JVs.mp4",
  "questions": ["Does the man on the right drop the red cup first?", "..."],
  "answers": ["Yes", "..."],
  "sub-questions": [["Is there a man on the right side?", "..."], "..."],
  "sub-answers": [["Yes", "..."], "..."]
}
```

Videos are referenced from a sibling directory and should be placed in
`raw_data/` (gitignored).

## Evaluation

Judge model answers using an LLM judge (Gemini Flash via OpenRouter):

```bash
python evaluation/judge_vanilla.py \
  --cache-dir cache/Qwen_Qwen3-VL-32B-Thinking__vanilla \
  --benchmark-dir benchmark \
  --out-dir cache/Qwen_Qwen3-VL-32B-Thinking__vanilla__judged
```

Re-score cached pipeline answers with the Gemini native API:

```bash
python evaluation/llm_judge_accuracy.py --concurrency 8
```

Inspect accuracy across pipeline configurations:

```bash
python evaluation/inspect_accuracy.py
```

Detect conflicting question-answer pairs in benchmark files:

```bash
python evaluation/inspect_question_answer_conflicts.py --benchmark_dir benchmark
```

## Tests

```bash
make test
```

Or directly:

```bash
python -m pytest tests/ -v
```

## Project structure

```
.
├── method/
│   ├── run_pipeline.py          # Three-stage pipeline (A1 → B → C)
│   └── run_baseline.py          # Vanilla VLM baseline evaluation
├── evaluation/
│   ├── judge_vanilla.py         # LLM-judge for vanilla cache outputs
│   ├── llm_judge_accuracy.py    # LLM-judge re-scorer (Gemini native API)
│   ├── inspect_accuracy.py      # Accuracy summary across pipelines
│   └── inspect_question_answer_conflicts.py
├── stages/                      # Pipeline stage implementations
│   ├── eval_tier2_flash_aggregator.py
│   ├── eval_tier2_flash_aggregator_planner.py
│   └── stage_a_planner.py
├── src/                         # Core library
│   ├── answer_processing.py     # Yes/no extraction from free-form answers
│   ├── eval_module.py           # Evaluation orchestration + scoring
│   ├── load_data.py             # Benchmark data loader
│   ├── cache/                   # Question-level answer cache
│   ├── metrics/                 # Accuracy, consistency (Cons@All/TC/TW)
│   ├── models/                  # Model backends (Gemini, Qwen, Claude, vLLM, …)
│   └── frame_selectors/         # Frame selection strategies (CLIP, AKS, EFS)
├── benchmark/                   # Benchmark data (~88 video samples)
├── tests/                       # Unit tests
├── docs/
│   └── methodology.md           # Full methodology description
├── outputs/                     # Results (gitignored)
├── pyproject.toml
├── Makefile
└── .python-version
```

## Cache layout

Stage outputs are cached under the `--states_cache_dir` directory:

```
<cache_dir>/<video_stem>/
├── chunks.json                          # Stage A1 per-chunk states
├── stage_a_concat.txt                   # Concatenated timeline
├── plan.json                            # Extractor plan
├── aliases.txt                          # Cross-chunk identity table
├── filter[_<model_tag>]/<qid>.json      # Stage B filter outputs
└── answers_<prompt_method>/<qid>.json   # Stage C final answers
```

## Notes

- `--max_new_tokens 16384` is required for thinking-model answerers (e.g.
  `q3vl`). Use `4096` for non-thinking answerers.
- B1 and D1 share `cache/pipeline_b1_d1/` so Stage A1 chunks are extracted
  once. Different `--prompt_method` tags keep Stage C answers separate.
- All scripts should be run from the project root directory.
