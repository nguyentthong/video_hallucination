# STRAND

Code and benchmark data for **"STRAND: Benchmarking and Improving Object-Centric
Spatio-Temporal Monitoring in Video Large Language Models"**.

Project page: https://stemo2026.github.io

This repository holds two things:

1. **STRAND** — a benchmark of human-verified, object-centric facts. Every
   compositional *target* question is paired with the atomic *sub-questions* it
   rests on, so evaluation can ask not only whether a model answered, but
   whether it got the supporting facts right. 88 videos, 977 targets, 2,516
   sub-questions (2.58 per target), all yes/no.
2. **An object-centric framework** that answers from explicitly constructed
   object trajectories rather than from raw frames alone.

## Metrics

STRAND is scored on **Faithful Accuracy (`A_faith`)**, the primary metric. A
target counts only when the model answers it *and* every supporting
sub-question correctly, over **all** targets:

```
A_faith = (1/N) * Σ_i 1[ŷ_i = y_i] * Π_j 1[ŷ_ij = y_ij]
```

The fixed denominator is the point: `A_faith` is monotone in both target and
sub-question correctness, is zero for a model that answers nothing, and is
therefore comparable across models.

| Key                        | Symbol     | Role                                            |
| -------------------------- | ---------- | ----------------------------------------------- |
| `faithful_accuracy`        | `A_faith`  | **Primary.** Target + all sub-questions correct. |
| `accuracy`                 | `A_target` | Target questions only.                           |
| `sub_accuracy`             | `A_sub`    | All sub-questions, flat.                         |
| `conditional_consistency`  | `A_cons`   | Reference only. See the caveat below.            |
| `strand`                   | —          | All four at once, in reporting order.            |

```bash
python method/run_pipeline.py ... --metrics strand
```

> **`A_cons` is reference only.** It averages over the targets a model happens
> to answer correctly, so two models are scored on different subsets. It is
> maximised by low-recall selectivity and does not fall when a model misses a
> target outright. State consistency claims in terms of `A_faith`.

> **Deprecated:** the older `Cons@All` / `Cons@TC` / `Cons@TW` family in
> `src/metrics/consistency.py` averages over the whole question group, *target
> included*. `Cons@TC` is therefore **not** the paper's `A_cons` and reads
> systematically higher. It is retained only for backwards compatibility.

Recompute the headline table from dumped predictions, with the definitional
invariants (`A_faith <= A_target`, `A_faith <= A_target * A_cons`) checked:

```bash
python evaluation/compute_faithful_accuracy.py predictions.jsonl
```

## The framework

Stages 1–5 run **once per video** and are reused by every question about it;
stages 6–7 run **once per question**.

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

1. **Chunk-wise state extraction — `E_φ` (VLM).** Splits the video into
   disjoint 15 s chunks, samples 60 frames from each, and runs each chunk
   *independently* (so extraction parallelises), emitting per-chunk JSON
   states. A state is relational, not merely attributive: it carries the
   predicate together with its arguments.
2. **Temporal aggregation — `A_ψ` (deterministic, symbolic).** Consolidates
   chunk-level observations into global trajectories. **This issues no model
   call**; `ψ = (Δt_max, τ_conf)` are fixed hyperparameters, not learned
   weights. Four steps: similarity scoring over visual attributes and spatial
   proximity, constraint filtering on temporal proximity and a confidence
   threshold, bipartite conflict resolution prioritised by temporal adjacency,
   and trajectory generation in which an observation matching nothing survives
   as a singleton. (`--aggregator_backend llm_summarize` is an *ablation*, not
   the default.)
3. **Query-based trajectory retrieval — `R_ω` (text LLM).** Selects the
   query-relevant subset of trajectories.
4. **Trajectory-guided answering — `G_η` (VLM).** Answers conditioned on the
   retrieved trajectories, the question, and a fixed uniformly sampled 64-frame
   budget that is independent of the question — so among the video-derived
   inputs only the retrieved subset varies per query.

See [`docs/methodology.md`](docs/methodology.md) for the full description.

The two configurations reported in the paper:

| Configuration           | `E_φ` extractor                 | `R_ω` retrieval        | `G_η` answerer  |
| ----------------------- | ------------------------------- | ---------------------- | --------------- |
| **Ours (Gemini-3-Flash)** | `gemini-3-flash-preview`      | `gemini-3-flash-preview` | `gemini-3-flash-preview` |
| **Ours (Qwen3-VL-235B)**  | `qwen3-vl-235b-a22b-thinking` | `qwen3-235b-a22b`      | `qwen3.5-27b`   |

Unless ablated, both use 15 s chunks, 60 frames per chunk, 64 frames at the
answerer, and identity linking.

### Ours (Gemini-3-Flash)

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
  --mode all --metrics strand \
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
  --mode all --metrics strand \
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
  --mode all --metrics strand \
  --max_new_tokens 16384 \
  --questions_dir benchmark
```

### Vanilla VLM baseline (no pipeline)

Run a single VLM end-to-end without the three-stage decomposition:

```bash
python method/run_baseline.py \
  --model_id gemini-3-flash-preview \
  --metrics strand \
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

The benchmark lives under `benchmark/` (unpack `strand_questions.tar.gz`),
organised as one JSON file per video sample (88 samples). Each file contains:

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
│   ├── compute_faithful_accuracy.py  # A_faith table + invariant checks
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
│   ├── metrics/                 # A_faith (primary), A_target, A_sub, A_cons
│   ├── models/                  # Model backends (Gemini, Qwen, Claude, vLLM, …)
│   └── frame_selectors/         # Frame selection strategies (CLIP, AKS, EFS)
├── benchmark/                   # STRAND data (88 video samples)
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
