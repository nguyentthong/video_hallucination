"""
run_baseline.py — Vanilla VLM baseline evaluation.

Evaluates a single model end-to-end on the sub-question benchmark
(no pipeline decomposition).

Usage (from project root):
    python method/run_baseline.py --model_id <MODEL> --metrics all

Examples:
    python method/run_baseline.py --model_id Qwen/Qwen3-VL-8B-Instruct --metrics all
    python method/run_baseline.py --model_id gemini-3-flash-preview --metrics accuracy

# Use a different prompt method (creates a separate cache namespace)
python benchmark_sub.py --model_id Qwen/Qwen3-VL-8B-Instruct --prompt_method cot --metrics all

# API model (stub — will raise NotImplementedError until implemented)
python benchmark_sub.py --model_id claude-3-7-sonnet-20250219 --metrics accuracy consistency

Available metric keys
---------------------
  all               → every metric below
  accuracy          → target-question accuracy
  sub_accuracy      → sub-question accuracy
  consistency_all   → Cons@All
  consistency_tc    → Cons@TC  (target-correct groups only)
  consistency_tw    → Cons@TW  (target-wrong groups only)
  consistency       → all three Cons metrics at once
"""

import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from tqdm import tqdm

from src.cache import make_cache
from src.eval_module import build_question_groups, evaluate, fill_predictions
from src.load_data import load_benchmark
from src.metrics import build_metrics
from src.models import load_model


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def _aggregate_results(
    per_sample_results: List[Dict[str, float]],
    accuracy_counts: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """
    Average per-sample metric dicts into a single summary dict.

    The target accuracy metric is optionally aggregated as a pooled
    question-level score so videos with more target questions carry the right
    weight. Other metrics remain sample-level macro averages.
    """
    if not per_sample_results:
        return {}
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for r in per_sample_results:
        for k, v in r.items():
            if v == v:  # skip NaN
                totals[k] += v
                counts[k] += 1
    result = {k: totals[k] / counts[k] for k in totals}

    if accuracy_counts and "accuracy" in result:
        total_correct = sum(correct for correct, _ in accuracy_counts)
        total_targets = sum(total for _, total in accuracy_counts)
        if total_targets:
            result["accuracy"] = total_correct / total_targets

    return result


def _print_results(results: Dict[str, float]) -> None:
    max_key_len = max(len(k) for k in results) if results else 0
    print("\n" + "=" * 50)
    print("  Benchmark Results")
    print("=" * 50)
    for key, value in sorted(results.items()):
        if value != value:  # NaN
            print(f"  {key:<{max_key_len}}  =  N/A (no qualifying groups)")
        else:
            print(f"  {key:<{max_key_len}}  =  {value:.4f}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    # ------------------------------------------------------------------
    # 1. Load data, model, cache, metrics
    # ------------------------------------------------------------------
    benchmark_data = load_benchmark(args.video_dir, args.mode, args.questions_dir)

    model = load_model(
        model_id=args.model_id,
        prompt_method=args.prompt_method,
        debug_with_n_frames=args.debug_with_n_frames,
        force_fps=args.force_fps,
    )
    cache = make_cache(model, cache_root=args.cache_dir)
    metrics = build_metrics(args.metrics)

    tqdm.write(f"Model     : {model}")
    tqdm.write(f"Namespace : {model.cache_namespace}")
    tqdm.write(f"Metrics   : {[m.name for m in metrics]}")
    tqdm.write(f"Samples   : {len(benchmark_data)}\n")

    # ------------------------------------------------------------------
    # 2. Run per sample
    # ------------------------------------------------------------------
    per_sample_results: List[Dict[str, float]] = []
    accuracy_counts: List[Tuple[int, int]] = []
    skipped_examples: List[str] = []

    for idx, sample in enumerate(tqdm(benchmark_data, desc="Evaluating")):
        groups = build_question_groups(sample)
        example_path = sample.get("example_path", sample.get("video_path", "<unknown>"))

        try:
            fill_predictions(
                groups=groups,
                sample=sample,
                model=model,
                cache=cache,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            skipped_examples.append(str(example_path))
            tqdm.write(
                f"[{idx + 1:>4}/{len(benchmark_data)}] Skipping sample due to model/API error: {example_path}"
            )
            tqdm.write(f"Error: {exc}")
            continue

        sample_result = evaluate(groups, metrics)
        per_sample_results.append(sample_result)
        if "accuracy" in sample_result:
            accuracy_counts.append(
                (sum(1 for group in groups if group.is_target_correct()), len(groups))
            )

        tqdm.write(
            f"[{idx + 1:>4}/{len(benchmark_data)}] "
            + "  ".join(f"{k}={v:.3f}" for k, v in sample_result.items() if v == v)
        )

    if skipped_examples:
        tqdm.write(f"Skipped {len(skipped_examples)} samples due to model/API errors.")

    # ------------------------------------------------------------------
    # 3. Aggregate and report
    # ------------------------------------------------------------------
    final = _aggregate_results(per_sample_results, accuracy_counts=accuracy_counts)
    _print_results(final)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": args.model_id,
                    "prompt_method": args.prompt_method,
                    "force_fps": args.force_fps,
                    "cache_namespace": model.cache_namespace,
                    "metrics": final,
                    "per_sample": per_sample_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        tqdm.write(f"Results saved to {args.output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()

    parser = ArgumentParser(description="Run model against the sub-question benchmark")

    # Model
    parser.add_argument(
        "--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace repo ID or API model name (e.g. claude-3-7-sonnet-20250219)"
    )
    parser.add_argument(
        "--prompt_method", type=str, default="vanilla",
        help=(
            "Short label for the prompt template.  Changing this creates a "
            "separate cache namespace so old results are not reused.  "
            "Default: 'vanilla'"
        ),
    )

    # Metrics  ← key new flag
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        metavar="METRIC",
        help=(
            "Which metrics to compute.  Pass 'all' for everything, or a "
            "space-separated list from: accuracy sub_accuracy consistency_all "
            "consistency_tc consistency_tw consistency.  "
            "Default: all"
        ),
    )

    # Paths
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--video_dir", type=str, default="raw_data")
    parser.add_argument("--questions_dir", type=str, default="benchmark")
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write full results as JSON"
    )

    # Benchmark mode
    parser.add_argument("--mode", type=str, default="all")

    # Model tuning
    parser.add_argument("--debug_with_n_frames", type=int, default=None)
    parser.add_argument(
        "--force_fps",
        type=float,
        default=None,
        help=(
            "Force a specific FPS for the local Qwen3-VL video path. "
            "Useful for Qwen3-compatible checkpoints like Cosmos-Reason2 "
            "that expect FPS=4."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()
    main(args)
