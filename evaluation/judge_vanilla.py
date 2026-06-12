# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.40.0",
#     "python-dotenv>=1.2.2",
#     "tqdm>=4.67.3",
# ]
# ///
"""
judge_vanilla.py
----------------
Judge free-form model answers stored in the vanilla cache layout
(``cache/<namespace>/<video_id>.json`` with ``{q_hash: pred_text}``)
against the ground-truth yes/no answers in ``benchmark/sample_<video_id>.json``.

For each (question, gt, pred) triple — both target and sub-questions — we ask
Gemini (via OpenRouter, OpenAI-compatible API) whether the model's free-form
answer agrees with the gt verdict. Results are written one JSON per video
under ``--out-dir`` and an aggregate summary is printed.

Cache key convention (from src/cache/cache_sys.py):
    q_hash = md5(question.strip().lower())[:12]
"""

import argparse
import hashlib
import json
import os
import numpy as np
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


CACHE_DIR = Path("cache/Qwen_Qwen3-VL-32B-Thinking__vanilla")
BENCHMARK_DIR = Path("benchmark")
DEFAULT_OUT_DIR = Path("cache/Qwen_Qwen3-VL-32B-Thinking__vanilla__judged")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_ID = "google/gemini-2.5-flash"
JUDGE_SOURCE_TAG = "llm_judge_openrouter_gemini_flash"

JUDGE_PROMPT_TMPL = """You are an impartial judge deciding whether a model's free-form answer agrees with the ground-truth yes/no answer for a question about a video.

Question: {question}

Ground truth: {gt}

Model answer: {pred}

Decide whether the model's answer conveys the SAME verdict as the ground truth (yes vs. no). Focus on what verdict the model commits to, not on hedging, justification, or extra narration.

Respond with exactly one word: YES if the model answer agrees with the ground truth, NO otherwise. No punctuation, no explanation."""


def question_hash(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:12]


def benchmark_path_for(benchmark_dir: Path, cache_filename: str) -> Path:
    return benchmark_dir / f"sample_{cache_filename}"


def call_judge(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 16,
    retries: int = 3,
) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(max_tokens),
            )
            choices = getattr(resp, "choices", None) or []
            if not choices:
                last_exc = RuntimeError("no choices in response")
                time.sleep(1.5 * (2 ** attempt))
                continue
            content = getattr(choices[0].message, "content", "") or ""
            if isinstance(content, list):
                content = "".join(
                    (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")) or ""
                    for c in content
                )
            return content.strip()
        except Exception as e:
            last_exc = e
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"judge call failed after {retries} attempts: {last_exc}")


def parse_verdict(raw: str) -> Optional[bool]:
    if not raw:
        return None
    first = raw.strip().split()[0].strip(".,:;!?").upper()
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    return None


def judge_triple(
    client: OpenAI,
    model: str,
    question: str,
    gt: str,
    pred: str,
) -> Tuple[Optional[bool], str]:
    prompt = JUDGE_PROMPT_TMPL.format(
        question=question.strip(),
        gt=gt.strip(),
        pred=pred.strip(),
    )
    raw = call_judge(client, model, prompt)
    return parse_verdict(raw), raw


def collect_items(
    cache_data: Dict[str, str],
    benchmark: Dict,
    include_subquestions: bool,
) -> List[Dict]:
    """Pair each benchmark question (and optionally sub-question) with the
    cached prediction (if any)."""
    items: List[Dict] = []
    questions = benchmark["questions"]
    answers = benchmark["answers"]
    sub_qs = benchmark.get("sub-questions", [[] for _ in questions])
    sub_as = benchmark.get("sub-answers", [[] for _ in questions])

    for idx, (q, a, sqs, sas) in enumerate(zip(questions, answers, sub_qs, sub_as)):
        items.append({
            "group_id": idx,
            "kind": "target",
            "question": q,
            "gt": a,
            "pred": cache_data.get(question_hash(q)),
        })
        if include_subquestions:
            for sub_idx, (sq, sa) in enumerate(zip(sqs, sas)):
                items.append({
                    "group_id": idx,
                    "kind": "sub",
                    "sub_index": sub_idx,
                    "question": sq,
                    "gt": sa,
                    "pred": cache_data.get(question_hash(sq)),
                })
    return items


def judge_one_file(
    client: OpenAI,
    model: str,
    cache_path: Path,
    benchmark_dir: Path,
    out_path: Path,
    include_subquestions: bool,
    force: bool,
) -> Tuple[str, Optional[Dict]]:
    """Returns (status, summary). status in {"ok", "skip", "error"}."""
    bench_path = benchmark_path_for(benchmark_dir, cache_path.name)
    if not bench_path.exists():
        return ("error", {"detail": f"missing benchmark: {bench_path}"})

    if out_path.exists() and not force:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            if prior.get("source") == JUDGE_SOURCE_TAG:
                return ("skip", prior.get("summary"))
        except Exception:
            pass  # fall through and re-judge

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        with open(bench_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    except Exception as e:
        return ("error", {"detail": f"read-error:{cache_path}:{e}"})

    items = collect_items(cache_data, benchmark, include_subquestions)
    judged = []
    for item in items:
        pred = item["pred"]
        if pred is None:
            judged.append({**item, "is_correct": None, "judge_raw": None,
                           "note": "no cached prediction"})
            continue
        try:
            verdict, raw = judge_triple(client, model, item["question"], item["gt"], pred)
        except Exception as e:
            judged.append({**item, "is_correct": None, "judge_raw": None,
                           "note": f"api-error:{e}"})
            continue
        judged.append({**item, "is_correct": verdict, "judge_raw": raw})

    target_items = [r for r in judged if r["kind"] == "target"]
    sub_items = [r for r in judged if r["kind"] == "sub"]

    def acc(rs):
        if not rs:
            return None
        # missing pred or unparseable verdict counts as wrong
        return sum(1 for r in rs if r["is_correct"] is True) / len(rs)

    summary = {
        "video": cache_path.stem,
        "n_target": len(target_items),
        "n_sub": len(sub_items),
        "n_target_missing_pred": sum(1 for r in target_items if r["pred"] is None),
        "n_sub_missing_pred": sum(1 for r in sub_items if r["pred"] is None),
        "target_accuracy": acc(target_items),
        "sub_accuracy": acc(sub_items),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source": JUDGE_SOURCE_TAG,
        "model": model,
        "cache_file": str(cache_path),
        "benchmark_file": str(bench_path),
        "include_subquestions": include_subquestions,
        "summary": summary,
        "items": judged,
    }
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)

    return ("ok", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=(
            "OpenRouter model slug for the judge "
            "(e.g. 'google/gemini-2.5-flash', 'google/gemini-2.5-pro')."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Env var holding the OpenRouter API key.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=OPENROUTER_BASE_URL,
        help="OpenAI-compatible base URL (default: OpenRouter).",
    )
    parser.add_argument(
        "--no-subquestions",
        action="store_true",
        help="Judge target questions only, skip sub-questions.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge files that already have output written.",
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional cap on number of cache files (smoke test).")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        sys.exit(f"{args.api_key_env} is not set (check your .env or shell env).")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    cache_files = sorted(p for p in args.cache_dir.iterdir() if p.suffix == ".json")
    if args.limit:
        cache_files = cache_files[: args.limit]

    if not cache_files:
        print(f"No cache JSON files in {args.cache_dir}", flush=True)
        return

    include_sub = not args.no_subquestions
    print(
        f"Judging {len(cache_files)} cache file(s) via {args.base_url} "
        f"model={args.model} "
        f"(concurrency={args.concurrency}, include_subquestions={include_sub}, "
        f"force={args.force})",
        flush=True,
    )

    summaries: List[Dict] = []
    ok_n = skip_n = err_n = 0
    errors: List[str] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for cache_path in cache_files:
            out_path = args.out_dir / cache_path.name
            futures[pool.submit(
                judge_one_file, client, args.model, cache_path,
                args.benchmark_dir, out_path, include_sub, args.force
            )] = cache_path

        pbar = tqdm(as_completed(futures), total=len(futures))
        for fut in pbar:
            cache_path = futures[fut]
            status, payload = fut.result()
            if status == "ok":
                ok_n += 1
                if isinstance(payload, dict):
                    summaries.append(payload)
            elif status == "skip":
                skip_n += 1
                if isinstance(payload, dict):
                    summaries.append(payload)
            else:
                err_n += 1
                detail = (payload or {}).get("detail", "unknown")
                errors.append(f"{cache_path.name}: {detail}")
            pbar.set_postfix(ok=ok_n, skip=skip_n, err=err_n)

    print(f"\nDone. judged={ok_n} skipped={skip_n} errors={err_n}", flush=True)
    if errors:
        print(f"First {min(len(errors), 20)} error(s):")
        for e in errors[:20]:
            print("  " + e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    # Aggregate
    def safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    macro_target = safe_mean(s["target_accuracy"] for s in summaries)
    macro_sub = safe_mean(s["sub_accuracy"] for s in summaries)

    target_correct = sum(round((s["target_accuracy"] or 0) * s["n_target"])
                         for s in summaries)
    target_total = sum(s["n_target"] for s in summaries)
    sub_correct = sum(round((s["sub_accuracy"] or 0) * s["n_sub"])
                      for s in summaries)
    sub_total = sum(s["n_sub"] for s in summaries)

    print("=" * 60)
    print(f"Videos judged          : {len(summaries)}")
    print(f"Macro target accuracy  : "
          f"{macro_target:.4f}" if macro_target is not None else "Macro target accuracy  : N/A")
    if include_sub:
        print(f"Macro sub accuracy     : "
              f"{macro_sub:.4f}" if macro_sub is not None else "Macro sub accuracy     : N/A")
    print(f"Micro target accuracy  : "
          f"{target_correct}/{target_total} = "
          f"{(target_correct / target_total):.4f}" if target_total else "Micro target accuracy  : N/A")
    if include_sub:
        print(f"Micro sub accuracy     : "
              f"{sub_correct}/{sub_total} = "
              f"{(sub_correct / sub_total):.4f}" if sub_total else "Micro sub accuracy     : N/A")
    print("=" * 60)

    summary_path = args.out_dir / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "base_url": args.base_url,
                "cache_dir": str(args.cache_dir),
                "benchmark_dir": str(args.benchmark_dir),
                "include_subquestions": include_sub,
                "n_videos": len(summaries),
                "macro_target_accuracy": macro_target,
                "macro_sub_accuracy": macro_sub,
                "micro_target_accuracy": (
                    target_correct / target_total if target_total else None
                ),
                "micro_sub_accuracy": (
                    sub_correct / sub_total if sub_total else None
                ),
                "per_video": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
