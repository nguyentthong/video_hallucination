#!/usr/bin/env python3
"""
eval_tier2_flash_aggregator_planner.py
---------------------------------------
Variant of eval_tier2_flash_aggregator.py that adds a per-video sampling
planner. A Gemini Flash call looks at a handful of preview frames and
recommends `chunk_seconds` and `frames_per_chunk` BEFORE Stage A1 runs,
instead of using fixed 45s / 12 frames for every video.

Motivation: a uniform 45s x 12-frame schedule = ~0.27 fps. That under-
samples fast-action content (throws, ring/tire tosses, balloon pops) and
over-samples slow content. Let a planner pick per-video.

Pipeline delta vs. eval_tier2_flash_aggregator.py:
  - NEW Step 0: sampling planner (Gemini Flash, ~10 preview frames).
  - Stage A1 uses the PLANNED chunk_seconds + frames_per_chunk.
  - A cached chunks.json whose params don't match the plan is re-chunked.
  - Plan is cached at {cache_dir}/{stem}/plan.json. Delete to replan.
  - Downstream Stage A2 (Flash aggregator) + Stage B (Flash video QA)
    are unchanged and share helpers from eval_tier2_flash_aggregator.

Run:
    python eval_tier2_flash_aggregator_planner.py \\
        --questions_dir benchmark_small \\
        --video_dir     raw_data \\
        --chunks_cache_dir outputs_tier2_flash_agg_planner/cache/chunks \\
        --cache_dir        outputs_tier2_flash_agg_planner/cache \\
        --output_json      outputs_tier2_flash_agg_planner/results.json \\
        --max_concurrency_questions 4
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from dotenv import load_dotenv

import google.genai as genai
from google.genai import types as genai_types

from .eval_tier2_flash_aggregator import (
    DEFAULT_AGGREGATOR_MODEL,
    DEFAULT_CHUNK_VLM_MODEL,
    DEFAULT_STAGE_B_MODEL,
    _atomic_write_json,
    _atomic_write_text,
    _build_frame_parts,
    _build_gemini_client,
    _build_openrouter_client,
    _chunks_cache_path,
    _extract_chunk_frames_b64,
    _flash_cache_paths,
    _gemini_extract_text,
    _get_video_metadata,
    _load_json,
    _plan_chunks,
    _strip_code_fence,
    _strip_thinking,
    aggregator_flash,
    answers_match,
    extract_yes_no,
    load_samples,
    qhash,
    stage_a1_extract_chunk,
    stage_a1_extract_chunk_flash,
    stage_b_answer,
    stage_b_cache_key,
    stage_b_dispatch,
    upload_video,
)

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc


DEFAULT_PLANNER_MODEL = "gemini-3-flash-preview"

CHUNK_SECONDS_MIN = 15.0
CHUNK_SECONDS_MAX = 90.0
FRAMES_PER_CHUNK_MIN = 8
FRAMES_PER_CHUNK_MAX = 32


PLANNER_PROMPT_TEMPLATE = """You are planning how to chunk a long video
(~{duration_sec:.1f}s total) so a later vision-language model can
describe each chunk. You are shown {n_preview} preview frames sampled
uniformly across the FULL video.

Pick two parameters for the per-chunk sampler:
  - chunk_seconds      — length of each chunk, in seconds.
                         Range: [{cs_min:g}, {cs_max:g}].
  - frames_per_chunk   — frames the VLM sees per chunk.
                         Range: [{fpc_min}, {fpc_max}].

Effective sampling rate = frames_per_chunk / chunk_seconds.

Guidance:
  - Fast action with brief discrete events (throws, ring/tire tosses,
    balloon pops, hits/misses, competitive sports, game shows): aim
    for ~0.5-0.8 fps. Typical pick: chunk_seconds=25, frames_per_chunk=16.
  - Medium pace (cooking, tutorials, dialogue with cuts): aim for
    ~0.25-0.4 fps. Typical pick: chunk_seconds=45, frames_per_chunk=12.
  - Slow / largely static (lectures, long documentary shots): aim for
    ~0.15-0.25 fps. Typical pick: chunk_seconds=60, frames_per_chunk=10.

Return STRICT JSON only. No prose, no markdown fences:
{{
  "content_type": "fast_action" | "medium" | "slow",
  "chunk_seconds": <number>,
  "frames_per_chunk": <int>,
  "rationale": "<one short sentence>"
}}
"""


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def plan_sampling(
    client: genai.Client,
    model: str,
    video_path: str,
    preview_frames: int = 10,
    preview_max_side: int = 512,
    preview_jpeg_quality: int = 80,
    max_retries: int = 3,
) -> Dict[str, Any]:
    meta = _get_video_metadata(video_path)
    duration_sec = float(meta["duration_sec"])

    frames_b64 = _extract_chunk_frames_b64(
        video_path,
        start_sec=0.0,
        end_sec=duration_sec,
        frames_per_chunk=preview_frames,
        max_side=preview_max_side,
        jpeg_quality=preview_jpeg_quality,
    )
    if not frames_b64:
        raise RuntimeError(f"Planner: could not extract preview frames from {video_path}")

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        duration_sec=duration_sec,
        n_preview=len(frames_b64),
        cs_min=CHUNK_SECONDS_MIN,
        cs_max=CHUNK_SECONDS_MAX,
        fpc_min=FRAMES_PER_CHUNK_MIN,
        fpc_max=FRAMES_PER_CHUNK_MAX,
    )

    contents: List[Any] = [prompt]
    for fb64 in frames_b64:
        contents.append(
            genai_types.Part.from_bytes(
                data=base64.b64decode(fb64),
                mime_type="image/jpeg",
            )
        )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=512,
                    temperature=0.0,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        text, finish_reason = _gemini_extract_text(resp)
        payload = _strip_code_fence(text)
        try:
            data = json.loads(payload)
        except Exception as e:
            last_err = RuntimeError(
                f"Planner output is not valid JSON "
                f"(finish_reason={finish_reason}): {e}\n--- raw ---\n{text[:400]}"
            )
            time.sleep(2 ** attempt)
            continue

        try:
            cs = float(data["chunk_seconds"])
            fpc = int(data["frames_per_chunk"])
        except (KeyError, TypeError, ValueError) as e:
            last_err = RuntimeError(f"Planner JSON missing fields: {e} -- {data!r}")
            time.sleep(2 ** attempt)
            continue

        cs = float(_clamp(cs, CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX))
        fpc = int(_clamp(fpc, FRAMES_PER_CHUNK_MIN, FRAMES_PER_CHUNK_MAX))

        return {
            "content_type": str(data.get("content_type", "unknown")),
            "chunk_seconds": cs,
            "frames_per_chunk": fpc,
            "rationale": str(data.get("rationale", "")).strip(),
            "video_duration_sec": duration_sec,
            "planner_model": model,
            "n_preview_frames": len(frames_b64),
        }

    raise RuntimeError(f"Planner failed after {max_retries} retries: {last_err}")


def plan_sampling_qwen(
    client: OpenAI,
    model: str,
    video_path: str,
    preview_frames: int = 10,
    preview_max_side: int = 512,
    preview_jpeg_quality: int = 80,
    max_retries: int = 3,
    max_new_tokens: int = 2048,
) -> Dict[str, Any]:
    """Planner variant that calls Qwen (OpenRouter) on inline preview frames.
    Same return schema as plan_sampling."""
    meta = _get_video_metadata(video_path)
    duration_sec = float(meta["duration_sec"])

    frames_b64 = _extract_chunk_frames_b64(
        video_path,
        start_sec=0.0,
        end_sec=duration_sec,
        frames_per_chunk=preview_frames,
        max_side=preview_max_side,
        jpeg_quality=preview_jpeg_quality,
    )
    if not frames_b64:
        raise RuntimeError(f"Qwen planner: could not extract preview frames from {video_path}")

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        duration_sec=duration_sec,
        n_preview=len(frames_b64),
        cs_min=CHUNK_SECONDS_MIN,
        cs_max=CHUNK_SECONDS_MAX,
        fpc_min=FRAMES_PER_CHUNK_MIN,
        fpc_max=FRAMES_PER_CHUNK_MAX,
    )

    messages = [{
        "role": "user",
        "content": [
            *_build_frame_parts(frames_b64),
            {"type": "text", "text": prompt},
        ],
    }]

    extra_body: Dict[str, Any] = {}
    if "thinking" in model.lower():
        extra_body["provider"] = {"order": ["novita"], "allow_fallbacks": False}

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                timeout=600,
                extra_body=extra_body or None,
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        choices = getattr(resp, "choices", None) or []
        if not choices:
            last_err = RuntimeError("Qwen planner: no choices in response")
            time.sleep(2 ** attempt)
            continue
        msg = choices[0].message
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")) or ""
                for c in content
            )
        text = _strip_thinking(content).strip()
        payload = _strip_code_fence(text)
        try:
            data = json.loads(payload)
        except Exception as e:
            last_err = RuntimeError(
                f"Qwen planner output is not valid JSON: {e}\n--- raw (first 400) ---\n{text[:400]}"
            )
            time.sleep(2 ** attempt)
            continue

        try:
            cs = float(data["chunk_seconds"])
            fpc = int(data["frames_per_chunk"])
        except (KeyError, TypeError, ValueError) as e:
            last_err = RuntimeError(f"Qwen planner JSON missing fields: {e} -- {data!r}")
            time.sleep(2 ** attempt)
            continue

        cs = float(_clamp(cs, CHUNK_SECONDS_MIN, CHUNK_SECONDS_MAX))
        fpc = int(_clamp(fpc, FRAMES_PER_CHUNK_MIN, FRAMES_PER_CHUNK_MAX))

        return {
            "content_type": str(data.get("content_type", "unknown")),
            "chunk_seconds": cs,
            "frames_per_chunk": fpc,
            "rationale": str(data.get("rationale", "")).strip(),
            "video_duration_sec": duration_sec,
            "planner_model": model,
            "n_preview_frames": len(frames_b64),
        }

    raise RuntimeError(f"Qwen planner failed after {max_retries} retries: {last_err}")


def _plan_cache_path(cache_dir: Path, video_name: str) -> Path:
    return cache_dir / Path(video_name).stem / "plan.json"


def _chunks_match_plan(chunks_cache: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    try:
        cs_cached = float(chunks_cache.get("chunk_seconds", -1))
        fpc_cached = int(chunks_cache.get("frames_per_chunk", -1))
    except (TypeError, ValueError):
        return False
    return (
        abs(cs_cached - float(plan["chunk_seconds"])) < 0.5
        and fpc_cached == int(plan["frames_per_chunk"])
    )


def evaluate_sample_planned(
    sample: Dict[str, Any],
    chunks_cache_dir: Path,
    cache_dir: Path,
    gemini_client: genai.Client,
    openrouter_client: Any,
    planner_model: str,
    aggregator_model: str,
    stage_b_model: str,
    chunk_vlm_model: str,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_aggregator: int,
    max_new_tokens_stage_b: int,
    max_new_tokens_chunk: int,
    max_concurrency_questions: int,
    max_concurrency_chunks: int,
    planner_backend: str = "flash",
    state_backend: str = "qwen",
    stage_b_mode: str = "plain",
    max_reextract_chunks: int = 3,
) -> Dict[str, Any]:
    questions: List[str] = sample["questions"]
    answers: List[str] = sample["answers"]
    if len(questions) != len(answers):
        raise ValueError(
            f"{sample['video_name']}: |questions|={len(questions)} "
            f"!= |answers|={len(answers)}"
        )

    video_path = sample["video_path"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing video: {video_path}")

    # Step 0 — planner (cached per video).
    plan_path = _plan_cache_path(cache_dir, sample["video_name"])
    plan = _load_json(plan_path)
    if plan is None:
        print(f"  [planner/{planner_backend}] {planner_model} — deciding chunk params ...")
        if planner_backend == "qwen":
            plan = plan_sampling_qwen(openrouter_client, planner_model, video_path)
        else:
            plan = plan_sampling(gemini_client, planner_model, video_path)
        _atomic_write_json(plan_path, plan)
    print(
        f"  [planner/{planner_backend}] content_type={plan['content_type']} "
        f"chunk_seconds={plan['chunk_seconds']:g} "
        f"frames_per_chunk={plan['frames_per_chunk']}  "
        f"({plan.get('rationale', '')})"
    )

    chunk_seconds = float(plan["chunk_seconds"])
    frames_per_chunk = int(plan["frames_per_chunk"])

    # Stage A1 — reuse chunks.json only if it matches the plan.
    chunks_path = _chunks_cache_path(chunks_cache_dir, sample["video_name"])
    chunks_cache = _load_json(chunks_path)
    if chunks_cache is not None and not _chunks_match_plan(chunks_cache, plan):
        print(
            f"  [stage_a1] existing chunks.json used "
            f"chunk_seconds={chunks_cache.get('chunk_seconds')} "
            f"frames_per_chunk={chunks_cache.get('frames_per_chunk')} "
            f"-- plan differs, re-chunking."
        )
        chunks_cache = None

    if chunks_cache is None:
        meta = _get_video_metadata(video_path)
        ranges = _plan_chunks(meta["duration_sec"], chunk_seconds)
        print(
            f"  [stage_a1/{state_backend}] {chunk_vlm_model} — chunking "
            f"{len(ranges)} x ~{chunk_seconds:g}s "
            f"(video {meta['duration_sec']:.1f}s, "
            f"{frames_per_chunk} frames/chunk)"
        )

        def run_chunk(i: int) -> Dict[str, Any]:
            cs, ce = ranges[i]
            if state_backend == "flash":
                return stage_a1_extract_chunk_flash(
                    gemini_client,
                    chunk_vlm_model,
                    video_path,
                    chunk_idx=i,
                    chunk_start=cs,
                    chunk_end=ce,
                    frames_per_chunk=frames_per_chunk,
                    frame_max_side=frame_max_side,
                    frame_jpeg_quality=frame_jpeg_quality,
                    max_new_tokens=max_new_tokens_chunk,
                )
            return stage_a1_extract_chunk(
                openrouter_client,
                chunk_vlm_model,
                video_path,
                chunk_idx=i,
                chunk_start=cs,
                chunk_end=ce,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens=max_new_tokens_chunk,
            )

        workers = max(1, min(max_concurrency_chunks, len(ranges)))
        results: List[Dict[str, Any] | None] = [None] * len(ranges)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_i = {pool.submit(run_chunk, i): i for i in range(len(ranges))}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                results[i] = fut.result()
                print(
                    f"  [stage_a1/{state_backend}] {i + 1}/{len(ranges)} done "
                    f"({len(results[i]['events'])} events)"
                )
        chunks_cache = {
            "video_name": sample["video_name"],
            "video_duration_sec": meta["duration_sec"],
            "chunk_seconds": chunk_seconds,
            "frames_per_chunk": frames_per_chunk,
            "chunk_vlm_model": chunk_vlm_model,
            "planner": {
                "content_type": plan["content_type"],
                "rationale": plan.get("rationale", ""),
                "planner_model": plan.get("planner_model", planner_model),
            },
            "chunks": results,
        }
        _atomic_write_json(chunks_path, chunks_cache)
        print(f"  [stage_a1/{state_backend}] wrote {chunks_path}")

    # Stage A2 — Flash aggregator (text-only).
    stage_a_path, stage_b_path = _flash_cache_paths(cache_dir, sample["video_name"])
    timeline = stage_a_path.read_text(encoding="utf-8") if stage_a_path.exists() else None
    if timeline is None:
        print(
            f"  [aggregator/flash] {aggregator_model} — merging "
            f"{len(chunks_cache['chunks'])} chunks ..."
        )
        timeline = aggregator_flash(
            gemini_client,
            aggregator_model,
            chunks_cache["chunks"],
            max_new_tokens=max_new_tokens_aggregator,
        )
        _atomic_write_text(stage_a_path, timeline)
        print(f"  [aggregator/flash] wrote {stage_a_path} ({len(timeline)} chars)")

    # Stage B — Flash text-only QA (dispatched by --stage_b_mode).
    stage_b_cache: Dict[str, str] = _load_json(stage_b_path) or {}
    uncached_idx = [
        i for i, q in enumerate(questions)
        if stage_b_cache_key(q, stage_b_mode) not in stage_b_cache
    ]

    if uncached_idx:
        print(
            f"  [stage_b/flash] mode={stage_b_mode} QA for "
            f"{len(uncached_idx)}/{len(questions)} pending questions ..."
        )
        cache_lock = Lock()

        def work(i: int) -> None:
            q = questions[i]
            key = stage_b_cache_key(q, stage_b_mode)
            if key in stage_b_cache:
                return
            t0 = time.time()
            pred = stage_b_dispatch(
                mode=stage_b_mode,
                question=q,
                timeline=timeline,
                chunks_cache=chunks_cache,
                video_path=video_path,
                gemini_client=gemini_client,
                openrouter_client=openrouter_client,
                stage_b_model=stage_b_model,
                chunk_vlm_model=chunk_vlm_model,
                state_backend=state_backend,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens_stage_b=max_new_tokens_stage_b,
                max_new_tokens_chunk=max_new_tokens_chunk,
                max_reextract_chunks=max_reextract_chunks,
            )
            with cache_lock:
                stage_b_cache[key] = pred
                _atomic_write_json(stage_b_path, stage_b_cache)
            print(
                f"  [stage_b/{stage_b_mode}] q{i + 1}/{len(questions)} "
                f"done in {time.time() - t0:.1f}s"
            )

        workers = max(1, min(max_concurrency_questions, len(uncached_idx)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(work, i) for i in uncached_idx]
            for fut in as_completed(futs):
                fut.result()

    predictions = [stage_b_cache[stage_b_cache_key(q, stage_b_mode)] for q in questions]
    per_q = [
        {
            "question": q,
            "ground_truth": gt,
            "prediction": p,
            "pred_yn": extract_yes_no(p),
            "gt_yn": extract_yes_no(gt),
            "correct": answers_match(p, gt),
        }
        for q, gt, p in zip(questions, answers, predictions)
    ]
    correct = sum(1 for r in per_q if r["correct"])
    total = len(per_q)
    return {
        "video_name": sample["video_name"],
        "example_path": sample["example_path"],
        "plan": plan,
        "chunks_path": str(chunks_path),
        "stage_a_path": str(stage_a_path),
        "stage_a_chars": len(timeline or ""),
        "num_chunks": len(chunks_cache["chunks"]),
        "num_questions": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else float("nan"),
        "per_question": per_q,
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument(
        "--chunks_cache_dir",
        required=True,
        help="Directory for {stem}/chunks.json. Use a planner-specific path — "
             "chunks here vary per video based on the plan, so sharing with the "
             "fixed-param tier-2 cache would cause re-chunking.",
    )
    ap.add_argument(
        "--cache_dir",
        required=True,
        help="Cache dir for plan.json, stage_a.txt, stage_b.json.",
    )
    ap.add_argument("--output_json", required=True)

    ap.add_argument("--planner_model", default=None,
                    help="Model slug for the sampling planner. Default depends on "
                         f"--planner_backend: flash → {DEFAULT_PLANNER_MODEL}; "
                         f"qwen → {DEFAULT_CHUNK_VLM_MODEL}.")
    ap.add_argument("--planner_backend", choices=["flash", "qwen"], default="flash",
                    help="Backend for the sampling planner. flash → Gemini; "
                         "qwen → OpenRouter. Default: flash.")
    ap.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL)
    ap.add_argument("--stage_b_model", default=DEFAULT_STAGE_B_MODEL)
    ap.add_argument("--chunk_vlm_model", default=None,
                    help="Model slug for Stage A1. Default depends on --state_backend: "
                         f"qwen → {DEFAULT_CHUNK_VLM_MODEL}; "
                         f"flash → {DEFAULT_AGGREGATOR_MODEL}.")
    ap.add_argument("--state_backend", choices=["qwen", "flash"], default="qwen",
                    help="Backend for Stage A1 state extraction. Default: qwen.")

    ap.add_argument("--stage_b_mode", choices=["plain", "cot", "feedback"],
                    default="plain",
                    help="Stage B answering strategy. See "
                         "eval_tier2_flash_aggregator.py for details.")
    ap.add_argument("--max_reextract_chunks", type=int, default=3,
                    help="Cap on chunks to re-extract per question in feedback mode.")

    ap.add_argument("--max_new_tokens_aggregator", type=int, default=65536)
    ap.add_argument("--max_new_tokens_stage_b", type=int, default=256)
    ap.add_argument("--max_new_tokens_chunk", type=int, default=4096)
    ap.add_argument("--max_concurrency_questions", type=int, default=4)
    ap.add_argument("--max_concurrency_chunks", type=int, default=4)

    ap.add_argument("--frame_max_side", type=int, default=768)
    ap.add_argument("--frame_jpeg_quality", type=int, default=85)

    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--google_api_key_env", default="GOOGLE_API_KEY")
    ap.add_argument("--openrouter_api_key_env", default="OPENROUTER_API_KEY")
    args = ap.parse_args()

    if args.planner_model is None:
        args.planner_model = (
            DEFAULT_CHUNK_VLM_MODEL if args.planner_backend == "qwen"
            else DEFAULT_PLANNER_MODEL
        )
    if args.chunk_vlm_model is None:
        args.chunk_vlm_model = (
            DEFAULT_AGGREGATOR_MODEL if args.state_backend == "flash"
            else DEFAULT_CHUNK_VLM_MODEL
        )

    gemini_client = _build_gemini_client(args.google_api_key_env)
    openrouter_client = _build_openrouter_client(args.openrouter_api_key_env)

    samples = load_samples(args.questions_dir, args.video_dir)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.questions_dir}")
    print(f"Planner ({args.planner_backend}):   {args.planner_model}")
    print(f"Stage A1 ({args.state_backend}):    {args.chunk_vlm_model}")
    print(f"Aggregator:        {args.aggregator_model}")
    print(f"Stage B:           {args.stage_b_model}")
    print(f"Chunks cache dir:  {args.chunks_cache_dir}")
    print(f"Flash-agg + plan:  {args.cache_dir}")

    per_sample: List[Dict[str, Any]] = []
    total_correct = 0
    total_questions = 0
    chunks_root = Path(args.chunks_cache_dir)
    cache_root = Path(args.cache_dir)
    out_path = Path(args.output_json)

    for i, sample in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] {sample['video_name']}")
        result = evaluate_sample_planned(
            sample=sample,
            chunks_cache_dir=chunks_root,
            cache_dir=cache_root,
            gemini_client=gemini_client,
            openrouter_client=openrouter_client,
            planner_model=args.planner_model,
            aggregator_model=args.aggregator_model,
            stage_b_model=args.stage_b_model,
            chunk_vlm_model=args.chunk_vlm_model,
            frame_max_side=args.frame_max_side,
            frame_jpeg_quality=args.frame_jpeg_quality,
            max_new_tokens_aggregator=args.max_new_tokens_aggregator,
            max_new_tokens_stage_b=args.max_new_tokens_stage_b,
            max_new_tokens_chunk=args.max_new_tokens_chunk,
            max_concurrency_questions=args.max_concurrency_questions,
            max_concurrency_chunks=args.max_concurrency_chunks,
            planner_backend=args.planner_backend,
            state_backend=args.state_backend,
            stage_b_mode=args.stage_b_mode,
            max_reextract_chunks=args.max_reextract_chunks,
        )
        per_sample.append(result)
        total_correct += result["num_correct"]
        total_questions += result["num_questions"]
        running = (total_correct / total_questions) if total_questions else float("nan")
        print(
            f"    acc={result['accuracy']:.4f}  running={running:.4f}  "
            f"(n_chunks={result['num_chunks']}, "
            f"chunk_sec={result['plan']['chunk_seconds']:g}, "
            f"fpc={result['plan']['frames_per_chunk']})"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "planner_model": args.planner_model,
                    "aggregator_model": args.aggregator_model,
                    "stage_b_model": args.stage_b_model,
                    "stage_b_mode": args.stage_b_mode,
                    "planner_backend": args.planner_backend,
                    "state_backend": args.state_backend,
                    "mode": "tier2 flash-aggregator + flash-stage-b + per-video sampling planner",
                    "chunks_cache_dir": args.chunks_cache_dir,
                    "aggregate": {
                        "num_samples_done": i,
                        "num_samples_total": len(samples),
                        "num_questions": total_questions,
                        "num_correct": total_correct,
                        "accuracy": running,
                    },
                    "per_sample": per_sample,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    print("=" * 60)
    print(f"Planner:    {args.planner_model}")
    print(f"Aggregator: {args.aggregator_model}")
    print(f"Stage B:    {args.stage_b_model}")
    print(f"Samples:    {len(samples)}")
    print(f"Questions:  {total_questions}")
    print(f"Correct:    {total_correct}")
    if total_questions:
        print(f"Accuracy:   {total_correct / total_questions:.4f}")
    print(f"Results ->  {args.output_json}")


if __name__ == "__main__":
    main()
