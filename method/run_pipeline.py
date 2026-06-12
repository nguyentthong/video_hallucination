#!/usr/bin/env python3
"""
run_pipeline.py — Three-stage video QA pipeline (A1 → B → C).

Runs the full hallucination-mitigation pipeline on the sub-question
benchmark:
  Stage A1: Extract per-chunk structured timeline from video.
  Stage B:  Filter relevant events + cross-chunk identity linking.
  Stage C:  Answer yes/no questions with evidence trace.

Usage (from project root):
    python method/run_pipeline.py [OPTIONS]

See README.md for full experiment commands.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import base64
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from stages.eval_tier2_flash_aggregator import (
    AGGREGATOR_PROMPT_TEMPLATE,
    DEFAULT_AGGREGATOR_MODEL,
    DEFAULT_CHUNK_VLM_MODEL,
    _NonRetryableError,
    _atomic_write_json,
    _atomic_write_text,
    _build_gemini_client,
    _build_openrouter_client,
    _chunks_cache_path,
    _flash_cache_paths,
    _gemini_extract_text,
    _get_video_metadata,
    _load_json,
    _parse_json_list_or_raise,
    _permissive_safety_settings,
    _plan_chunks,
    _strip_code_fence,
    _strip_thinking,
    aggregator_concat,
    aggregator_flash,
    annotate_timeline_with_subtype_order,
    format_alias_header,
    identity_link_via_flash,
    identity_link_via_openrouter,
    stage_a1_extract_chunk,
    stage_a1_extract_chunk_flash,
)
from stages.eval_tier2_flash_aggregator_planner import (
    _chunks_match_plan,
    _plan_cache_path,
)
from stages.stage_a_planner import (
    build_biased_timeline_cached,
    sanitize_model_tag as _planner_sanitize_tag,
)

from google.genai import types as genai_types

from src.eval_module import (  # noqa: E402
    SimpleAnswerProcessor,
    build_question_groups,
    evaluate,
)
from src.load_data import load_benchmark  # noqa: E402
from src.metrics import build_metrics  # noqa: E402
from src.models.vllm_openai import VLLMOpenAIModel  # noqa: E402


DEFAULT_MODEL_ID = "vllm/Qwen/Qwen3-VL-32B-Thinking"
DEFAULT_VLLM_BASE_URL = "http://localhost:8700/v1"
_VLLM_PREFIX = "vllm/"

FORCED_CHUNK_SECONDS = 15.0
FORCED_FRAMES_PER_CHUNK = 32  # default; override via --frames_per_chunk
FRAMES_PER_CHUNK_CHOICES = (32, 60)

DEFAULT_STAGE_B_MODEL = "gemini-3-flash-preview"

BACKEND_GEMINI = "gemini"
BACKEND_OPENROUTER = "openrouter"
BACKEND_LITELLM = "litellm"
BACKEND_CONCAT = "concat"
BACKENDS = (BACKEND_GEMINI, BACKEND_OPENROUTER, BACKEND_LITELLM)
AGGREGATOR_BACKENDS = (BACKEND_GEMINI, BACKEND_OPENROUTER, BACKEND_CONCAT)

STRATEGY_NARRATIVE = "narrative"
STRATEGY_FILTER = "filter"
STRATEGY_PLANNER = "planner"
STRATEGIES = (STRATEGY_NARRATIVE, STRATEGY_FILTER, STRATEGY_PLANNER)

PLANNER_BACKENDS = (BACKEND_GEMINI, BACKEND_OPENROUTER)
DEFAULT_PLANNER_MODEL = "gemini-3-flash-preview"
LITELLM_PROXY_URL_ENV = "LITELLM_PROXY_URL"
LITELLM_BASE_URL_ENV = "LITELLM_BASE_URL"
LITELLM_MODEL_PREFIX = "litellm/"


# ---------------------------------------------------------------------------
# Stage A: extract per-video JSON timeline (forced chunking, no planner).
# ---------------------------------------------------------------------------

def _forced_plan(duration_sec: float, frames_per_chunk: int) -> Dict[str, Any]:
    return {
        "content_type": "forced",
        "chunk_seconds": float(FORCED_CHUNK_SECONDS),
        "frames_per_chunk": int(frames_per_chunk),
        "rationale": (
            f"Forced {FORCED_CHUNK_SECONDS:g}s / {frames_per_chunk}-frame "
            "chunking (planner skipped)."
        ),
        "video_duration_sec": float(duration_sec),
        "planner_model": "forced",
        "n_preview_frames": 0,
    }


def _stage_a_path(states_cache_dir: Path, video_name: str, aggregator_backend: str) -> Path:
    base = states_cache_dir / Path(video_name).stem
    if aggregator_backend == BACKEND_OPENROUTER:
        return base / "stage_a_qwen.txt"
    if aggregator_backend == BACKEND_CONCAT:
        return base / "stage_a_concat.txt"
    return _flash_cache_paths(states_cache_dir, video_name)[0]


def _aliases_path(states_cache_dir: Path, video_name: str) -> Path:
    """Cache for the post-Stage-A identity-link header. Plain text so it can
    be read and prepended to per-question Context with no parsing."""
    return states_cache_dir / Path(video_name).stem / "aliases.txt"


def _build_aliases_if_missing(
    *,
    states_cache_dir: Path,
    video_name: str,
    timeline_str: str,
    gemini_client,
    openrouter_client,
    litellm_client,
    aliases_model: str,
    aliases_backend: str,
    max_new_tokens: int,
) -> str:
    """Build (or load) the alias header for one video. Returns the header
    string, possibly empty. One model call per video, cached.

    When aliases_backend is OpenAI-compatible (OpenRouter or LiteLLM), uses
    identity_link_via_openrouter with that client; otherwise falls back to
    Gemini Flash."""
    openai_client = None
    if aliases_backend == BACKEND_OPENROUTER:
        openai_client = openrouter_client
    elif aliases_backend == BACKEND_LITELLM:
        openai_client = litellm_client
    use_openai_compatible = openai_client is not None
    if not use_openai_compatible and not gemini_client:
        return ""
    cache_path = _aliases_path(states_cache_dir, video_name)
    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except OSError:
            pass
    try:
        if use_openai_compatible:
            model_name = (
                _litellm_model_name(aliases_model)
                if aliases_backend == BACKEND_LITELLM
                else aliases_model
            )
            alias_map = identity_link_via_openrouter(
                openai_client, model_name, timeline_str,
                max_new_tokens=max_new_tokens,
            )
        else:
            alias_map = identity_link_via_flash(
                gemini_client, aliases_model, timeline_str,
                max_new_tokens=max_new_tokens,
            )
    except Exception as exc:
        tqdm.write(f"  [identity_link] {video_name}: failed ({exc}); proceeding without aliases.")
        alias_map = None
    header = format_alias_header(alias_map)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(cache_path, header)
    if alias_map is not None:
        _atomic_write_json(cache_path.with_name("aliases.json"), alias_map)
    return header


def _extract_timeline(
    *,
    sample: Dict[str, Any],
    states_cache_dir: Path,
    gemini_client,
    extractor_client,
    aggregator_openrouter_client,
    chunk_vlm_model: str,
    aggregator_model: str,
    aggregator_backend: str,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_chunk: int,
    max_new_tokens_aggregator: int,
    max_concurrency_chunks: int,
    chunk_prompt_version: str = "v1",
    state_extractor_backend: str = "openrouter",
    frames_per_chunk: int = FORCED_FRAMES_PER_CHUNK,
) -> str:
    video_path = sample["video_path"]
    video_name = sample["video_name"]
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing video: {video_path}")

    plan_path = _plan_cache_path(states_cache_dir, video_name)
    plan = _load_json(plan_path)
    needs_plan = (
        plan is None
        or float(plan.get("chunk_seconds", 0)) != FORCED_CHUNK_SECONDS
        or int(plan.get("frames_per_chunk", 0)) != int(frames_per_chunk)
    )
    if needs_plan:
        meta = _get_video_metadata(video_path)
        plan = _forced_plan(meta["duration_sec"], frames_per_chunk)
        _atomic_write_json(plan_path, plan)

    chunks_path = _chunks_cache_path(states_cache_dir, video_name)
    chunks_cache = _load_json(chunks_path)
    if chunks_cache is not None and not _chunks_match_plan(chunks_cache, plan):
        chunks_cache = None
    if chunks_cache is None:
        meta = _get_video_metadata(video_path)
        ranges = _plan_chunks(meta["duration_sec"], FORCED_CHUNK_SECONDS)

        def run_chunk(i: int) -> Dict[str, Any]:
            cs, ce = ranges[i]
            if state_extractor_backend == BACKEND_GEMINI:
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
                    chunk_prompt_version=chunk_prompt_version,
                )
            return stage_a1_extract_chunk(
                extractor_client,
                chunk_vlm_model,
                video_path,
                chunk_idx=i,
                chunk_start=cs,
                chunk_end=ce,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens=max_new_tokens_chunk,
                chunk_prompt_version=chunk_prompt_version,
            )

        workers = max(1, min(max_concurrency_chunks, len(ranges)))
        results: List[Optional[Dict[str, Any]]] = [None] * len(ranges)
        n = len(ranges)
        tqdm.write(
            f"  [stage A1] {video_name}: extracting {n} chunk(s) "
            f"with {chunk_vlm_model} ({workers} workers)"
        )
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_i = {pool.submit(run_chunk, i): i for i in range(n)}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                try:
                    results[i] = fut.result()
                    done += 1
                    n_events = len(results[i].get("events", []))
                    tqdm.write(
                        f"  [stage A1] chunk {i + 1}/{n} done "
                        f"({done}/{n} total, {n_events} events, "
                        f"elapsed {time.time() - t0:.1f}s)"
                    )
                except Exception as exc:
                    tqdm.write(
                        f"  [stage A1] chunk {i + 1}/{n} FAILED "
                        f"(elapsed {time.time() - t0:.1f}s): {exc}"
                    )
                    raise
        chunks_cache = {
            "video_name": video_name,
            "video_duration_sec": meta["duration_sec"],
            "chunk_seconds": FORCED_CHUNK_SECONDS,
            "frames_per_chunk": int(frames_per_chunk),
            "chunk_vlm_model": chunk_vlm_model,
            "planner": {
                "content_type": plan.get("content_type"),
                "rationale": plan.get("rationale", ""),
                "planner_model": plan.get("planner_model", "forced"),
            },
            "chunks": results,
        }
        _atomic_write_json(chunks_path, chunks_cache)

    stage_a_path = _stage_a_path(states_cache_dir, video_name, aggregator_backend)
    if stage_a_path.exists():
        timeline = stage_a_path.read_text(encoding="utf-8")
    else:
        if aggregator_backend == BACKEND_OPENROUTER:
            if aggregator_openrouter_client is None:
                raise RuntimeError(
                    "aggregator_backend=openrouter requires an OpenRouter client; "
                    "none was provided."
                )
            timeline = aggregator_openrouter(
                aggregator_openrouter_client,
                aggregator_model,
                chunks_cache["chunks"],
                max_new_tokens=max_new_tokens_aggregator,
            )
        elif aggregator_backend == BACKEND_CONCAT:
            timeline = aggregator_concat(chunks_cache["chunks"])
        else:
            timeline = aggregator_flash(
                gemini_client,
                aggregator_model,
                chunks_cache["chunks"],
                max_new_tokens=max_new_tokens_aggregator,
            )
        _atomic_write_text(stage_a_path, timeline)

    return annotate_timeline_with_subtype_order(timeline)


# ---------------------------------------------------------------------------
# Stage A2: Qwen text-only aggregator (OpenRouter).
# ---------------------------------------------------------------------------

def aggregator_openrouter(
    client,
    model: str,
    chunks: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    prompt = AGGREGATOR_PROMPT_TEMPLATE.format(
        n_chunks=len(chunks),
        chunks_block=json.dumps(chunks, ensure_ascii=False, indent=2),
    )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(max_new_tokens),
            )
        except Exception as e:
            print(
                f"[openrouter][aggregator] attempt {attempt} failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            last_err = e
            time.sleep(2 ** attempt)
            continue

        choices = getattr(resp, "choices", None) or []
        if not choices:
            print(
                f"[openrouter][aggregator] attempt {attempt} empty choices; "
                f"raw resp={resp!r}",
                flush=True,
            )
            last_err = RuntimeError("OpenRouter aggregator: no choices in response")
            time.sleep(2 ** attempt)
            continue

        msg = choices[0].message
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")) or ""
                for c in content
            )
        finish_reason = str(getattr(choices[0], "finish_reason", None) or "UNKNOWN").upper()
        if finish_reason == "LENGTH":
            raise _NonRetryableError(
                f"OpenRouter aggregator hit token cap ({max_new_tokens}). "
                f"Bump --max_new_tokens_aggregator."
            )

        text = _strip_code_fence(_strip_thinking(content))
        try:
            _parse_json_list_or_raise(text, "OpenRouter aggregator", finish_reason)
        except Exception as e:
            print(
                f"[openrouter][aggregator] attempt {attempt} parse failed: "
                f"{type(e).__name__}: {e} | finish_reason={finish_reason} | "
                f"text[:500]={text[:500]!r}",
                flush=True,
            )
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return text

    raise RuntimeError(
        f"OpenRouter aggregator failed after {max_retries} retries: {last_err}"
    )


# ---------------------------------------------------------------------------
# Stage B: shared text helpers (Gemini Flash + OpenAI-compatible backends).
# ---------------------------------------------------------------------------

def _gemini_text_call(
    client,
    model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=int(max_new_tokens),
                    temperature=0.0,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    safety_settings=_permissive_safety_settings(),
                ),
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        text, finish_reason = _gemini_extract_text(resp)
        if text:
            return _strip_code_fence(text)
        last_err = RuntimeError(f"Empty Flash response (finish_reason={finish_reason})")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Flash call failed after {max_retries} retries: {last_err}")


def _openai_compatible_text_call(
    client,
    model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    backend_label: str,
    max_retries: int = 4,
) -> str:
    if client is None:
        raise RuntimeError(f"{backend_label} text call requested without a client")
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(max_new_tokens),
            )
        except Exception as e:
            print(
                f"[{backend_label}][text] attempt {attempt} failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            last_err = e
            time.sleep(2 ** attempt)
            continue
        choices = getattr(resp, "choices", None) or []
        if not choices:
            print(
                f"[{backend_label}][text] attempt {attempt} empty choices; "
                f"raw resp={resp!r}",
                flush=True,
            )
            last_err = RuntimeError(f"{backend_label} text call: no choices in response")
            time.sleep(2 ** attempt)
            continue
        msg = choices[0].message
        content = getattr(msg, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")) or ""
                for c in content
            )
        finish_reason = str(getattr(choices[0], "finish_reason", None) or "UNKNOWN").upper()
        if finish_reason == "LENGTH":
            raise _NonRetryableError(
                f"{backend_label} text call hit token cap ({max_new_tokens})."
            )
        text = _strip_code_fence(_strip_thinking(content))
        if text:
            return text
        print(
            f"[{backend_label}][text] attempt {attempt} empty text "
            f"(finish_reason={finish_reason}); raw content[:500]={str(content)[:500]!r}",
            flush=True,
        )
        last_err = RuntimeError(f"Empty {backend_label} response (finish_reason={finish_reason})")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{backend_label} text call failed after {max_retries} retries: {last_err}")


def _openrouter_text_call(
    client,
    model: str,
    prompt: str,
    *,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    return _openai_compatible_text_call(
        client,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        backend_label=BACKEND_OPENROUTER,
        max_retries=max_retries,
    )


def _resolve_litellm_base_url(cli_value: Optional[str]) -> str:
    base_url = (
        cli_value
        or os.environ.get(LITELLM_PROXY_URL_ENV)
        or os.environ.get(LITELLM_BASE_URL_ENV)
    )
    if not base_url:
        raise RuntimeError(
            f"--litellm_base_url or environment variable {LITELLM_PROXY_URL_ENV!r} "
            f"/ {LITELLM_BASE_URL_ENV!r} is required for --stage_b_backend litellm."
        )
    return base_url


def _build_litellm_client(*, base_url: Optional[str], api_key_env: str) -> Tuple[OpenAI, str]:
    resolved_base_url = _resolve_litellm_base_url(base_url)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env!r} is required for "
            "--stage_b_backend litellm."
        )
    return OpenAI(api_key=api_key, base_url=resolved_base_url), resolved_base_url


# ---------------------------------------------------------------------------
# Stage B (narrative): one Flash call per video, cached.
# ---------------------------------------------------------------------------

_NARRATIVE_PROMPT = """You are given a JSON event timeline extracted from a video.
Rewrite it as a concise chronological narrative in plain English. Preserve:
- the order of events,
- the objects involved (e.g. "blue ball", "wooden ramp"),
- any state changes / outcomes (e.g. "knocked down", "released").

Use complete sentences, not bullets. Do not invent details that are not in
the JSON. Aim for 6 to 12 sentences.

JSON timeline:
{timeline}

Narrative:"""


def _sanitize_model_tag(model: str) -> str:
    """Produce a filesystem-safe, model-specific tag for cache folder names.
    Drops the vendor prefix (``qwen/`` etc.) and replaces unsafe chars."""
    tag = model.rsplit("/", 1)[-1]
    tag = re.sub(r"[^a-zA-Z0-9._-]", "-", tag).strip("-._") or "model"
    return tag


def _litellm_model_name(model: str) -> str:
    """Allow either 'gemini/...' or 'litellm/gemini/...' on the CLI."""
    if model.startswith(LITELLM_MODEL_PREFIX):
        return model[len(LITELLM_MODEL_PREFIX):]
    return model


def _stage_b_suffix(filter_backend: str, filter_model: Optional[str]) -> str:
    """Cache-path suffix for Stage B artifacts. Gemini keeps the empty
    suffix (backward-compatible with existing ``filter/`` caches);
    OpenRouter and LiteLLM get backend/model-specific namespaces so
    providers never clobber each other."""
    if filter_backend == BACKEND_OPENROUTER and filter_model:
        return f"_{_sanitize_model_tag(filter_model)}"
    if filter_backend == BACKEND_LITELLM and filter_model != 'gemini-3-flash':
        return f"_litellm_{_sanitize_model_tag(filter_model)}"
    return ""


def _uses_qwen_filter_prompt(filter_backend: str, model: str) -> bool:
    """Preserve the historical OpenRouter prompt, and use it for LiteLLM
    Qwen models too. LiteLLM Gemini/OpenAI-style models use the generic
    filter prompt."""
    if filter_backend == BACKEND_OPENROUTER:
        return True
    if filter_backend == BACKEND_LITELLM:
        return "qwen" in (model or "").lower()
    return False


def _narrative_cache_path(
    states_cache_dir: Path,
    video_name: str,
    filter_backend: str = BACKEND_GEMINI,
    filter_model: Optional[str] = None,
) -> Path:
    suffix = _stage_b_suffix(filter_backend, filter_model)
    return states_cache_dir / Path(video_name).stem / f"stage_b_narrative{suffix}.txt"


def _stage_b_narrative(
    *,
    video_name: str,
    timeline: str,
    states_cache_dir: Path,
    gemini_client,
    openrouter_client,
    litellm_client,
    filter_backend: str,
    model: str,
    max_new_tokens: int,
) -> str:
    cache_path = _narrative_cache_path(states_cache_dir, video_name, filter_backend, model)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    prompt = _NARRATIVE_PROMPT.format(timeline=timeline)
    if filter_backend == BACKEND_OPENROUTER:
        text = _openrouter_text_call(
            openrouter_client, model, prompt, max_new_tokens=max_new_tokens,
        )
    elif filter_backend == BACKEND_LITELLM:
        text = _openai_compatible_text_call(
            litellm_client,
            _litellm_model_name(model),
            prompt,
            max_new_tokens=max_new_tokens,
            backend_label=BACKEND_LITELLM,
        )
    else:
        text = _gemini_text_call(
            gemini_client, model, prompt, max_new_tokens=max_new_tokens,
        )
    _atomic_write_text(cache_path, text)
    return text


# ---------------------------------------------------------------------------
# Stage B (filter): one Flash call per (video, question), cached.
# ---------------------------------------------------------------------------

_FILTER_PROMPT = """You are given a yes/no question about a video and a JSON
event timeline extracted from that video. Pick between 3 and {top_k}
event_index values whose events are most likely to help answer the question.
Prefer including a few neighboring events for context even when one event
seems most directly relevant; err on the side of more rather than fewer.

Choose events that mention the same objects, actors, or actions as the
question, or whose ordinal position matters for the question (e.g. "first",
"3rd", "last"). Do not invent indices that are not present in the timeline.

When the timeline carries a ``round_index`` field on events and the
question refers to an Nth game / Nth round / Nth match (e.g. "the second
game", "the fourth round"), restrict your picks to events whose
``round_index`` matches that ordinal. Use ``segment_label_resolved`` as a
secondary signal when multiple rounds share a number. Include the full
range of event_index values inside that round, not just the decisive
moment.

Timeline:
{timeline}

Question: {question}

Respond ONLY with a JSON array of integers, e.g. [1, 4, 7]."""


# Qwen-specific variant: Qwen struggles to jointly satisfy keyword matching
# and ordinal matching, so we ask it to assemble TWO groups of events and
# merge them. This gives broader recall when the question combines a
# content cue ("pink ball") with an ordinal cue ("fourth game").
_FILTER_PROMPT_QWEN = """You are given a yes/no question about a video and a JSON
event timeline extracted from that video. Pick between 3 and {top_k}
event_index values whose events are most likely to help answer the question.
Prefer including a few neighboring events for context even when one event
seems most directly relevant; err on the side of more rather than fewer.

Choose events that mention the same objects, actors, or actions as the
question, or whose ordinal position matters for the question (e.g. "first",
"3rd", "last"). Do not invent indices that are not present in the timeline.

When events carry a ``round_index`` field and the question refers to an
Nth game / Nth round / Nth match, restrict picks to events whose
``round_index`` equals that ordinal; include the full range of event_index
values inside that round, not just the decisive moment.

Timeline:
{timeline}

Question: {question}

Respond ONLY with a JSON array of integers, e.g. [1, 4, 7]."""


# Keyword patterns that signal an aggregation / ordinal / trajectory question.
# For these, we boost the filter top_k to let the selector pull evidence spread
# across the full timeline instead of a single local cluster — observed failure
# mode on multi-round / Nth-object questions.
_AGGREGATION_PATTERNS = (
    r"\bhow many\b",
    r"\btotal\b",
    r"\bmore than\b",
    r"\bfewer than\b",
    r"\bless than\b",
    r"\bat least\b",
    r"\bat most\b",
    r"\bexactly \d+\b",
    r"\bonly\b",
    r"\beach\b",
    r"\bevery\b",
    r"\bfirst\b",
    r"\bsecond\b",
    r"\bthird\b",
    r"\bfourth\b",
    r"\bfifth\b",
    r"\bsixth\b",
    r"\bseventh\b",
    r"\beighth\b",
    r"\bninth\b",
    r"\btenth\b",
    r"\blast\b",
    r"\bfinal\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\buntil\b",
    r"\bthen\b",
    r"\bmore .* than\b",
    r"\bfewer .* than\b",
    r"\bincrease\b",
    r"\bdecrease\b",
    r"\bgradually\b",
)

_AGGREGATION_REGEX = re.compile("|".join(_AGGREGATION_PATTERNS), re.IGNORECASE)


def _is_aggregation_question(question: str) -> bool:
    """Cheap keyword heuristic — returns True for ordinal / counting /
    trajectory Qs that benefit from a broader filter top_k."""
    return bool(_AGGREGATION_REGEX.search(question or ""))


def _filter_dir(
    states_cache_dir: Path,
    video_name: str,
    filter_backend: str = BACKEND_GEMINI,
    filter_model: Optional[str] = None,
) -> Path:
    suffix = _stage_b_suffix(filter_backend, filter_model)
    return states_cache_dir / Path(video_name).stem / f"filter{suffix}"


def _filter_cache_path(
    states_cache_dir: Path,
    video_name: str,
    question: str,
    filter_backend: str = BACKEND_GEMINI,
    filter_model: Optional[str] = None,
) -> Path:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    return _filter_dir(states_cache_dir, video_name, filter_backend, filter_model) / f"{h}.json"


def _parse_index_list(text: str) -> List[int]:
    payload = _strip_code_fence(text)
    m = re.search(r"\[.*?\]", payload, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    out: List[int] = []
    for x in parsed:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _events_by_index(timeline: str, indices: List[int]) -> List[Dict[str, Any]]:
    if not indices:
        return []
    try:
        parsed = json.loads(timeline)
    except (ValueError, TypeError):
        return []
    # annotate_timeline_with_subtype_order wraps the list in
    # {"subtype_counts": ..., "events": [...]}; Stage A alone returns a list.
    if isinstance(parsed, dict):
        parsed = parsed.get("events", [])
    if not isinstance(parsed, list):
        return []
    keep = set(indices)
    return [
        ev for ev in parsed
        if isinstance(ev, dict) and ev.get("event_index") in keep
    ]


def _stage_b_filter_indices(
    *,
    video_name: str,
    timeline: str,
    question: str,
    states_cache_dir: Path,
    gemini_client,
    openrouter_client,
    litellm_client,
    filter_backend: str,
    model: str,
    max_new_tokens: int,
    top_k: int,
) -> List[int]:
    cache_path = _filter_cache_path(states_cache_dir, video_name, question, filter_backend, model)
    cached = _load_json(cache_path)
    if cached is not None and isinstance(cached.get("indices"), list):
        indices = [int(i) for i in cached["indices"]][:top_k]
        if "events" not in cached:
            # Migrate older cache entries so the user can inspect what the
            # filter picked without cross-referencing stage_a.txt.
            cached["events"] = _events_by_index(timeline, indices)
            _atomic_write_json(cache_path, cached)
        return indices
    prompt_template = (
        _FILTER_PROMPT_QWEN
        if _uses_qwen_filter_prompt(filter_backend, model)
        else _FILTER_PROMPT
    )
    prompt = prompt_template.format(timeline=timeline, question=question, top_k=top_k)
    if filter_backend == BACKEND_OPENROUTER:
        text = _openrouter_text_call(
            openrouter_client, model, prompt, max_new_tokens=max_new_tokens,
        )
    elif filter_backend == BACKEND_LITELLM:
        text = _openai_compatible_text_call(
            litellm_client,
            _litellm_model_name(model),
            prompt,
            max_new_tokens=max_new_tokens,
            backend_label=BACKEND_LITELLM,
        )
    else:
        text = _gemini_text_call(
            gemini_client, model, prompt, max_new_tokens=max_new_tokens,
        )
    indices = _parse_index_list(text)[:top_k]
    events = _events_by_index(timeline, indices)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        cache_path,
        {"question": question, "top_k": top_k, "indices": indices, "events": events},
    )
    return indices


def _slice_timeline(timeline: str, indices: List[int]) -> str:
    sliced = _events_by_index(timeline, indices)
    if not sliced:
        return ""
    return json.dumps(sliced, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Answerer: vLLM Qwen3-VL-32B-Thinking + per-question / per-video state.
# ---------------------------------------------------------------------------

_ANSWERER_GROUNDING_V1 = (
    "IMPORTANT: only use information you can directly verify from the "
    "video frames. Answer the question directly. When possible, cite "
    "rough timestamps. Remember to include a clear 'Yes' or 'No' "
    "somewhere in your answer, and do not enumerate frame numbers or "
    "list long sequences of integers."
)

_ANSWERER_GROUNDING_V2 = (
    "You are answering a yes/no question about a SPECIFIC VIDEO. Your only "
    "evidence is (a) the frames shown below and (b) the extracted event "
    "timeline given as Context (may be incomplete).\n\n"
    "STRICT RULES:\n"
    "1. Treat this as an unknown video. Even if it looks like a famous event "
    "(a sports match, a news clip, a movie, a real-life incident), do NOT "
    "use outside knowledge. Never introduce player names, team rosters, "
    "final scores, historical outcomes, or any named real-world entity that "
    "is not directly visible in the frames or explicitly listed in Context. "
    "Refer to people only by what you see: 'player in white jersey number 10', "
    "'goalkeeper in green', 'person in red shirt'. Never resolve a jersey "
    "number, face, or uniform to a real person's name.\n"
    "2. Answer YES only if the frames or Context positively establish the "
    "claim. If the evidence is missing, ambiguous, or only indirect, answer "
    "NO. 'Could have happened' and 'is consistent with' are not sufficient.\n"
    "3. Facial expressions, body language, crowd reactions, dejected players, "
    "and celebrations are NOT sufficient evidence for what happened during an "
    "action (e.g., whether a shot scored, who missed, who saved). Only the "
    "action itself counts.\n"
    "4. For questions about round numbers, 'Nth' ordinals, or the score at a "
    "specific round, answer YES only if the frames or Context explicitly "
    "label that ordinal or show an explicit scoreboard tied to that point. "
    "Do NOT infer ordinals from real-world knowledge of the event.\n"
    "5. Self-check before answering: does your reasoning cite any name, "
    "number, score, or event not present in the frames or Context? If yes, "
    "remove it and re-answer using only the supplied evidence.\n\n"
    "OUTPUT FORMAT:\n"
    "End your response with exactly these two lines (no other text after):\n"
    "  Evidence: <one short quote from Context, or brief frame description, "
    "or 'none'>\n"
    "  Answer: Yes  (or)  Answer: No"
)


# Two-pass answerer: forces the model to enumerate Yes-evidence and
# No-evidence SEPARATELY before committing, to counter the observed
# "bail to No under uncertainty" bias of Qwen3-VL-32B-Thinking.
# Yes-recall on v4 was only 46% despite balanced ground truth.
_ANSWERER_GROUNDING_TWO_PASS = (
    "You are answering a yes/no question about a SPECIFIC VIDEO. Your "
    "evidence is (a) the frames shown below and (b) the extracted event "
    "timeline given as Context (may be incomplete).\n\n"
    "ANSWER PROTOCOL — follow all three steps in order, do not skip:\n\n"
    "STEP 1 — Evidence FOR Yes:\n"
    "List every piece of Context (quoting verbatim) or frame observation "
    "that would support a YES answer. Be exhaustive. If the question asks "
    "about an Nth occurrence or ordinal position (e.g. 'the third ball', "
    "'the last spin', 'the fourth game'), enumerate ALL matching events "
    "in chronological order and number them 1, 2, 3, ..., then identify "
    "the one the question asks about. If no evidence supports Yes, write "
    "'Yes-evidence: none'.\n\n"
    "STEP 2 — Evidence FOR No:\n"
    "List every piece of Context (quoting verbatim) or frame observation "
    "that would support a NO answer, including evidence that contradicts "
    "the claim or points to a different actor, outcome, or ordering. If "
    "no evidence supports No, write 'No-evidence: none'.\n\n"
    "STEP 3 — Commit:\n"
    "Compare the two lists. Answer YES only if the Yes-evidence list "
    "contains DIRECTLY-QUOTED Context or clearly-observed frame evidence "
    "and is stronger than the No-evidence list. Otherwise answer NO. Do "
    "NOT answer Yes based solely on the absence of contrary evidence.\n\n"
    "GROUNDING CONSTRAINTS (apply during all three steps):\n"
    "- Do NOT use outside knowledge. No real player names, no team names, "
    "  no historical scores, no real-world identities. Refer to people by "
    "  their visible clothing or jersey number (e.g. 'player in white "
    "  jersey #4', 'woman in red shirt').\n"
    "- Facial expressions, body language, crowd reactions, celebrations, "
    "  and dejection are NOT sufficient evidence for what happened during "
    "  an action (did the shot score, who missed, etc.). Only the action "
    "  itself counts.\n"
    "- Paraphrase IS fine (e.g. 'step over' and 'jump over' describe the "
    "  same action; 'push' and 'throw' a ball can be the same action). "
    "  But do NOT stitch multiple separate events into an inferred "
    "  conclusion — each Yes-evidence item must stand on its own quote.\n"
    "- A direct on-screen text quote (e.g. 'CHUCHU WINS', 'ARGENTINA 4 - "
    "  FRANCE 2') IS explicit evidence if it answers the question.\n\n"
    "OUTPUT FORMAT:\n"
    "Write the three steps as plain text labelled 'STEP 1:', 'STEP 2:', "
    "'STEP 3:'. Then end your response with exactly these two lines "
    "(no other text after):\n"
    "  Evidence: <one short quote from Context or brief frame description>\n"
    "  Answer: Yes  (or)  Answer: No"
)


# v3: single-call reasoning prompt with four targeted rules aimed at the
# failure modes that v2 leaves on the table:
#   (1) Counting questions -- the model was treating "how many X" as a
#       lookup against the Context instead of enumerating matching events
#       and tallying them. Rule 1 forces enumeration-then-count.
#   (2) Event compositing -- the model sometimes fabricated a new event by
#       stitching fragments from different events (e.g. "box 8" + "bird's
#       nest" even though those appear on separate unrelated events).
#       Rule 2 blocks synthesis.
#   (3) Event fan-out vs object count -- the extractor can split one
#       box-open into 5-7 adjacent events (take, place, open, reveal,
#       examine, describe). The model was counting events, not object
#       instances. Rule 3 forces coalescing of adjacent events about the
#       same underlying object.
#   (4) Evidence/Answer mismatch -- the model would sometimes emit an
#       Evidence line that clearly supports Yes and then commit to "No"
#       (or vice versa). Rule 4 is a final consistency check.
# Designed to be used with reasoning-capable answerer models (e.g.
# Qwen3-VL-32B-Thinking) which already produce chain-of-thought --
# no structured 3-step protocol needed.
_ANSWERER_GROUNDING_V3 = (
    "You are answering a yes/no question about a SPECIFIC VIDEO. Your only "
    "evidence is (a) the frames shown below and (b) the extracted event "
    "timeline given as Context (may be incomplete).\n\n"
    "GROUNDING CONSTRAINTS:\n"
    "- Treat this as an unknown video. Even if it looks like a famous "
    "event (sports match, news clip, TV show, movie, real-life incident), "
    "do NOT use outside knowledge. No player names, team rosters, final "
    "scores, historical outcomes, named real-world identities. Refer to "
    "people by visible attributes only: 'blue-shirt man', 'woman in red "
    "dress', 'player in white jersey #10'.\n"
    "- If the frames disagree with the Context, trust the frames.\n"
    "- Facial expressions, body language, crowd reactions, celebrations, "
    "and dejected looks are NOT sufficient evidence for what happened "
    "during an action (did the shot score, who missed, etc.). Only the "
    "action itself counts.\n"
    "- A direct on-screen text quote (e.g. 'CHUCHU WINS', 'ARGENTINA 4 - "
    "FRANCE 2', 'ROUND 4') IS explicit evidence when it answers the "
    "question.\n\n"
    "REASONING RULES (apply every time):\n"
    "0. PHRASING SEMANTICS. Existential phrasing -- 'Was there N "
    "X', 'Is there N X', 'Were there N X', 'Did N X happen', 'Did "
    "any X', 'Was there an X who [verb]' -- means AT LEAST N "
    "occurrences exist in the video. Answer Yes if you can identify "
    "at least N matching instances. The questions 'Was there one X' "
    "and 'Was there two X' can BOTH be Yes simultaneously when the "
    "video contains two or more X (one is at least one, two is at "
    "least two). Only the explicit exactness markers 'exactly N', "
    "'only N', 'just N', 'precisely N', 'total N' (e.g. 'were there "
    "TOTAL two X'), 'no more than N' demand exact cardinality and "
    "should answer No if the actual count differs from N. When in "
    "doubt between existential and exact reading, default to "
    "existential.\n"
    "1. COUNTING. If the question asks 'how many', 'exactly N', 'at "
    "least N', 'at most N', 'more than', 'fewer than', 'only', 'each', "
    "or any variant requiring a count: you MUST enumerate the matching "
    "instances yourself. Write them out in chronological order as '1. "
    "<event>', '2. <event>', ..., then state the total. Do NOT search "
    "the Context for a pre-computed total and give up when you cannot "
    "find one. If the Context has zero relevant events, the count is "
    "zero. (For existential 'was there N' questions handled by Rule 0, "
    "you only need to confirm at least N -- exhaustive enumeration is "
    "not required.)\n"
    "2. NO EVENT SYNTHESIS. Your Evidence must correspond to a SINGLE "
    "event that actually appears in the Context (quote it verbatim) or "
    "to what you can directly see in a specific frame. Do NOT combine "
    "details from two or more separate events into a new composite "
    "event. If you find yourself writing 'box 8 ... bird's nest' but "
    "those come from different Context entries, stop -- that is "
    "synthesis, not evidence.\n"
    "2a. NO CHAINING ON SINGLE-STEP TRANSITIONS. If the question asks "
    "about a direct transition 'A -> B' (e.g. 'does the mask change "
    "from red to black?', 'does the ball change from blue to green?'), "
    "you need ONE single event whose ``object_before`` = A and "
    "``object_after`` = B. You may NOT chain two events ('A -> C' plus "
    "'C -> B') to infer 'A -> B' -- that is still synthesis. Two "
    "intermediate transitions do not prove a direct one unless the "
    "Context explicitly records the direct one.\n"
    "3. OBJECT vs EVENT COUNTING. The extractor often splits one real "
    "action into several adjacent events (take -> place -> open -> "
    "reveal -> examine -> describe). When counting objects (boxes, "
    "balls, items, rounds, shots), GROUP events that describe the same "
    "underlying instance -- same numbered item, same color object, same "
    "round_index, or same few-second window -- and count each instance "
    "ONCE. Seven events about box 3 = one box, not seven.\n"
    "3e. OBJECT SELECTION vs CONTENT MANIPULATION. Picking up or "
    "examining the CONTENTS of a box/container that is already open "
    "on the table does NOT constitute opening a new box. Only a "
    "'lift from shelf + open' sequence counts as one box being opened. "
    "If you see a 'lift a bird's nest / plate / item' event after a "
    "box-reveal, that lift refers to the same box already opened -- "
    "do NOT count it as a separate box opening.\n"
    "3f. OBJECT LABELS FROM CONTEXT ONLY. Box numbers, jersey numbers, "
    "podium positions, and similar discrete labels MUST come from the "
    "Context (where the extractor explicitly recorded them, e.g. 'Box 3', "
    "'Box 9') or from clearly legible on-screen text in the frames. Do "
    "NOT infer a box number from its visual position on a shelf "
    "('it looks like the 7th box') or from frame appearance when the "
    "label is too small to read reliably. If the Context does not name "
    "a box number for an event, say 'an unnamed box' -- never substitute "
    "a guessed number. A box number that appears nowhere in the Context "
    "and cannot be clearly read on-screen is a hallucination.\n"
    "3a. PERSON IDENTITY COREFERENCE -- THE ALIAS MAP IS AUTHORITATIVE.\n"
    "    If Context begins with a header named 'PEOPLE IN THIS VIDEO', "
    "that header is an AUTHORITATIVE alias map produced by a dedicated "
    "identity-linking step. Each bullet lists descriptor variants that "
    "refer to ONE person, and may carry a screen-position prefix "
    "(LEFT player / RIGHT player / CENTER player).\n"
    "    SCREEN POSITION: When a bullet is labelled 'LEFT player' or "
    "'RIGHT player', that label is the answer to any 'man on the left' "
    "/ 'woman on the right' question -- you do NOT need to inspect "
    "frames to determine which side they are on. All descriptor "
    "variants in that bullet refer to the same left-side or right-side "
    "person. Example: if the alias map has\n"
    "      - LEFT player (guest): \"blue-suit man\" / \"plaid-shirt man\"\n"
    "      - RIGHT player (host): \"suit man\" / \"dark-suit man\"\n"
    "    then ANY action by \"blue-suit man\" or \"plaid-shirt man\" is "
    "an action by the LEFT person, and ANY action by \"suit man\" or "
    "\"dark-suit man\" is an action by the RIGHT person. When counting "
    "how many boxes the LEFT person opened, sum every lift/open event "
    "whose actor matches ANY descriptor in the LEFT bullet.\n"
    "    DESCRIPTOR COREFERENCE: Before you treat two descriptors as "
    "referring to different people anywhere in your reasoning, you MUST "
    "scan the alias map and check whether those descriptors appear in "
    "the SAME bullet. If they do, they are the SAME person -- this is "
    "binding and OVERRIDES any frame-level intuition that they look "
    "different.\n"
    "    Self-check before emitting your final answer: scan your own "
    "reasoning for any pair of person-descriptors. For each pair, ask "
    "'do these appear in the same alias-map bullet?' If yes, they "
    "are one person and your reasoning must treat them as one.\n"
    "    Even when no alias map is present (singleton-only video, or "
    "identity-link disabled), the chunk extractor still produces 2-4 "
    "slightly varying labels for one recurring person across chunks "
    "as lighting / angle / garment-noun / dominant-color word "
    "changes. Treat near-synonym variations (e.g. 'grey-suit man' / "
    "'grey-jacket man' / 'tan-jacket professor' for one professor; "
    "'red-hoodie student' / 'maroon-hoodie student' for one student) "
    "as ONE person UNLESS two of those descriptors co-occur in the "
    "SAME event (which would prove they're different). When counting "
    "distinct persons, merge near-synonym descriptors before "
    "counting; when answering 'did person X do Y', accept any of X's "
    "variant labels.\n"
    "3b. CONTEXT IS INCOMPLETE -- USE FRAMES TO RECOVER. The Context "
    "is a SUMMARY of dominant events; brief or secondary actions "
    "(a celebration, a high-five, a quick board manipulation, a "
    "glance, a peer reaction) are routinely dropped. Therefore: if "
    "the question asks about a specific action and the Context "
    "contains NO matching event, do NOT default to No on that basis "
    "alone. Fall back to the FRAMES as your primary evidence -- scan "
    "them for the action the question asks about. If the frames "
    "clearly show that action happening at any point, answer Yes and "
    "cite the frame in Evidence (e.g. 'Evidence: frames near 03:30 "
    "show the tan-shirt student high-fiving a peer.'). If the frames "
    "clearly show the action's ABSENCE during the relevant time "
    "window (e.g. the student is visible but not celebrating), "
    "answer No. Only when BOTH Context and frames provide no "
    "evidence either way should you answer No on insufficient-"
    "evidence grounds.\n"
    "3c. SPATIAL CLAIMS -- FRAME EVIDENCE IS MANDATORY, NOT OPTIONAL. "
    "Context's spatial labels (left / middle / right of a multi-panel "
    "structure, top / bottom of a stack, leftmost / rightmost of a "
    "row, near / far podium, upper / lower drawer, Box 1 / 2 / 3 in a "
    "row) are ONE OF THE EXTRACTOR'S MOST ERROR-PRONE OUTPUTS. The "
    "extractor often calls a rightmost panel 'middle' because the "
    "moving panel ends up near the visual centre of the cropped frame. "
    "When a question asks about a SPECIFIC spatial position (left / "
    "middle / right / leftmost / rightmost / top / bottom / Box N / "
    "podium N / row N / etc.), you MUST:\n"
    "  (i) Treat any spatial label in Context as a HYPOTHESIS, not a "
    "fact.\n"
    "  (ii) Look at the frames YOURSELF across the time window of the "
    "relevant event. Count and locate all visible sub-parts of the "
    "multi-part structure (e.g. 3 chalkboard panels left/middle/right, "
    "5 boxes numbered 1-5 from the left, etc.). Identify which "
    "sub-part actually moves or changes between BEFORE and AFTER "
    "frames. Label it by reference to the sub-parts that DID NOT move.\n"
    "  (iii) Your final Evidence line for this question MUST be a "
    "FRAME observation, NOT a Context quote. Acceptable forms:\n"
    "    'Evidence: frames at ~01:38 show the rightmost chalkboard "
    "panel slide down while the leftmost and middle panels remain "
    "stationary.'\n"
    "    'Evidence: frames at ~02:15 show Box 3 (the middle box of "
    "the row of 5) being opened while the others stay closed.'\n"
    "  Forms that are NOT acceptable for a spatial question:\n"
    "    'Evidence: Context states \"the professor pulls DOWN the "
    "MIDDLE chalkboard section\".'  <-- this only repeats the "
    "extractor's label without verification; rejected.\n"
    "    'Evidence: event 40 says ...'   <-- same problem.\n"
    "  (iv) If your frame observation disagrees with Context's "
    "spatial label, TRUST THE FRAMES.\n"
    "If you genuinely cannot tell from the frames which sub-part "
    "moved (camera angle blocks the answer, action too brief, "
    "etc.), say 'Evidence: frames do not unambiguously identify "
    "which sub-part was affected.' and answer No on insufficient-"
    "evidence grounds. That is preferable to confidently echoing "
    "the extractor's label.\n"
    "3d. round_index IS VIDEO-SEGMENT STRUCTURE, NOT GAME TURNS. The "
    "``round_index`` and ``segment_label_resolved`` fields reflect how "
    "the extractor split the video into broad SEGMENTS (e.g. 'Intro', "
    "'Box of Lies', 'Outro') -- NOT individual player turns or game "
    "rounds WITHIN a segment. A label like 'Outro', 'Credits', or "
    "'Title card' is a video segment, never a game round.\n"
    "    When the question asks about 'the Nth round', 'the Nth turn', "
    "'the Nth game', 'the Nth attempt', etc., first check whether all "
    "game events share the same ``round_index``. If they do, the "
    "extractor treated the entire game as ONE segment, and the "
    "question's ordinal refers to the Nth GAME-TURN WITHIN that "
    "segment -- NOT to ``round_index``.\n"
    "    To find the Nth game turn, count the repeating core action "
    "in chronological order:\n"
    "    - Box game: each box lifted FROM THE SHELF = one turn.\n"
    "    - Penalty shootout: each kick attempt = one turn.\n"
    "    - Quiz show: each question posed to a contestant = one turn.\n"
    "    - Throwing game: each throw/toss by a player = one turn.\n"
    "    Count these directly by ``event_index`` order and locate the "
    "Nth one -- that is the 'Nth round' the question means.\n"
    "    MANDATORY SELF-CHECK before answering any 'Nth round' question: "
    "if ``round_index`` of the candidate event is 1 but the 'Outro' / "
    "non-game segment has ``round_index`` 2 or higher, do NOT say 'the "
    "second round is the Outro'. The second round of the GAME must be "
    "the second game-turn event.\n"
    "4. EVIDENCE / ANSWER CONSISTENCY. Before emitting the final "
    "Answer, re-read your Evidence line. Ask: 'Would a reader, seeing "
    "only my Evidence, reach the Answer I am about to commit to?' If "
    "the Evidence supports the opposite verdict, one of them is wrong "
    "-- fix it before you output. Never commit to an Answer that "
    "contradicts your own quoted Evidence.\n"
    "5. REASONING HYGIENE. You see the frames directly as images; "
    "reason about them SILENTLY. Do NOT write out imagined per-frame "
    "descriptions (no 'Frame 1: ...', 'Frame 5: ...' style lists). "
    "That output is wasted tokens and often hallucinated, and it has "
    "caused prior runs to exhaust the token budget before ever "
    "reaching the Answer line. Keep your reasoning compact: at most "
    "about 12 short bullet points or a few paragraphs, then emit the "
    "final Evidence and Answer lines. The Answer line is MANDATORY -- "
    "if you run out of space, skip the reasoning and emit Evidence + "
    "Answer directly.\n\n"
    "TERM CONVENTIONS (how question vocabulary maps to what you see):\n"
    "- 'teleport' / 'teleportation': in object-tracking / cups-and-balls / "
    "magic-trick videos this refers to the perceptual effect of a "
    "covered object appearing to have moved between covered positions "
    "when revealed. It does NOT require literal physics-defying "
    "instantaneous movement. If Context shows object X is hidden at "
    "position A and the same object X is later revealed at position B "
    "(without an explicit visible carry between them), that satisfies "
    "'teleport from A to B' in the question -- answer Yes. The fact "
    "that a human hand was involved off-screen is irrelevant to the "
    "question's use of the word.\n"
    "- 'immediately' / 'immediate' / 'right after' / 'directly after': "
    "in a multi-round game (penalty shootout, quiz round, throwing "
    "round, contest, turn-based game), means 'in the next round / "
    "turn / instance'. It does NOT mean literal temporal adjacency in "
    "seconds. A save in round N+1 happens 'immediately after' a goal "
    "in round N, even if 60+ seconds of clock time pass between them. "
    "Do NOT answer No to an 'immediate' question solely because a time "
    "gap exists -- check whether the events are in adjacent turns.\n"
    "- COLOR / DESCRIPTOR FLEXIBILITY when matching question to "
    "Context: a color word with a pattern qualifier ('white with "
    "green patterns', 'red with black trim', 'blue with white stripes') "
    "still counts as the base color for question-matching. If the "
    "question asks about 'the white object' and Context describes 'a "
    "white cup with green floral patterns', that IS the white object. "
    "Same for clothing ('tan shirt' ~ 'white shirt' ~ 'cream shirt' "
    "for the same person, see Rule 3a) and for objects with a "
    "dominant color plus accent (a 'red mask with black lines' is a "
    "'red mask' for matching).\n"
    "- HAND-RAISE = ANY 'RAISE A [HAND-HELD OBJECT]' PHRASE. Treat the "
    "following Context phrasings as DIRECTLY EQUIVALENT to 'raises "
    "their hand' for any question about hand-raising (to answer, "
    "speak, vote, ask, signal, etc.):\n"
    "    'raises a pen', 'raises a pencil', 'raises a marker',\n"
    "    'raises a paper', 'raises a notebook', 'raises a phone',\n"
    "    'raises a book', 'raises a finger', 'raises an arm',\n"
    "    'raises their pen / pencil / paper / phone / notebook',\n"
    "    'lifts a pen / pencil / paper / phone',\n"
    "    'holds up a pen / pencil / paper / hand / phone',\n"
    "    'puts up their hand / a hand',\n"
    "    'gestures with a raised pen / pencil / paper',\n"
    "    'raises a [color] pen' (e.g. 'raises a silver pen', 'raises "
    "      a blue pen'),\n"
    "    'raises a pen in his/her right/left hand',\n"
    "    'gestures with a pen [in his/her hand] while speaking',\n"
    "    'speaks while holding [up] a pen',\n"
    "    'holds a pen [up] while speaking [to the professor / host]',\n"
    "    'pen [held] in his/her hand while addressing the [professor / "
    "      host / class]'.\n"
    "  When an audience member or student is described in Context as "
    "speaking, gesturing, or addressing the lecturer/host while a pen "
    "(or pencil, paper, marker, phone, notebook) is in their hand, "
    "treat that as the hand-raise event for any question about "
    "hand-raising. The whole 'pen + active engagement with the "
    "leader' pattern IS the hand-raise pattern in classroom / "
    "audience / panel / Q&A contexts -- the phrasing does not need "
    "to include the literal word 'raise'.\n"
    "  These are the same physical action -- the object in the hand "
    "is INCIDENTAL. Do NOT distinguish them from 'raises their hand'. "
    "Do NOT require the Context to use the literal word 'hand' for "
    "this to count. If a student in Context 'raises a silver pen in "
    "his right hand while speaking', that IS a hand-raise; for a "
    "question 'did the [color]-shirt student raise their hand to "
    "answer?', if that matches the actor of the raise-pen event, "
    "answer Yes.\n\n"
    "RUNNING-SCORE INFERENCE (when the question asks about a specific "
    "score the Context doesn't print verbatim):\n"
    "When a question asks about a particular score state in a "
    "multi-round game (e.g. 'was the score 2-1 after round 3?', 'did "
    "X secure the 4-2 victory?', 'did the score reach 3-3 at any "
    "point?'), do NOT default to No just because the exact score "
    "string is absent from Context. Instead, infer it:\n"
    "  (i) Find the most recent scoreboard reading in Context that "
    "occurs at or before the moment in question (look at "
    "on_screen_text fields and explicit scoreboard descriptions).\n"
    "  (ii) Walk forward through subsequent scoring events in "
    "chronological order, adding +1 to the appropriate team's count "
    "per successful score. Save / miss / wide / blocked events do "
    "NOT change the score.\n"
    "  (iii) Commit to the inferred score at the queried moment.\n"
    "Example: Context contains 'ARGENTINA 2 - 1 FRANCE' at 4:30 and "
    "'Montiel #4 of Argentina scores' at 5:52. Inferred score after "
    "Montiel's goal is 3-1. If a fourth Argentine goal follows, "
    "the running tally goes to 4-1 then potentially 4-2. The "
    "'4-2 victory' question is answered Yes if the chain of events "
    "leads there, even though no event prints '4-2' verbatim.\n"
    "Same principle for any other running aggregate (number of misses "
    "before a victory, number of revealed answers before a strike, "
    "etc.) -- maintain the count yourself from explicit anchor "
    "readings + subsequent events. Do not require the exact "
    "aggregate to appear in Context.\n\n"
    "OBJECT-UNDER-COVER STATE INFERENCE (when the question asks about "
    "what changed under a cup / cylinder / box / cover):\n"
    "When a question asks whether the object hidden under a particular "
    "cover changed, stayed the same, or differs in any attribute "
    "(color, identity, count, position) between two moments, you do "
    "NOT need an explicit 'change' or 'transformation' event in "
    "Context. Infer the change from a placement event and a later "
    "reveal event involving the SAME cover:\n"
    "  (i) Find the placement event where the cover was put down on "
    "an object (the 'before' state -- what is now hidden under that "
    "specific cover).\n"
    "  (ii) Find the corresponding reveal / lift event where the "
    "SAME cover is taken away (the 'after' state -- what is now "
    "uncovered).\n"
    "  (iii) Compare the two hidden objects by their stated "
    "attributes (color, identity, count, etc., applying the COLOR "
    "/ DESCRIPTOR FLEXIBILITY rule above where relevant).\n"
    "  (iv) If they DIFFER -> the object under the cover changed; "
    "answer Yes to 'did it change?' / 'does it have a different "
    "color?'; answer No to 'is it the same?' / 'has the same "
    "color?'. If they MATCH -> the inverse answers.\n"
    "Example: Context says 'at 0:54 the woman places brown cylinder "
    "on top of yellow object' and 'at 1:12 she lifts the brown "
    "cylinder, revealing a white object'. The hidden object under "
    "the brown cylinder went from yellow (at placement) to white "
    "(at reveal). 'Did the object under the brown cylinder change?' "
    "= Yes. 'Does it have the same color before and after?' = No.\n"
    "Mirror case (teleport): if the SAME object is shown hidden "
    "under DIFFERENT covers at two times (under pink at 0:54, under "
    "brown at 1:12), that is the cups-and-balls 'teleport' pattern -- "
    "see TERM CONVENTIONS above. The two patterns are complementary: "
    "object-under-cover state inference is 'same cover, different "
    "contents'; teleport is 'different cover, same contents'.\n"
    "Apply the same logic for any cover -> reveal pair, including "
    "boxes (a question about whether the contents of Box 3 changed "
    "between two openings), drawers, and shelves.\n\n"
    "OUTPUT FORMAT:\n"
    "You may reason briefly first (see Rule 5). End your response with "
    "exactly these two lines (no other text after):\n"
    "  Evidence: <one short quote from Context, brief frame description, "
    "or 'none'>\n"
    "  Answer: Yes  (or)  Answer: No"
)


ANSWERER_PROMPT_VERSIONS: Dict[str, str] = {
    "v1": _ANSWERER_GROUNDING_V1,
    "v2": _ANSWERER_GROUNDING_V2,
    "v3": _ANSWERER_GROUNDING_V3,
    "two_pass": _ANSWERER_GROUNDING_TWO_PASS,
}


def get_answerer_grounding(version: str) -> str:
    try:
        return ANSWERER_PROMPT_VERSIONS[version]
    except KeyError:
        raise ValueError(
            f"Unknown answerer_prompt_version {version!r}; "
            f"choose from {sorted(ANSWERER_PROMPT_VERSIONS)}"
        )


def _apply_redaction(text: str, tokens: Iterable[str]) -> str:
    """Case-insensitive substring replacement with '[REDACTED]'. Longest-first
    so that multi-word phrases match before their constituent words."""
    if not text or not tokens:
        return text
    ordered = sorted({t for t in tokens if t}, key=len, reverse=True)
    out = text
    for tok in ordered:
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        out = pattern.sub("[REDACTED]", out)
    return out


class StatefulVLLM(VLLMOpenAIModel):
    """Same backend as benchmark_sub.py --model_id vllm/...; ``_build_content``
    prepends a state hint when one is registered for the question (per-question
    map first, then per-video fallback). Falls through to super() when no
    state is registered, so transport and behaviour are unchanged."""

    def __init__(
        self,
        *args: Any,
        answerer_prompt_version: str = "v1",
        redact_tokens: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._video_state: Optional[str] = None
        self._per_question_state: Dict[str, str] = {}
        self._answer_callback = None  # Optional[Callable[[str, str], None]]
        self._answerer_prompt_version = answerer_prompt_version
        self._grounding_text = get_answerer_grounding(answerer_prompt_version)
        self._redact_tokens = tuple(redact_tokens or ())

    def set_video_state(self, state: Optional[str]) -> None:
        self._video_state = state.strip() if state else None

    def set_per_question_state(self, mapping: Optional[Dict[str, str]]) -> None:
        self._per_question_state = {
            q: v.strip() for q, v in (mapping or {}).items() if v and v.strip()
        }

    def clear_states(self) -> None:
        self._video_state = None
        self._per_question_state = {}

    def set_answer_callback(self, fn) -> None:
        # Fired from the worker thread inside _answer_one once a fresh answer
        # comes back. Used by the streaming JSON writer to flush after every
        # individual answer (not just per sample).
        self._answer_callback = fn

    def _answer_one(
        self,
        question: str,
        frames_b64: List[str],
        max_new_tokens: int,
    ) -> str:
        ans = super()._answer_one(question, frames_b64, max_new_tokens)
        cb = self._answer_callback
        if cb is not None:
            try:
                cb(question, ans)
            except Exception:
                # Monitoring must never break inference.
                pass
        return ans

    def _build_content(self, question: str, frames_b64: List[str]) -> List[Dict[str, Any]]:
        state = self._per_question_state.get(question, self._video_state)
        if state and self._redact_tokens:
            state = _apply_redaction(state, self._redact_tokens)
        question_for_prompt = (
            _apply_redaction(question, self._redact_tokens)
            if self._redact_tokens else question
        )

        parts = [self._grounding_text]
        if state:
            if self._answerer_prompt_version == "v1":
                parts.append(
                    "Optional context (extracted from this video, may be "
                    "incomplete -- if it disagrees with the frames, trust the "
                    f"frames):\n\n{state}"
                )
            else:
                parts.append(
                    "Context (extracted events from this video; may be "
                    "incomplete or contain errors -- if it disagrees with "
                    "what you see in the frames, trust the frames. Do not "
                    f"invent facts not present here or in the frames):\n\n{state}"
                )
        parts.append(f"Question: {question_for_prompt}")
        prompt = "\n\n".join(parts)

        content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
            }
            for frame_b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        return content


# ---------------------------------------------------------------------------
# Gemini answerer: same interface as StatefulVLLM, but the per-question call
# goes to the native google-genai client instead of a local vLLM endpoint.
# Useful when the local Qwen answerer is the bottleneck and you have Gemini
# Flash quota available.
# ---------------------------------------------------------------------------

class GeminiAnswerer:
    """Answerer that runs on Google Gemini (native genai SDK).

    Mirrors the small subset of the StatefulVLLM interface that
    ``_fill_predictions_per_file`` needs: ``set_video_state``,
    ``set_per_question_state``, ``clear_states``, ``set_answer_callback``,
    ``answer_questions``."""

    def __init__(
        self,
        *,
        model_id: str,
        gemini_client,
        n_frames: int = 64,
        max_concurrency: int = 4,
        max_retries: int = 4,
        thinking_budget: int = 0,
        answerer_prompt_version: str = "v1",
        redact_tokens: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_id = model_id
        self._client = gemini_client
        self.n_frames = n_frames
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_retries = max(1, int(max_retries))
        self.thinking_budget = int(thinking_budget)
        self._answerer_prompt_version = answerer_prompt_version
        self._grounding_text = get_answerer_grounding(answerer_prompt_version)
        self._redact_tokens = tuple(redact_tokens or ())
        self._video_state: Optional[str] = None
        self._per_question_state: Dict[str, str] = {}
        self._answer_callback = None

    # ---- StatefulVLLM-compatible state hooks -------------------------------
    def set_video_state(self, state: Optional[str]) -> None:
        self._video_state = state.strip() if state else None

    def set_per_question_state(self, mapping: Optional[Dict[str, str]]) -> None:
        self._per_question_state = {
            q: v.strip() for q, v in (mapping or {}).items() if v and v.strip()
        }

    def clear_states(self) -> None:
        self._video_state = None
        self._per_question_state = {}

    def set_answer_callback(self, fn) -> None:
        self._answer_callback = fn

    # ---- Prompt assembly (mirrors StatefulVLLM._build_content) ------------
    def _build_text_prompt(self, question: str) -> str:
        state = self._per_question_state.get(question, self._video_state)
        if state and self._redact_tokens:
            state = _apply_redaction(state, self._redact_tokens)
        question_for_prompt = (
            _apply_redaction(question, self._redact_tokens)
            if self._redact_tokens else question
        )
        parts = [self._grounding_text]
        if state:
            if self._answerer_prompt_version == "v1":
                parts.append(
                    "Optional context (extracted from this video, may be "
                    "incomplete -- if it disagrees with the frames, trust the "
                    f"frames):\n\n{state}"
                )
            else:
                parts.append(
                    "Context (extracted events from this video; may be "
                    "incomplete or contain errors -- if it disagrees with "
                    "what you see in the frames, trust the frames. Do not "
                    f"invent facts not present here or in the frames):\n\n{state}"
                )
        parts.append(f"Question: {question_for_prompt}")
        return "\n\n".join(parts)

    # ---- Inference --------------------------------------------------------
    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 2048,
    ) -> List[str]:
        if not questions:
            return []
        # Reuse the same uniform-frame sampler the vLLM answerer uses, so
        # frame counts and timing are comparable across backends.
        from src.models.vllm_openai import _extract_frames_b64
        frames_b64 = _extract_frames_b64(video_path, self.n_frames, resize_to_square=None)

        normalized: List[str] = []
        for q in questions:
            qs = q.strip()
            if not qs:
                raise ValueError("Empty question passed to GeminiAnswerer")
            normalized.append(qs)

        results: List[str] = [""] * len(normalized)
        worker_count = min(self.max_concurrency, len(normalized))

        def _one(idx: int, q: str) -> None:
            ans = self._answer_one(q, frames_b64, max_new_tokens)
            results[idx] = ans
            cb = self._answer_callback
            if cb is not None:
                try:
                    cb(q, ans)
                except Exception:
                    pass

        if worker_count <= 1:
            for i, q in enumerate(normalized):
                _one(i, q)
            return results
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futs = [pool.submit(_one, i, q) for i, q in enumerate(normalized)]
            for f in as_completed(futs):
                f.result()
        return results

    def _answer_one(
        self,
        question: str,
        frames_b64: List[str],
        max_new_tokens: int,
    ) -> str:
        prompt_text = self._build_text_prompt(question)
        contents: List[Any] = [prompt_text]
        for fb64 in frames_b64:
            contents.append(
                genai_types.Part.from_bytes(
                    data=base64.b64decode(fb64),
                    mime_type="image/jpeg",
                )
            )

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=int(max_new_tokens),
                        temperature=0.0,
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_budget=self.thinking_budget,
                        ),
                        safety_settings=_permissive_safety_settings(),
                    ),
                )
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
                continue
            text, finish_reason = _gemini_extract_text(resp)
            if finish_reason in ("STOP", "UNKNOWN"):
                return text
            if finish_reason == "MAX_TOKENS":
                # Return whatever we got rather than raising; downstream
                # parser will handle a truncated answer.
                return text
            # Safety / other terminal reasons: return whatever text was
            # returned (may be empty); the parser will mark it unparseable
            # and `is_correct` becomes None for that question.
            return text or ""
        raise RuntimeError(
            f"GeminiAnswerer.{self.model_id} failed after {self.max_retries} retries: {last_err}"
        )



# ---------------------------------------------------------------------------
# Reporting helpers (mirror top-level benchmark_sub.py).
# ---------------------------------------------------------------------------

def _print_results(strategy: str, results: Dict[str, float]) -> None:
    if not results:
        return
    width = max(len(k) for k in results)
    print("\n" + "=" * 50)
    print(f"  Benchmark Results (strategy={strategy})")
    print("=" * 50)
    for key, value in sorted(results.items()):
        if value != value:
            print(f"  {key:<{width}}  =  N/A (no qualifying groups)")
        else:
            print(f"  {key:<{width}}  =  {value:.4f}")
    print("=" * 50 + "\n")


def _jsonify(results: Dict[str, float]) -> Dict[str, Optional[float]]:
    return {
        k: (None if isinstance(v, float) and math.isnan(v) else v)
        for k, v in results.items()
    }


_QA_PROCESSOR = SimpleAnswerProcessor()


def _is_correct(pred: str, gt: str) -> Optional[bool]:
    p = _QA_PROCESSOR(pred)
    g = _QA_PROCESSOR(gt)
    if p == -1:  # model output had no parseable yes/no
        return None
    return p == g


def _aggregate_metrics(
    per_sample_metrics: List[Dict[str, float]],
    accuracy_counts: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Aggregate per-sample metrics.

    Accuracy is pooled across target questions, matching benchmark_sub.py:
    total_correct_targets / total_target_questions. Other metrics remain
    sample-level macro averages.
    """
    if not per_sample_metrics:
        return {}
    keys = {k for r in per_sample_metrics for k in r}
    final: Dict[str, float] = {}
    for k in keys:
        vals = [r[k] for r in per_sample_metrics if k in r and r[k] == r[k]]
        if vals:
            final[k] = sum(vals) / len(vals)
    if accuracy_counts and "accuracy" in final:
        total_correct = sum(correct for correct, _ in accuracy_counts)
        total_targets = sum(total for _, total in accuracy_counts)
        if total_targets:
            final["accuracy"] = total_correct / total_targets
    return final


def _metrics_require_subquestions(metrics: List[Any]) -> bool:
    """Return True when the selected metrics need sub-question predictions."""
    return any(metric.name != "accuracy" for metric in metrics)


def _answer_dir(
    states_cache_dir: Path, video_name: str, prompt_method: str
) -> Path:
    return states_cache_dir / Path(video_name).stem / f"answers_{prompt_method}"


def _answer_file(answer_dir: Path, question: str) -> Path:
    qh = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
    return answer_dir / f"{qh}.json"


def _read_cached_answer(path: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("pred"), str):
        return data
    return None


def _flip_yes_no(gt: str) -> str:
    g = gt.strip().lower()
    if g.startswith("y"):
        return "No"
    if g.startswith("n"):
        return "Yes"
    return gt  # non yes/no gt: fall back to identity


def _scoring_pred(rec: Dict[str, Any], gt: str) -> str:
    """Pred to hand to the scorer for this question.

    If the JSON file has a boolean ``is_correct`` field (typically set by
    the user to override the automatic parser), synthesise a pred that will
    produce that verdict: ``gt`` for True, the flipped yes/no for False.
    Otherwise fall back to the stored ``pred`` string so the scorer parses
    it normally."""
    override = rec.get("is_correct")
    if isinstance(override, bool):
        return gt if override else _flip_yes_no(gt)
    return rec["pred"]


def _write_answer_file(
    path: Path,
    question: str,
    pred: str,
    gt: str,
    state: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        path,
        {
            "question": question,
            "pred": pred,
            "gt": gt,
            "is_correct": _is_correct(pred, gt),
            "state": state,
        },
    )


def _fill_predictions_per_file(
    *,
    groups,
    sample: Dict[str, Any],
    model: Any,
    answer_dir: Path,
    max_new_tokens: int,
    state_map: Optional[Dict[str, Any]] = None,
    include_subquestions: bool = False,
) -> None:
    """Answer selected questions once, writing one JSON file per answer under
    ``answer_dir``. Questions whose file already exists are reused
    (skip-on-rerun). When ``include_subquestions`` is true, sub-question
    predictions are filled too, matching benchmark_sub.py's metric inputs.

    ``state_map`` optionally maps ``question -> state`` (events list for the
    filter strategy, narrative string for the narrative strategy). The value
    is embedded in each answer JSON file under ``"state"`` so you can see
    what context the answerer actually received."""
    video_path = sample["video_path"]
    state_map = state_map or {}

    seen = set()
    unique: List[str] = []
    for g in groups:
        questions = g.all_questions if include_subquestions else [g.target_question]
        for question in questions:
            if question not in seen:
                seen.add(question)
                unique.append(question)

    gt_map: Dict[str, str] = {}
    for g in groups:
        gt_map.setdefault(g.target_question, g.target_gt)
        if include_subquestions:
            for sq, sa in zip(g.sub_questions, g.sub_gts):
                gt_map.setdefault(sq, sa)
    path_map = {q: _answer_file(answer_dir, q) for q in unique}
    answer_map: Dict[str, str] = {}

    uncached: List[str] = []
    for q in unique:
        rec = _read_cached_answer(path_map[q])
        if rec is not None:
            answer_map[q] = _scoring_pred(rec, gt_map[q])
        else:
            uncached.append(q)

    if uncached:
        def on_answered(q: str, a: str) -> None:
            gt = gt_map.get(q)
            path = path_map.get(q)
            if gt is not None and path is not None:
                _write_answer_file(path, q, a, gt, state=state_map.get(q))
        model.set_answer_callback(on_answered)
        try:
            new_answers = model.answer_questions(
                video_path, uncached, max_new_tokens=max_new_tokens,
            )
        finally:
            model.set_answer_callback(None)
        answer_map.update(dict(zip(uncached, new_answers)))

    for g in groups:
        g.target_pred = answer_map[g.target_question]
        if include_subquestions:
            g.sub_preds = [answer_map[sq] for sq in g.sub_questions]
        else:
            g.sub_preds = ["" for _ in g.sub_questions]


def _collect_unique_questions(groups, include_subquestions: bool = False) -> List[str]:
    seen = set()
    out: List[str] = []
    for g in groups:
        questions = g.all_questions if include_subquestions else [g.target_question]
        for question in questions:
            if question not in seen:
                seen.add(question)
                out.append(question)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=(
            "Sub-question benchmark with Stage B context for the Qwen3-VL-32B-"
            "Thinking answerer. --state_strategy picks 'narrative' (Flash "
            "rewrites timeline as prose, fed to every question) or 'filter' "
            "(per-question Flash call selects relevant event_indices)."
        )
    )

    parser.add_argument(
        "--state_strategy", choices=STRATEGIES, required=True,
        help="Stage B context strategy.",
    )

    parser.add_argument("--questions_dir", default="benchmark")
    parser.add_argument("--video_dir", default="raw_data")
    parser.add_argument("--mode", default="all")

    parser.add_argument("--states_cache_dir", required=True,
                        help="Per-video planner/chunks/timeline/stage-B cache root. "
                             "Per-answer JSON files are written under "
                             "<states_cache_dir>/<video_stem>/answers_<prompt_method>/.")

    parser.add_argument("--state_extractor_model", default=DEFAULT_CHUNK_VLM_MODEL)
    parser.add_argument(
        "--state_extractor_backend",
        choices=("openrouter", "vllm", BACKEND_GEMINI),
        default="openrouter",
        help="Where Stage A1 runs. 'openrouter' routes through OpenRouter "
             "(e.g. Qwen VLM); 'vllm' talks to a local OpenAI-compatible vLLM "
             "endpoint (see --state_extractor_vllm_base_url); 'gemini' calls "
             "the native google-genai client (inline frame Parts) -- pair with "
             "--state_extractor_model gemini-3-flash-preview (or similar).",
    )
    parser.add_argument(
        "--state_extractor_vllm_base_url",
        default="http://localhost:8200/v1",
        help="Base URL when --state_extractor_backend=vllm.",
    )
    parser.add_argument(
        "--state_extractor_vllm_api_key_env",
        default="VLLM_API_KEY",
        help="Env var holding the vLLM API key (server usually accepts any string).",
    )
    parser.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL)
    parser.add_argument(
        "--aggregator_backend", choices=AGGREGATOR_BACKENDS, default=BACKEND_GEMINI,
        help="'gemini' calls the native google-genai client; 'openrouter' "
             "routes the text-only aggregator through OpenRouter "
             "(use a text-capable Qwen model via --aggregator_model); "
             "'concat' skips the LLM and assembles the timeline by "
             "concatenating chunk events in chunk order.",
    )
    parser.add_argument("--stage_b_model", default=DEFAULT_STAGE_B_MODEL)
    parser.add_argument(
        "--stage_b_backend", choices=BACKENDS, default=BACKEND_GEMINI,
        help="Backend for the Stage B filter/narrative text call. 'litellm' "
             "uses an OpenAI-compatible LiteLLM proxy.",
    )
    parser.add_argument("--planner_model", default=DEFAULT_PLANNER_MODEL)
    parser.add_argument(
        "--planner_backend", choices=PLANNER_BACKENDS, default=BACKEND_GEMINI,
        help="Backend for the planner (--state_strategy=planner). 'gemini' "
             "calls google-genai with inline frames; 'openrouter' expects a "
             "VLM model slug (e.g. qwen/qwen3-vl-235b-a22b-thinking).",
    )
    parser.add_argument("--max_planner_frames", type=int, default=64,
                        help="Upper bound on frames fed to the planner.")
    parser.add_argument("--max_new_tokens_planner", type=int, default=1024)
    parser.add_argument("--frame_max_side", type=int, default=768)
    parser.add_argument("--frame_jpeg_quality", type=int, default=85)
    parser.add_argument(
        "--chunk_prompt_version", choices=("v1", "v2", "v3", "v4", "v5", "v6"), default="v1",
        help="Stage A1 chunk-extractor prompt variant. "
             "v1 = original; v2 = richer descriptors (hair/spatial, "
             "static landmarks, before/after state transitions, "
             "mandatory colors); v3 = re-ID-focused (bundles "
             "hair/build/accessories/jersey number/side, same descriptor "
             "reused within a chunk); v4 = v3 + optional `sports` "
             "sub-object on scoring events (team, result, direction, "
             "score_after, round_index, keeper_dive) and aggressive "
             "merging of reaction/celebration into their parent shot; "
             "v5 = v1 schema PLUS an ``on_screen_text`` field per event "
             "that transcribes burned-in text (scoreboards, chyrons, "
             "banners, jersey numbers, podium labels, captions) verbatim "
             "with a coarse location tag -- use this when the video "
             "contains scoreboard / caption / banner evidence the VLM "
             "would otherwise miss; "
             "v6 = v5 PLUS per-event ``segment_boundary`` (start / "
             "middle / end / unknown) and ``segment_label``, with a "
             "deterministic post-processor that assigns a global "
             "``round_index`` to every event -- use this for videos "
             "with multiple back-to-back rounds / games / matches whose "
             "boundaries are not explicitly numbered on screen. "
             "Use a distinct --states_cache_dir when switching versions "
             "so caches don't cross-contaminate.",
    )
    parser.add_argument(
        "--frames_per_chunk",
        type=int,
        choices=FRAMES_PER_CHUNK_CHOICES,
        default=FORCED_FRAMES_PER_CHUNK,
        help="Frames to sample per 15s chunk. 32 = default (matches prior "
             "runs); 60 = denser sampling to catch fast actions (jersey "
             "numbers, ball contact, scoreboard flashes). Use a fresh "
             "--states_cache_dir when switching so caches don't mix.",
    )
    parser.add_argument("--max_new_tokens_chunk", type=int, default=32768)
    parser.add_argument("--max_new_tokens_aggregator", type=int, default=65536)
    parser.add_argument("--max_new_tokens_narrative", type=int, default=4096)
    parser.add_argument("--max_new_tokens_filter", type=int, default=128)
    # Default depends on --stage_b_backend: Gemini handles both content and
    # ordinal signals in one pass so 10 is enough; Qwen uses a two-group
    # recall prompt and benefits from a bit more budget.
    parser.add_argument("--filter_top_k", type=int, default=None)
    parser.add_argument(
        "--filter_top_k_aggregation", type=int, default=None,
        help="Top_k used when the question looks like an aggregation / "
             "ordinal / trajectory question (keyword heuristic). Defaults to "
             "~2.5x --filter_top_k so the filter can sample evidence spread "
             "across the full video. Set equal to --filter_top_k to disable.",
    )
    parser.add_argument(
        "--aggregation_routing", action="store_true",
        help="When set, aggregation / ordinal / counting questions BYPASS "
             "the Stage B filter entirely and receive the FULL event "
             "timeline as Context. Designed to pair with "
             "--answerer_prompt_version two_pass, where the enumeration "
             "instructions exploit the full timeline to count instances "
             "deterministically. Non-aggregation questions still use the "
             "normal filter path.",
    )
    parser.add_argument(
        "--enable_identity_link", action="store_true",
        help="Run a one-shot Gemini Flash call after Stage A per video that "
             "groups participant descriptors by inferred identity (e.g. "
             "'tan-shirt student' / 'tan-plaid-shirt student' / 'tan-jacket "
             "student' -> one person). The resulting alias header is "
             "prepended to every per-question Context downstream so the "
             "answerer doesn't have to re-derive coreference each time. "
             "Cached as aliases.txt + aliases.json next to stage_a_concat.txt.",
    )
    parser.add_argument(
        "--identity_link_model", default=None,
        help="Model for --enable_identity_link. Defaults to --stage_b_model.",
    )
    parser.add_argument(
        "--max_new_tokens_identity_link", type=int, default=4096,
        help="Output cap for the identity-link Gemini call.",
    )
    parser.add_argument("--max_concurrency_chunks", type=int, default=4)
    parser.add_argument("--max_concurrency_filter", type=int, default=8)
    parser.add_argument("--google_api_key_env", default="GOOGLE_API_KEY")
    parser.add_argument("--openrouter_api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--litellm_base_url",
        default=None,
        help=(
            "OpenAI-compatible LiteLLM proxy base URL for "
            "--stage_b_backend litellm. Falls back to LITELLM_PROXY_URL, "
            "then LITELLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--litellm_api_key_env",
        default="LITELLM_API_KEY",
        help="Env var holding the LiteLLM proxy API key.",
    )

    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID,
                        help="Answerer model. Must start with 'vllm/'.")
    parser.add_argument("--vllm_base_url", default=None,
                        help=f"vLLM base URL. Falls back to env VLLM_BASE_URL, then {DEFAULT_VLLM_BASE_URL}.")
    parser.add_argument("--vllm_api_key_env", default="VLLM_API_KEY")
    parser.add_argument("--vllm_n_frames", type=int, default=64)
    parser.add_argument("--vllm_max_concurrency", type=int, default=4)
    parser.add_argument(
        "--vllm_fallback_base_url", default=None,
        help="Optional secondary OpenAI-compatible endpoint. When the "
             "primary --vllm_base_url raises a non-BadRequest exception "
             "for an answerer call (e.g. OpenRouter returns content=None "
             "or a malformed response), the request is retried once "
             "against this URL with --vllm_fallback_model_id.",
    )
    parser.add_argument(
        "--vllm_fallback_model_id", default=None,
        help="Model id to use against --vllm_fallback_base_url. "
             "Required if --vllm_fallback_base_url is set.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--answerer_backend", choices=("vllm", "gemini"), default="vllm",
        help="Backend for the per-question answerer. 'vllm' (default) uses "
             "the local Qwen vLLM endpoint via --vllm_base_url. 'gemini' "
             "calls the native google-genai client; pair with "
             "--model_id gemini-3-flash-preview (or any other Gemini model).",
    )
    parser.add_argument(
        "--gemini_answerer_thinking_budget", type=int, default=0,
        help="thinking_budget for the Gemini answerer (when "
             "--answerer_backend gemini). 0 = no extended thinking, the "
             "default; bump for harder questions at higher cost.",
    )
    parser.add_argument(
        "--gemini_answerer_max_concurrency", type=int, default=4,
        help="Max concurrent per-question calls to Gemini.",
    )
    parser.add_argument(
        "--answerer_prompt_version",
        choices=tuple(ANSWERER_PROMPT_VERSIONS),
        default="v1",
        help="Answerer grounding-prompt variant. v1 = original; v2 = "
             "hardened (forbids outside-knowledge, evidence-gated, strict "
             "Yes/No output). Encoded in the default prompt_method so "
             "answers land in a separate cache dir.",
    )
    parser.add_argument(
        "--redact_tokens", default="",
        help="Comma-separated list of case-insensitive substrings to replace "
             "with [REDACTED] in the state Context and in the question "
             "before the answerer sees them. Use to strip cues like team or "
             "event names that trigger training-data recall.",
    )
    parser.add_argument(
        "--prompt_method", default=None,
        help="Cache namespace tag for the answerer. If unset, defaults to "
             "'<strategy>_<answerer_prompt_version>' so different prompts "
             "do not collide.",
    )

    parser.add_argument("--metrics", nargs="+", default=["accuracy"])

    args = parser.parse_args()

    if args.filter_top_k is None:
        args.filter_top_k = (
            15 if _uses_qwen_filter_prompt(args.stage_b_backend, args.stage_b_model) else 10
        )
    if args.filter_top_k_aggregation is None:
        # ~2.5x the base top_k, capped at 40 to keep prompt size bounded.
        args.filter_top_k_aggregation = min(40, max(args.filter_top_k, int(round(args.filter_top_k * 2.5))))

    if args.answerer_backend == "vllm":
        if not args.model_id.startswith(_VLLM_PREFIX):
            raise ValueError(
                f"--model_id must start with '{_VLLM_PREFIX}' for "
                f"--answerer_backend vllm (got {args.model_id!r})."
            )
        real_model_id = args.model_id[len(_VLLM_PREFIX):]
        base_url = (
            args.vllm_base_url
            or os.environ.get("VLLM_BASE_URL")
            or DEFAULT_VLLM_BASE_URL
        )
    else:  # gemini
        # Strip the vllm/ prefix if the user accidentally left it on.
        real_model_id = args.model_id[len(_VLLM_PREFIX):] if args.model_id.startswith(_VLLM_PREFIX) else args.model_id
        base_url = None  # unused for Gemini
    version_tag = args.answerer_prompt_version
    answerer_tag = "" if args.answerer_backend == "vllm" else "_gemini"
    if args.prompt_method:
        prompt_method = args.prompt_method
    elif args.state_strategy == STRATEGY_PLANNER:
        prompt_method = (
            f"planner_{_planner_sanitize_tag(args.planner_model)}_{version_tag}{answerer_tag}"
        )
    else:
        backend_tag = _stage_b_suffix(args.stage_b_backend, args.stage_b_model)
        prompt_method = f"{args.state_strategy}{backend_tag}_{version_tag}{answerer_tag}"

    redact_tokens = tuple(
        t.strip() for t in args.redact_tokens.split(",") if t.strip()
    )

    benchmark_data = load_benchmark(args.video_dir, args.mode, args.questions_dir)
    metrics = build_metrics(args.metrics)
    include_subquestions = _metrics_require_subquestions(metrics)

    needs_gemini = (
        args.state_extractor_backend == BACKEND_GEMINI
        or args.aggregator_backend == BACKEND_GEMINI
        or args.stage_b_backend == BACKEND_GEMINI
        or args.answerer_backend == BACKEND_GEMINI
        or (
            args.state_strategy == STRATEGY_PLANNER
            and args.planner_backend == BACKEND_GEMINI
        )
        or (
            args.enable_identity_link
            and args.stage_b_backend == BACKEND_GEMINI
        )
    )
    gemini_client = (
        _build_gemini_client(args.google_api_key_env)
        if needs_gemini
        else None
    )
    litellm_client = None
    litellm_base_url = None
    if args.stage_b_backend == BACKEND_LITELLM:
        litellm_client, litellm_base_url = _build_litellm_client(
            base_url=args.litellm_base_url,
            api_key_env=args.litellm_api_key_env,
        )
    if args.state_extractor_backend == "vllm":
        extractor_client = OpenAI(
            api_key=os.environ.get(args.state_extractor_vllm_api_key_env) or "EMPTY",
            base_url=args.state_extractor_vllm_base_url,
        )
        extractor_label = f"{args.state_extractor_model} (vLLM @ {args.state_extractor_vllm_base_url})"
        openrouter_client = None
    elif args.state_extractor_backend == BACKEND_GEMINI:
        extractor_client = None
        extractor_label = f"{args.state_extractor_model} (native Gemini)"
        openrouter_client = None
    else:
        extractor_client = _build_openrouter_client(args.openrouter_api_key_env)
        extractor_label = f"{args.state_extractor_model} (OpenRouter)"
        openrouter_client = extractor_client

    needs_openrouter = (
        args.aggregator_backend == BACKEND_OPENROUTER
        or args.stage_b_backend == BACKEND_OPENROUTER
        or (args.state_strategy == STRATEGY_PLANNER
            and args.planner_backend == BACKEND_OPENROUTER)
    )
    if needs_openrouter and openrouter_client is None:
        openrouter_client = _build_openrouter_client(args.openrouter_api_key_env)

    if args.answerer_backend == "vllm":
        model = StatefulVLLM(
            model_id=real_model_id,
            prompt_method=prompt_method,
            n_frames=args.vllm_n_frames,
            api_key_env=args.vllm_api_key_env,
            base_url=base_url,
            max_concurrency=args.vllm_max_concurrency,
            answerer_prompt_version=args.answerer_prompt_version,
            redact_tokens=redact_tokens,
            fallback_model_id=args.vllm_fallback_model_id,
            fallback_base_url=args.vllm_fallback_base_url,
        )
    else:  # gemini
        model = GeminiAnswerer(
            model_id=real_model_id,
            gemini_client=gemini_client,
            n_frames=args.vllm_n_frames,
            max_concurrency=args.gemini_answerer_max_concurrency,
            thinking_budget=args.gemini_answerer_thinking_budget,
            answerer_prompt_version=args.answerer_prompt_version,
            redact_tokens=redact_tokens,
        )
    states_cache_dir = Path(args.states_cache_dir)
    states_cache_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Strategy         : {args.state_strategy}")
    tqdm.write(f"Chunking         : {FORCED_CHUNK_SECONDS}s / {args.frames_per_chunk} frames per chunk")
    tqdm.write(f"State extractor  : {extractor_label}")
    tqdm.write(f"Aggregator       : {args.aggregator_model} ({args.aggregator_backend})")
    tqdm.write(f"Stage B model    : {args.stage_b_model} ({args.stage_b_backend})")
    if litellm_base_url:
        tqdm.write(f"LiteLLM Stage B  : {litellm_base_url}")
    tqdm.write(f"Chunk prompt     : {args.chunk_prompt_version}")
    tqdm.write(f"Answerer (vLLM)  : {real_model_id} @ {base_url}")
    tqdm.write(f"Answerer prompt  : {args.answerer_prompt_version}")
    if redact_tokens:
        tqdm.write(f"Redact tokens    : {list(redact_tokens)}")
    tqdm.write(f"Prompt method    : {prompt_method}")
    tqdm.write(f"States cache dir : {states_cache_dir}")
    tqdm.write(f"Answers under    : {states_cache_dir}/<video_stem>/answers_{prompt_method}/")
    tqdm.write(f"Metrics          : {[m.name for m in metrics]}")
    tqdm.write(f"Sub-questions    : {'enabled' if include_subquestions else 'disabled'}")
    tqdm.write(f"Samples          : {len(benchmark_data)}\n")

    per_sample_metrics: List[Dict[str, float]] = []
    accuracy_counts: List[Tuple[int, int]] = []
    skipped_examples: List[str] = []

    for idx, sample in enumerate(tqdm(benchmark_data, desc="Evaluating")):
        example_path = sample.get("example_path", sample.get("video_path", "<unknown>"))
        timeline: Optional[str] = None
        planner_video_meta: Optional[Dict[str, Any]] = None
        planner_chunk_ranges: Optional[List] = None
        if args.state_strategy == STRATEGY_PLANNER:
            try:
                video_path = sample["video_path"]
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Missing video: {video_path}")
                planner_video_meta = _get_video_metadata(video_path)
                planner_chunk_ranges = _plan_chunks(
                    planner_video_meta["duration_sec"], FORCED_CHUNK_SECONDS
                )
            except Exception as exc:
                skipped_examples.append(str(example_path))
                tqdm.write(f"[{idx + 1:>4}/{len(benchmark_data)}] video probe failed: {example_path}")
                tqdm.write(f"Error: {exc}")
                continue
        else:
            try:
                timeline = _extract_timeline(
                    sample=sample,
                    states_cache_dir=states_cache_dir,
                    gemini_client=gemini_client,
                    extractor_client=extractor_client,
                    aggregator_openrouter_client=openrouter_client,
                    chunk_vlm_model=args.state_extractor_model,
                    aggregator_model=args.aggregator_model,
                    aggregator_backend=args.aggregator_backend,
                    frame_max_side=args.frame_max_side,
                    frame_jpeg_quality=args.frame_jpeg_quality,
                    max_new_tokens_chunk=args.max_new_tokens_chunk,
                    max_new_tokens_aggregator=args.max_new_tokens_aggregator,
                    max_concurrency_chunks=args.max_concurrency_chunks,
                    chunk_prompt_version=args.chunk_prompt_version,
                    state_extractor_backend=args.state_extractor_backend,
                    frames_per_chunk=args.frames_per_chunk,
                )
            except Exception as exc:
                skipped_examples.append(str(example_path))
                tqdm.write(f"[{idx + 1:>4}/{len(benchmark_data)}] state extraction failed: {example_path}")
                tqdm.write(f"Error: {exc}")
                continue

        # Identity-link header (cached). Adds a "PEOPLE IN THIS VIDEO"
        # block to every per-question Context so the answerer doesn't
        # have to re-derive coreference for each question.
        alias_header = ""
        if args.enable_identity_link:
            _aliases_backend = (
                args.stage_b_backend
                if not hasattr(args, "identity_link_backend")
                else args.identity_link_backend
            )
            alias_header = _build_aliases_if_missing(
                states_cache_dir=states_cache_dir,
                video_name=sample["video_name"],
                timeline_str=timeline,
                gemini_client=gemini_client,
                openrouter_client=openrouter_client,
                litellm_client=litellm_client,
                aliases_model=(args.identity_link_model or args.stage_b_model),
                aliases_backend=_aliases_backend,
                max_new_tokens=args.max_new_tokens_identity_link,
            )

        groups = build_question_groups(sample)
        model.clear_states()

        state_map: Dict[str, Any] = {}
        try:
            if args.state_strategy == STRATEGY_NARRATIVE:
                narrative = _stage_b_narrative(
                    video_name=sample["video_name"],
                    timeline=timeline,
                    states_cache_dir=states_cache_dir,
                    gemini_client=gemini_client,
                    openrouter_client=openrouter_client,
                    litellm_client=litellm_client,
                    filter_backend=args.stage_b_backend,
                    model=args.stage_b_model,
                    max_new_tokens=args.max_new_tokens_narrative,
                )
                model.set_video_state(narrative)
                state_map = {
                    q: narrative
                    for q in _collect_unique_questions(
                        groups, include_subquestions=include_subquestions
                    )
                }
            elif args.state_strategy == STRATEGY_PLANNER:
                unique_qs = _collect_unique_questions(
                    groups, include_subquestions=include_subquestions
                )
                slices: Dict[str, str] = {}
                planner_bar = tqdm(
                    unique_qs,
                    desc=f"  {sample['video_name']}",
                    leave=False,
                    unit="q",
                )
                for q in planner_bar:
                    planner_bar.set_postfix_str("start", refresh=False)

                    def _phase_cb(phase: str, _bar=planner_bar) -> None:
                        _bar.set_postfix_str(phase)

                    try:
                        timeline_q = build_biased_timeline_cached(
                            video_path=sample["video_path"],
                            video_name=sample["video_name"],
                            video_duration_sec=planner_video_meta["duration_sec"],
                            chunk_ranges=planner_chunk_ranges,
                            question=q,
                            states_cache_dir=states_cache_dir,
                            planner_backend=args.planner_backend,
                            planner_model=args.planner_model,
                            max_planner_frames=args.max_planner_frames,
                            max_new_tokens_planner=args.max_new_tokens_planner,
                            extractor_client=extractor_client,
                            extractor_model=args.state_extractor_model,
                            frames_per_chunk=args.frames_per_chunk,
                            frame_max_side=args.frame_max_side,
                            frame_jpeg_quality=args.frame_jpeg_quality,
                            max_new_tokens_chunk=args.max_new_tokens_chunk,
                            max_concurrency_chunks=args.max_concurrency_chunks,
                            aggregator_backend=args.aggregator_backend,
                            aggregator_model=args.aggregator_model,
                            max_new_tokens_aggregator=args.max_new_tokens_aggregator,
                            gemini_client=gemini_client,
                            openrouter_client=openrouter_client,
                            chunk_prompt_version=args.chunk_prompt_version,
                            on_phase=_phase_cb,
                        )
                    except Exception as exc:
                        tqdm.write(
                            f"  [planner] failed for one question, falling back to no-state: {exc}"
                        )
                        timeline_q = ""
                    slices[q] = (
                        f"{alias_header}\n\n{timeline_q}" if alias_header and timeline_q else timeline_q
                    )
                    try:
                        state_map[q] = json.loads(timeline_q) if timeline_q else []
                    except Exception:
                        state_map[q] = timeline_q
                model.set_per_question_state(slices)
            else:
                unique_qs = _collect_unique_questions(
                    groups, include_subquestions=include_subquestions
                )
                workers = max(1, min(args.max_concurrency_filter, len(unique_qs)))
                slices: Dict[str, str] = {}
                # Aggregation routing: for agg/ordinal/counting questions
                # (when --aggregation_routing is set) we skip the filter and
                # feed the FULL timeline, which lets the answerer enumerate
                # every matching instance deterministically.
                agg_qs = set()
                filter_qs = []
                for q in unique_qs:
                    if args.aggregation_routing and _is_aggregation_question(q):
                        agg_qs.add(q)
                    else:
                        filter_qs.append(q)
                for q in agg_qs:
                    slices[q] = (
                        f"{alias_header}\n\n{timeline}" if alias_header and timeline else timeline
                    )
                    try:
                        state_map[q] = json.loads(timeline) if timeline else []
                    except Exception:
                        state_map[q] = timeline
                if filter_qs:
                    with ThreadPoolExecutor(max_workers=max(1, min(args.max_concurrency_filter, len(filter_qs)))) as pool:
                        fut_to_q = {
                            pool.submit(
                                _stage_b_filter_indices,
                                video_name=sample["video_name"],
                                timeline=timeline,
                                question=q,
                                states_cache_dir=states_cache_dir,
                                gemini_client=gemini_client,
                                openrouter_client=openrouter_client,
                                litellm_client=litellm_client,
                                filter_backend=args.stage_b_backend,
                                model=args.stage_b_model,
                                max_new_tokens=args.max_new_tokens_filter,
                                top_k=(
                                    args.filter_top_k_aggregation
                                    if _is_aggregation_question(q)
                                    else args.filter_top_k
                                ),
                            ): q
                            for q in filter_qs
                        }
                        for fut in as_completed(fut_to_q):
                            q = fut_to_q[fut]
                            try:
                                indices = fut.result()
                            except Exception as exc:
                                tqdm.write(f"  [filter] failed for one question, falling back to no-state: {exc}")
                                indices = []
                            state_map[q] = _events_by_index(timeline, indices)
                            sliced = _slice_timeline(timeline, indices)
                            slices[q] = (
                                f"{alias_header}\n\n{sliced}" if alias_header and sliced else sliced
                            )
                model.set_per_question_state(slices)
        except Exception as exc:
            skipped_examples.append(str(example_path))
            tqdm.write(f"[{idx + 1:>4}/{len(benchmark_data)}] stage B failed: {example_path}")
            tqdm.write(f"Error: {exc}")
            continue

        answer_dir = _answer_dir(
            states_cache_dir, sample["video_name"], prompt_method
        )

        try:
            _fill_predictions_per_file(
                groups=groups,
                sample=sample,
                model=model,
                answer_dir=answer_dir,
                max_new_tokens=args.max_new_tokens,
                state_map=state_map,
                include_subquestions=include_subquestions,
            )
        except Exception as exc:
            skipped_examples.append(str(example_path))
            tqdm.write(f"[{idx + 1:>4}/{len(benchmark_data)}] answerer failed: {example_path}")
            tqdm.write(f"Error: {exc}")
            model.clear_states()
            continue
        model.clear_states()

        sample_result = evaluate(groups, metrics)
        per_sample_metrics.append(sample_result)
        if "accuracy" in sample_result:
            accuracy_counts.append(
                (sum(1 for group in groups if group.is_target_correct()), len(groups))
            )

        tqdm.write(
            f"[{idx + 1:>4}/{len(benchmark_data)}] "
            + "  ".join(f"{k}={v:.3f}" for k, v in sample_result.items() if v == v)
        )

    if skipped_examples:
        tqdm.write(f"Skipped {len(skipped_examples)} samples due to errors.")

    final = _aggregate_metrics(per_sample_metrics, accuracy_counts=accuracy_counts)
    _print_results(args.state_strategy, final)

    tqdm.write(
        f"Per-answer JSON files under {states_cache_dir}/<video_stem>/answers_{prompt_method}/"
    )


if __name__ == "__main__":
    main()
