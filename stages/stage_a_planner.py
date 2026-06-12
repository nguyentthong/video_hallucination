"""Planner-driven Stage A: use a VLM planner on sampled frames + question to
produce a focus hint, then re-run Stage A1 per chunk with the hint injected
into the extractor prompt. Replaces the Stage B filter when
--state_strategy=planner. Does not touch the existing filter/narrative paths."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.genai import types as genai_types

from concurrent.futures import ThreadPoolExecutor, as_completed

from .eval_tier2_flash_aggregator import (
    AGGREGATOR_PROMPT_TEMPLATE,
    CHUNK_PROMPT_TEMPLATE,
    CHUNK_PROMPT_TEMPLATES,
    get_chunk_prompt,
    _NonRetryableError,
    _atomic_write_json,
    _atomic_write_text,
    _build_frame_parts,
    _encode_frame_b64,
    _extract_chunk_frames_b64,
    _fmt_mmss,
    _gemini_extract_text,
    _load_json,
    _parse_json_list_or_raise,
    _strip_code_fence,
    _strip_thinking,
    aggregator_concat,
    aggregator_flash,
    annotate_timeline_with_subtype_order,
)


PLANNER_MAX_FRAMES_CAP = 64
PLANNER_SECONDS_PER_FRAME = 8.0
PLANNER_MIN_FRAMES = 4


def planner_n_frames(duration_sec: float, max_frames: int = PLANNER_MAX_FRAMES_CAP) -> int:
    n = int(round(duration_sec / PLANNER_SECONDS_PER_FRAME))
    return max(PLANNER_MIN_FRAMES, min(max_frames, n))


def sanitize_model_tag(model: str) -> str:
    tag = model.rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "-", tag).strip("-._") or "model"


def _question_hash(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]


def planner_cache_path(
    states_cache_dir: Path, video_name: str, question: str, planner_model: str
) -> Path:
    tag = sanitize_model_tag(planner_model)
    return (
        states_cache_dir
        / Path(video_name).stem
        / f"planner_{tag}"
        / f"{_question_hash(question)}.json"
    )


def biased_chunks_cache_path(
    states_cache_dir: Path, video_name: str, question: str, planner_model: str
) -> Path:
    tag = sanitize_model_tag(planner_model)
    return (
        states_cache_dir
        / Path(video_name).stem
        / f"planner_chunks_{tag}"
        / f"{_question_hash(question)}.json"
    )


def biased_timeline_cache_path(
    states_cache_dir: Path,
    video_name: str,
    question: str,
    planner_model: str,
    aggregator_backend: str,
) -> Path:
    tag = sanitize_model_tag(planner_model)
    return (
        states_cache_dir
        / Path(video_name).stem
        / f"planner_timeline_{tag}_{aggregator_backend}"
        / f"{_question_hash(question)}.txt"
    )


PLANNER_PROMPT_TEMPLATE = """You are a planner helping a downstream extractor
decide what to describe in detail when it watches a long video. You are shown
{n_frames} frames uniformly sampled from the video and one yes/no question
about the video.

Your job: list the objects and event-types that the extractor should pay the
most attention to so its output contains enough detail to answer the
question. Look at the frames carefully; base the hint on what you actually
see, not on guesses from the question alone.

Question:
{question}

Output STRICT JSON only, no prose and no markdown fences:
{{
  "focus_objects": [<short visible-attribute descriptors, e.g.
    "yellow ball", "blue-shirt man", "wooden ramp">, ...],
  "focus_events":  [<short event-type labels, e.g.
    "toss", "hit", "balloon pop", "ladder climb">, ...],
  "reasoning_brief": "<one short sentence>"
}}

Rules:
  - Keep each list between 1 and 8 items. Short phrases, no full sentences.
  - Use visible attributes (color, clothing) for people/objects; do NOT
    invent things you cannot see in the frames.
  - Event-type labels should match what an extractor would plausibly tag
    (e.g. "throw", "catch", "fall"), not raw question phrasing.
"""


def _sample_uniform_frames_b64(
    video_path: str,
    n_frames: int,
    max_side: int,
    jpeg_quality: int,
) -> List[str]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    if n_frames == 1:
        indices = [total // 2]
    else:
        indices = [
            round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)
        ]
    out: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        enc = _encode_frame_b64(frame, max_side, jpeg_quality)
        if enc is not None:
            out.append(enc)
    cap.release()
    return out


def _parse_planner_json(text: str) -> Dict[str, Any]:
    payload = _strip_code_fence(_strip_thinking(text))
    m = re.search(r"\{.*\}", payload, re.DOTALL)
    if not m:
        raise RuntimeError(f"Planner output has no JSON object: {payload[:300]!r}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Planner JSON is not an object: {type(obj).__name__}")
    fo = obj.get("focus_objects", [])
    fe = obj.get("focus_events", [])
    if not isinstance(fo, list) or not isinstance(fe, list):
        raise RuntimeError("Planner JSON missing focus_objects / focus_events lists.")
    return {
        "focus_objects": [str(x) for x in fo if x],
        "focus_events": [str(x) for x in fe if x],
        "reasoning_brief": str(obj.get("reasoning_brief", "")),
    }


def _call_planner_gemini(
    client,
    model: str,
    prompt: str,
    frames_b64: List[str],
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    contents: List[Any] = [prompt]
    for fb64 in frames_b64:
        contents.append(
            genai_types.Part.from_bytes(
                data=base64.b64decode(fb64),
                mime_type="image/jpeg",
            )
        )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=int(max_new_tokens),
                    temperature=0.0,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        text, finish_reason = _gemini_extract_text(resp)
        if finish_reason == "MAX_TOKENS":
            raise _NonRetryableError(
                f"Planner (Gemini) hit MAX_TOKENS (cap={max_new_tokens}). "
                f"Bump --max_new_tokens_planner."
            )
        if text:
            return text
        last_err = RuntimeError(f"Gemini planner empty text (finish_reason={finish_reason})")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini planner failed after {max_retries} retries: {last_err}")


def _call_planner_openrouter(
    client,
    model: str,
    prompt: str,
    frames_b64: List[str],
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
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
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                timeout=300,
                extra_body=extra_body or None,
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        choices = getattr(resp, "choices", None) or []
        if not choices:
            last_err = RuntimeError("OpenRouter planner: no choices in response")
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
                f"OpenRouter planner hit token cap ({max_new_tokens}). "
                f"Bump --max_new_tokens_planner."
            )
        text = _strip_code_fence(_strip_thinking(content))
        if text:
            return text
        last_err = RuntimeError(f"OpenRouter planner empty text (finish_reason={finish_reason})")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter planner failed after {max_retries} retries: {last_err}")


def plan_focus_cached(
    *,
    video_path: str,
    video_duration_sec: float,
    video_name: str,
    question: str,
    states_cache_dir: Path,
    backend: str,
    model: str,
    gemini_client,
    openrouter_client,
    max_frames: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    cache_path = planner_cache_path(states_cache_dir, video_name, question, model)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    n_frames = planner_n_frames(video_duration_sec, max_frames=max_frames)
    # At 64 frames, full-res planner prompts exceed OpenRouter's 19k free-tier cap.
    effective_max_side = 384 if n_frames == 64 else frame_max_side
    frames_b64 = _sample_uniform_frames_b64(
        video_path, n_frames, effective_max_side, frame_jpeg_quality
    )
    prompt = PLANNER_PROMPT_TEMPLATE.format(n_frames=len(frames_b64), question=question)
    if backend == "gemini":
        raw = _call_planner_gemini(
            gemini_client, model, prompt, frames_b64, max_new_tokens
        )
    elif backend == "openrouter":
        raw = _call_planner_openrouter(
            openrouter_client, model, prompt, frames_b64, max_new_tokens
        )
    else:
        raise ValueError(f"Unknown planner backend: {backend!r}")
    parsed = _parse_planner_json(raw)
    record = {
        "question": question,
        "n_frames": len(frames_b64),
        "backend": backend,
        "model": model,
        **parsed,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cache_path, record)
    return record


def _focus_hint_block(focus_objects: List[str], focus_events: List[str]) -> str:
    fo = ", ".join(focus_objects) if focus_objects else "(none specified)"
    fe = ", ".join(focus_events) if focus_events else "(none specified)"
    return (
        "\nPLANNER HINT (use this to decide what to describe in more detail):\n"
        f"  Focus objects:      {fo}\n"
        f"  Focus event types:  {fe}\n"
        "If any of the above appear in this chunk, describe them with extra\n"
        "detail (exact colors/attributes, what they do, any outcome). Do NOT\n"
        "ignore other events — be especially attentive to the focus items.\n"
    )


def stage_a1_extract_chunk_biased(
    client,
    model: str,
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
    focus_objects: List[str],
    focus_events: List[str],
    max_retries: int = 4,
    chunk_prompt_version: str = "v1",
) -> Dict[str, Any]:
    frames_b64 = _extract_chunk_frames_b64(
        video_path,
        start_sec=chunk_start,
        end_sec=chunk_end,
        frames_per_chunk=frames_per_chunk,
        max_side=frame_max_side,
        jpeg_quality=frame_jpeg_quality,
    )
    base_record = {
        "chunk_idx": chunk_idx,
        "chunk_start_sec": chunk_start,
        "chunk_end_sec": chunk_end,
        "chunk_start_mmss": _fmt_mmss(chunk_start),
        "chunk_end_mmss": _fmt_mmss(chunk_end),
    }
    if not frames_b64:
        return {**base_record, "events": [], "note": "no frames extracted"}
    prompt = get_chunk_prompt(chunk_prompt_version).format(
        chunk_start=_fmt_mmss(chunk_start),
        chunk_end=_fmt_mmss(chunk_end),
        chunk_idx=chunk_idx,
        chunk_duration_sec=chunk_end - chunk_start,
    ) + _focus_hint_block(focus_objects, focus_events)
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
    last_err: Optional[Exception] = None
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
            last_err = RuntimeError(f"chunk {chunk_idx}: no choices in response")
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
        finish_reason = str(getattr(choices[0], "finish_reason", None) or "UNKNOWN").upper()
        if finish_reason == "LENGTH":
            raise _NonRetryableError(
                f"Biased chunk {chunk_idx} Stage A1 hit token cap ({max_new_tokens}). "
                f"Bump --max_new_tokens_chunk."
            )
        if not text:
            last_err = RuntimeError(
                f"chunk {chunk_idx}: empty content (finish_reason={finish_reason})"
            )
            time.sleep(2 ** attempt)
            continue
        try:
            events = _parse_json_list_or_raise(
                text, f"Biased chunk {chunk_idx} A1", finish_reason
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return {**base_record, "events": events}
    raise RuntimeError(
        f"Biased Stage A1 chunk {chunk_idx} failed after {max_retries} retries: {last_err}"
    )


def aggregate_biased_chunks(
    *,
    chunks: List[Dict[str, Any]],
    aggregator_backend: str,
    aggregator_model: str,
    gemini_client,
    openrouter_client,
    max_new_tokens: int,
) -> str:
    if aggregator_backend == "concat":
        return aggregator_concat(chunks)
    if aggregator_backend == "gemini":
        return aggregator_flash(
            gemini_client, aggregator_model, chunks, max_new_tokens=max_new_tokens
        )
    if aggregator_backend == "openrouter":
        from benchmark_sub_with_states import aggregator_openrouter
        if openrouter_client is None:
            raise RuntimeError(
                "aggregator_backend=openrouter requires an OpenRouter client."
            )
        return aggregator_openrouter(
            openrouter_client, aggregator_model, chunks, max_new_tokens=max_new_tokens
        )
    raise ValueError(f"Unknown aggregator_backend: {aggregator_backend!r}")


def build_biased_timeline_cached(
    *,
    video_path: str,
    video_name: str,
    video_duration_sec: float,
    chunk_ranges: List[Tuple[float, float]],
    question: str,
    states_cache_dir: Path,
    # planner
    planner_backend: str,
    planner_model: str,
    max_planner_frames: int,
    max_new_tokens_planner: int,
    # extractor (biased Stage A1)
    extractor_client,
    extractor_model: str,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_chunk: int,
    max_concurrency_chunks: int,
    # aggregator
    aggregator_backend: str,
    aggregator_model: str,
    max_new_tokens_aggregator: int,
    # clients
    gemini_client,
    openrouter_client,
    # prompt
    chunk_prompt_version: str = "v1",
    # progress
    on_phase: Optional[Callable[[str], None]] = None,
) -> str:
    """Run the full per-question planner pipeline and return the annotated
    JSON timeline as a string. Artifacts are cached per (video, question,
    planner_model).

    ``on_phase(phase_name)`` is invoked right before each phase runs, with
    one of: ``"cached"``, ``"plan"``, ``"chunks"``, ``"aggregate"``. Phases
    that are served from cache are not announced (the caller won't see
    ``"plan"`` if the planner JSON is already on disk). Callers use it to
    update a tqdm postfix so progress is visible."""
    notify = on_phase or (lambda _: None)

    timeline_path = biased_timeline_cache_path(
        states_cache_dir, video_name, question, planner_model, aggregator_backend
    )
    if timeline_path.exists():
        notify("cached")
        return annotate_timeline_with_subtype_order(
            timeline_path.read_text(encoding="utf-8")
        )

    planner_cache_exists = planner_cache_path(
        states_cache_dir, video_name, question, planner_model
    ).exists()
    if not planner_cache_exists:
        notify("plan")
    focus = plan_focus_cached(
        video_path=video_path,
        video_duration_sec=video_duration_sec,
        video_name=video_name,
        question=question,
        states_cache_dir=states_cache_dir,
        backend=planner_backend,
        model=planner_model,
        gemini_client=gemini_client,
        openrouter_client=openrouter_client,
        max_frames=max_planner_frames,
        frame_max_side=frame_max_side,
        frame_jpeg_quality=frame_jpeg_quality,
        max_new_tokens=max_new_tokens_planner,
    )
    focus_objects = focus.get("focus_objects", [])
    focus_events = focus.get("focus_events", [])

    chunks_path = biased_chunks_cache_path(
        states_cache_dir, video_name, question, planner_model
    )
    chunks_cache = _load_json(chunks_path)
    if not (
        isinstance(chunks_cache, dict)
        and isinstance(chunks_cache.get("chunks"), list)
        and len(chunks_cache["chunks"]) == len(chunk_ranges)
    ):
        n = len(chunk_ranges)
        notify(f"chunks 0/{n}")
        done_counter = {"n": 0}

        def run_chunk(i: int) -> Dict[str, Any]:
            cs, ce = chunk_ranges[i]
            return stage_a1_extract_chunk_biased(
                extractor_client,
                extractor_model,
                video_path,
                chunk_idx=i,
                chunk_start=cs,
                chunk_end=ce,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens=max_new_tokens_chunk,
                focus_objects=focus_objects,
                focus_events=focus_events,
                chunk_prompt_version=chunk_prompt_version,
            )

        workers = max(1, min(max_concurrency_chunks, n))
        results: List[Optional[Dict[str, Any]]] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_i = {pool.submit(run_chunk, i): i for i in range(n)}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                results[i] = fut.result()
                done_counter["n"] += 1
                notify(f"chunks {done_counter['n']}/{n}")
        chunks_cache = {
            "video_name": video_name,
            "video_duration_sec": video_duration_sec,
            "question": question,
            "planner_model": planner_model,
            "focus_objects": focus_objects,
            "focus_events": focus_events,
            "chunks": results,
        }
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(chunks_path, chunks_cache)

    notify("aggregate")
    timeline = aggregate_biased_chunks(
        chunks=chunks_cache["chunks"],
        aggregator_backend=aggregator_backend,
        aggregator_model=aggregator_model,
        gemini_client=gemini_client,
        openrouter_client=openrouter_client,
        max_new_tokens=max_new_tokens_aggregator,
    )
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(timeline_path, timeline)
    return annotate_timeline_with_subtype_order(timeline)
