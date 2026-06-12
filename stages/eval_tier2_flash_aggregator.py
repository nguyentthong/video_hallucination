#!/usr/bin/env python3
"""
eval_tier2_flash_aggregator.py
-------------------------------
Tier-2, variant B: keep the open-source chunker, swap BOTH the aggregator
AND Stage B to gemini-3-flash-preview via the native google-genai API.

Rationale: on benchmark_small, tier-2 with an open-source aggregator scored
55%. Inspection says the per-chunk extractions look fine, so the failure is
in cross-chunk aggregation / ordinal indexing. Flash is strong at structured
reasoning, so we route Stage A2 (text) and Stage B (video QA with timeline)
both through Flash. The expensive stage at 300k scale is still the per-video
chunker (open-source); each Flash call here is cheap — one aggregation call
per video plus N yes/no questions.

Pipeline:
  - Stage A1 chunks are READ from an existing tier-2 cache (chunks.json).
    Not re-run. If chunks.json is missing for a video, you'll be told to
    run eval_tier2_chunked_aggregation.py first.
  - Stage A2 (aggregator): gemini-3-flash-preview, text-only, merges chunks
    into a global timeline.
  - Stage B (QA): gemini-3-flash-preview, video-grounded (Files API upload)
    + the aggregated timeline.

A separate cache dir is used so the Qwen-aggregator baseline (stage_a.txt
from tier-2) is preserved untouched for A/B comparison.

Setup:
    uv sync --group gemini
    echo 'GOOGLE_API_KEY=...' >> .env

Run:
    python eval_tier2_flash_aggregator.py \\
        --questions_dir benchmark_small \\
        --video_dir raw_data \\
        --chunks_cache_dir outputs_tier2/cache \\
        --cache_dir        outputs_tier2_flash_agg/cache \\
        --output_json      outputs_tier2_flash_agg/results.json \\
        --max_concurrency_questions 4
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

try:
    import google.genai as genai
    from google.genai import types as genai_types
except Exception as exc:
    raise RuntimeError(
        "Failed to import google-genai. Install with: uv sync --group gemini\n"
        f"Error: {exc}"
    ) from exc

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc


# OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_URL = "http://worker-74:8000/v1"

DEFAULT_AGGREGATOR_MODEL = "gemini-3-flash-preview"
DEFAULT_STAGE_B_MODEL = "gemini-3-flash-preview"
# Stage A1 chunker runs on Qwen thinking via OpenRouter when chunks.json is missing.
DEFAULT_CHUNK_VLM_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"


AGGREGATOR_PROMPT_TEMPLATE = """You are given {n_chunks} chunk-level
extractions from a long video. Each chunk is a contiguous slice covering a
specific global time range, and each extraction lists only the events
observed inside that chunk.

Your task: MERGE these chunk-level extractions into a single GLOBAL
chronological event timeline with correct ordinal indices.

Schema of the output:
  [
    {{
      "event_index":  <int>,              # 1-based GLOBAL index
      "event_type":   <str>,              # "game" / "round" / "obstacle" / ...
      "start_time":   "MM:SS",            # global
      "end_time":     "MM:SS",            # global
      "description":  <str>,
      "participants": [<str>, ...],
      "outcome":      <str>,
      "sub_events": [
        {{
          "sub_index":   <int>,           # 1-based within THIS parent event
          "actor":       <str>,
          "object":      <str>,
          "description": <str>,
          "outcome":     <str>
        }}, ...
      ]
    }},
    ...
  ]

RULES:
  - Chunks were produced independently; the same real-world event may
    appear PARTIALLY across multiple chunks (e.g. a "game" starts in
    chunk 5 and ends in chunk 7). Use `continuation_hint` in each chunk
    item to merge these into ONE event with combined start/end times.
  - Assign event_index strictly in chronological order (by start_time).
  - Sub-events (individual throws, rings, tire tosses, etc.) that belong
    INSIDE a parent event should be collected into that event's
    `sub_events` list with 1-based sub_index in the order they happen.
  - Preserve factual details verbatim from the chunk extractions. Do NOT
    invent events, actors, or outcomes not mentioned in any chunk.
  - If two chunks disagree about an event's outcome, prefer the chunk
    whose time range more tightly covers the event's end.
  - Participants' names: keep them consistent across the timeline
    ("blue-shirt man", "red-shirt man", ...).
  - Output STRICT JSON only: one top-level list. No prose, no markdown
    fences, no commentary before or after the JSON.

CHUNK EXTRACTIONS (chronological):
{chunks_block}
"""


ORDINAL_SEMANTICS_BLOCK = """ORDINAL SEMANTICS (READ CAREFULLY):
  - "Nth X" means the Nth occurrence of X in CHRONOLOGICAL ORDER.
    It NEVER means "last", "final", "most recent", or "latest".
  - The final X is the event whose `subtype_order` equals its
    `subtype_total`. Do not treat "Nth" as a synonym for that.
  - Each timeline event carries `subtype_order` (1-based chronological
    rank within its event_type) and `subtype_total` (how many events of
    that type exist in the whole video). `subtype_counts` at the top of
    the timeline lists totals per event_type.
  - To resolve "the Nth X", find the event with event_type ~ X and
    subtype_order == N. If the question's category term doesn't match
    any event_type verbatim, consider which event_types a viewer would
    group together under that term (e.g. "game" may cover both
    "ring_toss" and "ball_throw") and use their combined chronological
    order via `event_index`.
  - If N exceeds the relevant subtype_total, the referent does not exist
    in this video — answer accordingly."""


STAGE_B_PROMPT_TEMPLATE = """You are answering a yes/no question about a
video. A chronological event timeline for this video has already been
extracted and is provided below. Use the timeline as the authoritative
source of ordinal information ("first", "second", "Nth"), and cross-check
against the video.

EVENT TIMELINE (JSON):
{timeline}

{ordinal_semantics}

RULES:
  - Only use information you can verify from the timeline and/or video.
  - Start your answer with exactly "Yes" or "No".
  - Then give one short sentence of justification, citing the relevant
    event_index / sub_index from the timeline when the question is ordinal.

QUESTION: {question}
"""


STAGE_B_COT_PROMPT_TEMPLATE = """Answer a yes/no video question using this
pre-extracted event timeline.

EVENT TIMELINE (JSON):
{timeline}

{ordinal_semantics}

Think step by step: identify the relevant events, explain briefly how they
answer the question, then commit.

Output STRICT JSON only (no prose, no fences):
{{
  "evidence_spans": [
    {{"event_index": <int>, "sub_index": <int or null>, "note": "<short>"}}
  ],
  "reasoning": "<1-2 sentences>",
  "answer": "Yes" | "No"
}}

QUESTION: {question}
"""


STAGE_B_VL_COT_PROMPT_TEMPLATE = """Answer a yes/no video question.

You are given TWO sources:
  1. A pre-extracted chronological event timeline (JSON).
  2. A small set of video frames retrieved from the full video as visually
     relevant to this question. Each frame comes with its global timestamp
     (in MM:SS). The frames are ordered chronologically, not by relevance.

Use BOTH. The timeline gives structure and ordinals (event_index,
subtype_order, subtype_total). The frames give raw pixel evidence that
may contradict the timeline.

TIMELINE (JSON):
{timeline}

{ordinal_semantics}

FRAMES OVERRIDE TIMELINE ON OBSERVABLE FACTS.
The timeline is written by an upstream VLM from short clips and is often
wrong about observable details. When the question is about something a
viewer could SEE in a single frame or short frame span — trust the frames,
not the timeline text. Observable facts include, but are not limited to:
  - hit / miss / land / bounce outcomes (ball on target, ring on pole, ...)
  - object colors, counts, positions
  - who acted first / who is visible / who is celebrating / who is holding X
  - pass / fail / success / failure of a physical action
If the timeline says "miss" or "unknown" but a frame clearly shows the
ball/ring on the target, answer based on the frame. Same in reverse.

For any observable claim in your answer, you MUST cite at least one
specific frame timestamp in evidence_spans (frame_ts="MM:SS"), not just
an event_index. Use event_index/sub_index for structural/ordinal claims
(which event is the Nth, how many events of a type, etc.); use frame_ts
for what happens *inside* a frame.

Think step by step:
  1. What kind of claim does the question make? Structural/ordinal
     (use timeline) or observable (use frames)?
  2. Scan the frames for direct evidence of the observable fact.
  3. If timeline and frames disagree on an observable fact, trust frames.

Output STRICT JSON only (no prose, no fences):
{{
  "evidence_spans": [
    {{"event_index": <int or null>, "sub_index": <int or null>,
      "frame_ts": "<MM:SS or null>", "note": "<short>"}}
  ],
  "reasoning": "<1-2 sentences>",
  "answer": "Yes" | "No"
}}

QUESTION: {question}
"""

# Bump when STAGE_B_VL_COT_PROMPT_TEMPLATE changes in a way that would
# make old cached predictions misleading. Feeds into stage_b_cache_key().
# v2: frames-override patch — the best-performing config (65.89% overall).
# Ordinal pinning + frame grounding (v3) were tried and reverted — they
# regressed accuracy; see run_note.txt.
STAGE_B_VL_COT_PROMPT_VERSION = "v2"


STAGE_B_FEEDBACK_PROMPT_TEMPLATE = """Answer a yes/no video question using
this pre-extracted event timeline. If the timeline is insufficient, say so
and suggest which global time spans to re-examine.

EVENT TIMELINE (JSON):
{timeline}

{ordinal_semantics}

Output STRICT JSON only (no prose, no fences):
{{
  "status": "sufficient" | "insufficient",
  "answer": "Yes" | "No" | null,
  "missing": "<what specific info is missing, or empty>",
  "time_spans": [{{"start_sec": <number>, "end_sec": <number>}}],
  "rationale": "<1-2 sentences>"
}}

Rules:
  - If status="sufficient": fill "answer", set "time_spans"=[] and
    "missing"="".
  - If status="insufficient": set "answer"=null and list 1-3 short
    (<=30s each) global time spans where the missing info likely sits.

QUESTION: {question}
"""


STAGE_B_SUPPLEMENT_PROMPT_TEMPLATE = """Answer a yes/no video question.
Below is the pre-extracted timeline plus question-specific supplementary
evidence re-extracted to address an earlier information gap.

TIMELINE (JSON):
{timeline}

SUPPLEMENTARY EVIDENCE (JSON):
{supplement}

{ordinal_semantics}

RULES:
  - Only use information you can verify in the timeline or supplement.
  - Start with exactly "Yes" or "No".
  - Then give one short sentence of justification.

QUESTION: {question}
"""


REEXTRACT_CHUNK_PROMPT_TEMPLATE = """You are viewing a SHORT segment of a
long video (chunk {chunk_idx}, {chunk_start}-{chunk_end}, duration
{chunk_duration_sec:.1f}s).

A downstream Q&A system is trying to answer:
  "{question}"

It reported this information gap:
  "{missing}"

Re-describe every detail in THIS CHUNK that could resolve that gap.

For each event, emit JSON with: local_start (MM:SS), local_end (MM:SS),
event_type, description, participants, actor, object, outcome,
continuation_hint ("starts_before_chunk" / "continues_after_chunk" /
"complete").

RULES:
  - No ordinal words ("first", "Nth", "final").
  - Do NOT invent events not visible in this chunk.
  - Output STRICT JSON only: a single top-level list. No prose.
"""


# ---------------------------------------------------------------------------
# Answer normalisation + benchmark loading
# ---------------------------------------------------------------------------

def extract_yes_no(text: str) -> str:
    matches = re.findall(r"\b(yes|no)\b", (text or "").strip().lower())
    return matches[-1] if matches else "unknown"


def answers_match(pred: str, gt: str) -> bool:
    p, g = extract_yes_no(pred), extract_yes_no(gt)
    if p == "unknown" or g == "unknown":
        return False
    return p == g


def qhash(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()[:12]


def load_samples(questions_dir: str, video_dir: str) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(questions_dir, "**", "*.json"), recursive=True))
    samples: List[Dict[str, Any]] = []
    for f in files:
        with open(f, encoding="utf-8") as fin:
            data = json.load(fin)
        data["video_path"] = os.path.join(video_dir, data["video_name"])
        data["example_path"] = os.path.abspath(f)
        samples.append(data)
    return samples


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _build_gemini_client(api_key_env: str) -> genai.Client:
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")
    return genai.Client(api_key=api_key)


def _build_openrouter_client(api_key_env: str) -> OpenAI:
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Environment variable {api_key_env!r} is not set.")
    default_headers: Dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE")
    if referer:
        default_headers["HTTP-Referer"] = referer
    if title:
        default_headers["X-Title"] = title
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_URL,
        default_headers=default_headers or None,
    )


_THINK_TAG_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_TAG_RE.sub("", text or "").strip()


def upload_video(
    client: genai.Client,
    video_path: str,
    poll_interval: float = 2.0,
) -> Any:
    video_file = client.files.upload(file=video_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(poll_interval)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name != "ACTIVE":
        raise RuntimeError(
            f"Files API upload ended in state {video_file.state.name}: {video_path}"
        )
    return video_file


_SAFETY_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "IMAGE_SAFETY",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
    "LANGUAGE",
}


def _gemini_extract_text(resp: Any) -> Tuple[str, str]:
    """Returns (text, finish_reason). If the whole prompt was blocked before
    a candidate was generated, finish_reason is 'PROMPT_BLOCKED:<reason>' so
    the caller can distinguish it from a normal UNKNOWN."""
    text = getattr(resp, "text", None)
    candidates = getattr(resp, "candidates", None) or []
    if not text and candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        text = "".join(getattr(p, "text", "") or "" for p in parts)
    finish_reason = "UNKNOWN"
    if candidates:
        fr = getattr(candidates[0], "finish_reason", None)
        if fr is not None:
            finish_reason = getattr(fr, "name", None) or str(fr)
    # When the whole prompt is blocked, candidates is empty AND
    # prompt_feedback.block_reason is set. Surface that instead of UNKNOWN.
    pf = getattr(resp, "prompt_feedback", None)
    br = getattr(pf, "block_reason", None) if pf is not None else None
    if br is not None and (not candidates or not (text or "").strip()):
        br_name = getattr(br, "name", None) or str(br)
        if br_name and br_name != "BLOCK_REASON_UNSPECIFIED":
            finish_reason = f"PROMPT_BLOCKED:{br_name}"
    return (text or "").strip(), finish_reason


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _permissive_safety_settings() -> List[genai_types.SafetySetting]:
    """All safety categories set to BLOCK_NONE. We're analyzing user-supplied
    benchmark videos, not generating user-facing content, so the default
    thresholds produce spurious blocks on benign footage (sports tackles,
    stock news clips, etc.). Gemini still applies a service-level floor for
    illegal content."""
    categories = [
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
    ]
    return [
        genai_types.SafetySetting(
            category=c,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        )
        for c in categories
    ]


def _is_safety_finish(finish_reason: str) -> bool:
    if finish_reason.startswith("PROMPT_BLOCKED:"):
        return True
    return finish_reason in _SAFETY_FINISH_REASONS


def _parse_json_list_or_raise(text: str, stage: str, finish_reason: str) -> List[Any]:
    payload = _strip_code_fence(text)
    try:
        parsed = json.loads(payload)
    except Exception as e:
        raise RuntimeError(
            f"{stage} output is not valid JSON "
            f"(finish_reason={finish_reason}, len={len(text)} chars): {e}\n"
            f"--- raw (first 500 chars) ---\n{text[:500]}"
        ) from e
    if not isinstance(parsed, list):
        raise RuntimeError(
            f"{stage} JSON is not a top-level list (got {type(parsed).__name__})."
        )
    return parsed


class _NonRetryableError(RuntimeError):
    """Raised for deterministic failures (e.g. token-cap hit) so the retry
    loop aborts immediately instead of wasting attempts."""


def _assign_round_indices(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Walk events in order and assign a 1-based ``round_index`` to each,
    plus forward-fill ``segment_label_resolved``.

    Driven by the ``segment_boundary`` field emitted by the v6 chunk prompt:
      - "start"   -> open a new segment (increment round_index, unless this
                     is the first event and round_index is still unset).
      - "end"     -> close the current segment; the next "start" will open
                     a new one.
      - "middle"  -> stay in the current segment.
      - "unknown" / missing -> stay in the current segment (treated as
                               "middle"); do NOT silently open a new round
                               on ambiguity.

    Robustness rules:
      - Two adjacent "start" events with no intervening non-start event in
        between are collapsed: only the first bumps the counter, so a noisy
        extractor cannot silently multiply the round count.
      - If no event ever carries a ``segment_boundary`` hint, every event
        is assigned round_index=1 and segment_label_resolved is left blank
        — Stage B still gets a valid (if uninformative) field.

    Returns ``{"rounds": [{"round_index": int, "label": str,
    "event_index_range": [first, last]}, ...]}`` for Stage B inspection.
    """
    if not events:
        return {"rounds": []}

    any_boundary = any(
        isinstance(ev, dict) and ev.get("segment_boundary")
        for ev in events
    )
    if not any_boundary:
        for ev in events:
            if isinstance(ev, dict):
                ev["round_index"] = 1
                ev.setdefault("segment_label_resolved", "")
        return {"rounds": []}

    round_index = 0
    current_label = ""
    prev_was_start = False
    round_summaries: List[Dict[str, Any]] = []

    def _ensure_round_summary(idx: int, label: str, event_idx: Any) -> None:
        if not round_summaries or round_summaries[-1]["round_index"] != idx:
            round_summaries.append({
                "round_index": idx,
                "label": label,
                "event_index_range": [event_idx, event_idx],
            })
        else:
            round_summaries[-1]["event_index_range"][1] = event_idx
            if not round_summaries[-1]["label"] and label:
                round_summaries[-1]["label"] = label

    for ev in events:
        if not isinstance(ev, dict):
            continue
        boundary = str(ev.get("segment_boundary") or "").strip().lower()
        raw_label = ev.get("segment_label")
        label = str(raw_label or "").strip()

        if boundary == "start":
            if prev_was_start:
                # Collapse runs of consecutive "start" markers — the first
                # one already opened the segment. Still allow a fresh
                # label to upgrade an earlier empty one inside the run.
                if label and not current_label:
                    current_label = label
            else:
                round_index += 1
                # Reset the label at each new segment so we do not
                # inherit the previous round's label when the extractor
                # did not supply a new one.
                current_label = label
            prev_was_start = True
        else:
            prev_was_start = False
            # Within a segment, allow later events to fill in a label
            # the opening event was missing, but never overwrite a
            # label already set for this round.
            if label and not current_label:
                current_label = label

        # Ensure we never emit round_index=0 — clamp to 1 on the first
        # event even if the extractor didn't mark it "start".
        effective_round = round_index if round_index >= 1 else 1
        ev["round_index"] = effective_round
        ev["segment_label_resolved"] = current_label
        _ensure_round_summary(effective_round, current_label, ev.get("event_index"))

    return {"rounds": round_summaries}


def annotate_timeline_with_subtype_order(timeline_text: str) -> str:
    """Inject per-subtype ordinal bookkeeping into the aggregator timeline.

    For each event, add:
      - subtype_order: 1-based position among events sharing the same
        event_type (chronological).
      - subtype_total: total number of events with that event_type.
      - round_index: 1-based segment/round position derived from the v6
        chunk prompt's ``segment_boundary`` markers. Events with no
        boundary information all land in round 1.
      - segment_label_resolved: forward-filled label from the most recent
        non-empty ``segment_label`` (empty string when none was ever
        provided).

    Also wrap the events in a top-level object that carries
    ``subtype_counts`` and ``rounds``, so Stage B can sanity-check ordinal
    queries and round-indexed queries without re-scanning the timeline.

    If the timeline can't be parsed as a JSON list, return it unchanged so
    Stage B still sees something usable.
    """
    try:
        events = json.loads(_strip_code_fence(timeline_text))
    except Exception:
        return timeline_text
    if not isinstance(events, list):
        return timeline_text

    totals: Counter = Counter()
    for ev in events:
        if isinstance(ev, dict):
            totals[str(ev.get("event_type") or "unknown")] += 1

    seen: Counter = Counter()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        label = str(ev.get("event_type") or "unknown")
        seen[label] += 1
        ev["subtype_order"] = seen[label]
        ev["subtype_total"] = totals[label]

    round_summary = _assign_round_indices(events)

    wrapped: Dict[str, Any] = {
        "subtype_counts": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
        "events": events,
    }
    if round_summary.get("rounds"):
        wrapped["rounds"] = round_summary["rounds"]
    return json.dumps(wrapped, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Stage A2: Flash text-only aggregator
# ---------------------------------------------------------------------------

def aggregator_flash(
    client: genai.Client,
    model: str,
    chunks: List[Dict[str, Any]],
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    chunks_block = json.dumps(chunks, ensure_ascii=False, indent=2)
    prompt = AGGREGATOR_PROMPT_TEMPLATE.format(
        n_chunks=len(chunks),
        chunks_block=chunks_block,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        # Outer try: transient transport errors — these get retried.
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

        # Deterministic failures — do not retry.
        if finish_reason == "MAX_TOKENS":
            raise _NonRetryableError(
                f"Aggregator hit MAX_TOKENS (output cap = {max_new_tokens} tokens). "
                f"Bump --max_new_tokens_aggregator (Gemini 3 Flash supports up to 65536). "
                f"Returned {len(text)} chars before truncation."
            )
        if finish_reason not in ("STOP", "UNKNOWN"):
            raise _NonRetryableError(
                f"Aggregator finish_reason={finish_reason} (likely safety or filter). "
                f"Returned {len(text)} chars."
            )

        # JSON parse failure — may be transient (rare), retry cheaply.
        try:
            _parse_json_list_or_raise(text, "Flash aggregator", finish_reason)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        return _strip_code_fence(text)

    raise RuntimeError(
        f"Flash aggregator failed after {max_retries} retries: {last_err}"
    )


def aggregator_concat(chunks: List[Dict[str, Any]]) -> str:
    """Deterministic aggregator: emit one global event per chunk event, in
    chunk order, with 1-based ``event_index``. No LLM call.

    Per-chunk events already carry global ``local_start``/``local_end``
    (MM:SS) and the chunker emits them in local temporal order, so iterating
    chunks sorted by ``chunk_idx`` and concatenating produces a globally
    ordered timeline. ``sub_events`` is defaulted to ``[]`` so downstream
    Stage B sees the same schema it gets from the Flash aggregator."""
    ordered = sorted(chunks, key=lambda c: int(c.get("chunk_idx", 0)))
    events: List[Dict[str, Any]] = []
    next_index = 1
    for chunk in ordered:
        for ev in chunk.get("events", []) or []:
            if not isinstance(ev, dict):
                continue
            out = dict(ev)
            out["event_index"] = next_index
            next_index += 1
            if "local_start" in out and "start_time" not in out:
                out["start_time"] = out.pop("local_start")
            if "local_end" in out and "end_time" not in out:
                out["end_time"] = out.pop("local_end")
            out.setdefault("sub_events", [])
            events.append(out)
    return json.dumps(events, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Identity linking (post-aggregation): one Gemini Flash call per video that
# groups participant descriptors by inferred identity. Output is cached as
# ``aliases.txt`` next to ``stage_a_concat.txt`` and prepended to every
# per-question Context downstream so the answerer doesn't have to re-derive
# coreference on each question.
# ---------------------------------------------------------------------------

_IDENTITY_LINK_GENERIC_DESCRIPTORS = {
    "students", "audience", "crowd", "spectators", "people", "person",
    "team", "teams", "players", "contestants", "participants", "host",
    "hosts", "everyone", "others", "group", "kids", "children",
    "speakers", "panelists",
}


def _extract_unique_descriptors(timeline_str: str) -> List[str]:
    """Pull every distinct ``actor`` and ``participant`` string out of an
    aggregated timeline (JSON list-of-events). Strips trivially generic
    crowd labels like ``"students"`` / ``"audience"``."""
    try:
        events = json.loads(timeline_str)
    except (ValueError, TypeError):
        return []
    if isinstance(events, dict):
        events = events.get("events", [])
    if not isinstance(events, list):
        return []
    seen: Dict[str, None] = {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        for key in ("actor",):
            v = ev.get(key)
            if isinstance(v, str):
                k = v.strip()
                if k and k.lower() not in _IDENTITY_LINK_GENERIC_DESCRIPTORS:
                    seen.setdefault(k, None)
        for v in ev.get("participants", []) or []:
            if isinstance(v, str):
                k = v.strip()
                if k and k.lower() not in _IDENTITY_LINK_GENERIC_DESCRIPTORS:
                    seen.setdefault(k, None)
    return list(seen)


_IDENTITY_LINK_PROMPT_TEMPLATE = """The following are person descriptors
extracted from one video's events. The same person is often described
slightly differently across chunks (different lighting, camera angle, or
garment-noun swap). Group descriptors that you are confident refer to the
same person. When uncertain, leave a descriptor in its OWN group.

Heuristics for GROUPING (apply jointly):
  - Color near-synonyms within the same garment family belong together:
      tan ~ white ~ cream ~ beige ~ khaki ~ off-white ~ light
      red ~ maroon ~ crimson ~ burgundy
      blue ~ navy ~ teal ~ azure
      green ~ olive ~ teal ~ lime
      grey ~ silver ~ ash ~ slate
      orange ~ coral ~ salmon
      brown ~ tan ~ chestnut
  - Garment-noun swaps for the same garment, when the dominant color
    matches: shirt <-> hoodie <-> sweater <-> jacket <-> top <-> blazer.
  - Compound descriptors that share a dominant color belong together
    when the role is the same: e.g. "tan-shirt student" / "tan-plaid-
    shirt student" / "tan-jacket student" are likely one student.
  - Role hints (professor / lecturer / host / contestant / student)
    where one person recurs throughout a video with shifting garment
    descriptors. The lecturer in a single classroom video is one
    person even if called "grey-suit man" once and "tan-jacket
    professor" later.
  - SUIT VARIANTS for the same role: in a two-person game-show or
    interview setting, all suit-wearing descriptors for the HOST role
    belong together even when the color word shifts chunk-to-chunk
    (e.g. "suit man" / "black-suit man" / "dark-suit man" / "grey-suit
    man" / "blue-suit man" all refer to the same suited host when there
    is only one suited person in the video). Apply the same logic for
    the GUEST role.

Heuristics for KEEPING descriptors APART:
  - Genuinely different dominant colors AND different roles -> different
    people (a "red-shirt man" and a "blue-shirt man" with the same role
    may still be the same person if only one such person is in the video;
    use your judgment).
  - Same color but different role tags ("white-shirt student" vs
    "white-shirt professor") -> probably two people, unless the video
    only has one of that role.
  - When you have NO basis to merge, leave the descriptor as a singleton.

SCREEN POSITION — after grouping, determine where each person appears
on screen (their typical camera-left / camera-right / center position).
Use any of these cues from the descriptor text or video context:
  - Explicit spatial words in the descriptors ("left-side man", etc.).
  - Role inference: in a two-player game show, the HOST is typically on
    the RIGHT side of the frame; the GUEST is typically on the LEFT.
    In a classroom, the LECTURER/PROFESSOR is typically at the front
    (CENTER or RIGHT of frame); STUDENTS are at their seats (LEFT,
    CENTER, RIGHT varies).
  - If the video has only two main people and you can assign one group
    as "left" and one as "right" with reasonable confidence, do so.
  - When genuinely uncertain, use "unknown".

Descriptors:
{descriptors_block}

Output ONLY a JSON object with this exact shape, no markdown fences,
no commentary:

  {{"groups": [
      {{"label": "<short identity label, e.g. 'host in suit'>",
        "descriptors": ["<descriptor>", "<descriptor>", ...],
        "position": "left" | "right" | "center" | "unknown"}},
      ...
  ]}}

Every input descriptor must appear in exactly one group. Singletons are
fine (groups of size 1). Do not invent descriptors that are not in the
input list."""


def identity_link_via_flash(
    client: genai.Client,
    model: str,
    timeline_str: str,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run a single Gemini Flash call to group person descriptors by inferred
    identity. Returns ``{"groups": [{"label": ..., "descriptors": [...]}, ...]}``
    or ``None`` if the call fails or there are no descriptors to group."""
    descriptors = _extract_unique_descriptors(timeline_str)
    if len(descriptors) < 2:
        # Nothing to merge.
        return {"groups": [{"label": d, "descriptors": [d]} for d in descriptors]}

    descriptors_block = "\n".join(f"  - {d}" for d in descriptors)
    prompt = _IDENTITY_LINK_PROMPT_TEMPLATE.format(descriptors_block=descriptors_block)

    last_err: Exception | None = None
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
        if finish_reason == "MAX_TOKENS":
            last_err = RuntimeError(
                f"identity_link hit MAX_TOKENS at {max_new_tokens}; bump --max_new_tokens_identity_link."
            )
            break
        if finish_reason not in ("STOP", "UNKNOWN"):
            last_err = RuntimeError(f"identity_link finish_reason={finish_reason}")
            break

        cleaned = _strip_code_fence(text)
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        if not isinstance(parsed, dict) or "groups" not in parsed:
            last_err = RuntimeError("identity_link response missing 'groups'")
            time.sleep(2 ** attempt)
            continue

        # Sanity-check coverage and drop obviously broken outputs.
        seen = set()
        for grp in parsed.get("groups", []):
            if not isinstance(grp, dict):
                continue
            for d in grp.get("descriptors", []) or []:
                if isinstance(d, str):
                    seen.add(d.strip())
        # Allow up to 25% missing descriptors (model may drop a few),
        # but bail if it dropped most of them.
        if descriptors and len(seen) < 0.5 * len(descriptors):
            last_err = RuntimeError("identity_link covered <50% of input descriptors")
            time.sleep(2 ** attempt)
            continue

        return parsed

    # All retries failed; downstream falls back to "no alias header".
    return None


def identity_link_via_openrouter(
    client,  # openai.OpenAI
    model: str,
    timeline_str: str,
    max_new_tokens: int = 4096,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Same as identity_link_via_flash but routes through the OpenRouter
    OpenAI-compatible endpoint. ``client`` is an openai.OpenAI instance
    pointed at https://openrouter.ai/api/v1."""
    descriptors = _extract_unique_descriptors(timeline_str)
    if len(descriptors) < 2:
        return {"groups": [{"label": d, "descriptors": [d]} for d in descriptors]}

    descriptors_block = "\n".join(f"  - {d}" for d in descriptors)
    prompt = _IDENTITY_LINK_PROMPT_TEMPLATE.format(descriptors_block=descriptors_block)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=int(max_new_tokens),
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        choices = getattr(resp, "choices", None) or []
        if not choices:
            last_err = RuntimeError("identity_link_via_openrouter: no choices in response")
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
            last_err = RuntimeError(
                f"identity_link_via_openrouter hit token cap ({max_new_tokens})."
            )
            break

        cleaned = _strip_code_fence(_strip_thinking(content))
        try:
            parsed = json.loads(cleaned)
        except (ValueError, TypeError) as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue

        if not isinstance(parsed, dict) or "groups" not in parsed:
            last_err = RuntimeError("identity_link_via_openrouter response missing 'groups'")
            time.sleep(2 ** attempt)
            continue

        seen = set()
        for grp in parsed.get("groups", []):
            if not isinstance(grp, dict):
                continue
            for d in grp.get("descriptors", []) or []:
                if isinstance(d, str):
                    seen.add(d.strip())
        if descriptors and len(seen) < 0.5 * len(descriptors):
            last_err = RuntimeError("identity_link_via_openrouter covered <50% of input descriptors")
            time.sleep(2 ** attempt)
            continue

        return parsed

    return None


def format_alias_header(alias_map: Optional[Dict[str, Any]]) -> str:
    """Render the alias map as a human-readable header for the answerer
    Context.

    Multi-descriptor groups are always shown (they carry alias info).
    Singleton groups are shown only when they carry a known screen
    position (left/right/center) — useful for left/right questions even
    when the person has only one descriptor label.
    The position prefix (LEFT / RIGHT / CENTER player) is prepended when
    known so the answerer can resolve 'man on the left' questions without
    needing to inspect frames."""
    if not alias_map:
        return ""
    groups = alias_map.get("groups") or []
    if not groups:
        return ""

    _POS_PREFIX = {"left": "LEFT player", "right": "RIGHT player", "center": "CENTER player"}

    rows = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        ds = [d.strip() for d in (g.get("descriptors") or []) if isinstance(d, str) and d.strip()]
        if not ds:
            continue
        is_multi = len(ds) >= 2
        pos = (g.get("position") or "unknown").strip().lower()
        pos_known = pos in _POS_PREFIX
        if not is_multi and not pos_known:
            continue  # singleton with unknown position adds nothing
        label = (g.get("label") or "person").strip()
        pos_prefix = _POS_PREFIX.get(pos, "")
        if pos_prefix:
            display_label = f"{pos_prefix} ({label})"
        else:
            display_label = label
        if is_multi:
            joined = " / ".join(f'"{d}"' for d in ds)
            rows.append(f"  - {display_label}: {joined}")
        else:
            # singleton but position is known — show just the one descriptor
            rows.append(f"  - {display_label}: \"{ds[0]}\"")

    if not rows:
        return ""
    lines = [
        "PEOPLE IN THIS VIDEO (use screen-position labels to resolve "
        "'man on the left' / 'woman on the right' questions; descriptor "
        "variants listed for the same person are interchangeable when "
        "matching events or counting distinct people):"
    ]
    lines.extend(rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage A1: Qwen VLM per-chunk extraction (only used when chunks.json missing)
# ---------------------------------------------------------------------------

CHUNK_PROMPT_TEMPLATE = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

Describe every distinct event or notable sub-event you observe IN THIS
CHUNK ONLY, in the order they happen.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label (e.g. "game", "throw", "ring toss",
                        "tire toss", "balloon pop", "ladder climb")
  - description:        one concise sentence of what happens
  - participants:       list of short descriptors based on visible clothing
                        (e.g. ["blue-shirt man", "red-shirt man"])
  - actor:              who performed the event, if single-actor
  - object:             what object is involved (ball color, ring color,
                        tire color, balloon color, etc.) if visible
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / slipped / etc. if visible
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" — tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final". You only see ONE chunk — you cannot know global index.
  - Do NOT invent events you cannot see in this chunk.
  - Use the participants' visible clothing as their names, e.g.
    "blue-shirt man", "red-shirt man".
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk". Likewise if it was
    clearly already underway when the chunk opened, set
    continuation_hint = "starts_before_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


# Richer variant: expands description/participants/object to capture the
# detail required by ordinal/color/landmark questions that v1 was missing.
# Selected via --chunk_prompt_version=v2 in benchmark_sub_with_states.py.
CHUNK_PROMPT_TEMPLATE_V2 = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

Describe every distinct event or notable sub-event you observe IN THIS
CHUNK ONLY, in the order they happen.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label (e.g. "game", "throw", "ring toss",
                        "tire toss", "balloon pop", "ladder climb")
  - description:        one concise sentence of what happens. Name any
                        fixed structural element that is load-bearing to
                        the action (goal, net, finish line, specific
                        board section, numbered box) even when the
                        element itself is not moving. For visual state
                        changes, describe BOTH the BEFORE and AFTER
                        state (e.g. "mask changes from red to white",
                        not just "mask change").
  - participants:       list of short descriptors that disambiguate each
                        person. Combine visible clothing with at least
                        one additional feature when relevant: hair color
                        (e.g. "blonde woman"), gender, approximate age,
                        accessories (glasses, hat), and spatial position
                        ("on the left" / "on the right" / "in the
                        center") whenever two or more people share the
                        same clothing color.
  - actor:              who performed the event, if single-actor. Use
                        the same descriptor style as participants.
  - object:             what object is involved. Always name the object's
                        distinguishing visible attributes when present:
                        color (ball color, ring color, balloon color),
                        shirt number, count, size, etc. Do not abstract
                        colors or numbers away.
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / slipped / etc. if visible
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" — tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final". You only see ONE chunk — you cannot know global index.
  - Do NOT invent events you cannot see in this chunk.
  - Name people by clothing color PLUS at least one additional feature
    (hair color, gender, spatial position) whenever the clothing alone
    does not uniquely identify them. Example: "blonde woman in blue
    shirt on the left" is better than "blue-shirt woman".
  - Always record visible object colors, shirt numbers, and other
    distinguishing attributes — do not abstract them away.
  - For any visual state change (color, pose, position, shape,
    expression, appearance), describe BOTH the before and after state
    explicitly.
  - Name fixed structural landmarks (goal, net, finish line, specific
    board section, numbered box, chalkboard panel) in the description
    when they are load-bearing to the event, even if the landmark
    itself is stationary.
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk". Likewise if it was
    clearly already underway when the chunk opened, set
    continuation_hint = "starts_before_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


# Re-ID-focused variant: same event shape as v2, but the participants /
# actor strings are required to be a stable, re-identifiable descriptor
# bundle (hair, build, accessories, bib/jersey number, starting side,
# facing direction). The goal is that the SAME person described in two
# different chunks produces two descriptor strings that share enough
# surface tokens for a downstream identity linker to merge them.
# Selected via --chunk_prompt_version=v3 in benchmark_sub_with_states.py.
CHUNK_PROMPT_TEMPLATE_V3 = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

Describe every distinct event or notable sub-event you observe IN THIS
CHUNK ONLY, in the order they happen.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label (e.g. "game", "throw", "ring toss",
                        "tire toss", "balloon pop", "ladder climb")
  - description:        one concise sentence of what happens. Name any
                        fixed structural element that is load-bearing to
                        the action (goal, net, finish line, specific
                        board section, numbered box) even when the
                        element itself is not moving. For visual state
                        changes, describe BOTH the BEFORE and AFTER
                        state (e.g. "mask changes from red to white",
                        not just "mask change").
  - participants:       list of re-identifiable descriptor strings, one
                        per person involved. Each string MUST bundle
                        multiple stable attributes so the same person
                        can be recognized in a different segment of the
                        video. Include, whenever visible:
                          * upper-body clothing (color + pattern/logo)
                          * hair (color + length/style: "long black
                            ponytail", "bald", "short curly grey")
                          * build/height ("tall", "stocky", "slim")
                          * gender and rough age band ("young man",
                            "middle-aged woman")
                          * accessories (glasses, hat, cap, headband,
                            beard, wristband, bib, bag)
                          * jersey / bib / shirt number if visible
                          * starting spatial position in this chunk
                            ("on the left", "far side of the court",
                            "near the blue goal")
                        Use natural noun-phrase form, e.g.
                        "tall slim man in navy jersey #10 with short
                        black hair, starting on the left".
  - actor:              who performed the event, if single-actor. Use
                        the FULL descriptor string, identical in form to
                        the participants entries.
  - object:             what object is involved. Always name the object's
                        distinguishing visible attributes when present:
                        color (ball color, ring color, balloon color),
                        shirt number, count, size, etc. Do not abstract
                        colors or numbers away.
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / slipped / etc. if visible
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" — tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final". You only see ONE chunk — you cannot know global index.
  - Do NOT invent events or attributes you cannot see in this chunk.
  - For any given person, use the SAME descriptor string every time
    they appear as actor or participant within this chunk, even across
    multiple events. Do not re-describe the same person with different
    wording event-to-event.
  - A descriptor string with only ONE attribute (e.g. "blue-shirt man")
    is insufficient. Bundle at least two non-overlapping attributes
    unless no other attribute is visible.
  - Always record visible object colors, shirt numbers, and other
    distinguishing attributes — do not abstract them away.
  - For any visual state change (color, pose, position, shape,
    expression, appearance), describe BOTH the before and after state
    explicitly.
  - Name fixed structural landmarks (goal, net, finish line, specific
    board section, numbered box, chalkboard panel) in the description
    when they are load-bearing to the event, even if the landmark
    itself is stationary.
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk". Likewise if it was
    clearly already underway when the chunk opened, set
    continuation_hint = "starts_before_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


# Sports-aware variant: v3 re-ID rules plus an optional `sports` object
# that structured-shootout/goal questions need. When the event is a
# scorable sports moment (soccer kick, free throw, tennis point, darts
# throw, etc.), the model fills `sports` with team / result / direction /
# score_after / round_index / keeper_dive. For non-sports content the
# field is omitted, so v4 is a strict superset of v3.
#
# v4 also forbids emitting a standalone event for a celebration or
# reaction shot: those must be folded into the parent scoring event's
# description, so the timeline stays dominated by signal.
# Selected via --chunk_prompt_version=v4 in benchmark_sub_with_states.py.
CHUNK_PROMPT_TEMPLATE_V4 = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

Describe every distinct event or notable sub-event you observe IN THIS
CHUNK ONLY, in the order they happen.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label. For sports, prefer action-specific
                        labels: "penalty kick", "free kick", "goal",
                        "save", "miss", "foul", "free throw", "serve",
                        "point", "dart throw", "ring toss", "balloon
                        pop", "ladder climb". Avoid vague buckets like
                        "game" when a more specific label fits.
  - description:        one concise sentence of what happens. Name any
                        fixed structural element that is load-bearing to
                        the action (goal, net, finish line, specific
                        board section, numbered box) even when the
                        element itself is not moving. For visual state
                        changes, describe BOTH the BEFORE and AFTER
                        state (e.g. "mask changes from red to white",
                        not just "mask change"). If the shot/action was
                        immediately followed by a celebration or a
                        reaction shot within the same chunk, fold that
                        into this description (e.g. "... kicks the ball
                        into the top-right corner; the shooter runs off
                        celebrating") instead of emitting a separate
                        event.
  - participants:       list of re-identifiable descriptor strings, one
                        per person involved. Each string MUST bundle
                        multiple stable attributes so the same person
                        can be recognized in a different segment of the
                        video. Include, whenever visible:
                          * upper-body clothing (color + pattern/logo)
                          * hair (color + length/style)
                          * build/height
                          * gender and rough age band
                          * accessories (glasses, hat, headband,
                            beard, wristband, bib)
                          * jersey / bib / shirt number if visible
                          * starting spatial position in this chunk
                        Use natural noun-phrase form, e.g.
                        "tall slim man in navy jersey #10 with short
                        black hair, starting on the left".
  - actor:              who performed the event, if single-actor. Use
                        the FULL descriptor string, identical in form to
                        the participants entries.
  - object:             what object is involved. Always name the object's
                        distinguishing visible attributes when present:
                        color (ball color, ring color, balloon color),
                        shirt number, count, size, etc. Do not abstract
                        colors or numbers away.
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / saved / slipped / etc. if visible.
                        Leave empty for camera-only events (stadium
                        pans, wide shots) — do not fill with filler
                        like "in" when nothing is being scored.
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" — tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.
  - sports:             (OPTIONAL) present only when this event is a
                        scorable sports moment (a kick/serve/throw
                        whose outcome contributes to a game score or
                        contest result). When present, it is a JSON
                        object with these sub-fields (omit any whose
                        value is not visible):
                          * team:            the shooter/server's team,
                                             using a known team name
                                             (e.g. "Argentina",
                                             "France", "Lakers") when
                                             identifiable from kit,
                                             flag, logo, or context;
                                             else a color descriptor
                                             like "team in white and
                                             light blue stripes".
                          * kicker_number:   jersey / bib number if
                                             visible (e.g. "#10").
                          * result:          one of "goal", "saved",
                                             "missed_wide", "missed_
                                             high", "post", "crossbar",
                                             "blocked", "ace", "point",
                                             "hit", "miss".
                          * direction:       where the shot/throw went.
                                             For a goal, use a 3x3 grid
                                             label: "top-left",
                                             "top-center", "top-right",
                                             "middle-left", "center",
                                             "middle-right", "bottom-
                                             left", "bottom-center",
                                             "bottom-right". For other
                                             sports use the natural
                                             description (e.g. "down
                                             the line", "short put").
                          * keeper_dive:     for soccer/hockey saves,
                                             "left", "right", "stayed
                                             centered", "up", or
                                             "down".
                          * round_index:     1-based index of the round
                                             or frame within the
                                             contest if visible on a
                                             scoreboard or graphic
                                             (e.g. 4 for Round 4 of a
                                             shootout). Integer.
                          * kick_in_round:   "home_1of2", "away_2of2",
                                             "sudden_death", or
                                             similar, when the contest
                                             has a strict alternation
                                             pattern.
                          * score_after:     scoreboard state AFTER
                                             this event, as a plain
                                             string. Use the team
                                             names/colors that are
                                             visible, e.g. "ARG 3 - 2
                                             FRA" or "white 2 - 1 red".

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final". You only see ONE chunk — you cannot know global index.
    (The `sports.round_index` field is an exception: fill it ONLY when
    a round number is visibly displayed on the scoreboard or graphic.)
  - Do NOT invent events or attributes you cannot see in this chunk.
  - For any given person, use the SAME descriptor string every time
    they appear as actor or participant within this chunk, even across
    multiple events. Do not re-describe the same person with different
    wording event-to-event.
  - A descriptor string with only ONE attribute (e.g. "blue-shirt man")
    is insufficient. Bundle at least two non-overlapping attributes
    unless no other attribute is visible.
  - Always record visible object colors, shirt numbers, and other
    distinguishing attributes — do not abstract them away.
  - For any visual state change (color, pose, position, shape,
    expression, appearance), describe BOTH the before and after state
    explicitly.
  - Name fixed structural landmarks (goal, net, finish line, specific
    board section, numbered box, chalkboard panel) in the description
    when they are load-bearing to the event, even if the landmark
    itself is stationary.
  - Do NOT emit a standalone event whose only content is a celebration
    or a reaction/close-up shot. Fold those into the parent scoring
    event's description. A standalone celebration is allowed ONLY if
    the parent scoring event occurred in a different chunk.
  - For team naming in `sports.team`: you MAY use world knowledge to
    identify a team from its kit, flag, crest, or broadcast graphics
    (e.g. "Argentina" when the kit is light-blue and white stripes
    with an AFA crest). When uncertain, fall back to a color/kit
    descriptor and do not guess.
  - Do NOT fill `sports` on camera-only events (stadium pans, crowd
    cutaways, replays of already-logged shots). Replays may be marked
    with `event_type: "<action> replay"` and no `sports` object.
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk". Likewise if it was
    clearly already underway when the chunk opened, set
    continuation_hint = "starts_before_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


# OCR-aware variant: v1 schema plus a dedicated ``on_screen_text`` field
# per event. The extractor is instructed to transcribe any visible burned-in
# text (scoreboards, captions, chyrons, titles, name tags, pedestal labels,
# "GAME OVER" banners, jersey numbers, timers) verbatim, tagged with a
# coarse screen-region label so downstream stages can reason about it.
# Empty list when no text is visible. Selected via
# --chunk_prompt_version=v5 in benchmark_sub_with_states.py. Use a fresh
# --states_cache_dir when switching to v5; existing caches lack the OCR
# field and will not be re-extracted automatically.
CHUNK_PROMPT_TEMPLATE_V5 = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

Describe every distinct event or notable sub-event you observe IN THIS
CHUNK ONLY, in the order they happen.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label (e.g. "game", "throw", "ring toss",
                        "tire toss", "balloon pop", "ladder climb")
  - description:        one concise sentence of what happens
  - participants:       list of short descriptors based on visible clothing
                        (e.g. ["blue-shirt man", "red-shirt man"])
  - actor:              who performed the event, if single-actor
  - object:             what object is involved (ball color, ring color,
                        tire color, balloon color, etc.) if visible
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / slipped / etc. if visible
  - on_screen_text:     list of objects describing ALL burned-in text
                        visible during this event (scoreboard digits,
                        chyrons / lower-thirds, captions, "GAME OVER" /
                        "WINNER" banners, round labels, countdown timers,
                        jersey / bib numbers, numbered pedestal / podium
                        labels, contestant name tags, show logos, on-court
                        graphics, subtitles that are part of the footage).
                        Each entry is
                          {{"text": "<verbatim text, preserving case>",
                           "location": "<one of: scoreboard, caption,
                                        lower_third, title, banner,
                                        name_tag, jersey_number, podium,
                                        timer, subtitle, logo, other>"}}
                        Transcribe digits and punctuation exactly as
                        shown (e.g. "ARG 3 - 2 FRA", "ROUND 4", "#10",
                        "00:47"). Empty list if no text is visible.
                        Do NOT invent text that is not actually on screen.
                        Do NOT paraphrase or translate -- copy the
                        characters verbatim. If the same text persists
                        across multiple events in this chunk, include it
                        on each event it is visible for.
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" -- tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final". You only see ONE chunk -- you cannot know global index.
  - Do NOT invent events you cannot see in this chunk.
  - Use the participants' visible clothing as their names, e.g.
    "blue-shirt man", "red-shirt man".
  - When a scoreboard, banner, or caption conveys the RESULT of an
    action (e.g. "GAME OVER", "WINNER: RED TEAM", "4 - 3"), copy it into
    on_screen_text verbatim and also reflect the fact in the event's
    ``outcome`` or ``description`` so downstream readers do not need to
    re-parse the text.
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk". Likewise if it was
    clearly already underway when the chunk opened, set
    continuation_hint = "starts_before_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


# Round-tracking variant: v5 schema (on-screen OCR) plus explicit
# segment-boundary markers on every event. The extractor must decide,
# for each event, whether it OPENS a new game/round, continues an
# in-progress one, or CLOSES one, and forward a short label of what
# game/round is active. A deterministic post-processor
# (``annotate_timeline_with_subtype_order``) then walks the concatenated
# timeline and assigns a global 1-based ``round_index`` to every event,
# so downstream Stage B / answerer can answer "which player won the Nth
# game?" questions without guessing boundaries. Selected via
# --chunk_prompt_version=v6. Use a fresh --states_cache_dir.
CHUNK_PROMPT_TEMPLATE_V6 = """You are viewing a SHORT segment of a long video.
This chunk spans global timestamps {chunk_start} to {chunk_end}
(chunk index {chunk_idx}, duration {chunk_duration_sec:.1f}s).

YOUR ROLE: comprehensive event recording. Capture EVERY distinct
observable action, gesture, state change, interaction, and reaction in
this chunk, in the order they happen. A separate downstream filter
selects which events are relevant to each question -- you do NOT need
to predict which events matter. Err strongly toward over-capture; an
event you omit here is permanently lost from the pipeline.

This means specifically:
  * If TWO people perform DIFFERENT actions in overlapping or adjacent
    time windows (e.g. a professor lectures while a student raises
    their hand; a goalkeeper dives while the kicker shoots; one player
    celebrates while another walks away), emit TWO separate events --
    one per actor -- not one consolidated event with the second person
    in ``participants``.
  * Capture brief, secondary, or non-dominant actions (a hand-raise, a
    high-five, a head-nod, a glance, a quick object manipulation, a
    side conversation, an audience reaction, a short clip of on-screen
    text or graphics) as their own events. Do NOT consolidate them
    into the description of a longer dominant event happening in the
    same time window.
  * Capture every distinct visible person who does anything observable
    as the ACTOR of at least one event during this chunk -- not just
    listed in a generic ``participants`` array of someone else's event.
  * If the same person performs multiple distinct actions in this
    chunk (raises hand, then speaks, then writes), emit a separate
    event for each.
  * SPATIAL SPECIFICITY. When describing a manipulation of an object
    that has spatial structure (a multi-panel chalkboard, a row of
    boxes, a set of cups, multiple drawers, a door, a shelf, a grid of
    items, a row of contestants, left vs right side of a court), name
    the SPECIFIC sub-part affected (left / middle / right; top /
    bottom; first / second / third position from the left; the box
    labeled "3"; the leftmost cylinder; the upper drawer) AND the
    DIRECTION of the motion (down, up, open, closed, in, out, slid
    leftward, raised, lowered). 'Pulls a chalkboard section' is
    insufficient -- write 'pulls DOWN the RIGHTMOST chalkboard
    section'. 'Opens a box' is insufficient when several boxes are
    visible -- write 'opens the leftmost box (Box 3)'. Spatial
    position and motion direction are observable in frames; omitting
    them when multiple sub-parts exist makes the event ambiguous to
    every downstream stage.
    HOW TO IDENTIFY THE CORRECT SUB-PART -- USE BEFORE/AFTER
    COMPARISON, NOT A SINGLE FRAME. You see all 60 frames of this
    chunk. To label which panel / box / cup / drawer was affected:
      (i) From a frame BEFORE the manipulation, count and locate ALL
          visible sub-parts and their positions in left-to-right (and
          top-to-bottom) order. A standard lecture chalkboard usually
          has 3 panels (leftmost, middle, rightmost); a row of cups or
          boxes is numbered 1, 2, 3, ... from left.
      (ii) From a frame AFTER (or DURING) the manipulation, identify
           the ONE sub-part whose position / height / openness / state
           CHANGED. The neighbors that DID NOT move tell you which
           positions are NOT affected.
      (iii) Label the manipulated sub-part by its position RELATIVE TO
            those unchanged neighbors. E.g. if the leftmost and middle
            panels stayed in place and only the third visible panel
            slid, the moved panel is the RIGHTMOST.
    Do NOT commit to "middle" or "left" or "right" from the
    manipulation frame alone -- the manipulated sub-part may APPEAR
    near the visual centre of the cropped frame even when it is at the
    far right of the actual structure. Always cross-check against the
    stationary neighbors visible in the BEFORE frame.
  * NO INTENT HALLUCINATION. Describe what is OBSERVABLE in frames.
    Do NOT invent purpose, intent, or semantic distinctions that
    require audio, transcript, or world-knowledge you don't have. A
    raised hand is observable; whether the person is "asking",
    "answering", "voting", "requesting attention", or "celebrating"
    is intent and almost always not visually distinguishable. Write
    'raises their hand' (observable), NOT 'raises their hand to ask
    a question' or 'raises their hand to answer' (intent). Same for
    speech: write 'speaks' or 'says something to the professor'
    (observable), NOT 'asks a question' / 'answers a question' /
    'argues' / 'compliments' (intent). Same for an exchange of
    objects: 'hands a paper to X' (observable), NOT 'gives a graded
    test to X' unless the grade is visible on the paper. When in
    doubt about an intent / purpose word, omit it.
  * CLASSIFY BY PHYSICAL POSTURE, NOT BY HAND CONTENTS. Whenever a
    person's hand goes UP from a resting position to a raised
    position (above shoulder or near head level), that is a
    HAND-RAISE event. Use ``event_type: "hand_raise"``. This is a
    physical posture and does NOT depend on what the hand is
    holding. A raised hand holding a pen / pencil / marker / paper /
    notebook / phone / book / fan is STILL a hand-raise. Do NOT
    re-classify it as ``"gesture"``, ``"point"``, ``"raise_pen"``,
    ``"hold_up_paper"``, etc. just because something is in the
    hand. Mention the held object in the description as a
    parenthetical (e.g. ``"raises their hand (holding a blue pen)"``,
    ``"raises their hand (holding a sheet of paper)"``), but the
    event_type is ``"hand_raise"``. Reserve ``"gesture"`` and
    ``"point"`` for hand motions that are NOT a sustained raise (a
    pointing finger from chest level, a wave, a thumbs-up, an
    expressive flutter while speaking).
  * PEER-CONTACT ATTENTION. When two distinct people are visible
    together and ONE person's hand, arm, or body moves toward the
    OTHER person, you MUST track the motion across multiple frames
    and determine whether physical CONTACT actually occurred. If
    contact happened (a tap, pat, shoulder-touch, arm-touch, hand-
    on-shoulder, hug, high-five, fist-bump, handshake, lean against,
    pushed, kiss, hand-shake, holding the other's arm, helping them
    up, head-on-shoulder), emit a distinct event with an
    appropriate contact event_type ("tap", "pat", "shoulder_tap",
    "high_five", "fist_bump", "handshake", "hug", "celebration"
    when the contact is celebratory), with the initiating person as
    ``actor`` and the recipient named in BOTH ``participants`` and
    the description (e.g. ``"the tan-jacket student taps the red-
    hoodie student on the shoulder"``, ``"the orange-shirt man high-
    fives the blue-shirt man"``). Do NOT collapse a contact gesture
    into a generic ``"gestures with hands"`` / ``"reaches toward"``
    / ``"looks at"`` description -- those omit the load-bearing fact
    of whether contact happened. If you watched the motion across
    frames and contact did NOT occur (e.g. the hand stopped short
    or returned without touching), describe it as ``"reaches toward
    X without contact"`` -- still acknowledging the directed motion
    rather than a vague gesture. Sub-second contact (a brief tap
    that lasts ~1-2 frames) still counts -- this is exactly the
    kind of "brief, secondary, non-dominant action" the
    comprehensive-recording principle requires.
  * Do NOT reason about what is "important", "central", or "notable".
    That is the filter's job. Your job is to record what was visible.

For each event, emit a JSON object with these fields:
  - local_start:        timestamp within the FULL video (MM:SS)
  - local_end:          timestamp within the FULL video (MM:SS)
  - event_type:         short label describing the action. The
                        following examples are NON-EXHAUSTIVE -- use
                        whatever short verb-or-noun-phrase best
                        describes the action you observe:
                        "game", "throw", "ring toss", "tire toss",
                        "balloon pop", "ladder climb", "hand_raise",
                        "speak", "answer_question", "celebration",
                        "high_five", "fist_bump", "applaud",
                        "lecture", "write_on_board", "pull_board",
                        "open_box", "reveal", "place", "lift",
                        "walk", "sit", "stand", "point", "gesture",
                        "look", "react", "graphic", "title_card",
                        "scoreboard_update", "transition".
                        Coin a new short label freely if none of these
                        fits; do NOT pick the closest game/sports word
                        if your event isn't a game/sports action.
  - description:        one concise sentence of what happens
  - participants:       list of short descriptors based on visible clothing
                        (e.g. ["blue-shirt man", "red-shirt man"])
  - actor:              who performed the event, if single-actor
  - object:             what object is involved (ball color, ring color,
                        tire color, balloon color, etc.) if visible. For a
                        VISUAL STATE CHANGE event (color, pose, position,
                        shape, expression, mask, appearance), leave
                        ``object`` empty and use ``object_before`` /
                        ``object_after`` instead so the transition is
                        captured unambiguously.
  - object_before:      REQUIRED when this event is a visual state change
                        (``event_type`` contains "change", "transform",
                        "turn", "swap", "reveal", or similar). Describes
                        the object / attribute BEFORE the change using
                        its DOMINANT single color + noun form
                        (e.g. "red mask", "yellow mask", "green ball",
                        "gold fan"). Use simple color vocabulary:
                        red, yellow, blue, green, black, white, gold,
                        silver, pink, purple, orange, brown.
                        Accent patterns / minor trim colors
                        (e.g. thin black lines on a mostly-red mask,
                        small gold filigree on a blue mask) belong in
                        ``object_before_pattern``, NOT here.
                        Use compound form "A and B" ONLY if two colors
                        split the object's surface roughly equally
                        (e.g. a mask that is clearly half red and
                        half blue -> "red and blue mask"). If one
                        color clearly dominates, write only that
                        color even when small accents are visible.
                        Prefer "red mask" over "red and black mask"
                        when the black is just pattern detail.
                        Omit the field entirely on non-transition
                        events.
  - object_after:       REQUIRED when this event is a visual state
                        change. Same format as ``object_before`` --
                        DOMINANT single color + noun form (e.g.
                        "yellow mask"). Only use compound form for
                        truly bichromic objects. The pair
                        (object_before, object_after) must be
                        transcribed from frames you actually see --
                        never invent a color or guess a transition
                        that wasn't visible. If the AFTER state is
                        genuinely not visible in this chunk (e.g. the
                        shot cuts away mid-flip), set it to
                        "not_visible". If you can see SOMETHING but
                        cannot commit to the exact color, prefix it
                        with "uncertain: " (e.g. "uncertain: light-
                        colored mask"). CRITICAL CONSISTENCY: within
                        a single chunk, use the SAME color vocabulary
                        for the SAME mask across consecutive events --
                        if event N ends with object_after="yellow
                        mask", event N+1 must begin with
                        object_before="yellow mask" (unless a visible
                        flip occurred between them). Omit this field
                        entirely on non-transition events.
  - object_before_pattern / object_after_pattern:
                        OPTIONAL. Free-form accent / pattern detail
                        that was present but subordinate to the
                        dominant color (e.g. "thin black lines",
                        "gold filigree", "white swirls on forehead").
                        Use these ONLY when pattern info is load-
                        bearing for downstream questions; omit when
                        the object is plain colored or the pattern
                        is not distinctive. These fields never
                        replace ``object_before`` / ``object_after``
                        -- they supplement them.
  - cover / hidden:     Present ONLY on "cups-and-balls" style events
                        where one object is placed on top of another
                        and conceals it, or lifted to reveal what
                        was concealed.
                          * ``cover``  -- the covering object (cup,
                                          cylinder, cloth, hat, box,
                                          bowl, bucket, etc.) that the
                                          ACTOR'S HAND DIRECTLY
                                          CONTACTS AND LIFTS OR SETS
                                          DOWN in this event. Use its
                                          dominant-color form (e.g.
                                          "brown cylinder", "pink
                                          cup"). If two or more covers
                                          are visible in the shot,
                                          ``cover`` is ONLY the one
                                          being manipulated by the
                                          actor right now -- not a
                                          stationary neighbor.
                          * ``hidden`` -- a small object that BECOMES
                                          NEWLY VISIBLE (on a lift /
                                          reveal) or BECOMES NEWLY
                                          OCCLUDED (on a placement) AS
                                          A DIRECT MECHANICAL
                                          CONSEQUENCE of the cover
                                          being moved by the actor in
                                          this event. An object that
                                          was already visible on the
                                          table BEFORE the lift, or
                                          that remains stationary
                                          during the lift / placement,
                                          is a NEIGHBOR -- NOT
                                          ``hidden``.
                                          ``hidden`` is typically a
                                          small object the cover sits
                                          on top of (a ball, small
                                          cup, toy, card, fruit, coin,
                                          patterned cup). It is
                                          ALMOST NEVER another full-
                                          size cover (cylinder, cup,
                                          hat, bowl). If you are
                                          tempted to write
                                          ``hidden``='X cylinder' or
                                          'X cup' where X is the same
                                          form factor as ``cover``,
                                          STOP -- it is almost
                                          certainly a neighbor, not
                                          what's hidden; either set
                                          ``hidden`` to "uncertain"
                                          or omit the cover/hidden
                                          fields on this event
                                          entirely.
                        If multiple covers are placed or lifted in
                        the same shot, emit ONE event per cover,
                        each with its own (cover, hidden) pair --
                        never bundle two cover-content relations
                        into a single event.
                        If two or more covers are visible in the same
                        shot and the camera angle does NOT clearly
                        show the object directly underneath the
                        manipulated cover (e.g. the cover's body
                        occludes its contents from this viewpoint),
                        OR the hidden object's color flashed by too
                        briefly to commit to a value, set ``hidden``
                        to "uncertain: <best guess>" rather than
                        guessing a clean color or defaulting to a
                        nearby visible object.
                        CONSISTENCY: when a single cover is placed
                        and then later lifted in nearby events, the
                        ``hidden`` value on the placement and the
                        subsequent reveal of the SAME cover should
                        match. If they would be different, you are
                        probably misidentifying the cover -- prefer
                        ``hidden``="uncertain" on at least one of
                        them.
                        Omit both fields on events that are NOT
                        placement or reveal events.
  - outcome:            hit / miss / in / out / won / lost / popped /
                        scored / slipped / etc. if visible
  - on_screen_text:     list of objects describing ALL burned-in text
                        visible during this event (scoreboard digits,
                        chyrons / lower-thirds, captions, "GAME OVER" /
                        "WINNER" banners, round labels, countdown timers,
                        jersey / bib numbers, numbered pedestal / podium
                        labels, contestant name tags, show logos, on-court
                        graphics, subtitles that are part of the footage).
                        Each entry is
                          {{"text": "<verbatim text, preserving case>",
                           "location": "<one of: scoreboard, caption,
                                        lower_third, title, banner,
                                        name_tag, jersey_number, podium,
                                        timer, subtitle, logo, other>"}}
                        Transcribe digits and punctuation exactly as
                        shown (e.g. "ARG 3 - 2 FRA", "ROUND 4", "#10",
                        "00:47"). Empty list if no text is visible.
  - segment_boundary:   one of "start", "middle", "end", "unknown".
                        A "segment" is a self-contained round / game /
                        match / contest / challenge that has its own
                        setup, play, and resolution. Rules for choosing:
                          * "start"   -- this event OPENS a new segment.
                            Use when ANY of: (a) a title card / intro
                            graphic / logo for a new game is shown,
                            (b) the host announces, introduces, or
                            explains a new game / round, (c) a clean
                            cut from a "resolution" (scoreboard,
                            handshake, winner chyron, break slate) to
                            a visibly different setup starts new play,
                            (d) contestants take fresh starting
                            positions AFTER the previous game visibly
                            ended, (e) on_screen_text shows a new
                            round / game number or title.
                          * "end"     -- this event CLOSES the current
                            segment. Use for: final winning action,
                            "GAME OVER" / "WINNER" / "CHAMPION" banner,
                            scoreboard freeze, host declaring the
                            result, handshake, cut to commercial break
                            or results slate.
                          * "middle"  -- ongoing play within the
                            currently-active segment (most events will
                            be this).
                          * "unknown" -- genuinely cannot tell.
                        Be conservative: mark "start" only when you are
                        reasonably sure a NEW segment began. If two
                        adjacent events both look like starts, keep only
                        the earliest one as "start" and mark the other
                        "middle".
  - segment_label:      short phrase naming the currently-active segment
                        (e.g. "Blow Jump", "Ring Toss", "Penalty
                        Shootout -- Round 4", "Balloon Pop"). Copy
                        verbatim from on_screen_text / title cards when
                        available; otherwise give a descriptive label
                        based on the activity. Empty string ONLY when
                        you truly have no basis to name it.
  - continuation_hint:  "starts_before_chunk" / "continues_after_chunk" /
                        "complete" -- tells the aggregator whether this
                        event is partial so it can merge across chunk
                        boundaries.

RULES:
  - Do NOT use ordinal words like "first", "second", "third", "Nth",
    "final" inside ``description`` or ``segment_label`` based on
    position in the video -- you only see ONE chunk. Ordinals that are
    LITERALLY printed on screen (e.g. "ROUND 4", "Game 2") are fine to
    copy verbatim.
  - Do NOT invent events you cannot see in this chunk.
  - Use the participants' visible clothing as their names, e.g.
    "blue-shirt man", "red-shirt man".
  - Be decisive about segment_boundary. Marking every event "middle"
    wastes the signal. The typical pattern is one "start", several
    "middle"s, one "end", per segment.
  - For ANY visual state change (color, pose, position, shape,
    expression, mask, costume, appearance, count) you MUST describe
    BOTH the before and after state explicitly, via dedicated
    ``object_before`` and ``object_after`` fields. "A performer
    changes their mask" alone is insufficient -- the event MUST
    record both the starting mask color and the resulting mask
    color (e.g. object_before="red mask", object_after="yellow
    mask"). The single-valued ``object`` field is reserved for
    non-transition events. If one side of the transition is
    genuinely not visible (e.g. the shot cuts away mid-flip),
    use "not_visible" for that side -- never invent a color.
  - PREFER DOMINANT COLOR in ``object_before`` / ``object_after``.
    A mask that is mostly red with small black or white accent
    patterns is a "red mask", NOT a "red and black mask" or
    "red mask with black and white patterns". Use compound form
    ("A and B") ONLY when two colors genuinely split the surface
    roughly equally (e.g. a mask that is half red, half blue).
    Downstream questions are phrased in simple color terms like
    "red to yellow" or "black to white" -- if you pad with
    accent colors, the question will fail to match even when the
    underlying transition is exactly what's being asked about.
    When in doubt about whether a color is dominant or just
    accent, treat it as accent and put it in ``object_before_
    pattern`` / ``object_after_pattern`` instead of mixing it
    into the main color field.
  - COLOR CONSISTENCY ACROSS EVENTS. Within a single chunk, use
    the SAME color word for the SAME mask across consecutive
    events. If event N ends with object_after="yellow mask",
    then event N+1's object_before MUST be "yellow mask" (unless
    another flip happened in between that you also recorded).
    Do NOT re-describe a stable mask with different color words
    from one event to the next.
  - CONTAINMENT (cups-and-balls) TRACKING. When an object is
    placed on top of another (covering / hiding it) or lifted to
    reveal what was underneath -- regardless of the framing,
    whether explicit magic trick, a math / counting demo, a
    "which cup is the ball under" game, or a box-reveal segment
    -- you MUST record the relation as a (cover, hidden) pair on
    that event. The cover field names the covering object using
    its dominant color + noun form ("brown cylinder"); the
    hidden field names what goes under, or what is revealed,
    using the full color + noun form ("yellow object",
    "white cup with green patterns"). If the woman lifts TWO
    cups simultaneously to reveal TWO different objects,
    emit TWO separate events -- one per cup -- each with its
    own (cover, hidden) pair. NEVER write a vague description
    like "places the cylinders on top of other objects" or
    "reveals a cup and an object" without specifying which
    cover goes with which hidden object -- that loses the
    mapping and makes downstream tracking impossible. If the
    hidden color was only visible briefly before being covered,
    still record it; use "uncertain: <guess>" only when you
    genuinely could not see any color.
  - When a scoreboard, banner, or caption conveys the RESULT of an
    action (e.g. "GAME OVER", "WINNER: RED TEAM", "4 - 3"), copy it
    into on_screen_text verbatim AND set segment_boundary = "end" on
    that event.
  - When the chunk starts mid-segment (no new intro visible), mark the
    first event as "middle" (or "end" if it is actually closing one)
    and set continuation_hint = "starts_before_chunk".
  - If a game or round starts in this chunk but doesn't finish, set
    continuation_hint = "continues_after_chunk".
  - Output STRICT JSON only: a single top-level list of event objects.
    No prose, no markdown fences, no commentary outside the JSON.
"""


CHUNK_PROMPT_TEMPLATES: Dict[str, str] = {
    "v1": CHUNK_PROMPT_TEMPLATE,
    "v2": CHUNK_PROMPT_TEMPLATE_V2,
    "v3": CHUNK_PROMPT_TEMPLATE_V3,
    "v4": CHUNK_PROMPT_TEMPLATE_V4,
    "v5": CHUNK_PROMPT_TEMPLATE_V5,
    "v6": CHUNK_PROMPT_TEMPLATE_V6,
}


def get_chunk_prompt(version: str) -> str:
    try:
        return CHUNK_PROMPT_TEMPLATES[version]
    except KeyError:
        raise ValueError(
            f"Unknown chunk_prompt_version {version!r}; "
            f"known: {sorted(CHUNK_PROMPT_TEMPLATES)}"
        )


def _fmt_mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _encode_frame_b64(frame, max_side: int, jpeg_quality: int) -> str | None:
    import cv2
    h, w = frame.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        return None
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _get_video_metadata(video_path: str) -> Dict[str, Any]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        raise RuntimeError(f"Video reports 0 frames: {video_path}")
    if fps <= 0:
        fps = 30.0
    return {"fps": fps, "total_frames": total, "duration_sec": total / fps}


def _plan_chunks(duration_sec: float, chunk_seconds: float) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    t = 0.0
    while t < duration_sec:
        end = min(t + chunk_seconds, duration_sec)
        ranges.append((t, end))
        t = end
    if not ranges:
        ranges = [(0.0, duration_sec)]
    return ranges


def _extract_chunk_frames_b64(
    video_path: str,
    start_sec: float,
    end_sec: float,
    frames_per_chunk: int,
    max_side: int,
    jpeg_quality: int,
) -> List[str]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_f = int(round(start_sec * fps))
    end_f = min(total - 1, int(round(end_sec * fps)))
    if end_f <= start_f:
        cap.release()
        return []
    if frames_per_chunk == 1:
        indices = [(start_f + end_f) // 2]
    else:
        indices = [
            round(start_f + i * (end_f - start_f) / (frames_per_chunk - 1))
            for i in range(frames_per_chunk)
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


def _build_frame_parts(frames_b64: List[str]) -> List[Dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}}
        for fb64 in frames_b64
    ]


def stage_a1_extract_chunk(
    client: OpenAI,
    model: str,
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
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
    )
    messages = [{
        "role": "user",
        "content": [
            *_build_frame_parts(frames_b64),
            {"type": "text", "text": prompt},
        ],
    }]

    # Pin provider to novita for thinking models — SiliconFlow silently
    # truncates reasoning and returns content=None, which presents as a hang
    # from the client's perspective.
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
                f"Chunk {chunk_idx} Stage A1 hit token cap ({max_new_tokens}). "
                f"Bump --max_new_tokens_chunk."
            )
        if not text:
            last_err = RuntimeError(
                f"chunk {chunk_idx}: empty content (finish_reason={finish_reason})"
            )
            time.sleep(2 ** attempt)
            continue
        try:
            events = _parse_json_list_or_raise(text, f"Chunk {chunk_idx} A1", finish_reason)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return {**base_record, "events": events}

    raise RuntimeError(f"Stage A1 chunk {chunk_idx} failed after {max_retries} retries: {last_err}")


def stage_a1_extract_chunk_flash(
    client: genai.Client,
    model: str,
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
    max_retries: int = 4,
    chunk_prompt_version: str = "v1",
) -> Dict[str, Any]:
    """Stage A1 variant that runs Gemini Flash (inline frame Parts) instead
    of Qwen/OpenRouter. Same input/output contract as stage_a1_extract_chunk."""
    import base64
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
        if finish_reason == "MAX_TOKENS":
            raise _NonRetryableError(
                f"Chunk {chunk_idx} Stage A1 (flash) hit MAX_TOKENS "
                f"(cap={max_new_tokens}). Bump --max_new_tokens_chunk."
            )
        if _is_safety_finish(finish_reason):
            # Even with BLOCK_NONE, a hard service-level floor can still
            # refuse a chunk. Record the block and move on -- one 15 s chunk
            # shouldn't nuke a multi-minute video's timeline.
            return {
                **base_record,
                "events": [],
                "note": f"blocked by safety (finish_reason={finish_reason})",
                "block_reason": finish_reason,
            }
        if finish_reason not in ("STOP", "UNKNOWN"):
            last_err = RuntimeError(
                f"Chunk {chunk_idx} Stage A1 (flash) finish_reason={finish_reason}."
            )
            time.sleep(2 ** attempt)
            continue
        try:
            events = _parse_json_list_or_raise(
                text, f"Chunk {chunk_idx} A1 (flash)", finish_reason
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return {**base_record, "events": events}

    raise RuntimeError(
        f"Stage A1 chunk {chunk_idx} (flash) failed after {max_retries} retries: {last_err}"
    )


def stage_a1_reextract_chunk(
    client: OpenAI,
    model: str,
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    question: str,
    missing: str,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Question-aware re-extraction of a single chunk via OpenRouter (Qwen)."""
    frames_b64 = _extract_chunk_frames_b64(
        video_path,
        start_sec=chunk_start,
        end_sec=chunk_end,
        frames_per_chunk=frames_per_chunk,
        max_side=frame_max_side,
        jpeg_quality=frame_jpeg_quality,
    )
    base = {
        "chunk_idx": chunk_idx,
        "chunk_start_sec": chunk_start,
        "chunk_end_sec": chunk_end,
        "chunk_start_mmss": _fmt_mmss(chunk_start),
        "chunk_end_mmss": _fmt_mmss(chunk_end),
    }
    if not frames_b64:
        return {**base, "events": [], "note": "no frames"}
    prompt = REEXTRACT_CHUNK_PROMPT_TEMPLATE.format(
        chunk_start=_fmt_mmss(chunk_start),
        chunk_end=_fmt_mmss(chunk_end),
        chunk_idx=chunk_idx,
        chunk_duration_sec=chunk_end - chunk_start,
        question=question,
        missing=missing or "(unspecified)",
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
            last_err = RuntimeError(f"re-extract {chunk_idx}: no choices")
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
                f"re-extract chunk {chunk_idx} hit token cap ({max_new_tokens})."
            )
        if not text:
            last_err = RuntimeError(f"re-extract {chunk_idx}: empty content")
            time.sleep(2 ** attempt)
            continue
        try:
            events = _parse_json_list_or_raise(text, f"re-extract {chunk_idx}", finish_reason)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return {**base, "events": events}
    raise RuntimeError(
        f"re-extract chunk {chunk_idx} failed after {max_retries} retries: {last_err}"
    )


def stage_a1_reextract_chunk_flash(
    client: genai.Client,
    model: str,
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    question: str,
    missing: str,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Question-aware re-extraction via Gemini Flash (inline Parts)."""
    import base64
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
        return {**base_record, "events": [], "note": "no frames"}
    prompt = REEXTRACT_CHUNK_PROMPT_TEMPLATE.format(
        chunk_start=_fmt_mmss(chunk_start),
        chunk_end=_fmt_mmss(chunk_end),
        chunk_idx=chunk_idx,
        chunk_duration_sec=chunk_end - chunk_start,
        question=question,
        missing=missing or "(unspecified)",
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
        if finish_reason == "MAX_TOKENS":
            raise _NonRetryableError(
                f"re-extract {chunk_idx} (flash) hit MAX_TOKENS."
            )
        if _is_safety_finish(finish_reason):
            return {
                **base_record,
                "events": [],
                "note": f"blocked by safety (finish_reason={finish_reason})",
                "block_reason": finish_reason,
            }
        if finish_reason not in ("STOP", "UNKNOWN"):
            last_err = RuntimeError(
                f"re-extract {chunk_idx} (flash) finish_reason={finish_reason}"
            )
            time.sleep(2 ** attempt)
            continue
        try:
            events = _parse_json_list_or_raise(
                text, f"re-extract {chunk_idx} (flash)", finish_reason
            )
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        return {**base_record, "events": events}
    raise RuntimeError(
        f"re-extract chunk {chunk_idx} (flash) failed after {max_retries} retries: {last_err}"
    )


# ---------------------------------------------------------------------------
# Stage B: Flash video QA grounded on timeline
# ---------------------------------------------------------------------------

def stage_b_answer(
    client: genai.Client,
    model: str,
    question: str,
    timeline: str,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    prompt = STAGE_B_PROMPT_TEMPLATE.format(
        timeline=timeline,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )
    last_err: Exception | None = None
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
        if finish_reason not in ("STOP", "UNKNOWN", "MAX_TOKENS"):
            # Safety / filter — treat as non-retryable; return empty to mark
            # as unknown rather than cache a spurious yes/no.
            raise _NonRetryableError(
                f"Stage B finish_reason={finish_reason} for question {question!r}."
            )
        return text

    raise RuntimeError(f"Stage B call failed after {max_retries} retries: {last_err}")


def stage_b_answer_cot(
    client: genai.Client,
    model: str,
    question: str,
    timeline: str,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    """Chain-of-evidence Stage B: emit evidence_spans + reasoning + answer
    as JSON, then flatten to a short text string for the yes/no extractor."""
    prompt = STAGE_B_COT_PROMPT_TEMPLATE.format(
        timeline=timeline,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
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
                f"Stage B (cot) hit MAX_TOKENS (cap={max_new_tokens}). "
                f"Bump --max_new_tokens_stage_b."
            )
        if finish_reason not in ("STOP", "UNKNOWN"):
            raise _NonRetryableError(
                f"Stage B (cot) finish_reason={finish_reason} for question {question!r}."
            )
        try:
            payload = _strip_code_fence(text)
            data = json.loads(payload)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        answer = str(data.get("answer") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()
        return f"{answer}. {reasoning}".strip()
    raise RuntimeError(f"Stage B (cot) failed after {max_retries} retries: {last_err}")


def stage_b_answer_vl_cot(
    client: genai.Client,
    model: str,
    question: str,
    timeline: str,
    frames: List[Dict[str, Any]],
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    """Multimodal COT Stage B: timeline JSON + top-K retrieved frames +
    question → {frame_descriptions, evidence_spans, reasoning, answer}.

    `frames` is the output of clip_retrieval.retrieve_top_k — each item has
    `path` (jpg on disk) and `global_ts_sec` (float). Frames are sent in
    chronological order; each is preceded by a small text part carrying its
    MM:SS stamp so the model can cite it.
    """
    prompt = STAGE_B_VL_COT_PROMPT_TEMPLATE.format(
        timeline=timeline,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )
    contents: List[Any] = [prompt]
    for f in frames:
        ts = float(f.get("global_ts_sec") or 0.0)
        contents.append(f"FRAME @ {_fmt_mmss(ts)}")
        with open(f["path"], "rb") as fh:
            contents.append(
                genai_types.Part.from_bytes(
                    data=fh.read(),
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
                f"Stage B (vl_cot) hit MAX_TOKENS (cap={max_new_tokens}). "
                f"Bump --max_new_tokens_stage_b."
            )
        if finish_reason not in ("STOP", "UNKNOWN"):
            raise _NonRetryableError(
                f"Stage B (vl_cot) finish_reason={finish_reason} for question {question!r}."
            )
        try:
            data = json.loads(_strip_code_fence(text))
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        answer = str(data.get("answer") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()
        return f"{answer}. {reasoning}".strip()
    raise RuntimeError(f"Stage B (vl_cot) failed after {max_retries} retries: {last_err}")


def stage_b_answer_vl_cot_qwen(
    client: OpenAI,
    model: str,
    question: str,
    timeline: str,
    frames: List[Dict[str, Any]],
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    """Same contract as stage_b_answer_vl_cot but routes the multimodal
    answerer call through OpenRouter to a Qwen VL model instead of Gemini
    Flash. Use this to swap in a stronger visual reader when Flash's
    hit/miss misreads are the bottleneck.

    Frames are sent as `image_url` parts with data URIs (same shape used
    by the Qwen state extractor). JPEGs are read from the CLIP cache
    directly — no re-encoding — so the model sees exactly the frames
    ffmpeg decoded at 1 fps.
    """
    import base64
    prompt = STAGE_B_VL_COT_PROMPT_TEMPLATE.format(
        timeline=timeline,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )

    content: List[Dict[str, Any]] = []
    for f in frames:
        ts = float(f.get("global_ts_sec") or 0.0)
        content.append({"type": "text", "text": f"FRAME @ {_fmt_mmss(ts)}"})
        with open(f["path"], "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

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
            last_err = RuntimeError("stage_b_qwen: no choices in response")
            time.sleep(2 ** attempt)
            continue
        msg = choices[0].message
        raw = getattr(msg, "content", "") or ""
        if isinstance(raw, list):
            raw = "".join(
                (c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")) or ""
                for c in raw
            )
        text = _strip_thinking(raw).strip()
        finish_reason = str(getattr(choices[0], "finish_reason", None) or "UNKNOWN").upper()
        if finish_reason == "LENGTH":
            raise _NonRetryableError(
                f"Stage B (vl_cot/qwen) hit max_tokens (cap={max_new_tokens}). "
                f"Bump --max_new_tokens_stage_b."
            )
        try:
            data = json.loads(_strip_code_fence(text))
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        answer = str(data.get("answer") or "").strip()
        reasoning = str(data.get("reasoning") or "").strip()
        return f"{answer}. {reasoning}".strip()
    raise RuntimeError(
        f"Stage B (vl_cot/qwen) failed after {max_retries} retries: {last_err}"
    )


def stage_b_answer_feedback(
    client: genai.Client,
    model: str,
    question: str,
    timeline: str,
    max_new_tokens: int,
    max_retries: int = 4,
) -> Dict[str, Any]:
    """Feedback Stage B: emit {status, answer, missing, time_spans, rationale}."""
    prompt = STAGE_B_FEEDBACK_PROMPT_TEMPLATE.format(
        timeline=timeline,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
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
                f"Stage B (feedback) hit MAX_TOKENS (cap={max_new_tokens}). "
                f"Bump --max_new_tokens_stage_b."
            )
        if finish_reason not in ("STOP", "UNKNOWN"):
            raise _NonRetryableError(
                f"Stage B (feedback) finish_reason={finish_reason} "
                f"for question {question!r}."
            )
        try:
            payload = _strip_code_fence(text)
            data = json.loads(payload)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        status = str(data.get("status", "")).strip().lower()
        if status not in ("sufficient", "insufficient"):
            last_err = RuntimeError(f"Stage B feedback: unexpected status {status!r}")
            time.sleep(2 ** attempt)
            continue
        return {
            "status": status,
            "answer": data.get("answer"),
            "missing": str(data.get("missing") or "").strip(),
            "time_spans": data.get("time_spans") or [],
            "rationale": str(data.get("rationale") or "").strip(),
        }
    raise RuntimeError(f"Stage B (feedback) failed after {max_retries} retries: {last_err}")


def stage_b_answer_with_supplement(
    client: genai.Client,
    model: str,
    question: str,
    timeline: str,
    supplement: str,
    max_new_tokens: int,
    max_retries: int = 4,
) -> str:
    prompt = STAGE_B_SUPPLEMENT_PROMPT_TEMPLATE.format(
        timeline=timeline,
        supplement=supplement,
        question=question,
        ordinal_semantics=ORDINAL_SEMANTICS_BLOCK,
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
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
        if finish_reason not in ("STOP", "UNKNOWN", "MAX_TOKENS"):
            raise _NonRetryableError(
                f"Stage B (supplement) finish_reason={finish_reason} "
                f"for question {question!r}."
            )
        return text
    raise RuntimeError(
        f"Stage B (supplement) failed after {max_retries} retries: {last_err}"
    )


def _chunks_overlapping_spans(
    chunks: List[Dict[str, Any]],
    spans: List[Dict[str, Any]],
    max_chunks: int = 3,
) -> List[int]:
    """Return up to max_chunks chunk indices whose [start_sec, end_sec] most
    overlap with any of the given spans."""
    scored: List[Tuple[float, int]] = []
    for i, ck in enumerate(chunks):
        try:
            cs = float(ck["chunk_start_sec"])
            ce = float(ck["chunk_end_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        overlap = 0.0
        for sp in spans or []:
            try:
                ss = float(sp["start_sec"])
                se = float(sp["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap += max(0.0, min(ce, se) - max(cs, ss))
        if overlap > 0:
            scored.append((overlap, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:max_chunks]]


def run_stage_b_feedback(
    question: str,
    timeline: str,
    chunks_cache: Dict[str, Any],
    video_path: str,
    gemini_client: genai.Client,
    openrouter_client: OpenAI,
    stage_b_model: str,
    chunk_vlm_model: str,
    state_backend: str,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_stage_b: int,
    max_new_tokens_chunk: int,
    max_reextract_chunks: int = 3,
) -> str:
    """One iteration of the Stage B → Stage A feedback loop: ask the
    answerer whether the timeline suffices; if not, re-extract a few
    chunks question-aware and re-answer with the supplement."""
    feedback = stage_b_answer_feedback(
        gemini_client, stage_b_model, question, timeline, max_new_tokens_stage_b,
    )
    if feedback["status"] == "sufficient":
        ans = str(feedback.get("answer") or "").strip()
        rationale = feedback.get("rationale", "")
        return f"{ans}. {rationale}".strip()

    spans = feedback.get("time_spans") or []
    missing = feedback.get("missing", "")
    indices = _chunks_overlapping_spans(
        chunks_cache.get("chunks", []), spans, max_chunks=max_reextract_chunks,
    )
    if not indices:
        # Nothing concrete to re-examine; force a decision via COT on the
        # original timeline rather than caching a null answer.
        return stage_b_answer_cot(
            gemini_client, stage_b_model, question, timeline, max_new_tokens_stage_b,
        )

    supplement_items: List[Dict[str, Any] | None] = [None] * len(indices)

    def _reextract(k_idx: int) -> None:
        i = indices[k_idx]
        ck = chunks_cache["chunks"][i]
        cs = float(ck["chunk_start_sec"])
        ce = float(ck["chunk_end_sec"])
        if state_backend == "flash":
            supplement_items[k_idx] = stage_a1_reextract_chunk_flash(
                gemini_client, chunk_vlm_model, video_path,
                chunk_idx=i, chunk_start=cs, chunk_end=ce,
                question=question, missing=missing,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens=max_new_tokens_chunk,
            )
        else:
            supplement_items[k_idx] = stage_a1_reextract_chunk(
                openrouter_client, chunk_vlm_model, video_path,
                chunk_idx=i, chunk_start=cs, chunk_end=ce,
                question=question, missing=missing,
                frames_per_chunk=frames_per_chunk,
                frame_max_side=frame_max_side,
                frame_jpeg_quality=frame_jpeg_quality,
                max_new_tokens=max_new_tokens_chunk,
            )

    with ThreadPoolExecutor(max_workers=max(1, len(indices))) as pool:
        futs = [pool.submit(_reextract, k) for k in range(len(indices))]
        for fut in as_completed(futs):
            fut.result()

    supplement_json = json.dumps(supplement_items, ensure_ascii=False, indent=2)
    return stage_b_answer_with_supplement(
        gemini_client, stage_b_model, question, timeline, supplement_json,
        max_new_tokens_stage_b,
    )


def _effective_stage_b_max_tokens(mode: str, user_value: int) -> int:
    """COT / feedback / vl_cot emit JSON — bump the cap if the user left
    the default low."""
    floor = {"plain": 0, "cot": 1024, "feedback": 1024, "vl_cot": 1024}.get(mode, 0)
    return max(int(user_value), floor)


def stage_b_dispatch(
    mode: str,
    question: str,
    timeline: str,
    chunks_cache: Dict[str, Any],
    video_path: str,
    gemini_client: genai.Client,
    openrouter_client: OpenAI,
    stage_b_model: str,
    chunk_vlm_model: str,
    state_backend: str,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_stage_b: int,
    max_new_tokens_chunk: int,
    max_reextract_chunks: int = 3,
    clip_cache_dir: Path | None = None,
    clip_top_k: int = 16,
    clip_fps: float = 1.0,
    clip_model: str = "ViT-L-16-SigLIP-384",
    clip_pretrained: str = "webli",
    retrieval_mode: str = "single",
    decomposer_model: str = "gemini-3-flash-preview",
    vl_answerer_backend: str = "flash",
    qwen_vl_model: str = "qwen/qwen3-vl-235b-a22b-thinking",
) -> str:
    # Inject per-subtype ordinal bookkeeping into the timeline JSON once,
    # regardless of which cached stage_a.txt format produced it. Cheap, and
    # gives every Stage B mode (plain/cot/feedback/vl_cot) the subtype_order
    # / subtype_total fields the ordinal-semantics prompt block refers to.
    timeline = annotate_timeline_with_subtype_order(timeline)
    eff_tokens = _effective_stage_b_max_tokens(mode, max_new_tokens_stage_b)
    if mode == "cot":
        return stage_b_answer_cot(
            gemini_client, stage_b_model, question, timeline, eff_tokens,
        )
    if mode == "feedback":
        return run_stage_b_feedback(
            question=question, timeline=timeline, chunks_cache=chunks_cache,
            video_path=video_path,
            gemini_client=gemini_client, openrouter_client=openrouter_client,
            stage_b_model=stage_b_model, chunk_vlm_model=chunk_vlm_model,
            state_backend=state_backend,
            frames_per_chunk=frames_per_chunk,
            frame_max_side=frame_max_side,
            frame_jpeg_quality=frame_jpeg_quality,
            max_new_tokens_stage_b=eff_tokens,
            max_new_tokens_chunk=max_new_tokens_chunk,
            max_reextract_chunks=max_reextract_chunks,
        )
    if mode == "vl_cot":
        if clip_cache_dir is None:
            raise RuntimeError(
                "stage_b_mode='vl_cot' requires clip_cache_dir (wire it from "
                "evaluate_sample)."
            )
        # Local imports so 'plain'/'cot'/'feedback' runs don't pay the torch /
        # open_clip import cost.
        if retrieval_mode == "decomposed":
            from decomposed_retrieval import retrieve_top_k_decomposed
            frames = retrieve_top_k_decomposed(
                question=question,
                video_path=video_path,
                cache_dir=clip_cache_dir,
                gemini_client=gemini_client,
                decomposer_model=decomposer_model,
                top_k=clip_top_k,
                fps=clip_fps,
                model_name=clip_model,
                pretrained=clip_pretrained,
            )
        elif retrieval_mode == "uniform":
            from clip_retrieval import retrieve_uniform
            frames = retrieve_uniform(
                video_path=video_path,
                cache_dir=clip_cache_dir,
                top_k=clip_top_k,
                fps=clip_fps,
            )
        else:
            from clip_retrieval import retrieve_top_k
            frames = retrieve_top_k(
                question=question,
                video_path=video_path,
                cache_dir=clip_cache_dir,
                top_k=clip_top_k,
                fps=clip_fps,
                model_name=clip_model,
                pretrained=clip_pretrained,
            )
        if vl_answerer_backend == "qwen":
            return stage_b_answer_vl_cot_qwen(
                openrouter_client, qwen_vl_model, question, timeline, frames,
                eff_tokens,
            )
        return stage_b_answer_vl_cot(
            gemini_client, stage_b_model, question, timeline, frames, eff_tokens,
        )
    return stage_b_answer(
        gemini_client, stage_b_model, question, timeline, eff_tokens,
    )


def stage_b_cache_key(
    question: str,
    mode: str,
    retrieval_mode: str = "single",
    vl_answerer_backend: str = "flash",
) -> str:
    """Mode-prefixed cache key so switching --stage_b_mode doesn't hit a
    stale cached answer from a different mode. Plain keeps the legacy
    qhash-only key for backward compat with existing caches. vl_cot also
    pins a prompt version, retrieval mode, and answerer backend so
    prompt/retrieval/backend edits invalidate stale answers."""
    if mode == "plain":
        return qhash(question)
    if mode == "vl_cot":
        # Preserve the historic bare `vl_cot:<qhash>` format for the default
        # single+flash configuration so the 65.89% cache at
        # outputs_cmp/density_vlcot/cache stays reusable. Non-default
        # retrieval modes / backends get an explicit suffix that carries
        # the prompt version too.
        if retrieval_mode == "single" and vl_answerer_backend == "flash":
            return f"vl_cot:{qhash(question)}"
        backend_suffix = "" if vl_answerer_backend == "flash" else f"_{vl_answerer_backend}"
        return (
            f"vl_cot_{STAGE_B_VL_COT_PROMPT_VERSION}{backend_suffix}"
            f"_{retrieval_mode}:{qhash(question)}"
        )
    return f"{mode}:{qhash(question)}"


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _chunks_cache_path(chunks_cache_dir: Path, video_name: str) -> Path:
    return chunks_cache_dir / Path(video_name).stem / "chunks.json"


def _flash_cache_paths(cache_dir: Path, video_name: str) -> Tuple[Path, Path]:
    stem = Path(video_name).stem
    base = cache_dir / stem
    return base / "stage_a.txt", base / "stage_b.json"


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-sample
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample: Dict[str, Any],
    chunks_cache_dir: Path,
    cache_dir: Path,
    gemini_client: genai.Client,
    openrouter_client: OpenAI,
    aggregator_model: str,
    stage_b_model: str,
    chunk_vlm_model: str,
    chunk_seconds: float,
    frames_per_chunk: int,
    frame_max_side: int,
    frame_jpeg_quality: int,
    max_new_tokens_aggregator: int,
    max_new_tokens_stage_b: int,
    max_new_tokens_chunk: int,
    max_concurrency_questions: int,
    max_concurrency_chunks: int,
    state_backend: str = "qwen",
    stage_b_mode: str = "plain",
    max_reextract_chunks: int = 3,
    clip_cache_dir: Path | None = None,
    clip_top_k: int = 16,
    clip_fps: float = 1.0,
    clip_model: str = "ViT-L-16-SigLIP-384",
    clip_pretrained: str = "webli",
    retrieval_mode: str = "single",
    decomposer_model: str = "gemini-3-flash-preview",
    vl_answerer_backend: str = "flash",
    qwen_vl_model: str = "qwen/qwen3-vl-235b-a22b-thinking",
) -> Dict[str, Any]:
    questions: List[str] = sample["questions"]
    answers: List[str] = sample["answers"]
    if len(questions) != len(answers):
        raise ValueError(
            f"{sample['video_name']}: |questions|={len(questions)} "
            f"!= |answers|={len(answers)}"
        )

    chunks_path = _chunks_cache_path(chunks_cache_dir, sample["video_name"])
    stage_a_path, stage_b_path = _flash_cache_paths(cache_dir, sample["video_name"])

    chunks_cache = _load_json(chunks_path)
    if chunks_cache is None:
        video_path = sample["video_path"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Missing video for Stage A1 chunking: {video_path}")
        meta = _get_video_metadata(video_path)
        plan = _plan_chunks(meta["duration_sec"], chunk_seconds)
        print(
            f"  [stage_a1/{state_backend}] {chunk_vlm_model} — chunking "
            f"{len(plan)} x ~{chunk_seconds:.0f}s "
            f"(video duration {meta['duration_sec']:.1f}s)"
        )

        def run_chunk(i: int) -> Dict[str, Any]:
            cs, ce = plan[i]
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

        workers = max(1, min(max_concurrency_chunks, len(plan)))
        results: List[Dict[str, Any] | None] = [None] * len(plan)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_to_i = {pool.submit(run_chunk, i): i for i in range(len(plan))}
            for fut in as_completed(fut_to_i):
                i = fut_to_i[fut]
                results[i] = fut.result()
                print(
                    f"  [stage_a1/{state_backend}] {i + 1}/{len(plan)} done "
                    f"({len(results[i]['events'])} events)"
                )
        chunks_cache = {
            "video_name": sample["video_name"],
            "video_duration_sec": meta["duration_sec"],
            "chunk_seconds": chunk_seconds,
            "frames_per_chunk": frames_per_chunk,
            "chunk_vlm_model": chunk_vlm_model,
            "chunks": results,
        }
        _atomic_write_json(chunks_path, chunks_cache)
        print(f"  [stage_a1/qwen] wrote {chunks_path}")

    # Stage A2 — Flash aggregator (text-only).
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
        if stage_b_cache_key(q, stage_b_mode, retrieval_mode, vl_answerer_backend)
        not in stage_b_cache
    ]
    video_path = sample["video_path"]

    # If we're in the multimodal mode, pre-build the frame cache for this
    # video once so per-question retrieval calls just read cached frames.
    # Uniform mode skips CLIP embeddings entirely — decode only.
    if stage_b_mode == "vl_cot" and uncached_idx and clip_cache_dir is not None:
        if retrieval_mode == "uniform":
            from clip_retrieval import _decode_frames, _clip_cache_root
            stem = Path(video_path).stem
            frames_dir = _clip_cache_root(clip_cache_dir, stem) / "frames"
            frame_paths = _decode_frames(video_path, frames_dir, fps=clip_fps)
            print(
                f"  [uniform] {len(frame_paths)} frames @ {clip_fps}fps "
                f"(no CLIP model loaded)"
            )
        else:
            from clip_retrieval import build_frame_cache
            info = build_frame_cache(
                video_path=video_path,
                cache_dir=clip_cache_dir,
                fps=clip_fps,
                model_name=clip_model,
                pretrained=clip_pretrained,
            )
            print(
                f"  [clip] {info['manifest']['frame_count']} frames @ "
                f"{clip_fps}fps  model={clip_model}/{clip_pretrained}"
            )

    if uncached_idx:
        backend_label = (
            f"qwen ({qwen_vl_model})"
            if stage_b_mode == "vl_cot" and vl_answerer_backend == "qwen"
            else f"flash ({stage_b_model})"
        )
        print(
            f"  [stage_b/{backend_label}] mode={stage_b_mode} QA for "
            f"{len(uncached_idx)}/{len(questions)} pending questions ..."
        )
        cache_lock = Lock()

        def work(i: int) -> None:
            q = questions[i]
            key = stage_b_cache_key(q, stage_b_mode, retrieval_mode, vl_answerer_backend)
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
                clip_cache_dir=clip_cache_dir,
                clip_top_k=clip_top_k,
                clip_fps=clip_fps,
                clip_model=clip_model,
                clip_pretrained=clip_pretrained,
                retrieval_mode=retrieval_mode,
                decomposer_model=decomposer_model,
                vl_answerer_backend=vl_answerer_backend,
                qwen_vl_model=qwen_vl_model,
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

    predictions = [
        stage_b_cache[
            stage_b_cache_key(q, stage_b_mode, retrieval_mode, vl_answerer_backend)
        ]
        for q in questions
    ]
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
        "chunks_path": str(chunks_path),
        "stage_a_path": str(stage_a_path),
        "stage_a_chars": len(timeline or ""),
        "num_chunks": len(chunks_cache["chunks"]),
        "num_questions": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else float("nan"),
        "per_question": per_q,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--chunks_cache_dir", required=True,
                    help="Existing tier-2 cache dir containing {stem}/chunks.json.")
    ap.add_argument("--cache_dir", required=True,
                    help="Separate cache dir for Flash-aggregator outputs "
                         "(stage_a.txt + stage_b.json). Keeps the Qwen baseline intact.")
    ap.add_argument("--output_json", required=True)

    ap.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL,
                    help=f"Gemini model for aggregation. Default: {DEFAULT_AGGREGATOR_MODEL}")
    ap.add_argument("--stage_b_model", default=DEFAULT_STAGE_B_MODEL,
                    help=f"Gemini model for Stage B QA. Default: {DEFAULT_STAGE_B_MODEL}")
    ap.add_argument("--chunk_vlm_model", default=None,
                    help="Model slug for Stage A1 chunking when chunks.json is "
                         "missing. Default depends on --state_backend: "
                         f"qwen → {DEFAULT_CHUNK_VLM_MODEL}; "
                         f"flash → {DEFAULT_AGGREGATOR_MODEL}.")
    ap.add_argument("--state_backend", choices=["qwen", "flash"], default="qwen",
                    help="Backend for Stage A1 state extraction. qwen routes to "
                         "OpenRouter; flash routes to Gemini Flash with inline "
                         "frame parts. Default: qwen.")

    ap.add_argument("--stage_b_mode", choices=["plain", "cot", "feedback", "vl_cot"],
                    default="plain",
                    help="Stage B answering strategy. plain: one text call. "
                         "cot: JSON chain-of-evidence (evidence_spans + reasoning "
                         "+ answer). feedback: Stage B emits a sufficiency check; "
                         "on 'insufficient' the pipeline re-extracts 1-3 chunks "
                         "with a question-aware A1 prompt and re-answers with "
                         "the supplement (capped at 1 iteration). "
                         "vl_cot: multimodal COT — CLIP-retrieves top-K frames "
                         "from the video and passes them alongside the timeline "
                         "to the answerer.")
    ap.add_argument("--max_reextract_chunks", type=int, default=3,
                    help="Cap on chunks to re-extract per question in feedback "
                         "mode. Default 3.")

    ap.add_argument("--clip_cache_dir", default=None,
                    help="Where to store per-video decoded frames + CLIP "
                         "embeddings for vl_cot mode. Default: "
                         "<cache_dir>/clip. Reused across runs.")
    ap.add_argument("--clip_top_k", type=int, default=16,
                    help="Top-K frames retrieved per question in vl_cot mode. "
                         "Default 16.")
    ap.add_argument("--clip_fps", type=float, default=1.0,
                    help="Frame-decode fps for the CLIP cache in vl_cot mode. "
                         "Default 1.0.")
    ap.add_argument("--clip_model", default="ViT-L-16-SigLIP-384",
                    help="open_clip model name for vl_cot retrieval. "
                         "Default ViT-L-16-SigLIP-384.")
    ap.add_argument("--clip_pretrained", default="webli",
                    help="open_clip pretrained tag. Default webli (SigLIP).")

    ap.add_argument("--retrieval_mode",
                    choices=["single", "decomposed", "uniform"],
                    default="single",
                    help="Frame-selection strategy for vl_cot mode. single: "
                         "ordinal-stripped question -> one CLIP top-K query. "
                         "decomposed: Flash splits the question into 2-4 "
                         "object-focused sub-queries; CLIP ranks per "
                         "sub-query; Reciprocal Rank Fusion merges to top-K. "
                         "uniform: no CLIP — decode frames at --clip_fps and "
                         "return --clip_top_k frames sampled uniformly over "
                         "the full video (baseline to test whether retrieval "
                         "is actually helping). Default single.")
    ap.add_argument("--decomposer_model", default="gemini-3-flash-preview",
                    help="Gemini model used to decompose questions into "
                         "CLIP sub-queries. Used only when --retrieval_mode="
                         "decomposed. Default gemini-3-flash-preview.")

    ap.add_argument("--vl_answerer_backend", choices=["flash", "qwen"],
                    default="flash",
                    help="Backend for the vl_cot multimodal answerer. "
                         "flash: Gemini 3 Flash (current default). "
                         "qwen: OpenRouter -> --qwen_vl_model. Swap to qwen "
                         "when Flash's visual misreads dominate the residual "
                         "error bucket.")
    ap.add_argument("--qwen_vl_model",
                    default="qwen/qwen3-vl-235b-a22b-thinking",
                    help="OpenRouter model slug for the vl_cot answerer when "
                         "--vl_answerer_backend=qwen. Default matches the "
                         "state-extractor model.")

    ap.add_argument("--max_new_tokens_aggregator", type=int, default=65536)
    ap.add_argument("--max_new_tokens_stage_b", type=int, default=256)
    ap.add_argument("--max_new_tokens_chunk", type=int, default=4096)
    ap.add_argument("--max_concurrency_questions", type=int, default=4)
    ap.add_argument("--max_concurrency_chunks", type=int, default=4)

    ap.add_argument("--chunk_seconds", type=float, default=45.0)
    ap.add_argument("--frames_per_chunk", type=int, default=12)
    ap.add_argument("--frame_max_side", type=int, default=768)
    ap.add_argument("--frame_jpeg_quality", type=int, default=85)

    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--google_api_key_env", default="GOOGLE_API_KEY")
    ap.add_argument("--openrouter_api_key_env", default="OPENROUTER_API_KEY")
    args = ap.parse_args()

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
    backend_label = "Flash" if args.state_backend == "flash" else "OpenRouter"
    print(f"Loaded {len(samples)} samples from {args.questions_dir}")
    stage_b_label = (
        f"qwen (OpenRouter): {args.qwen_vl_model}"
        if args.stage_b_mode == "vl_cot" and args.vl_answerer_backend == "qwen"
        else f"Flash: {args.stage_b_model}"
    )
    print(f"Stage A1 ({backend_label}): {args.chunk_vlm_model} (only runs if chunks.json missing)")
    print(f"Aggregator (Flash):            {args.aggregator_model}")
    print(f"Stage B (mode={args.stage_b_mode}):      {stage_b_label}")
    print(f"Reading chunks from:           {args.chunks_cache_dir}")
    print(f"Flash-agg cache dir:           {args.cache_dir}")

    per_sample: List[Dict[str, Any]] = []
    total_correct = 0
    total_questions = 0
    chunks_root = Path(args.chunks_cache_dir)
    cache_root = Path(args.cache_dir)
    out_path = Path(args.output_json)
    clip_cache_root = (
        Path(args.clip_cache_dir) if args.clip_cache_dir else cache_root
    )

    for i, sample in enumerate(samples, start=1):
        print(f"[{i}/{len(samples)}] {sample['video_name']}")
        result = evaluate_sample(
            sample=sample,
            chunks_cache_dir=chunks_root,
            cache_dir=cache_root,
            gemini_client=gemini_client,
            openrouter_client=openrouter_client,
            aggregator_model=args.aggregator_model,
            stage_b_model=args.stage_b_model,
            chunk_vlm_model=args.chunk_vlm_model,
            chunk_seconds=args.chunk_seconds,
            frames_per_chunk=args.frames_per_chunk,
            frame_max_side=args.frame_max_side,
            frame_jpeg_quality=args.frame_jpeg_quality,
            max_new_tokens_aggregator=args.max_new_tokens_aggregator,
            max_new_tokens_stage_b=args.max_new_tokens_stage_b,
            max_new_tokens_chunk=args.max_new_tokens_chunk,
            max_concurrency_questions=args.max_concurrency_questions,
            max_concurrency_chunks=args.max_concurrency_chunks,
            state_backend=args.state_backend,
            stage_b_mode=args.stage_b_mode,
            max_reextract_chunks=args.max_reextract_chunks,
            clip_cache_dir=clip_cache_root,
            clip_top_k=args.clip_top_k,
            clip_fps=args.clip_fps,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            retrieval_mode=args.retrieval_mode,
            decomposer_model=args.decomposer_model,
            vl_answerer_backend=args.vl_answerer_backend,
            qwen_vl_model=args.qwen_vl_model,
        )
        per_sample.append(result)
        total_correct += result["num_correct"]
        total_questions += result["num_questions"]
        running = (total_correct / total_questions) if total_questions else float("nan")
        print(
            f"    acc={result['accuracy']:.4f}  "
            f"running={running:.4f}  "
            f"(n_chunks={result['num_chunks']})"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "aggregator_model": args.aggregator_model,
                    "stage_b_model": args.stage_b_model,
                    "stage_b_mode": args.stage_b_mode,
                    "state_backend": args.state_backend,
                    "mode": "tier2 flash-aggregator + flash-stage-b (chunks reused from tier-2 cache)",
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
