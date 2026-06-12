"""
src/models/traveler.py
-----------------------
TraveLER-style multi-agent pipeline for video QA.

Architecture: Planner → Retriever → Extractor → Summarizer → Evaluator
Model backend: any vision-language model served via vLLM (OpenAI-compatible API).

The pipeline is question-specific — each question gets its own memory bank and
iterative reasoning loop.  This fits naturally into ``answer_questions`` by
running the full Answerer once per question.

Routing convention
------------------
    load_model("traveler/Qwen/Qwen3.5-2B")

The part after ``"traveler/"`` is the vLLM model name.
The vLLM server URL is read from the env var ``VLLM_BASE_URL``
(default: ``"http://localhost:8123/v1"``).

vLLM setup (run once before benchmarking)
------------------------------------------
    CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.5-2B \\
        --port 8123 \\
        --gpu-memory-utilization 0.25 \\
        --max-model-len 8192 \\
        --reasoning-parser qwen3 \\
        --default-chat-template-kwargs '{"enable_thinking": false}' \\
        --enable-prefix-caching \\
        --max-cudagraph-capture-size 256

Required Python package: ``openai`` (already in the ``openai`` dependency group).
No transformers or GPU memory needed in the benchmark process itself.
"""

import ast
import base64
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage

from .base import BaseVideoQAModel

# ---------------------------------------------------------------------------
# Prompt templates (adapted from TraveLER — MCQ references removed)
# ---------------------------------------------------------------------------

_PLANNER_FIRST = """\
You are given a question about a video and some initial observations from individual frames.

Question: {question}

The video is {length:.1f} seconds long.
Initial observations (dict keyed by timestamp):
{info}

Create a concise step-by-step plan (at most 5 steps) to gather the visual evidence needed to answer the question.

Rules:
1. You only have access to individual frames — no audio.
2. You can navigate to any timestamp and ask visual questions about that frame.
3. Make sure you actually locate the key moment(s) the question refers to.

Return ONLY the numbered plan after PLAN:

PLAN:"""

_PLANNER_CONT = """\
You are given a question about a video, the current memory of observations, and a reason why the question hasn't been answered yet.

Question: {question}

The video is {length:.1f} seconds long.
Current memory (dict keyed by timestamp):
{info}

Reason the question is still unanswered: {explanation}

Create a new concise step-by-step plan (at most 5 steps) to collect the missing evidence.

Rules:
1. You only have access to individual frames — no audio.
2. Build on what you already know; don't revisit timestamps already in memory.

Return ONLY the numbered plan after PLAN:

PLAN:"""

_RETRIEVER = """\
You are navigating a {length:.1f}-second video to answer: "{question}"

Current memory (keyed by timestamp):
{info}

Plan to follow:
{plan}

You are currently at second {curr:.1f}.

Choose the next timestamp (in seconds) to inspect. Requirements:
- At least 5 seconds away from current position.
- Not a timestamp already covered in memory.
- Most informative for the plan.

Return ONLY a single Python float (e.g. 12.5). No explanation."""

_EXTRACTOR = """\
You are analysing a {length:.1f}-second video to answer: "{question}"

Current memory:
{info}

Plan:
{plan}

You are now looking at the frame at second {current_time:.1f}.
Frame caption: {caption}

Formulate up to {num_questions} concise questions about this frame to help answer the main question.
Rules:
- Questions must be answerable from a SINGLE frame (no temporal or cross-frame questions).
- Focus on visual evidence directly relevant to the question.

Return a Python list of strings only — no backticks, no extra text.
Example: ["Is someone wearing a blue shirt?", "Is a ball visible near the net?"]"""

_SUMMARIZER = """\
Summarise the following observations about a video in order to best answer the question: "{question}"

Observations (Python dict, keys = timestamps, values = lists of strings):
{info}

Keep only information relevant to the question. Do NOT answer the question here.

Return the summary in the SAME Python dict format (dict[str, list[str]]).
Use double quotes. No backticks, no language tags.

OUTPUT:"""

_EVALUATOR = """\
Evaluate whether there is enough visual evidence to answer the following question.

Question: {question}
Video length: {length:.1f} seconds.

Plan that was followed:
{plan}

Memory of observations (keyed by timestamp):
{info}

If the evidence is sufficient and conclusive, provide your answer after "Final Answer:".
If evidence is insufficient, explain why and write "Final Answer: None".

Be strict — only commit to an answer if the evidence clearly supports it.
Let's think step by step.

Final Answer:"""

_FINAL_ANSWER = """\
Answer the following question about a video using all available visual evidence.

Question: {question}
Video length: {length:.1f} seconds.

All collected observations (keyed by timestamp):
{info}

Be concise and direct. If the question is yes/no, start with "Yes" or "No" then explain briefly.

Final Answer:"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_sec(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{int(s):02d}.{int((s - int(s)) * 100):02d}"


def _timestamp_key(sec: float, total: float) -> str:
    return f"{_fmt_sec(sec)}/{_fmt_sec(total)}"


def _frange(start: float, stop: float, step: float = 1.0) -> List[float]:
    vals, v = [], start
    while v <= stop + 1e-9:
        vals.append(round(v, 3))
        v += step
    return vals


def _merge_info(
    a: Dict[str, List[str]],
    b: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, []) + v
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# VideoState — internal to the pipeline
# ---------------------------------------------------------------------------

class _VideoState:
    def __init__(
        self,
        frames: List[PILImage.Image],
        duration: float,
        question: str,
        fps: int,
    ) -> None:
        self.frames = frames
        self.duration = duration
        self.question = question
        self.fps = fps
        self.info: Dict[str, List[str]] = {}

    def get_frame(self, second: float) -> PILImage.Image:
        idx = max(0, min(int(second * self.fps), len(self.frames) - 1))
        return self.frames[idx]


def _load_video(video_path: str, target_fps: int) -> Tuple[List[PILImage.Image], float]:
    """Decode video to PIL frames at target_fps via OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = original_fps / target_fps

    frames: List[PILImage.Image] = []
    frame_id = 0.0
    while frame_id < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_id += ratio
    cap.release()
    duration = len(frames) / target_fps
    return frames, duration


# ---------------------------------------------------------------------------
# vLLM client wrapper
# ---------------------------------------------------------------------------

class _VLMClient:
    """
    Thin OpenAI-compatible client for a vLLM-served vision-language model.
    Separates LLM (text-only) and VLM (image+text) sampling params per the
    Qwen3.5 official recommendation.
    """

    # Recommended sampling params
    _LLM_PARAMS = dict(temperature=1.0, top_p=1.0, presence_penalty=2.0, top_k=20)
    _VLM_PARAMS = dict(temperature=0.7, top_p=0.8, presence_penalty=1.5, top_k=20)

    def __init__(self, base_url: str, model_name: str, api_key: str = "token-abc123") -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_name

    @staticmethod
    def _encode(image: PILImage.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def _call(self, messages: list, params: dict, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"top_k": params["top_k"]},
            temperature=params["temperature"],
            top_p=params["top_p"],
            presence_penalty=params["presence_penalty"],
        )
        return resp.choices[0].message.content

    def text(self, prompt: str, max_tokens: int = 512) -> str:
        return self._call(
            [{"role": "user", "content": prompt}],
            self._LLM_PARAMS,
            max_tokens,
        )

    def vision(self, image: PILImage.Image, prompt: str, max_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": self._encode(image)}},
            {"type": "text", "text": prompt},
        ]}]
        return self._call(messages, self._VLM_PARAMS, max_tokens)


# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------

class _Planner:
    def __init__(self, client: _VLMClient) -> None:
        self.client = client
        self.explanation: str = "Insufficient evidence collected so far."

    def forward(self, video: _VideoState, first: bool = False) -> str:
        if first:
            prompt = _PLANNER_FIRST.format(
                question=video.question, length=video.duration, info=video.info
            )
        else:
            prompt = _PLANNER_CONT.format(
                question=video.question,
                length=video.duration,
                info=video.info,
                explanation=self.explanation,
            )
        return self.client.text(prompt)


class _Retriever:
    def __init__(self, client: _VLMClient, view_range: int) -> None:
        self.client = client
        self.view_range = view_range
        self.curr: float = 0.0

    def forward(
        self, video: _VideoState, plan: str
    ) -> Tuple[List[float], List[PILImage.Image]]:
        prompt = _RETRIEVER.format(
            length=video.duration,
            question=video.question,
            info=video.info,
            plan=plan,
            curr=self.curr,
        )
        raw = self.client.text(prompt).strip()
        try:
            goto = float(raw)
        except ValueError:
            nums = re.findall(r"\d+\.?\d*", raw)
            goto = float(nums[0]) if nums else video.duration / 2
        goto = max(0.0, min(goto, video.duration))
        self.curr = goto
        lo = max(0.0, goto - self.view_range)
        hi = min(video.duration, goto + self.view_range)
        seconds = _frange(lo, hi, 1.0)
        return seconds, [video.get_frame(s) for s in seconds]


class _Extractor:
    def __init__(self, client: _VLMClient, num_questions: int, max_retries: int) -> None:
        self.client = client
        self.num_questions = num_questions
        self.max_retries = max_retries

    def forward(
        self,
        video: _VideoState,
        plan: str,
        seconds: List[float],
        frames: List[PILImage.Image],
    ) -> Dict[str, List[str]]:
        output: Dict[str, List[str]] = {}
        for sec, frame in zip(seconds, frames):
            key = _timestamp_key(sec, video.duration)
            caption = self.client.vision(frame, "Describe the scene in less than 2 sentences.")
            results = [caption]

            questions: List[str] = []
            for _ in range(self.max_retries):
                prompt = _EXTRACTOR.format(
                    length=video.duration,
                    question=video.question,
                    info=video.info or "(none yet)",
                    plan=plan,
                    current_time=sec,
                    caption=caption,
                    num_questions=self.num_questions,
                )
                raw = self.client.text(prompt)
                try:
                    qs = ast.literal_eval(raw)
                    if isinstance(qs, list):
                        questions = qs
                        break
                except Exception:
                    pass

            prefix = "If there are factual errors in the question, point them out; otherwise answer briefly."
            for q in questions:
                a = self.client.vision(frame, f"{prefix}\nQuestion: {q}")
                results.append(f"Q: {q} | A: {a}")

            output[key] = results
        return output


class _Summarizer:
    def __init__(self, client: _VLMClient, max_retries: int) -> None:
        self.client = client
        self.max_retries = max_retries

    def forward(
        self, video: _VideoState, new_info: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        combined = _merge_info(video.info, new_info)
        prompt = _SUMMARIZER.format(question=video.question, info=combined)
        for _ in range(self.max_retries):
            raw = self.client.text(prompt)
            raw = raw.split("OUTPUT:")[-1].strip()
            try:
                summary = ast.literal_eval(raw)
                if isinstance(summary, dict):
                    return summary
            except Exception:
                pass
        return combined


class _Evaluator:
    def __init__(self, client: _VLMClient) -> None:
        self.client = client

    def forward(
        self,
        video: _VideoState,
        plan: str,
        final: bool = False,
    ) -> Tuple[Optional[str], str]:
        if final:
            prompt = _FINAL_ANSWER.format(
                question=video.question, length=video.duration, info=video.info
            )
        else:
            prompt = _EVALUATOR.format(
                question=video.question,
                length=video.duration,
                plan=plan,
                info=video.info,
            )
        raw = self.client.text(prompt)
        parts = raw.split("Final Answer:")
        explanation = parts[0].strip() if len(parts) > 1 else ""
        answer_str = parts[-1].strip()
        if answer_str.lower() in ("none", ""):
            return None, explanation
        return answer_str, explanation


# ---------------------------------------------------------------------------
# Answerer — orchestrator
# ---------------------------------------------------------------------------

class _Answerer:
    def __init__(
        self,
        client: _VLMClient,
        max_iters: int,
        max_retries: int,
        view_range: int,
        num_questions: int,
        init_frames: int,
    ) -> None:
        self.client = client
        self.max_iters = max_iters
        self.init_frames = init_frames
        self.planner = _Planner(client)
        self.retriever = _Retriever(client, view_range)
        self.extractor = _Extractor(client, num_questions, max_retries)
        self.summarizer = _Summarizer(client, max_retries)
        self.evaluator = _Evaluator(client)

    def _init_memory(self, video: _VideoState) -> None:
        gotos = list(np.linspace(0, video.duration, self.init_frames, endpoint=False))
        init_info: Dict[str, List[str]] = {}
        for sec in gotos:
            frame = video.get_frame(sec)
            caption = self.client.vision(frame, "Describe the scene in less than 2 sentences.")
            init_info[_timestamp_key(sec, video.duration)] = [caption]
        video.info = init_info
        self.retriever.curr = video.duration / 2

    def forward(self, video: _VideoState) -> str:
        self._init_memory(video)
        plan = ""
        for iteration in range(self.max_iters):
            plan = self.planner.forward(video, first=(iteration == 0))
            if iteration == 0:
                seconds = list(np.linspace(0, video.duration, self.init_frames, endpoint=False))
                frames = [video.get_frame(s) for s in seconds]
            else:
                seconds, frames = self.retriever.forward(video, plan)
            new_info = self.extractor.forward(video, plan, seconds, frames)
            video.info = self.summarizer.forward(video, new_info)
            answer, explanation = self.evaluator.forward(video, plan)
            self.planner.explanation = explanation
            if answer is not None:
                return answer
        # Exhausted — force final answer
        answer, _ = self.evaluator.forward(video, plan, final=True)
        return answer or "Unable to determine answer from available visual evidence."


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class TraveLERModel(BaseVideoQAModel):
    """
    TraveLER-style multi-agent pipeline for video QA.

    The model must be served externally via vLLM before running the benchmark.
    See module docstring for the serve command.

    Parameters
    ----------
    model_id : str
        Full identifier: ``"traveler/<vllm_model_name>"``, e.g.
        ``"traveler/Qwen/Qwen3.5-2B"``.
    prompt_method : str
        Cache namespace suffix.  Default: ``"vanilla"``.
    vllm_base_url : str | None
        Override the vLLM server URL.  Falls back to the ``VLLM_BASE_URL``
        env var, then ``"http://localhost:8123/v1"``.
    max_iters : int
        Max Planner→Evaluator loops per question.  Default: 3.
    view_range : int
        ± seconds around the retrieved timestamp.  Default: 2.
    num_questions : int
        Extractor questions per frame.  Default: 3.
    init_frames : int
        Evenly-spaced frames for memory initialisation.  Default: 5.
    video_fps : int
        FPS to which the video is down-sampled before loading.  Default: 10.
    max_retries : int
        Retries on LLM parse errors (Extractor, Summarizer).  Default: 3.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        vllm_base_url: Optional[str] = None,
        max_iters: int = 3,
        view_range: int = 2,
        num_questions: int = 3,
        init_frames: int = 5,
        video_fps: int = 10,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model_id, prompt_method)

        # Resolve vLLM model name from model_id ("traveler/Qwen/Qwen3.5-2B")
        _, vllm_model_name = model_id.split("/", 1)

        base_url = (
            vllm_base_url
            or os.environ.get("VLLM_BASE_URL", "http://localhost:8123/v1")
        )
        self._client = _VLMClient(base_url=base_url, model_name=vllm_model_name)
        self._max_iters = max_iters
        self._view_range = view_range
        self._num_questions = num_questions
        self._init_frames = init_frames
        self._video_fps = video_fps
        self._max_retries = max_retries

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Run the full TraveLER pipeline once per question.

        Each question gets its own memory bank and reasoning loop, which is
        correct for the multi-agent design but expensive (N × pipeline cost).
        Answers are cached by the benchmark framework so each (video, question)
        pair is only computed once.
        """
        frames, duration = _load_video(video_path, self._video_fps)
        results = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            video = _VideoState(
                frames=frames,
                duration=duration,
                question=question,
                fps=self._video_fps,
            )
            answerer = _Answerer(
                client=self._client,
                max_iters=self._max_iters,
                max_retries=self._max_retries,
                view_range=self._view_range,
                num_questions=self._num_questions,
                init_frames=self._init_frames,
            )
            results.append(answerer.forward(video))
        return results
