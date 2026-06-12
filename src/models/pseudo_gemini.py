"""
src/models/pseudo_gemini.py
---------------------------
Pseudo Gemini backend for preflight token estimation.

This backend mirrors the frame-based Gemini prompt shape but calls
``count_tokens(...)`` instead of ``generate_content(...)``. It is useful when
you want to estimate how many Gemini tokens a batch would consume before
running the real model.

Use it with a model ID prefix such as:

    pseudo_gemini/gemini-2.5-pro

The prefix is kept in the cache namespace so estimates do not collide with
real Gemini answer caches.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .gemini import (
    GeminiModel,
    _DEFAULT_MAX_CONCURRENCY,
    _DEFAULT_N_FRAMES,
    _extract_frames_bytes,
)

_ESTIMATE_KEYS = (
    "requests",
    "estimated_input_tokens",
    "estimated_output_tokens",
    "estimated_total_tokens",
    "cached_input_tokens",
)
_IMAGE_TOKENS_PER_TILE = 258
_SMALL_IMAGE_MAX_DIM = 384
_IMAGE_TILE_DIM = 768
_TEXT_CHARS_PER_TOKEN = 4


def _empty_estimate_stats() -> Dict[str, int]:
    return {key: 0 for key in _ESTIMATE_KEYS}


def _merge_estimate_stats(target: Dict[str, int], update: Dict[str, int]) -> None:
    for key in _ESTIMATE_KEYS:
        target[key] += int(update.get(key, 0) or 0)


class PseudoGeminiModel(GeminiModel):
    """
    Gemini-backed token estimator using frame prompts only.

    Parameters
    ----------
    model_id : str
        Display model identifier kept for cache namespace separation. This may
        include a pseudo-model prefix such as ``"pseudo_gemini/gemini-2.5-pro"``.
    api_model_id : str | None
        Actual Gemini model name to send to the API. If omitted, falls back to
        ``model_id``.
    n_frames : int
        Number of frames to sample and include in the estimation prompt.
        Default: 128.
    estimated_output_tokens_per_answer : int | None
        Optional fixed output-token estimate per question. If omitted, the
        estimator uses ``max_new_tokens`` as a conservative upper bound.
    count_mode : str
        One of ``"auto"``, ``"api"``, or ``"offline"``.
        ``"auto"`` uses the Gemini API when a key is available and falls back
        to offline estimation otherwise. Default: ``"auto"``.
    print_estimate : bool
        If True, print a summary for each estimated batch. Default: True.
    estimate_log_path : str | None
        Optional JSONL log path to append estimate summaries to. Default: None.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        api_model_id: Optional[str] = None,
        n_frames: int = _DEFAULT_N_FRAMES,
        api_key_env: str = "GEMINI_API_KEY",
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        estimated_output_tokens_per_answer: Optional[int] = None,
        count_mode: str = "auto",
        print_estimate: bool = True,
        estimate_log_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            prompt_method=prompt_method,
            n_frames=n_frames,
            video_upload=False,
            api_key_env=api_key_env,
            max_concurrency=max_concurrency,
            print_usage=False,
            usage_log_path=None,
        )
        if estimated_output_tokens_per_answer is not None and estimated_output_tokens_per_answer < 0:
            raise ValueError(
                "estimated_output_tokens_per_answer must be non-negative when provided"
            )
        if count_mode not in {"auto", "api", "offline"}:
            raise ValueError(
                f"count_mode must be one of 'auto', 'api', or 'offline', got {count_mode!r}"
            )

        self.api_model_id = api_model_id or model_id
        self.estimated_output_tokens_per_answer = estimated_output_tokens_per_answer
        self.count_mode = count_mode
        self.print_estimate = print_estimate
        self.estimate_log_path = estimate_log_path
        self.estimate_totals: Dict[str, int] = _empty_estimate_stats()
        self.last_batch_estimate: Dict[str, int] = _empty_estimate_stats()
        self.last_batch_estimate_mode = self._resolve_count_mode()

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        if not questions:
            return []

        normalized_questions: List[str] = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            normalized_questions.append(question)

        frames = _extract_frames_bytes(video_path, self.n_frames)
        answers, batch_estimate = self._estimate_questions_with_frames(
            normalized_questions,
            frames,
            max_new_tokens,
        )
        self._finalize_batch_estimate(
            video_path=video_path,
            n_questions=len(normalized_questions),
            max_new_tokens=max_new_tokens,
            batch_estimate=batch_estimate,
        )
        return answers

    def _estimate_questions_with_frames(
        self,
        questions: List[str],
        frames: List[bytes],
        max_new_tokens: int,
    ) -> tuple[List[str], Dict[str, int]]:
        results = [""] * len(questions)
        batch_estimate = _empty_estimate_stats()
        worker_count = min(self.max_concurrency, len(questions))

        if worker_count == 1:
            for idx, question in enumerate(questions):
                answer, estimate = self._estimate_one_with_frames(question, frames, max_new_tokens)
                results[idx] = answer
                _merge_estimate_stats(batch_estimate, estimate)
            return results, batch_estimate

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._estimate_one_with_frames, question, frames, max_new_tokens): idx
                for idx, question in enumerate(questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                answer, estimate = future.result()
                results[idx] = answer
                _merge_estimate_stats(batch_estimate, estimate)

        return results, batch_estimate

    def _estimate_one_with_frames(
        self,
        question: str,
        frames: List[bytes],
        max_new_tokens: int,
    ) -> tuple[str, Dict[str, int]]:
        mode = self._resolve_count_mode()
        self.last_batch_estimate_mode = mode
        estimated_input_tokens = 0
        cached_input_tokens = 0

        if mode == "api":
            client = self._get_client()
            contents = self._build_frame_contents(question, frames)
            response = client.models.count_tokens(
                model=self.api_model_id,
                contents=contents,
            )
            estimated_input_tokens = int(getattr(response, "total_tokens", 0) or 0)
            cached_input_tokens = int(getattr(response, "cached_content_token_count", 0) or 0)
        else:
            estimated_input_tokens = self._estimate_offline_input_tokens(question, frames)

        estimated_output_tokens = self._estimate_output_tokens(max_new_tokens)
        estimated_total_tokens = estimated_input_tokens + estimated_output_tokens

        estimate = {
            "requests": 1,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "cached_input_tokens": cached_input_tokens,
        }
        return self._build_placeholder_answer(estimate), estimate

    def _estimate_output_tokens(self, max_new_tokens: int) -> int:
        if self.estimated_output_tokens_per_answer is not None:
            return int(self.estimated_output_tokens_per_answer)
        return int(max_new_tokens)

    def _resolve_count_mode(self) -> str:
        if self.count_mode == "offline":
            return "offline"
        api_key = os.environ.get(self.api_key_env)
        if self.count_mode == "api":
            if not api_key:
                raise RuntimeError(
                    f"count_mode='api' requires environment variable {self.api_key_env!r}."
                )
            return "api"
        return "api" if api_key else "offline"

    def _estimate_offline_input_tokens(self, question: str, frames: List[bytes]) -> int:
        prompt = self._build_prompt(question, source_name="video frames")
        text_tokens = self._estimate_text_tokens(prompt)
        image_tokens = sum(self._estimate_image_tokens_from_bytes(frame) for frame in frames)
        return text_tokens + image_tokens

    def _estimate_text_tokens(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        return max(1, math.ceil(len(text) / _TEXT_CHARS_PER_TOKEN))

    def _estimate_image_tokens_from_bytes(self, frame_bytes: bytes) -> int:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode JPEG bytes while estimating image tokens")

        height, width = image.shape[:2]
        if width <= _SMALL_IMAGE_MAX_DIM and height <= _SMALL_IMAGE_MAX_DIM:
            return _IMAGE_TOKENS_PER_TILE

        tiles_w = math.ceil(width / _IMAGE_TILE_DIM)
        tiles_h = math.ceil(height / _IMAGE_TILE_DIM)
        return _IMAGE_TOKENS_PER_TILE * tiles_w * tiles_h

    def _build_placeholder_answer(self, estimate: Dict[str, int]) -> str:
        return (
            "[pseudo-gemini] "
            f"estimated_input_tokens={estimate['estimated_input_tokens']} "
            f"estimated_output_tokens={estimate['estimated_output_tokens']} "
            f"estimated_total_tokens={estimate['estimated_total_tokens']}"
        )

    def _finalize_batch_estimate(
        self,
        video_path: str,
        n_questions: int,
        max_new_tokens: int,
        batch_estimate: Dict[str, int],
    ) -> None:
        self.last_batch_estimate = dict(batch_estimate)
        _merge_estimate_stats(self.estimate_totals, batch_estimate)

        payload = {
            "model_id": self.model_id,
            "api_model_id": self.api_model_id,
            "count_mode": self.last_batch_estimate_mode,
            "prompt_method": self.prompt_method,
            "video_path": video_path,
            "n_questions": n_questions,
            "n_frames": self.n_frames,
            "max_new_tokens": int(max_new_tokens),
            "estimated_output_tokens_per_answer": self._estimate_output_tokens(max_new_tokens),
            "batch_estimate": dict(batch_estimate),
            "cumulative_estimate": dict(self.estimate_totals),
        }

        if self.print_estimate:
            print(self._format_estimate_message(payload), flush=True)

        if self.estimate_log_path:
            log_dir = os.path.dirname(self.estimate_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.estimate_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _format_estimate_message(self, payload: Dict[str, Any]) -> str:
        batch = payload["batch_estimate"]
        cumulative = payload["cumulative_estimate"]
        video_name = os.path.basename(payload["video_path"])
        return (
            "[Pseudo Gemini estimate] "
            f"video={video_name} "
            f"mode={payload['count_mode']} "
            f"frames={payload['n_frames']} "
            f"questions={payload['n_questions']} "
            f"per_answer_output_assumption={payload['estimated_output_tokens_per_answer']} "
            f"batch_input={batch['estimated_input_tokens']} "
            f"batch_output_est={batch['estimated_output_tokens']} "
            f"batch_total_est={batch['estimated_total_tokens']} "
            f"cumulative_input={cumulative['estimated_input_tokens']} "
            f"cumulative_output_est={cumulative['estimated_output_tokens']} "
            f"cumulative_total_est={cumulative['estimated_total_tokens']} "
            f"cached_input={cumulative['cached_input_tokens']}"
        )
