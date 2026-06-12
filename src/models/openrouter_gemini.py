"""
src/models/openrouter_gemini.py
------------------------------
OpenRouter Gemini backend using raw video input with frame fallback.
"""

import base64
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

from .base import BaseVideoQAModel
from .openai_gpt import _extract_frames_b64

_DEFAULT_MAX_CONCURRENCY = 4
_FALLBACK_N_FRAMES = 32
_DEFAULT_MAX_FRAME_LONGEST_SIDE = 480
_SUPPORTED_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
}


def _guess_video_mime_type(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if mime_type in _SUPPORTED_VIDEO_MIME_TYPES:
        return mime_type
    return "video/mp4"


def _encode_video_data_url(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with open(video_path, "rb") as handle:
        video_bytes = handle.read()

    if not video_bytes:
        raise ValueError(f"Video file is empty: {video_path}")

    mime_type = _guess_video_mime_type(video_path)
    encoded = base64.b64encode(video_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _serialize_response(response: Any) -> Any:
    return response.model_dump() if hasattr(response, "model_dump") else repr(response)


def _response_error_payload(response: Any) -> Optional[Dict[str, Any]]:
    payload = _serialize_response(response)
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return error
    return None


def _should_fallback_to_frames(response: Any) -> bool:
    error = _response_error_payload(response)
    if not error:
        return False

    code = error.get("code")
    message = str(error.get("message") or "").lower()
    return code in {502, 503, 504} or "aborted" in message or "timeout" in message


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        payload = _serialize_response(response)
        raise RuntimeError(f"API response did not include any choices: {payload}")

    message = getattr(choices[0], "message", None)
    if message is None:
        payload = _serialize_response(response)
        raise RuntimeError(f"API response choice did not include a message: {payload}")

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
                continue
            text_part = getattr(item, "text", None)
            if isinstance(text_part, str) and text_part.strip():
                parts.append(text_part.strip())
        if parts:
            return "\n".join(parts)

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    payload = _serialize_response(response)
    raise RuntimeError(f"API response did not contain text content: {payload}")


class OpenRouterGeminiModel(BaseVideoQAModel):
    """
    OpenRouter Gemini backend that prefers raw video and falls back to frames.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        n_frames: int = _FALLBACK_N_FRAMES,
        prefer_video: bool = True,
        max_frame_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        if max_frame_longest_side is not None and max_frame_longest_side <= 0:
            raise ValueError(
                f"max_frame_longest_side must be positive, got {max_frame_longest_side}"
            )
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.n_frames = n_frames
        self.prefer_video = prefer_video
        self.max_frame_longest_side = max_frame_longest_side
        self._client: Optional[OpenAI] = None

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

        video_data_url = _encode_video_data_url(video_path) if self.prefer_video else None
        frames_b64 = _extract_frames_b64(
            video_path,
            self.n_frames,
            max_longest_side=self.max_frame_longest_side,
        )
        results = [""] * len(normalized_questions)
        worker_count = min(self.max_concurrency, len(normalized_questions))

        if worker_count == 1:
            for idx, question in enumerate(normalized_questions):
                results[idx] = self._answer_one(question, video_data_url, frames_b64, max_new_tokens)
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._answer_one, question, video_data_url, frames_b64, max_new_tokens): idx
                for idx, question in enumerate(normalized_questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set; "
                f"required for model {self.model_id!r}."
            )

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _answer_one(
        self,
        question: str,
        video_data_url: Optional[str],
        frames_b64: List[str],
        max_new_tokens: int,
    ) -> str:
        client = self._get_client()
        if video_data_url is not None:
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": self._build_video_content(question, video_data_url)}],
                max_tokens=int(max_new_tokens),
            )
            if not _should_fallback_to_frames(response):
                return _extract_response_text(response)

        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": self._build_frame_content(question, frames_b64)}],
            max_tokens=int(max_new_tokens),
        )
        return _extract_response_text(response)

    def _build_video_content(self, question: str, video_data_url: str) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. Answer the question directly. When possible, cite rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"
        return [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_data_url}},
        ]

    def _build_frame_content(self, question: str, frames_b64: List[str]) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video frames. Answer the question directly. When possible, cite rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"
        content: List[Dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_b64}",
                    "detail": "low",
                },
            }
            for frame_b64 in frames_b64
        ]
        content.append({"type": "text", "text": prompt})
        return content
