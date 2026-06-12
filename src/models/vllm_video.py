"""
src/models/vllm_video.py
------------------------
OpenAI-compatible backend for locally served vLLM models using native video
file paths instead of extracted image frames.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import av
except Exception as exc:
    raise RuntimeError(
        "Failed to import PyAV. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

try:
    from openai import OpenAI
except Exception as exc:
    raise RuntimeError(
        "Failed to import the OpenAI SDK. Install with: uv sync --group openai\n"
        f"Error: {exc}"
    ) from exc

from .base import BaseVideoQAModel

_DEFAULT_MAX_CONCURRENCY = 1
_DEFAULT_VLLM_API_KEY = "EMPTY"
_DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def _serialize_response(response: Any) -> Any:
    return response.model_dump() if hasattr(response, "model_dump") else repr(response)


def _get_message_field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)

    value = getattr(message, name, None)
    if value is not None:
        return value

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict) and name in model_extra:
        return model_extra[name]

    if hasattr(message, "model_dump"):
        payload = message.model_dump()
        if isinstance(payload, dict):
            return payload.get(name)

    return None


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(f"API response did not include any choices: {_serialize_response(response)}")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError(f"API response choice did not include a message: {_serialize_response(response)}")

    content = _get_message_field(message, "content")
    if isinstance(content, str) and content.strip():
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

    for attr_name in ("reasoning", "reasoning_content"):
        reasoning = _get_message_field(message, attr_name)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

    refusal = _get_message_field(message, "refusal")
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    raise RuntimeError(f"API response did not contain text content: {_serialize_response(response)}")


def _resolve_video_path(video_path: str) -> Path:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"Video file does not contain any video streams: {video_path}")

    return path


def _video_path_to_file_uri(video_path: str) -> str:
    return _resolve_video_path(video_path).as_uri()


class VLLMVideoModel(BaseVideoQAModel):
    """
    vLLM OpenAI-compatible backend that sends a local video file URI.

    The vLLM server must be launched with ``--allowed-local-media-path`` that
    includes the benchmark video directory, and with video inputs enabled via
    ``--limit-mm-per-prompt.video``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        api_key_env: str = "VLLM_API_KEY",
        base_url: Optional[str] = None,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_concurrency = max_concurrency
        self.extra_body = dict(extra_body) if extra_body else None
        self._client: Optional[OpenAI] = None

    @property
    def cache_namespace(self) -> str:
        safe_id = f"vllm_video/{self.model_id}".replace("/", "_").replace(":", "-")
        return f"{safe_id}__{self.prompt_method}"

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

        video_uri = _video_path_to_file_uri(video_path)
        results = [""] * len(normalized_questions)
        worker_count = min(self.max_concurrency, len(normalized_questions))

        if worker_count == 1:
            for idx, question in enumerate(normalized_questions):
                results[idx] = self._answer_one(question, video_uri, max_new_tokens)
            return results

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._answer_one, question, video_uri, max_new_tokens): idx
                for idx, question in enumerate(normalized_questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return results

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env, _DEFAULT_VLLM_API_KEY)
        base_url = self.base_url or os.environ.get("VLLM_BASE_URL", _DEFAULT_VLLM_BASE_URL)

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def _answer_one(self, question: str, video_uri: str, max_new_tokens: int) -> str:
        client = self._get_client()
        request_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": self._build_content(question, video_uri)}],
            "max_tokens": int(max_new_tokens),
        }
        if self.extra_body:
            request_kwargs["extra_body"] = self.extra_body
        else:
            request_kwargs["extra_body"] = {"mm_processor_kwargs": {"fps": 2 ,"do_sample_frames": True}}

        # breakpoint()
        response = client.chat.completions.create(**request_kwargs)
        return _extract_response_text(response)

    def _build_content(self, question: str, video_uri: str) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. Answer the question directly. When possible, cite rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"
        return [
            {"type": "text", "text": prompt},
            {
                "type": "video_url",
                "video_url": {"url": video_uri},
                "uuid": video_uri,
            },
        ]
