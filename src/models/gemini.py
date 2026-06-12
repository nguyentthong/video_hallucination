"""
src/models/gemini.py
---------------------
Google Gemini API backend.

This backend supports two Gemini-native input modes:

1. Upload the whole video once via the Files API and reuse the uploaded asset
   across all questions in the batch.
2. Extract evenly-spaced JPEG frames locally, downscale them, and send them
   inline as image parts when ``video_upload=False``.

Authentication uses the native Gemini env var ``GEMINI_API_KEY`` by default.
"""

import mimetypes
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

try:
    import av
except Exception as exc:
    raise RuntimeError(
        "Failed to import PyAV. Install with: uv sync --group gemini\n"
        f"Error: {exc}"
    ) from exc

import cv2

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    raise RuntimeError(
        "Failed to import the Google GenAI SDK. Install with: uv sync --group gemini\n"
        f"Error: {exc}"
    ) from exc

from .base import BaseVideoQAModel

_DEFAULT_N_FRAMES = 64
_DEFAULT_MAX_FRAME_LONGEST_SIDE = 480
_DEFAULT_MAX_CONCURRENCY = 4
_DEFAULT_UPLOAD_TIMEOUT_S = 300.0
_DEFAULT_UPLOAD_POLL_INTERVAL_S = 2.0
_DEFAULT_VIDEO_MIME_TYPE = "video/mp4"
_USAGE_KEYS = (
    "requests",
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "thought_tokens",
    "total_tokens",
)


def _resize_longest_side_bgr(bgr_frame: Any, max_longest_side: Optional[int]) -> Any:
    if max_longest_side is None:
        return bgr_frame
    if max_longest_side <= 0:
        raise ValueError(f"max_longest_side must be positive, got {max_longest_side}")

    h, w = bgr_frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid frame shape: {bgr_frame.shape}")

    longest_side = max(h, w)
    if longest_side <= max_longest_side:
        return bgr_frame

    scale = max_longest_side / longest_side
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg_bytes(
    frame: av.VideoFrame,
    max_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
) -> bytes:
    bgr_frame = frame.to_ndarray(format="bgr24")
    bgr_frame = _resize_longest_side_bgr(bgr_frame, max_longest_side)
    ok, buf = cv2.imencode(".jpg", bgr_frame)
    if not ok:
        raise ValueError("Failed to encode sampled video frame as JPEG")
    return buf.tobytes()


def _sample_evenly(items: List[bytes], n: int) -> List[bytes]:
    if not items:
        raise ValueError("Cannot sample from an empty frame list")
    if n <= 1:
        return [items[len(items) // 2]]
    if len(items) == 1:
        return items * n
    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    return [items[idx] for idx in indices]


def _extract_frames_bytes_sequential(
    video_path: str,
    n_frames: int,
    max_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
) -> List[bytes]:
    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        decoded: List[bytes] = []
        for frame in container.decode(stream):
            decoded.append(_encode_frame_jpeg_bytes(frame, max_longest_side=max_longest_side))

        if not decoded:
            raise ValueError(f"Failed to decode any frames sequentially from video: {video_path}")

        return _sample_evenly(decoded, n_frames)


def _get_video_stream(container: av.container.input.InputContainer) -> av.video.stream.VideoStream:
    if not container.streams.video:
        raise ValueError("Video file does not contain any video streams")
    return container.streams.video[0]


def _extract_frames_bytes(
    video_path: str,
    n_frames: int,
    max_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
) -> List[bytes]:
    """
    Sample exactly ``n_frames`` uniformly-spaced frame slots as JPEG bytes.

    If the source video has fewer than ``n_frames`` decoded frames, some slots
    may map to the same underlying frame so the prompt remains a fixed size.
    When ``max_longest_side`` is set, larger sampled frames are downscaled
    before JPEG encoding.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with av.open(video_path) as container:
        stream = _get_video_stream(container)
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            return _extract_frames_bytes_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        if n_frames == 1:
            indices = [total_frames // 2]
        else:
            indices = [
                round(i * (total_frames - 1) / (n_frames - 1))
                for i in range(n_frames)
            ]

        wanted = Counter(indices)
        cache: Dict[int, bytes] = {}
        last_needed = max(wanted)

        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx > last_needed:
                break
            if frame_idx in wanted and frame_idx not in cache:
                cache[frame_idx] = _encode_frame_jpeg_bytes(
                    frame,
                    max_longest_side=max_longest_side,
                )
                if len(cache) == len(wanted):
                    break

        if not cache:
            return _extract_frames_bytes_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        if any(idx not in cache for idx in wanted):
            return _extract_frames_bytes_sequential(
                video_path,
                n_frames,
                max_longest_side=max_longest_side,
            )

        return [cache[idx] for idx in indices]


def _guess_video_mime_type(video_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(video_path)
    if isinstance(mime_type, str) and mime_type.startswith("video/"):
        return mime_type
    return _DEFAULT_VIDEO_MIME_TYPE


def _serialize_response(payload: Any) -> Any:
    return payload.model_dump() if hasattr(payload, "model_dump") else repr(payload)


def _extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = getattr(response, "parts", None)
    if isinstance(parts, list):
        texts: List[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                texts.append(part_text.strip())
        if texts:
            return "\n".join(texts)

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        texts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            candidate_parts = getattr(content, "parts", None)
            if not isinstance(candidate_parts, list):
                continue
            for part in candidate_parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    texts.append(part_text.strip())
        if texts:
            return "\n".join(texts)

    payload = _serialize_response(response)
    raise RuntimeError(f"Gemini response did not contain any text content: {payload}")


def _enum_name(value: Any) -> str:
    if value is None:
        return ""

    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    if isinstance(value, str):
        return value.upper()

    rendered = str(value)
    if "." in rendered:
        rendered = rendered.rsplit(".", 1)[-1]
    return rendered.upper()


def _format_file_error(uploaded_file: Any) -> str:
    error = getattr(uploaded_file, "error", None)
    if error is None:
        return "Unknown file-processing error"

    message = getattr(error, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()

    payload = _serialize_response(error)
    return str(payload)


def _empty_usage_stats() -> Dict[str, int]:
    return {key: 0 for key in _USAGE_KEYS}


def _extract_usage_stats(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return _empty_usage_stats()

    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    cached_input_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
    thought_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens

    return {
        "requests": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
    }


def _merge_usage_stats(target: Dict[str, int], update: Dict[str, int]) -> None:
    for key in _USAGE_KEYS:
        target[key] += int(update.get(key, 0) or 0)


class GeminiModel(BaseVideoQAModel):
    """
    Google Gemini API backend using the official ``google-genai`` SDK.

    Parameters
    ----------
    model_id : str
        Gemini model name, e.g. ``"gemini-2.5-pro"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    n_frames : int
        Frames to extract when NOT using the Files API.  Default: 64.
    video_upload : bool
        If True, upload the whole video via the Files API instead of sending
        resized inline frames.  Default: False.
    api_key_env : str
        Env-var name for the Gemini API key.  Default: ``"GEMINI_API_KEY"``.
    max_frame_longest_side : int | None
        Optional cap for the longest side of sampled JPEG frames. When set,
        larger frames are resized before being encoded and sent to Gemini.
        Default: 480.
    print_usage : bool
        If True, print token-usage summaries for every answer batch. Default: True.
    usage_log_path : str | None
        Optional JSONL file to append usage summaries to. Default: None.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        n_frames: int = _DEFAULT_N_FRAMES,
        video_upload: bool = False,
        api_key_env: str = "GEMINI_API_KEY",
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
        max_frame_longest_side: Optional[int] = _DEFAULT_MAX_FRAME_LONGEST_SIDE,
        upload_timeout_s: float = _DEFAULT_UPLOAD_TIMEOUT_S,
        upload_poll_interval_s: float = _DEFAULT_UPLOAD_POLL_INTERVAL_S,
        print_usage: bool = True,
        usage_log_path: Optional[str] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        if max_frame_longest_side is not None and max_frame_longest_side <= 0:
            raise ValueError(
                f"max_frame_longest_side must be positive, got {max_frame_longest_side}"
            )
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        if upload_timeout_s <= 0:
            raise ValueError(f"upload_timeout_s must be positive, got {upload_timeout_s}")
        if upload_poll_interval_s <= 0:
            raise ValueError(
                f"upload_poll_interval_s must be positive, got {upload_poll_interval_s}"
            )
        self.n_frames = n_frames
        self.video_upload = video_upload
        self.api_key_env = api_key_env
        self.max_concurrency = max_concurrency
        self.max_frame_longest_side = max_frame_longest_side
        self.upload_timeout_s = upload_timeout_s
        self.upload_poll_interval_s = upload_poll_interval_s
        self.print_usage = print_usage
        self.usage_log_path = usage_log_path
        self._client: Optional[genai.Client] = None
        self.usage_totals: Dict[str, int] = _empty_usage_stats()
        self.last_batch_usage: Dict[str, int] = _empty_usage_stats()

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

        if self.video_upload:
            uploaded_file = self._upload_video(video_path)
            try:
                answers, batch_usage = self._answer_questions_with_uploaded_video(
                    normalized_questions,
                    uploaded_file,
                    max_new_tokens,
                )
            finally:
                self._delete_uploaded_file(uploaded_file)
            self._finalize_batch_usage(
                video_path=video_path,
                source_name="uploaded_video",
                n_questions=len(normalized_questions),
                batch_usage=batch_usage,
            )
            return answers

        frames = _extract_frames_bytes(
            video_path,
            self.n_frames,
            max_longest_side=self.max_frame_longest_side,
        )
        answers, batch_usage = self._answer_questions_with_frames(
            normalized_questions,
            frames,
            max_new_tokens,
        )
        self._finalize_batch_usage(
            video_path=video_path,
            source_name="frames",
            n_questions=len(normalized_questions),
            batch_usage=batch_usage,
        )
        return answers

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {self.api_key_env!r} is not set; "
                f"required for model {self.model_id!r}."
            )

        # Pass the key explicitly so we use the requested env var even when
        # other Google auth env vars are present in the shell.
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _answer_questions_with_uploaded_video(
        self,
        questions: List[str],
        uploaded_file: Any,
        max_new_tokens: int,
    ) -> tuple[List[str], Dict[str, int]]:
        results = [""] * len(questions)
        batch_usage = _empty_usage_stats()
        worker_count = min(self.max_concurrency, len(questions))

        if worker_count == 1:
            for idx, question in enumerate(questions):
                answer, usage = self._answer_one_with_uploaded_video(
                    question,
                    uploaded_file,
                    max_new_tokens,
                )
                results[idx] = answer
                _merge_usage_stats(batch_usage, usage)
            return results, batch_usage

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(
                    self._answer_one_with_uploaded_video,
                    question,
                    uploaded_file,
                    max_new_tokens,
                ): idx
                for idx, question in enumerate(questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                answer, usage = future.result()
                results[idx] = answer
                _merge_usage_stats(batch_usage, usage)

        return results, batch_usage

    def _answer_questions_with_frames(
        self,
        questions: List[str],
        frames: List[bytes],
        max_new_tokens: int,
    ) -> tuple[List[str], Dict[str, int]]:
        results = [""] * len(questions)
        batch_usage = _empty_usage_stats()
        worker_count = min(self.max_concurrency, len(questions))

        if worker_count == 1:
            for idx, question in enumerate(questions):
                answer, usage = self._answer_one_with_frames(question, frames, max_new_tokens)
                results[idx] = answer
                _merge_usage_stats(batch_usage, usage)
            return results, batch_usage

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_idx = {
                executor.submit(self._answer_one_with_frames, question, frames, max_new_tokens): idx
                for idx, question in enumerate(questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                answer, usage = future.result()
                results[idx] = answer
                _merge_usage_stats(batch_usage, usage)

        return results, batch_usage

    def _answer_one_with_uploaded_video(
        self,
        question: str,
        uploaded_file: Any,
        max_new_tokens: int,
    ) -> tuple[str, Dict[str, int]]:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_id,
            contents=self._build_uploaded_video_contents(question, uploaded_file),
            config=types.GenerateContentConfig(max_output_tokens=int(max_new_tokens)),
        )
        return _extract_response_text(response), _extract_usage_stats(response)

    def _answer_one_with_frames(
        self,
        question: str,
        frames: List[bytes],
        max_new_tokens: int,
    ) -> tuple[str, Dict[str, int]]:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_id,
            contents=self._build_frame_contents(question, frames),
            config=types.GenerateContentConfig(max_output_tokens=int(max_new_tokens)),
        )
        return _extract_response_text(response), _extract_usage_stats(response)

    def _upload_video(self, video_path: str) -> Any:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        client = self._get_client()
        mime_type = _guess_video_mime_type(video_path)
        uploaded_file = client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(
                display_name=os.path.basename(video_path),
                mime_type=mime_type,
            ),
        )
        return self._wait_for_uploaded_file(uploaded_file)

    def _wait_for_uploaded_file(self, uploaded_file: Any) -> Any:
        client = self._get_client()
        deadline = time.monotonic() + self.upload_timeout_s
        current = uploaded_file

        while True:
            state = _enum_name(getattr(current, "state", None))
            if state == "ACTIVE":
                return current
            if state == "FAILED":
                raise RuntimeError(
                    f"Gemini file upload failed for {getattr(current, 'name', '<unknown>')!r}: "
                    f"{_format_file_error(current)}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for Gemini to finish processing uploaded video "
                    f"{getattr(current, 'name', '<unknown>')!r} after {self.upload_timeout_s:.0f}s."
                )

            name = getattr(current, "name", None)
            if not isinstance(name, str) or not name:
                raise RuntimeError(
                    f"Uploaded Gemini file is missing a retrievable name: {_serialize_response(current)}"
                )

            time.sleep(self.upload_poll_interval_s)
            current = client.files.get(name=name)

    def _delete_uploaded_file(self, uploaded_file: Any) -> None:
        name = getattr(uploaded_file, "name", None)
        if not isinstance(name, str) or not name:
            return

        try:
            self._get_client().files.delete(name=name)
        except Exception:
            # Cleanup should not mask a successful answer batch.
            pass

    def _build_uploaded_video_contents(self, question: str, uploaded_file: Any) -> List[Any]:
        prompt = self._build_prompt(question, source_name="video")
        mime_type = getattr(uploaded_file, "mime_type", None) or _DEFAULT_VIDEO_MIME_TYPE
        uri = getattr(uploaded_file, "uri", None)
        if not isinstance(uri, str) or not uri:
            raise RuntimeError(
                f"Uploaded Gemini file did not expose a usable URI: {_serialize_response(uploaded_file)}"
            )
        return [
            types.Part.from_text(text=prompt),
            types.Part.from_uri(file_uri=uri, mime_type=mime_type),
        ]

    def _build_frame_contents(self, question: str, frames: List[bytes]) -> List[Any]:
        contents: List[Any] = [
            types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg")
            for frame_bytes in frames
        ]
        contents.append(types.Part.from_text(text=self._build_prompt(question, source_name="video frames")))
        return contents

    def _build_prompt(self, question: str, source_name: str) -> str:
        grounding = (
            f"IMPORTANT: only use information you can directly verify from the {source_name}. "
            "Answer the question directly. When possible, cite rough timestamps."
        )
        return f"{grounding}\n\nQuestion: {question}"

    def _finalize_batch_usage(
        self,
        video_path: str,
        source_name: str,
        n_questions: int,
        batch_usage: Dict[str, int],
    ) -> None:
        self.last_batch_usage = dict(batch_usage)
        _merge_usage_stats(self.usage_totals, batch_usage)

        payload = {
            "model_id": self.model_id,
            "prompt_method": self.prompt_method,
            "video_path": video_path,
            "source_name": source_name,
            "n_questions": n_questions,
            "batch_usage": dict(batch_usage),
            "cumulative_usage": dict(self.usage_totals),
        }

        if self.print_usage:
            print(self._format_usage_message(payload), flush=True)

        if self.usage_log_path:
            log_dir = os.path.dirname(self.usage_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _format_usage_message(self, payload: Dict[str, Any]) -> str:
        batch = payload["batch_usage"]
        cumulative = payload["cumulative_usage"]
        video_name = os.path.basename(payload["video_path"])
        return (
            "[Gemini usage] "
            f"video={video_name} "
            f"source={payload['source_name']} "
            f"questions={payload['n_questions']} "
            f"batch_input={batch['input_tokens']} "
            f"batch_output={batch['output_tokens']} "
            f"batch_total={batch['total_tokens']} "
            f"cumulative_input={cumulative['input_tokens']} "
            f"cumulative_output={cumulative['output_tokens']} "
            f"cumulative_total={cumulative['total_tokens']} "
            f"cached_input={cumulative['cached_input_tokens']} "
            f"thought_tokens={cumulative['thought_tokens']}"
        )
