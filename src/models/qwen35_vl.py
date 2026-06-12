"""
src/models/qwen35_vl.py
------------------------
Local HuggingFace backend for Qwen3.5-VL models.

Requires the ``qwen35vl`` dependency group:
    uv sync --group qwen35vl

Key differences from Qwen3VLModel (qwen3_vl.py)
------------------------------------------------
* Uses ``Qwen3_5ForConditionalGeneration`` instead of
  ``Qwen3VLForConditionalGeneration``.
* Supports thinking mode: append ``/think`` or ``/no_think`` to the prompt.
  Controlled via the ``thinking`` constructor flag.
* Video-mode inference samples at ``force_fps``; the default is 2 FPS.
* ``process_vision_info`` can return fps as a list; this file normalises it to
  a scalar before calling the processor when metadata is not requested.
* Torchvision ≥ 0.21 removed ``torchvision.io.read_video``; patched via PyAV
  at import time so the transformers video loader does not crash.
* Generation uses sampling by default (temperature / top_p / top_k) to match
  the recommended Qwen3.5 inference settings, falling back to greedy when
  thinking=False and the caller wants deterministic output.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import BaseVideoQAModel

# ---------------------------------------------------------------------------
# torchvision.io.read_video compatibility patch
# ---------------------------------------------------------------------------
try:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "read_video"):
        import av as _av
        import numpy as _np

        def _read_video_av(
            filename,
            start_pts=0,
            end_pts=None,
            pts_unit="pts",
            output_format="TCHW",
        ):
            container = _av.open(str(filename))
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            frames = []
            for frame in container.decode(stream):
                frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            container.close()
            video = torch.stack(frames).permute(0, 3, 1, 2)  # (T,3,H,W)
            return video, torch.empty(0), {"video_fps": fps}

        _tvio.read_video = _read_video_av
except Exception:
    pass  # non-fatal; will surface only if the processor actually calls read_video

# ---------------------------------------------------------------------------
# Heavy imports — deferred to _load_weights() via the try/except at class level
# ---------------------------------------------------------------------------
try:
    from transformers import AutoProcessor
    from transformers import Qwen3_5ForConditionalGeneration
except Exception as exc:
    raise RuntimeError(
        "Failed to import Qwen3_5ForConditionalGeneration.\n"
        "Install the dev transformers with: uv sync --group qwen35vl\n"
        f"Error: {exc}"
    ) from exc

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:
    raise RuntimeError(
        "Failed to import qwen_vl_utils. Install with: pip install qwen-vl-utils\n"
        f"Error: {exc}"
    ) from exc


def _to_file_uri(p: str) -> str:
    return Path(p).expanduser().resolve().as_uri()


def _format_fps_tag(fps: float) -> str:
    fps = float(fps)
    if fps.is_integer():
        return str(int(fps))
    return str(fps).replace(".", "p")


@dataclass
class _LoadedWeights:
    model_id: str
    model: Any
    processor: Any


class Qwen35VLModel(BaseVideoQAModel):
    """
    Runs inference locally using a Qwen3.5-VL checkpoint.

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID, e.g. ``"Qwen/Qwen3.5-VL-7B-Instruct"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: ``"vanilla"``.
        Use ``"thinking"`` to signal thinking-mode runs in the cache namespace.
    thinking : bool
        If True, append ``/think`` to every question so the model emits a
        ``<think>…</think>`` block before the final answer.  The think block is
        stripped from the returned answer string.  Default: False.
    max_frames : int
        Frames sampled uniformly from the video.  Overridden by
        ``debug_with_n_frames`` when that is set.  Default: 16.
    max_pixels : int
        Max pixels per frame passed to the processor.  Default: 360 × 420.
    debug_with_n_frames : int | None
        When set, overrides ``max_frames`` (mirrors the Qwen3VL CLI flag).
    force_fps : float | None
        When set, sample video inputs at this FPS.  Default: 2.0.  Set to
        None to use fixed ``nframes`` sampling via ``max_frames``.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        thinking: bool = False,
        max_frames: int = 16,
        max_pixels: int = 360 * 420,
        debug_with_n_frames: Optional[int] = None,
        force_fps: Optional[float] = 2.0,
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.thinking = thinking
        self.debug_with_n_frames = debug_with_n_frames
        self.max_frames = debug_with_n_frames if debug_with_n_frames is not None else max_frames
        self.max_pixels = max_pixels
        if force_fps is not None and float(force_fps) <= 0:
            raise ValueError("force_fps must be > 0 when provided")
        self.force_fps = float(force_fps) if force_fps is not None else None
        self._loaded: Optional[_LoadedWeights] = None

    # ------------------------------------------------------------------
    # BaseVideoQAModel interface
    # ------------------------------------------------------------------

    def answer_questions(
        self,
        video_path: str,
        questions: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]:
        results = []
        for question in questions:
            question = question.strip()
            if not question:
                raise ValueError(f"Empty question passed to {self!r}")
            messages = self._build_messages(video_path, question)
            answer, _ = self._generate_one(messages, max_new_tokens)
            results.append(answer)
        return results

    @property
    def cache_namespace(self) -> str:
        base_namespace = super().cache_namespace
        if self.force_fps is None:
            return base_namespace
        return f"{base_namespace}__fps{_format_fps_tag(self.force_fps)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_weights(self) -> _LoadedWeights:
        if self._loaded is not None and self._loaded.model_id == self.model_id:
            return self._loaded

        processor = AutoProcessor.from_pretrained(self.model_id)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        self._loaded = _LoadedWeights(self.model_id, model, processor)
        return self._loaded

    # ------------------------------------------------------------------
    # Frame-based inference (used by FramePipelineModel)
    # ------------------------------------------------------------------

    def answer_from_frames(
        self,
        frames: List[Any],
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Run inference on pre-extracted PIL Image frames instead of a video path.
        Called by ``FramePipelineModel`` after a frame selector has run.
        """
        question = question.strip()
        messages = self._build_messages_from_frames(frames, question)
        answer, _ = self._generate_one_from_frames(messages, max_new_tokens)
        return answer

    def _build_messages_from_frames(
        self, frames: List[Any], question: str
    ) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. If you are unsure, say 'Not sure'. When possible, cite "
            "rough timestamps."
        )
        suffix = "/think" if self.thinking else "/no_think"
        prompt = f"{grounding}\n\nQuestion: {question} {suffix}"
        content: List[Dict[str, Any]] = [
            *[{"type": "image", "image": f} for f in frames],
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]

    def _generate_one_from_frames(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Like _generate_one but for image-frame messages (no video_kwargs)."""
        loaded = self._load_weights()
        model, processor = loaded.model, loaded.processor

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # process_vision_info without return_video_kwargs — frames are images
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6 if self.thinking else 0.7,
            top_p=0.95 if self.thinking else 0.80,
            top_k=20,
        )

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        answer = self._parse_answer(raw)

        debug = {
            "model_id": self.model_id,
            "max_new_tokens": int(max_new_tokens),
            "thinking": self.thinking,
            "latency_sec": round(dt, 3),
        }
        return answer, debug

    # ------------------------------------------------------------------

    def _build_messages(
        self, video_path: str, question: str
    ) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. If you are unsure, say 'Not sure'. When possible, cite "
            "rough timestamps."
        )
        suffix = "/think" if self.thinking else "/no_think"
        prompt = f"{grounding}\n\nQuestion: {question} {suffix}"

        video_content: Dict[str, Any] = {
            "type": "video",
            "video": _to_file_uri(video_path),
            "max_pixels": self.max_pixels,
        }
        if self.debug_with_n_frames is not None:
            video_content["nframes"] = self.debug_with_n_frames
        elif self.force_fps is not None:
            video_content["fps"] = self.force_fps
            video_content["max_frames"] = self.max_frames
        else:
            video_content["nframes"] = self.max_frames

        content: List[Dict[str, Any]] = [
            video_content,
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]

    def _generate_one(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        loaded = self._load_weights()
        model, processor = loaded.model, loaded.processor

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )

        video_metadatas = None
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

        video_kwargs = dict(video_kwargs or {})
        # process_vision_info may return fps as a list when metadata is not requested.
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(next(model.parameters()).device)

        # Qwen3.5 recommended sampling settings
        if self.thinking:
            gen_kwargs: Dict[str, Any] = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
            )
        else:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.80,
                top_k=20,
            )

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        # keep special tokens so we can parse <think>…</think>
        raw = processor.batch_decode(
            trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        answer = self._parse_answer(raw)

        debug = {
            "model_id": self.model_id,
            "max_new_tokens": int(max_new_tokens),
            "thinking": self.thinking,
            "force_fps": self.force_fps,
            "latency_sec": round(dt, 3),
        }
        return answer, debug

    @staticmethod
    def _parse_answer(raw: str) -> str:
        """Strip <think>…</think> and leftover XML tags, return clean answer."""
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        answer = re.sub(r"<[^>]+>", "", answer)
        return answer.strip()
