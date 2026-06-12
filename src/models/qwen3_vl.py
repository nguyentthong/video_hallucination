"""
src/models/qwen3_vl.py
-----------------------
Local HuggingFace backend for Qwen3-VL models.
Supports optional fixed-FPS injection for Qwen3-compatible checkpoints that
expect a specific video sampling rate.
"""

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .base import BaseVideoQAModel

try:
    from transformers import AutoProcessor, GenerationConfig
    from transformers import Qwen3VLForConditionalGeneration
except Exception as exc:
    raise RuntimeError(
        "Failed to import Qwen3VLForConditionalGeneration. "
        "Install a recent Transformers that supports Qwen3-VL.\n"
        f"Error: {exc}"
    ) from exc

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:
    raise RuntimeError(
        "Failed to import qwen_vl_utils. Install with: pip install qwen-vl-utils\n"
        f"Error: {exc}"
    ) from exc

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


class Qwen3VLModel(BaseVideoQAModel):
    """
    Runs inference locally using a Qwen3-VL checkpoint loaded via HuggingFace
    Transformers.  Weights are lazy-loaded on the first call to
    `answer_questions` and then cached in memory for subsequent calls.

    Parameters
    ----------
    model_id : str
        HuggingFace repo ID, e.g. ``"Qwen/Qwen3-VL-8B-Instruct"``.
    prompt_method : str
        Prompt template label (affects cache namespace).  Default: "vanilla".
    debug_with_n_frames : int | None
        When set, only this many evenly-spaced frames are sampled from the
        video and written to ``sampled_frames/`` for inspection.
    force_fps : float | None
        When set, inject this FPS into the video message and processor kwargs.
        Useful for Qwen3-compatible checkpoints that expect a fixed sampling
        rate, such as Cosmos-Reason2 at 4 FPS.
    """

    def __init__(
        self,
        model_id: str,
        prompt_method: str = "vanilla",
        debug_with_n_frames: Optional[int] = None,
        force_fps: Optional[float] = None,
    ) -> None:
        super().__init__(model_id, prompt_method)
        self.debug_with_n_frames = debug_with_n_frames
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
            answer, _ = self._generate_one(
                messages,
                max_new_tokens,
                force_fps=self.force_fps,
            )
            results.append(answer.strip())
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
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, dtype="auto", device_map="auto"
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", device_map="auto"
            )
        model.eval()
        self._loaded = _LoadedWeights(self.model_id, model, processor)
        return self._loaded

    def _build_messages(
        self, video_path: str, question: str
    ) -> List[Dict[str, Any]]:
        grounding = (
            "IMPORTANT: only use information you can directly verify from the "
            "video. If you are unsure, say 'Not sure'. When possible, cite "
            "rough timestamps."
        )
        prompt = f"{grounding}\n\nQuestion: {question}"
        video_content: Dict[str, Any] = {
            "type": "video",
            "video": _to_file_uri(video_path),
        }
        if self.debug_with_n_frames is not None:
            video_content["nframes"] = self.debug_with_n_frames
        if self.force_fps is not None:
            video_content["fps"] = self.force_fps

        content: List[Dict[str, Any]] = [
            video_content,
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]

    @staticmethod
    def _inject_force_fps(
        messages: List[Dict[str, Any]],
        force_fps: Optional[float],
    ) -> List[Dict[str, Any]]:
        if force_fps is None:
            return messages

        injected_messages: List[Dict[str, Any]] = []
        for message in messages:
            new_message = dict(message)
            content = message.get("content")
            if not isinstance(content, list):
                injected_messages.append(new_message)
                continue

            new_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue

                new_item = dict(item)
                if new_item.get("type") == "video":
                    new_item["fps"] = float(force_fps)
                new_content.append(new_item)
            new_message["content"] = new_content
            injected_messages.append(new_message)
        return injected_messages

    def _generate_one(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        force_fps: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        loaded = self._load_weights()
        model, processor = loaded.model, loaded.processor
        effective_fps = self.force_fps if force_fps is None else float(force_fps)
        messages = self._inject_force_fps(messages, effective_fps)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        patch_size = getattr(
            getattr(processor, "image_processor", None), "patch_size", 16
        )

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        if self.debug_with_n_frames is not None:
            self._dump_frames(videos)

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        video_kwargs = dict(video_kwargs or {})
        if "fps" in video_kwargs and isinstance(video_kwargs["fps"], list):
            video_kwargs["fps"] = video_kwargs["fps"][0]

        if effective_fps is not None:
            video_kwargs["fps"] = effective_fps

        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors="pt",
            do_resize=False,
            **(video_kwargs or {}),
        )
        inputs = inputs.to(model.device)

        gen_config = GenerationConfig(
            do_sample=False, temperature=0.0, top_p=1.0, num_beams=1
        )

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                generation_config=gen_config,
            )
        dt = time.time() - t0

        in_ids = inputs.input_ids
        trimmed = [generated_ids[i, in_ids.shape[1]:] for i in range(generated_ids.shape[0])]
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        debug = {
            "model_id": self.model_id,
            "max_new_tokens": int(max_new_tokens),
            "greedy": True,
            "force_fps": effective_fps,
            "latency_sec": round(dt, 3),
        }
        return out_text, debug

    @staticmethod
    def _dump_frames(videos: Any) -> None:
        if os.path.exists("sampled_frames"):
            shutil.rmtree("sampled_frames")
        os.makedirs("sampled_frames", exist_ok=True)
        for i, frame in enumerate(videos[0][0]):
            arr = frame.numpy().transpose(1, 2, 0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"sampled_frames/sample_frame_{i:04d}.jpg", arr)