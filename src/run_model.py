import argparse
import json
import os
import time
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import AutoProcessor
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import GenerationConfig
except Exception as e:
    raise RuntimeError(
        "Failed to import Qwen3VLForConditionalGeneration. "
        "Install a recent Transformers that supports Qwen3-VL.\n"
        "Error: " + str(e)
    )

try:
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise RuntimeError(
        "Failed to import qwen_vl_utils. Install it with: pip install qwen-vl-utils\n"
        "Error: " + str(e)
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _to_file_uri(p: str) -> str:
    return Path(p).expanduser().resolve().as_uri()


@dataclass
class LoadedModel:
    model_id: str
    model: Any
    processor: Any


class ModelManager:
    def __init__(self, debug_with_n_frames: Optional[int]) -> None:
        self.debug_with_n_frames = debug_with_n_frames
        self._loaded: Optional[LoadedModel] = None

    def load(self, model_id: str) -> LoadedModel:
        if self._loaded is not None and self._loaded.model_id == model_id:
            return self._loaded

        processor = AutoProcessor.from_pretrained(model_id)
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, dtype="auto", device_map="auto"
            )
        except TypeError:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto"
            )

        model.eval()
        self._loaded = LoadedModel(model_id=model_id, model=model, processor=processor)
        return self._loaded

    def generate_one(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        force_fps: Optional[float] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        loaded = self.load(model_id)
        model, processor = loaded.model, loaded.processor

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
        if patch_size is None:
            patch_size = 16

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if self.debug_with_n_frames is not None:
            if os.path.exists("sampled_frames"):
                shutil.rmtree("sampled_frames")
            os.makedirs("sampled_frames", exist_ok=True)
            for i in range(len(videos[0][0])):
                frame = videos[0][0][i].numpy()
                frame = np.transpose(frame, (1,2,0))
                if frame.max() <= 1.0:
                    frame = frame * 255.0

                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"sampled_frames/sample_frame_{i:04d}.jpg", frame)

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)

        if force_fps is not None:
            video_kwargs = dict(video_kwargs or {})
            video_kwargs["fps"] = float(force_fps)

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
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
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
        trimmed = []
        for i in range(generated_ids.shape[0]):
            trimmed.append(generated_ids[i, in_ids.shape[1] :])

        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        debug = {
            "model_id": model_id,
            "max_new_tokens": int(max_new_tokens),
            "greedy": True,
            "force_fps": force_fps,
            "video_kwargs": video_kwargs,
            "latency_sec": round(dt, 3),
        }
        return out_text, debug

def build_messages_for_video_qa(video_path: str, question: str, n_frames: Optional[int]) -> List[Dict[str, Any]]:
    grounding = (
        "IMPORTANT: only use information you can directly verify from the video. "
        "If you are unsure, say 'Not sure'. When possible, cite rough timestamps."
    )
    prompt = f"{grounding}\n\nQuestion: {question.strip()}"
    if n_frames is not None:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": _to_file_uri(video_path), "nframes": n_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": _to_file_uri(video_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]


def load_model(model_id: str, debug_with_n_frames: Optional[int]) -> ModelManager:
    mm = ModelManager(debug_with_n_frames)
    return mm

def run_model(
    mm: ModelManager,
    model_id: str,
    sample: Dict[str, Any], 
    sample_id: int, 
    max_new_tokens: int = 256,
    force_fps: float = 0.0,
    debug_json: bool = False,
    debug_with_n_frames: Optional[int] = None,
) -> List[str]:
    video_path = Path(sample['video_path'])
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    results = []
    for question in sample['questions']:
        question = question.strip()
        if not question:
            raise ValueError(f"A question in sample {sample_id} is empty. Please fix that!")

        _force_fps = None if force_fps <= 0 else float(force_fps)

        messages = build_messages_for_video_qa(str(video_path), question, debug_with_n_frames)
        answer, debug = mm.generate_one(
            model_id,
            messages,
            max_new_tokens=max_new_tokens,
            force_fps=_force_fps,
        )

        results.append(answer.strip())
        if debug_json:
            print(json.dumps(debug, indent=2))

        return results
