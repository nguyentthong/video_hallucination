import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
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
    def __init__(self) -> None:
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


def build_messages_for_video_qa(video_path: str, question: str) -> List[Dict[str, Any]]:
    grounding = (
        "IMPORTANT: only use information you can directly verify from the video. "
        "If you are unsure, say 'Not sure'. When possible, cite rough timestamps."
    )
    prompt = f"{grounding}\n\nQuestion: {question.strip()}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": _to_file_uri(video_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask Qwen3-VL-8B a question about a local video."
    )
    parser.add_argument("--video-path", required=True, help="Path to the input video.")
    parser.add_argument("--question", required=True, help="Question about the video.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for the answer.",
    )
    parser.add_argument(
        "--force-fps",
        type=float,
        default=0.0,
        help="Optional FPS for video sampling (0 to disable).",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Print debug generation settings/timings as JSON.",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    question = args.question.strip()
    if not question:
        raise ValueError("Question must be non-empty.")

    force_fps = None if args.force_fps <= 0 else float(args.force_fps)

    mm = ModelManager()
    messages = build_messages_for_video_qa(str(video_path), question)
    answer, debug = mm.generate_one(
        args.model_id,
        messages,
        max_new_tokens=args.max_new_tokens,
        force_fps=force_fps,
    )

    print(answer.strip())
    if args.debug_json:
        print(json.dumps(debug, indent=2))


if __name__ == "__main__":
    main()

    # python vibe_check_video_qa.py --video-path /home/thong/weride_project/hallucination_vibecheck/vibe_check_examples/videos/_1CKVcXe06A.mp4 --question "what is the man in the video doing?"
