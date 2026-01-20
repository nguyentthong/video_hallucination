import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import jsonlines
import structlog
import torch
import numpy as np

logger = structlog.get_logger()
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
    def __init__(self):
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


def build_messages_for_video_qa(video_path: str, user_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": _to_file_uri(video_path)},
                {"type": "text", "text": user_text},
            ],
        }
    ]


def _strict_yes_no(text: str) -> Optional[str]:
    t = text.strip().lower()
    if t in ("yes", "no"):
        return t
    if t in ("y", "n"):
        return "yes" if t == "y" else "no"
    return None


def parse_yes_no(
    text: str,
    mode: str,
) -> Tuple[str, bool]:
    """
    Returns (answer, followed_instruction).
    If not followed, answer is forced to "no".
    """
    if mode == "direct" or mode == "refine":
        strict = _strict_yes_no(text)
        if strict is None:
            return "no", False
        return strict, True

    if mode == "explain":
        match = re.search(r"final answer\s*:\s*(yes|no)\b", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower(), True
        strict = _strict_yes_no(text)
        if strict is None:
            return "no", False
        return strict, True

    return "no", False


def normalize_yes_no(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and value in (0, 1):
        return "yes" if value == 1 else "no"
    if isinstance(value, str):
        t = value.strip().lower().rstrip(".")
        if t in ("yes", "no"):
            return t
        if t in ("true", "false"):
            return "yes" if t == "true" else "no"
    return None


def resolve_video_path(sample: Dict[str, Any], videos_dir: Path) -> Optional[str]:
    sample_id = sample.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        return None
    prefix = sample_id.split("_")[0]
    subdir = "interleave" if prefix == "mix" else prefix
    candidate = videos_dir / subdir / f"{sample_id}.mp4"
    if candidate.exists():
        return str(candidate)
    return None


def resolve_questions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    questions = sample.get("questions", [])
    if not isinstance(questions, list):
        return []
    return questions


def build_direct_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY 'YES' or 'NO'. "
        "If you are unsure, answer 'NO'.\n\n"
        f"Question: {question}"
    )


def build_explain_prompt(question: str) -> str:
    return (
        "Explain your reasoning grounded in the video, then answer.\n"
        "Format:\n"
        "Reasoning: ...\n"
        "Final answer: YES or NO\n"
        "If you are unsure, answer NO.\n\n"
        f"Question: {question}"
    )


def build_init_prompt(question: str) -> str:
    return (
        "Answer the question using ONLY 'YES' or 'NO'. "
        "If you are unsure, answer 'NO'.\n\n"
        f"Question: {question}"
    )


def build_desc_prompt(question: str) -> str:
    return (
        "Describe the video in detail focusing on evidence relevant to the question.\n"
        "Include approximate timestamps if possible.\n\n"
        f"Question: {question}"
    )


def build_refine_prompt(question: str, init_answer: str, description: str) -> str:
    return (
        "You will refine your answer using the description below.\n"
        "If the description contradicts the initial answer, correct it.\n"
        "Answer using ONLY 'YES' or 'NO'. If unsure, answer 'NO'.\n\n"
        f"Question: {question}\n\n"
        f"Initial answer: {init_answer.strip()}\n\n"
        f"Description:\n{description.strip()}"
    )


def evaluate_sample(
    mm: ModelManager,
    sample: Dict[str, Any],
    videos_dir: Path,
    model_id: str,
    approach: str,
    max_new_tokens: int,
    max_desc_tokens: int,
    force_fps: Optional[float],
) -> Dict[str, Any]:
    questions = resolve_questions(sample)
    if not questions:
        raise ValueError("Missing questions in sample.")
    video_path = resolve_video_path(sample, videos_dir)
    if video_path is None:
        raise ValueError("Missing or invalid video path in sample.")

    outputs: Dict[str, Any] = {
        "id": sample.get("id"),
        "video_path": video_path,
        "questions": [],
    }

    for q in questions:
        question_text = q.get("question")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError("Missing question text in sample.")

        gt = normalize_yes_no(q.get("answer"))
        q_out: Dict[str, Any] = {
            "question": question_text.strip(),
            "type": q.get("type"),
            "ground_truth": gt,
        }

        if approach == "direct":
            prompt = build_direct_prompt(question_text)
            msg = build_messages_for_video_qa(video_path, prompt)
            raw, debug = mm.generate_one(model_id, msg, max_new_tokens, force_fps=force_fps)
            pred, followed = parse_yes_no(raw, "direct")
            q_out["direct"] = {
                "raw_output": raw,
                "predicted": pred,
                "followed_instruction": followed,
                "ground_truth": gt,
                "is_correct": (pred == gt) if gt is not None else None,
                "debug": debug,
            }

        if approach == "explain":
            prompt = build_explain_prompt(question_text)
            msg = build_messages_for_video_qa(video_path, prompt)
            raw, debug = mm.generate_one(model_id, msg, max_new_tokens, force_fps=force_fps)
            pred, followed = parse_yes_no(raw, "explain")
            q_out["explain"] = {
                "raw_output": raw,
                "predicted": pred,
                "followed_instruction": followed,
                "ground_truth": gt,
                "is_correct": (pred == gt) if gt is not None else None,
                "debug": debug,
            }

        if approach == "refine":
            init_prompt = build_init_prompt(question_text)
            init_msg = build_messages_for_video_qa(video_path, init_prompt)
            init_raw, init_debug = mm.generate_one(
                model_id, init_msg, max_new_tokens, force_fps=force_fps
            )
            init_pred, init_followed = parse_yes_no(init_raw, "direct")

            desc_prompt = build_desc_prompt(question_text)
            desc_msg = build_messages_for_video_qa(video_path, desc_prompt)
            desc_raw, desc_debug = mm.generate_one(
                model_id, desc_msg, max_desc_tokens, force_fps=force_fps
            )

            refine_prompt = build_refine_prompt(question_text, init_raw, desc_raw)
            refine_msg = build_messages_for_video_qa(video_path, refine_prompt)
            refine_raw, refine_debug = mm.generate_one(
                model_id, refine_msg, max_new_tokens, force_fps=force_fps
            )
            refine_pred, refine_followed = parse_yes_no(refine_raw, "refine")

            q_out["refine"] = {
                "initial_answer": {
                    "raw_output": init_raw,
                    "predicted": init_pred,
                    "followed_instruction": init_followed,
                    "debug": init_debug,
                },
                "description": {
                    "raw_output": desc_raw,
                    "debug": desc_debug,
                },
                "refined_answer": {
                    "raw_output": refine_raw,
                    "predicted": refine_pred,
                    "followed_instruction": refine_followed,
                    "ground_truth": gt,
                    "is_correct": (refine_pred == gt) if gt is not None else None,
                    "debug": refine_debug,
                },
            }

        outputs["questions"].append(q_out)

    return outputs


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_accuracy(rows: List[Dict[str, Any]], approach: str) -> List[int]:
    accuracy_list: List[int] = []
    for row in rows:
        for q in row.get("questions", []):
            entry = q.get(approach)
            if entry is None:
                continue
            if approach == "refine":
                entry = entry.get("refined_answer", entry)
            is_correct = entry.get("is_correct")
            if is_correct is None:
                continue
            accuracy_list.append(int(is_correct))
    return accuracy_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL models on EventHallusion question JSON."
    )
    parser.add_argument(
        "--questions-path",
        required=True,
        help="Path to EventHallusion question JSON (e.g., entire_questions.json).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="direct",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for answer generations.",
    )
    parser.add_argument(
        "--max-desc-tokens",
        type=int,
        default=512,
        help="Max tokens for description generation in refine approach.",
    )
    parser.add_argument(
        "--force-fps",
        type=float,
        default=0.0,
        help="Optional FPS for video sampling (0 to disable).",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions_path)
    if not questions_path.is_absolute():
        questions_path = Path("EventHallusion") / "questions" / questions_path
    if not questions_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {questions_path}")

    output_dir = Path("outputs") / "EventHallusion" / questions_path.stem / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.approach}.json"

    force_fps = None if args.force_fps <= 0 else float(args.force_fps)

    samples = load_dataset(questions_path)
    videos_dir = Path("EventHallusion") / "videos"
    if not videos_dir.exists():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")

    mm = ModelManager()

    if output_path.exists():
        generated_outputs = list(jsonlines.open(output_path))
        generated_ids = set(range(len(generated_outputs)))
        accuracy_list = collect_accuracy(generated_outputs, args.approach)
    else:
        generated_outputs = []
        generated_ids = set()
        accuracy_list = []

    for i, sample in enumerate(tqdm(samples)):
        if i in generated_ids:
            if accuracy_list:
                current_accuracy = np.mean(accuracy_list)
                logger.info(f"Accuracy: {current_accuracy * 100:.2f}")
            continue

        model_out = evaluate_sample(
            mm,
            sample,
            videos_dir,
            args.model_id,
            args.approach,
            max_new_tokens=args.max_new_tokens,
            max_desc_tokens=args.max_desc_tokens,
            force_fps=force_fps,
        )

        with output_path.open("a", encoding="utf-8") as f:
            json.dump(model_out, f)
            f.write("\n")
            f.flush()

        accuracy_list.extend(collect_accuracy([model_out], args.approach))
        if accuracy_list:
            current_accuracy = np.mean(accuracy_list)
            logger.info(f"Accuracy: {current_accuracy * 100:.2f}")

    if accuracy_list:
        avg_accuracy = np.mean(accuracy_list)
        logger.info(f"Final accuracy: {avg_accuracy * 100:.2f}")


if __name__ == "__main__":
    main()
