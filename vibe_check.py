import os
import time
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import torch

# Optional: choose video reader backend if you hit issues:
# os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"  # or "decord" or "torchcodec" :contentReference[oaicite:4]{index=4}

try:
    from transformers import AutoProcessor
    # Qwen3-VL model class (requires sufficiently new transformers). :contentReference[oaicite:5]{index=5}
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import GenerationConfig
except Exception as e:
    raise RuntimeError(
        "Failed to import Qwen3VLForConditionalGeneration. "
        "Install a recent Transformers that supports Qwen3-VL. "
        "Qwen's model card suggests installing from source if needed.\n"
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
    # qwen-vl-utils accepts file:// URIs for local media. :contentReference[oaicite:6]{index=6}
    return Path(p).expanduser().resolve().as_uri()


def _normalize_gradio_video_input(video_in: Any) -> Optional[str]:
    """
    Gradio's Video component may return:
      - a filepath string
      - a dict with a path
      - a tuple/list like (filepath, ...)
    We'll try to robustly extract a filepath.
    """
    if video_in is None:
        return None
    if isinstance(video_in, str):
        return video_in
    if isinstance(video_in, dict):
        # common keys in some gradio versions
        for k in ("path", "video", "name", "data"):
            if k in video_in and isinstance(video_in[k], str):
                return video_in[k]
        return None
    if isinstance(video_in, (tuple, list)) and len(video_in) > 0 and isinstance(video_in[0], str):
        return video_in[0]
    return None

@dataclass
class LoadedModel:
    model_id: str
    model: Any
    processor: Any


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._loaded: Optional[LoadedModel] = None
        self._gen_lock = threading.Lock()


    def load(self, model_id):
        with self._lock:
            if self._loaded is not None and self._loaded.model_id == model_id:
                return self._loaded
            
            processor = AutoProcessor.from_pretrained(model_id)

            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id, dtype='auto', device_map='auto'
                )
            except TypeError:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype='auto', device_map='auto'
                )
            
            model.eval()
            self._loaded = LoadedModel(model_id=model_id, model=model, processor=processor)
            return self._loaded


    def generate_one(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        force_fps: Optional[float] = None,
        resized_hw: Optional[Tuple[int, int]] = None
    ):
        loaded = self.load(model_id)
        model, processor = loaded.model, loaded.processor

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        patch_size = getattr(getattr(processor, 'image_processor', None), 'patch_size', None)
        if patch_size is None:
            patch_size = 16

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        
        if force_fps is not None:
            video_kwargs = dict(video_kwargs or {})
            video_kwargs['fps'] = float(force_fps)
        
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            return_tensors='pt',
            do_resize=False,
            **(video_kwargs or {})
        )

        inputs = inputs.to(model.device)
        gen_config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1
        )

        t0 = time.time()
        with self._gen_lock:
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(max_new_tokens),
                    generation_config=gen_config
                )
        
        dt = time.time() - t0
        in_ids = inputs.input_ids
        trimmed = []
        for i in range(generated_ids.shape[0]):
            trimmed.append(generated_ids[i, in_ids.shape[1]:])
        
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        debug = {
            "model_id": model_id,
            "max_new_tokens": int(max_new_tokens),
            "greedy": True,
            "force_fps": force_fps,
            "video_kwargs": video_kwargs,
            "latency_sec": round(dt, 3)
        }
        return out_text, debug



MM = ModelManager()

def build_messages_for_video_qa(
    video_path,
    user_text
):
    video_uri = _to_file_uri(video_path)
    return [
        {
            'role': 'user',
            'content': [
                {"type": "video", "video": video_uri},
                {"type": "text", "text": user_text},
            ]
        }
    ]


def run_three_settings(
    video_in,
    question,
    model_id,
    max_new_tokens,
    fps: Optional[float]
):
    """
    Returns:
        direct_answer, explain_answer, (answer/desc/refine combined), debug_json
    """
    video_path = _normalize_gradio_video_input(video_in)
    if not video_path or not Path(video_path).exists():
        return (
            "",
            "",
            "",
            "ERROR: please upload a valid video file."
        )
    
    if not question.strip():
        return (
            "",
            "",
            "",
            "ERROR: please enter a question."
        )
    
    grounding = (
        "IMPORTANT: only use information you can directly verify from the video. "
        "If you are unsure, say 'Not sure'. When possible, cite rough timestamps."
    )

    # 1) Direct answer
    prompt_direct = f"{grounding}\n\nQuestion: {question.strip()}"
    msg_direct = build_messages_for_video_qa(video_path, prompt_direct)
    ans_direct, dbg1 = MM.generate_one(model_id, msg_direct, max_new_tokens=max_new_tokens, force_fps=fps)

    # 2) Explain then answer
    prompt_explain = (
        f"{grounding}\n\n"
        f"Question: {question.strip()}\n\n"
        "First explain your reasoning grounded in the video (with timestamps), then give the final answer.\n"
        "Format:\n"
        "Reasoning:\n"
        "- ...\n"
        "Final answer:\n"
        "- ..."
    )
    msg_explain = build_messages_for_video_qa(video_path, prompt_explain)
    ans_explain, dbg2 = MM.generate_one(model_id, msg_explain, max_new_tokens=max_new_tokens, force_fps=fps)

    # 3) Answer -> describe video -> refine
    # 3a) initial answer (same as direct but shorter)
    prompt_init = f"{grounding}\n\nQuestion: {question.strip()}\nAnswer concisely"
    msg_init = build_messages_for_video_qa(video_path, prompt_init)
    ans_init, dbg3a = MM.generate_one(model_id, msg_init, max_new_tokens=max_new_tokens, force_fps=fps)

    # 3b) video description
    prompt_desc = (
        f"{grounding}\n\n"
        "Describe the video in detail. Focus on entities/actions/events that are relevant for answering the question:\n"
        f"Question: {question.strip()}\n\n"
        "Output a timestamped description (approximate is fine)"
    )
    msg_desc = build_messages_for_video_qa(video_path, prompt_desc)
    ans_desc, dbg3b = MM.generate_one(model_id, msg_desc, max_new_tokens=max_new_tokens, force_fps=fps)

    # 3c) refine using the description
    prompt_refine = (
        f"{grounding}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Your initial answer:\n{ans_init.strip()}\n\n"
        f"Your video description:\n{ans_desc.strip()}\n\n"
        "Now refine your answer. If the description contradicts your initial answer, correct it."
        "If still uncertain, say 'Not sure'"
    )
    msg_refine = build_messages_for_video_qa(video_path, prompt_refine)
    ans_refine, dbg3c = MM.generate_one(model_id, msg_refine, max_new_tokens=max_new_tokens, force_fps=fps)

    combined = (
        "=== 3a) Initial answer ===\n"
        f"{ans_init.strip()}\n\n"
        "=== 3b) Video description ===\n"
        f"{ans_desc.strip()}\n\n"
        "=== 3c) Refined answer ===\n"
        f"{ans_refine.strip()}\n"
    )

    debug = {
        'direct': dbg1,
        'explain': dbg2,
        'chain': {
            "init": dbg3a,
            "desc": dbg3b,
            "refine": dbg3c
        }
    }
    return ans_direct, ans_explain, combined, json.dumps(debug, indent=2)


with gr.Blocks(title="Qwen3-VL Video Hallucination Vibe Check") as demo:
    gr.Markdown("""
# Qwen3-VL: Video Hallucination Vibe-Check (Greedy Decoding)

Upload a video, ask a question, then compare:
1) Direct answer  
2) Explain → Answer  
3) Answer → Describe Video → Refine  

Greedy decoding is enforced (`do_sample=False`, `temperature=0`, `num_beams=1`).
""")

    with gr.Row():
        model_id = gr.Textbox(
            label="Model ID (huggingface)",
            value="Qwen/Qwen3-VL-8B-Thinking"
        )

        max_new_tokens = gr.Slider(
            label="max_new_tokens",
            minimum=32,
            maximum=2048,
            step=32,
            value=512
        )
        fps = gr.Slider(
            label="Optional video sampling fps (leave as-is unless needed)",
            minimum=0.0,
            maximum=10.0,
            step=0.5,
            value=0.0
        )

    video = gr.Video(label="Upload video", sources=['upload'])
    question = gr.Textbox(
        label="Question about the video",
        lines=3,
    )
    run_btn = gr.Button("Run all 3 settings (greedy)")
    with gr.Row():
        out_direct = gr.Textbox(label="1) Direct answer", lines=10)
        out_explain = gr.Textbox(label="2) Explain -> Answer", lines=10)

    out_chain = gr.Textbox(label="3) Answer → Describe → Refine", lines=18)
    out_debug = gr.Code(label="Debug (generation settings / timings)", language='json')

    def _fps_or_none(x: float):
        return None if x is None or float(x) <= 0 else float(x)
    
    run_btn.click(
        fn=lambda v, q, m, t, f: run_three_settings(v, q, m, int(t), _fps_or_none(f)),
        inputs=[video, question, model_id, max_new_tokens, fps],
        outputs=[out_direct, out_explain, out_chain, out_debug],
    )

demo.queue().launch()
