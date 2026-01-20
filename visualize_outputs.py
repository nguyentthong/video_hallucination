import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def infer_method(records: List[Dict[str, Any]]) -> str:
    if not records:
        return "direct"
    first = records[0]
    if "refine" in first:
        return "refine"
    if "explain" in first:
        return "explain"
    if "direct" in first:
        return "direct"
    return "direct"


def is_correct_record(rec: Dict[str, Any], method: str) -> Any:
    if method == "refine":
        return rec.get("refine", {}).get("refined_answer", {}).get("is_correct")
    return rec.get(method, {}).get("is_correct")


def filter_records(
    records: List[Dict[str, Any]],
    method: str,
    correctness: str,
) -> List[Dict[str, Any]]:
    if correctness == "all":
        return records
    want = correctness == "correct"
    out: List[Dict[str, Any]] = []
    for rec in records:
        val = is_correct_record(rec, method)
        if val is True and want:
            out.append(rec)
        elif val is False and not want:
            out.append(rec)
    return out


def render_sample(
    records: List[Dict[str, Any]],
    method: str,
    index: int,
):
    if not records:
        return (
            gr.update(value="No samples to display."),
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value="", visible=True),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="0 / 0"),
        )

    index = max(0, min(index, len(records) - 1))
    rec = records[index]

    question = rec.get("question", "")
    video_path = rec.get("video_path", "")
    video_exists = bool(video_path) and Path(video_path).exists()
    video_note = ""
    if video_path and not video_exists:
        video_note = f"Video file not found: {video_path}"

    if method == "refine":
        init = rec.get("refine", {}).get("initial_answer", {}).get("raw_output", "")
        desc = rec.get("refine", {}).get("description", {}).get("raw_output", "")
        refined = rec.get("refine", {}).get("refined_answer", {}).get("raw_output", "")
        return (
            gr.update(value=question),
            gr.update(value=video_path if video_exists else None),
            gr.update(value=video_note),
            gr.update(value="", visible=False),
            gr.update(value=init, visible=True),
            gr.update(value=desc, visible=True),
            gr.update(value=refined, visible=True),
            gr.update(value=f"{index + 1} / {len(records)}"),
        )

    raw = rec.get(method, {}).get("raw_output", "")
    return (
        gr.update(value=question),
        gr.update(value=video_path if video_exists else None),
        gr.update(value=video_note),
        gr.update(value=raw, visible=True),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value="", visible=False),
        gr.update(value=f"{index + 1} / {len(records)}"),
    )


def build_ui(default_path: str):
    with gr.Blocks() as demo:
        gr.Markdown("## Output Visualizer")

        state_records = gr.State([])
        state_method = gr.State("direct")
        state_filtered = gr.State([])

        with gr.Row():
            path_in = gr.Textbox(
                label="JSONL output path",
                value=default_path,
                scale=5,
            )
            load_btn = gr.Button("Load", scale=1)

        method_info = gr.Markdown("Method: -")

        with gr.Row():
            correctness = gr.Dropdown(
                choices=["all", "correct", "incorrect"],
                value="all",
                label="Show",
            )
            index = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Sample index",
            )
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")

        count_info = gr.Markdown("0 / 0")

        question = gr.Textbox(label="Question", lines=3)
        video = gr.Video(label="Video")
        video_note = gr.Markdown("")

        raw_answer = gr.Textbox(label="Model raw answer", lines=8)
        init_answer = gr.Textbox(label="Initial raw answer", lines=4, visible=False)
        desc_answer = gr.Textbox(label="Description raw answer", lines=6, visible=False)
        refine_answer = gr.Textbox(label="Refine raw answer", lines=4, visible=False)

        def on_load(path: str, correctness_val: str):
            records = load_jsonl(path)
            method = infer_method(records)
            filtered = filter_records(records, method, correctness_val)
            method_text = f"Method: {method} | Total: {len(records)} | Filtered: {len(filtered)}"
            max_index = max(0, len(filtered) - 1)
            return (
                records,
                method,
                filtered,
                gr.update(value=method_text),
                gr.update(maximum=max_index, value=0),
            )

        def on_filter(records, method, correctness_val):
            filtered = filter_records(records, method, correctness_val)
            max_index = max(0, len(filtered) - 1)
            return (
                filtered,
                gr.update(maximum=max_index, value=0),
                gr.update(value=f"Method: {method} | Total: {len(records)} | Filtered: {len(filtered)}"),
            )

        def on_nav(step, idx, filtered, method):
            if not filtered:
                empty = render_sample(filtered, method, 0)
                return (gr.update(value=0),) + empty
            new_idx = max(0, min(idx + step, len(filtered) - 1))
            return (gr.update(value=new_idx),) + render_sample(filtered, method, new_idx)

        load_btn.click(
            on_load,
            inputs=[path_in, correctness],
            outputs=[state_records, state_method, state_filtered, method_info, index],
        ).then(
            render_sample,
            inputs=[state_filtered, state_method, index],
            outputs=[
                question,
                video,
                video_note,
                raw_answer,
                init_answer,
                desc_answer,
                refine_answer,
                count_info,
            ],
        )

        correctness.change(
            on_filter,
            inputs=[state_records, state_method, correctness],
            outputs=[state_filtered, index, method_info],
        ).then(
            render_sample,
            inputs=[state_filtered, state_method, index],
            outputs=[
                question,
                video,
                video_note,
                raw_answer,
                init_answer,
                desc_answer,
                refine_answer,
                count_info,
            ],
        )

        index.change(
            render_sample,
            inputs=[state_filtered, state_method, index],
            outputs=[
                question,
                video,
                video_note,
                raw_answer,
                init_answer,
                desc_answer,
                refine_answer,
                count_info,
            ],
        )

        prev_btn.click(
            on_nav,
            inputs=[gr.State(-1), index, state_filtered, state_method],
            outputs=[
                index,
                question,
                video,
                video_note,
                raw_answer,
                init_answer,
                desc_answer,
                refine_answer,
                count_info,
            ],
        )

        next_btn.click(
            on_nav,
            inputs=[gr.State(1), index, state_filtered, state_method],
            outputs=[
                index,
                question,
                video,
                video_note,
                raw_answer,
                init_answer,
                desc_answer,
                refine_answer,
                count_info,
            ],
        )

    return demo


def _infer_allowed_paths(default_path: str) -> List[str]:
    allowed = []
    try:
        records = load_jsonl(default_path)
    except Exception:
        return allowed
    for rec in records[:50]:
        video_path = rec.get("video_path")
        if video_path:
            allowed.append(str(Path(video_path).expanduser().resolve().parent))
    # de-dup while preserving order
    seen = set()
    deduped = []
    for p in allowed:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation outputs.")
    parser.add_argument(
        "--path",
        default="outputs/video-hallucer-hallucinated.json/Qwen/Qwen3-VL-8B-Instruct/direct.json",
        help="Path to JSONL output file.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Gradio host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio port.",
    )
    parser.add_argument(
        "--allow-path",
        action="append",
        default=[],
        help="Extra allowed video directories for Gradio.",
    )
    args = parser.parse_args()

    demo = build_ui(args.path)
    inferred = _infer_allowed_paths(args.path)
    allowed_paths = []
    for p in inferred + args.allow_path:
        if p and p not in allowed_paths:
            allowed_paths.append(p)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        allowed_paths=allowed_paths or None,
    )


if __name__ == "__main__":
    main()
