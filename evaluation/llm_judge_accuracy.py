# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.67.0",
#     "python-dotenv>=1.2.2",
#     "tqdm>=4.67.3",
# ]
# ///
"""
llm_judge_accuracy.py
----------------------
Re-score cached answer JSON files using Gemini Flash as an LLM judge.

For each ``(question, pred, gt)`` tuple stored under the four cache roots
used by ``inspect_accuracy.py``, ask the Gemini native API whether ``pred``
agrees with ``gt``. Overwrites the ``is_correct`` field in place.

A sidecar ``is_correct_source`` field is written alongside ``is_correct``
so re-running this script skips files already judged; pass ``--force`` to
re-judge them.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()


CACHE_ROOT = os.environ.get("CACHE_ROOT", "cache")

# (cache_root_dir, answer_subdir_name)
CACHE_ROOTS: List[Tuple[str, str]] = [
    (
        f"{CACHE_ROOT}/states_15s_32f-planner_gemini-3-flash_fresh",
        "answers_planner_gemini-3-flash-preview_v1",
    ),
    (
        f"{CACHE_ROOT}/states_15s_32f-planner_qwen3-vl-235b-thinking_fresh",
        "answers_planner_qwen3-vl-235b-a22b-thinking_v1",
    ),
    (
        f"{CACHE_ROOT}/states_15s_32f-qwen3-vl-235b-a22b-thinking-concat_fresh",
        "answers_filter_v1",
    ),
    (
        f"{CACHE_ROOT}/states_15s_32f-qwen3-vl-235b-a22b-thinking-concat_qwen3-235b-instruct-2507_fresh",
        "answers_filter_qwen3-235b-a22b-2507_v1",
    ),
]

MODEL_ID = "gemini-3-flash-preview"
JUDGE_SOURCE_TAG = "llm_judge_gemini_3_flash"

JUDGE_PROMPT_TMPL = """You are an impartial judge deciding whether a model's free-form answer agrees with the ground-truth yes/no answer for a question about a video.

Question: {question}

Ground truth: {gt}

Model answer: {pred}

Decide whether the model's answer conveys the SAME verdict as the ground truth (yes vs. no). Focus on what verdict the model commits to, not on hedging, justification, or extra narration.

Respond with exactly one word: YES if the model answer agrees with the ground truth, NO otherwise. No punctuation, no explanation."""


def iter_answer_files() -> Iterable[str]:
    for root, answer_subdir in CACHE_ROOTS:
        if not os.path.isdir(root):
            continue
        for video_id in sorted(os.listdir(root)):
            ans_dir = os.path.join(root, video_id, answer_subdir)
            if not os.path.isdir(ans_dir):
                continue
            for name in sorted(os.listdir(ans_dir)):
                if name.endswith(".json"):
                    yield os.path.join(ans_dir, name)


def _call_judge(client: genai.Client, prompt: str, retries: int = 3) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=16384,
                ),
            )
            text = getattr(response, "text", "") or ""
            return text.strip()
        except Exception as e:
            last_exc = e
            if attempt == retries - 1:
                break
            time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"judge call failed after {retries} attempts: {last_exc}")


def _parse_verdict(raw: str) -> Optional[bool]:
    if not raw:
        return None
    first = raw.strip().split()[0].strip(".,:;!?").upper()
    if first.startswith("YES"):
        return True
    if first.startswith("NO"):
        return False
    return None


def judge_one(client: genai.Client, path: str, force: bool) -> Tuple[str, Optional[str]]:
    """Returns (status, detail). status in {"ok", "skip", "error"}."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return ("error", f"read-error:{path}:{e}")

    if not isinstance(data, dict):
        return ("error", f"bad-record:{path}")

    question = data.get("question")
    pred = data.get("pred")
    gt = data.get("gt")
    if not isinstance(question, str) or not isinstance(pred, str) or not isinstance(gt, str):
        return ("error", f"missing-fields:{path}")

    if not force and data.get("is_correct_source") == JUDGE_SOURCE_TAG:
        return ("skip", None)

    prompt = JUDGE_PROMPT_TMPL.format(
        question=question.strip(),
        gt=gt.strip(),
        pred=pred.strip(),
    )

    try:
        raw = _call_judge(client, prompt)
    except Exception as e:
        return ("error", f"api-error:{path}:{e}")

    verdict = _parse_verdict(raw)
    if verdict is None:
        return ("error", f"parse-error:{path}:{raw!r}")

    data["is_correct"] = verdict
    data["is_correct_source"] = JUDGE_SOURCE_TAG

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return ("error", f"write-error:{path}:{e}")

    return ("ok", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge files already tagged by this script.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of files (for a smoke test).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set (check your .env or shell env).")
    client = genai.Client(api_key=api_key)

    files = list(iter_answer_files())
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("No answer JSON files found under the configured cache roots.", flush=True)
        return

    print(
        f"Judging {len(files)} answer files with model={MODEL_ID} "
        f"(concurrency={args.concurrency}, force={args.force})",
        flush=True,
    )

    ok_n = skip_n = err_n = 0
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(judge_one, client, p, args.force): p for p in files}
        pbar = tqdm(as_completed(futures), total=len(futures))
        for fut in pbar:
            status, detail = fut.result()
            if status == "ok":
                ok_n += 1
            elif status == "skip":
                skip_n += 1
            else:
                err_n += 1
                if detail:
                    errors.append(detail)
            pbar.set_postfix(ok=ok_n, skip=skip_n, err=err_n)

    print(f"\nDone. judged={ok_n} skipped={skip_n} errors={err_n}", flush=True)
    if errors:
        print(f"First {min(len(errors), 50)} error(s):")
        for e in errors[:50]:
            print("  " + e)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")


if __name__ == "__main__":
    main()
