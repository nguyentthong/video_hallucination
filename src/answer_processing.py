"""
src/answer_processing.py
-------------------------
Shared answer normalisation utilities.

Why this exists
---------------
Model outputs are free-form strings like:
    "Yes, the man on the left side drops the red cup."
    "No. I do not see a green cup being dropped."
    "Not sure."

A naive string equality check ("yes" == "yes, the man...") always fails.
This module provides the canonical yes/no extractor used by both:
  - src/eval_module.py  (EvaluationMaster / AccuracyEvaluator pipeline)
  - src/metrics/base.py (QuestionGroup.is_target_correct / correctness_vector)

Logic (preserved exactly from the original eval_module.py)
-----------------------------------------------------------
1. Lower-case the string.
2. Find all whole-word occurrences of "yes" or "no" via regex.
3. Take the LAST match as the answer.
4. Return a canonical token: "yes", "no", or "unknown" (no match found).

Using the last match handles hedged answers:
    "No, wait — yes, I can see it."  →  "yes"
"""

import re
from typing import Optional


# Sentinel returned when no yes/no token is found.
UNKNOWN = "unknown"


def extract_yes_no(answer: str) -> str:
    """
    Extract the final yes/no decision from a free-form answer string.

    Parameters
    ----------
    answer : str
        Raw model output.

    Returns
    -------
    str
        ``"yes"``, ``"no"``, or ``"unknown"`` if neither token appears.
    """
    matches = re.findall(r"\b(yes|no)\b", answer.strip().lower())
    if not matches:
        return UNKNOWN
    return matches[-1]  # last match wins


def answers_match(prediction: str, ground_truth: str) -> bool:
    """
    Return True if *prediction* and *ground_truth* resolve to the same
    yes/no decision.

    Both sides are normalised through ``extract_yes_no`` before comparison,
    so verbose model outputs are handled correctly.

    ``"unknown"`` never matches anything — including another ``"unknown"``
    — so unanswered questions always count as wrong.

    Parameters
    ----------
    prediction : str
        Raw model prediction string.
    ground_truth : str
        Ground-truth answer string (e.g. ``"Yes"`` or ``"No"``).

    Returns
    -------
    bool
    """
    pred_token = extract_yes_no(prediction)
    gt_token = extract_yes_no(ground_truth)
    if pred_token == UNKNOWN or gt_token == UNKNOWN:
        return False
    return pred_token == gt_token