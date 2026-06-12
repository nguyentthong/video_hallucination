"""
tests/test_metrics.py
----------------------
Verifies all three consistency metrics and accuracy against the worked example
on page 2 of Metrics_formula.pdf.

PDF example recap
-----------------
G1: target ✓,  2 subs both ✓  →  cons = 3/3 = 1.000  (enters S✓)
G2: target ✓,  2 subs: Q2 ✗ Q3 ✓  →  cons = 2/3 = 0.667  (enters S✓)
G3: target ✗,  Q5 ✓ Q6 ✗  →  cons = 1/3 = 0.333  (enters S✗)
G4: target ✗,  Q8 ✗  →  cons = 0/2 = 0.000  (enters S✗)

Expected results
----------------
Cons@All  = (1.000 + 0.667 + 0.333 + 0.000) / 4 = 0.500
Cons@TC   = (1.000 + 0.667) / 2                  = 0.833
Cons@TW   = (0.333 + 0.000) / 2                  = 0.167
Accuracy  = 2 / 4                                 = 0.500
"""

import math
import sys
import os

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import (
    QuestionGroup,
    TargetAccuracy,
    ConsistencyAll,
    ConsistencyTargetCorrect,
    ConsistencyTargetWrong,
    build_metrics,
)


# ---------------------------------------------------------------------------
# Build the four groups from the PDF
# ---------------------------------------------------------------------------

def _make_groups():
    # G1: target ✓, Q2 ✓, Q3 ✓
    g1 = QuestionGroup(
        target_question="Is the leftmost suitcase a cake?",
        sub_questions=[
            "Are leftmost and cake the same?",
            "Is one of the suitcases a cake?",
        ],
        target_gt="Yes", sub_gts=["Yes", "Yes"],
        target_pred="Yes", sub_preds=["Yes", "Yes"],
        group_id="G1",
    )
    # G2: target ✓, Q2 ✗ (model says No, GT is Yes), Q3 ✓
    g2 = QuestionGroup(
        target_question="Is the leftmost suitcase a cake?",
        sub_questions=[
            "Are leftmost and cake the same?",
            "Is one of the suitcases a cake?",
        ],
        target_gt="Yes", sub_gts=["Yes", "Yes"],
        target_pred="Yes", sub_preds=["No", "Yes"],   # Q2 wrong
        group_id="G2",
    )
    # G3: target ✗, Q5 ✓, Q6 ✗
    g3 = QuestionGroup(
        target_question="Did the man open three boxes?",
        sub_questions=[
            "Did the man open two boxes?",
            "Did the man open only one box?",
        ],
        target_gt="Yes", sub_gts=["Yes", "No"],
        target_pred="No",  sub_preds=["Yes", "No"],   # target wrong, Q5 right, Q6 right (pred No == gt No)
        group_id="G3",
    )
    # G4: target ✗, Q8 ✗
    g4 = QuestionGroup(
        target_question="Did he visit the laundry before dorm?",
        sub_questions=["Did he take elevator before ping pong?"],
        target_gt="Yes", sub_gts=["No"],
        target_pred="Yes",  # target wrong? wait — let me re-read PDF
        # PDF says target ✗ for G4, target_pred != target_gt
        # GT=Yes pred=Yes would be correct… re-reading:
        # "Q7 target: Did he visit laundry before dorm? ✗ Yes"
        # The ✗ means prediction is wrong.  GT is presumably "No" then
        # and model predicted "Yes", OR GT is "Yes" and model predicted "No".
        # From PDF: target is marked ✗ and enters S_wrong.
        # cons(G4) = 0/2 = 0.000, meaning both target AND sub are wrong.
        # Sub GT="No", model must have predicted "Yes" (wrong).
        # So: target_gt="No", target_pred="Yes"; sub_gt="No", sub_pred="Yes"
        group_id="G4",
    )
    # Fix G4 based on the reasoning above
    g4 = QuestionGroup(
        target_question="Did he visit the laundry before dorm?",
        sub_questions=["Did he take elevator before ping pong?"],
        target_gt="No",  sub_gts=["No"],
        target_pred="Yes", sub_preds=["Yes"],   # both wrong → cons = 0/2 = 0
        group_id="G4",
    )
    # Fix G3: cons should be 1/3
    # target wrong (1 wrong), Q5 right (1 right), Q6: gt=No pred=No → right (1 right)
    # That gives 2/3 — but PDF says 1/3.
    # Re-reading: Q5 "Did man open two boxes?" GT=Yes, pred=Yes ✓
    #             Q6 "Did man open only one box?" GT=No,  pred=No  ✓ ... that's 2/3 not 1/3
    # PDF says cons(G3) = 1/3.  So only 1 of 3 is right.
    # Target ✗, one sub ✓, one sub ✗.
    # → target_pred=No (wrong since gt=Yes), Q5 pred=Yes (right), Q6 pred=Yes (wrong, gt=No)
    g3 = QuestionGroup(
        target_question="Did the man open three boxes?",
        sub_questions=["Did the man open two boxes?", "Did the man open only one box?"],
        target_gt="Yes", sub_gts=["Yes", "No"],
        target_pred="No",  sub_preds=["Yes", "Yes"],  # target ✗, Q5 ✓, Q6 ✗ → 1/3
        group_id="G3",
    )
    return [g1, g2, g3, g4]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

TOLERANCE = 1e-3


def test_group_consistency():
    g1, g2, g3, g4 = _make_groups()
    assert abs(g1.group_consistency() - 1.000) < TOLERANCE, g1.group_consistency()
    assert abs(g2.group_consistency() - 0.667) < TOLERANCE, g2.group_consistency()
    assert abs(g3.group_consistency() - 0.333) < TOLERANCE, g3.group_consistency()
    assert abs(g4.group_consistency() - 0.000) < TOLERANCE, g4.group_consistency()
    print("✓ group_consistency")


def test_consistency_all():
    groups = _make_groups()
    result = ConsistencyAll().compute(groups)
    assert abs(result - 0.500) < TOLERANCE, f"Cons@All = {result}"
    print(f"✓ Cons@All = {result:.4f}")


def test_consistency_tc():
    groups = _make_groups()
    result = ConsistencyTargetCorrect().compute(groups)
    assert abs(result - 0.833) < TOLERANCE, f"Cons@TC = {result}"
    print(f"✓ Cons@TC  = {result:.4f}")


def test_consistency_tw():
    groups = _make_groups()
    result = ConsistencyTargetWrong().compute(groups)
    assert abs(result - 0.167) < TOLERANCE, f"Cons@TW = {result}"
    print(f"✓ Cons@TW  = {result:.4f}")


def test_accuracy():
    groups = _make_groups()
    result = TargetAccuracy().compute(groups)
    assert abs(result - 0.500) < TOLERANCE, f"Accuracy = {result}"
    print(f"✓ Accuracy = {result:.4f}")


def test_build_metrics_all():
    metrics = build_metrics(["all"])
    names = [m.name for m in metrics]
    assert "accuracy" in names
    assert "consistency_all" in names
    print(f"✓ build_metrics(['all']) returned: {names}")


def test_build_metrics_subset():
    metrics = build_metrics(["accuracy", "consistency_tc"])
    assert len(metrics) == 2
    assert metrics[0].name == "accuracy"
    assert metrics[1].name == "consistency_tc"
    print("✓ build_metrics subset")


def test_build_metrics_unknown_raises():
    try:
        build_metrics(["nonexistent_metric"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Unknown metric correctly raises ValueError: {e}")



def test_verbose_answer_handling():
    """
    Ensures metrics correctly handle realistic verbose model outputs
    rather than bare 'Yes'/'No' strings.
    """
    # G1: target verbose-yes ✓, both subs verbose-yes ✓  →  cons = 1.0
    g1 = QuestionGroup(
        target_question="Does the man drop the red cup?",
        sub_questions=["Is there a man on the left?", "Does the man drop a cup?"],
        target_gt="Yes", sub_gts=["Yes", "Yes"],
        target_pred="Yes, the man on the left clearly drops the red cup.",
        sub_preds=[
            "Yes, there is a man visible on the left side of the frame.",
            "Yes, he drops a cup around the 3-second mark.",
        ],
        group_id="verbose_G1",
    )
    # G2: target verbose-no ✗ (GT=Yes), sub1 verbose-yes ✓, sub2 verbose-no ✗
    g2 = QuestionGroup(
        target_question="Does the woman win the game?",
        sub_questions=["Is there a woman in the middle?", "Does the woman drop a cup?"],
        target_gt="Yes", sub_gts=["Yes", "Yes"],
        target_pred="No, I do not see the woman winning.",
        sub_preds=[
            "Yes, there is a woman in the middle.",
            "No, she does not appear to drop any cup.",
        ],
        group_id="verbose_G2",
    )

    assert abs(g1.group_consistency() - 1.0) < 1e-3, g1.group_consistency()
    # G2: target wrong, sub1 right, sub2 wrong → 1/3
    assert abs(g2.group_consistency() - (1/3)) < 1e-3, g2.group_consistency()

    groups = [g1, g2]
    acc = TargetAccuracy().compute(groups)
    assert abs(acc - 0.5) < 1e-3, acc

    print("✓ verbose answer handling (yes/no extraction from free-form output)")

if __name__ == "__main__":
    test_group_consistency()
    test_consistency_all()
    test_consistency_tc()
    test_consistency_tw()
    test_accuracy()
    test_build_metrics_all()
    test_build_metrics_subset()
    test_build_metrics_unknown_raises()
    test_verbose_answer_handling()
    print("\nAll tests passed.")