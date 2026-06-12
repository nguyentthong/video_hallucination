"""
src/metrics/consistency.py
---------------------------
The three consistency metrics defined in Metrics_formula.pdf.

For a question group G_j = {Q_target, Q_sub1, ..., Q_subK}:

    cons(G_j) = (1 / |G_j|) x Σ c(Q_i)      where c(Q_i) ∈ {0, 1}

Consistency@All   = (1/N) x Σ cons(G_j)           over all N groups
Consistency@TC    = (1/|S✓|) x Σ cons(G_j)         for j where target is ✓
Consistency@TW    = (1/|S✗|) x Σ cons(G_j)         for j where target is ✗

What each metric reveals (from the PDF)
----------------------------------------
* Cons@All   — overall reasoning coherence.
* Cons@TC    — when the model gets the target right, does it also get the
               supporting sub-questions right?  A gap from 1.0 reveals
               lucky guesses.
* Cons@TW    — when the model gets the target wrong, how much does it still
               get right?  High Cons@TW with low accuracy → broken
               compositional reasoning (easier to fix).  Low Cons@TW →
               deep perceptual failures.
"""

from typing import Dict, List, Optional

from .base import BaseMetric, QuestionGroup


class ConsistencyAll(BaseMetric):
    """
    Consistency@All — average group consistency over every question group.
    """

    @property
    def name(self) -> str:
        return "consistency_all"

    def compute(self, groups: List[QuestionGroup]) -> float:
        if not groups:
            return 0.0
        return sum(g.group_consistency() for g in groups) / len(groups)


class ConsistencyTargetCorrect(BaseMetric):
    """
    Consistency@TC — average group consistency conditioned on the target
    question being answered correctly.

    Returns ``float('nan')`` when no group has a correct target answer,
    so callers can distinguish "undefined" from zero.
    """

    @property
    def name(self) -> str:
        return "consistency_tc"

    def compute(self, groups: List[QuestionGroup]) -> float:
        correct_groups = [g for g in groups if g.is_target_correct()]
        if not correct_groups:
            return float("nan")
        return sum(g.group_consistency() for g in correct_groups) / len(correct_groups)


class ConsistencyTargetWrong(BaseMetric):
    """
    Consistency@TW — average group consistency conditioned on the target
    question being answered incorrectly.

    Returns ``float('nan')`` when every group has a correct target answer.
    """

    @property
    def name(self) -> str:
        return "consistency_tw"

    def compute(self, groups: List[QuestionGroup]) -> float:
        wrong_groups = [g for g in groups if not g.is_target_correct()]
        if not wrong_groups:
            return float("nan")
        return sum(g.group_consistency() for g in wrong_groups) / len(wrong_groups)


class AllConsistencyMetrics(BaseMetric):
    """
    Convenience wrapper that computes all three consistency metrics at once.

    ``compute`` returns Cons@All (the primary scalar).
    ``compute_with_details`` returns all three in one dict.
    """

    @property
    def name(self) -> str:
        return "consistency_all"

    def compute(self, groups: List[QuestionGroup]) -> float:
        return ConsistencyAll().compute(groups)

    def compute_with_details(
        self, groups: List[QuestionGroup]
    ) -> Dict[str, float]:
        return {
            "consistency_all": ConsistencyAll().compute(groups),
            "consistency_tc": ConsistencyTargetCorrect().compute(groups),
            "consistency_tw": ConsistencyTargetWrong().compute(groups),
        }