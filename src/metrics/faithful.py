"""
src/metrics/faithful.py
------------------------
Faithful Accuracy (A_faith) — the primary STRAND metric — and Conditional
Consistency (A_cons), reported alongside it for reference only.

Faithful Accuracy
-----------------
A target counts only when the model answers it *and* every one of its
supporting sub-questions correctly:

    A_faith = (1/N) * Σ_i  1[ŷ_i = y_i] * Π_j 1[ŷ_ij = y_ij]

The denominator is **all N targets**, not just the ones the model answered
correctly. That fixed denominator is the whole point: it makes A_faith
monotone in both target and sub-question correctness, zero for a model that
answers nothing, and directly comparable across models.

Conditional Consistency
-----------------------
    A_cons = (1/|C|) * Σ_{i in C} (1/m_i) * Σ_j 1[ŷ_ij = y_ij]

where C is the set of targets the model answered correctly. Note two things:

1. The inner average runs over **sub-questions only** — the target is not
   included. This differs from ``Cons@TC`` in :mod:`src.metrics.consistency`,
   which averages over the whole group (target included) and therefore reads
   systematically higher. ``A_cons`` is the definition used in the paper;
   ``Cons@TC`` is kept only for backwards compatibility.
2. Because C is chosen by the model, two models are scored on different
   subsets. A_cons is maximised by low-recall selectivity and does not fall
   when a model misses a target outright, so state consistency claims in terms
   of A_faith and report A_cons for reference.

Invariants (asserted by :func:`check_invariants`)
-------------------------------------------------
    A_faith <= A_target
    A_faith <= A_target * A_cons
"""

from typing import Dict, List

from .base import BaseMetric, QuestionGroup


def _sub_correctness(group: QuestionGroup) -> List[bool]:
    """Correctness of the sub-questions only, dropping the leading target."""
    return group.correctness_vector()[1:]


class FaithfulAccuracy(BaseMetric):
    """
    A_faith — the primary STRAND metric.

    Credits a target only when the target answer and every supporting
    sub-question are correct, over all targets.
    """

    @property
    def name(self) -> str:
        return "faithful_accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        if not groups:
            return 0.0
        faithful = sum(
            1
            for g in groups
            if g.is_target_correct() and all(_sub_correctness(g))
        )
        return faithful / len(groups)


class ConditionalConsistency(BaseMetric):
    """
    A_cons — secondary, reference only.

    Among targets answered correctly, the mean proportion of sub-questions
    also answered correctly. Sub-questions only; the target is excluded.

    Returns ``float('nan')`` when the model answers no target correctly, so
    callers can tell "undefined" apart from zero.
    """

    @property
    def name(self) -> str:
        return "conditional_consistency"

    def compute(self, groups: List[QuestionGroup]) -> float:
        correct = [g for g in groups if g.is_target_correct()]
        scored = [g for g in correct if _sub_correctness(g)]
        if not scored:
            return float("nan")
        return sum(
            sum(_sub_correctness(g)) / len(_sub_correctness(g)) for g in scored
        ) / len(scored)


class StrandMetrics(BaseMetric):
    """
    All four headline STRAND numbers in one pass, in reporting order.

    ``compute`` returns A_faith, the primary scalar.
    """

    @property
    def name(self) -> str:
        return "faithful_accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        return FaithfulAccuracy().compute(groups)

    def compute_with_details(
        self, groups: List[QuestionGroup]
    ) -> Dict[str, float]:
        from .accuracy import SubQuestionAccuracy, TargetAccuracy

        return {
            "faithful_accuracy": FaithfulAccuracy().compute(groups),
            "accuracy": TargetAccuracy().compute(groups),
            "sub_accuracy": SubQuestionAccuracy().compute(groups),
            "conditional_consistency": ConditionalConsistency().compute(groups),
        }


def check_invariants(details: Dict[str, float], tol: float = 1e-9) -> List[str]:
    """
    Return the definitional invariants violated by a metric dict.

    An empty list means the numbers are internally consistent. A non-empty
    list means the evaluation is wrong — do not report those numbers.
    """
    problems: List[str] = []
    faith = details.get("faithful_accuracy")
    target = details.get("accuracy")
    cons = details.get("conditional_consistency")

    if faith is None or target is None:
        return problems

    if faith > target + tol:
        problems.append(
            f"A_faith {faith:.4f} > A_target {target:.4f} (definitional)"
        )
    if cons is not None and cons == cons:  # skip NaN
        ceiling = target * cons
        if faith > ceiling + tol:
            problems.append(
                f"A_faith {faith:.4f} > A_target * A_cons {ceiling:.4f} (definitional)"
            )
    return problems
