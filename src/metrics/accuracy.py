"""
src/metrics/accuracy.py
------------------------
Simple accuracy over target questions only.

accuracy = (number of groups where target_pred == target_gt) / N
"""

from typing import List

from .base import BaseMetric, QuestionGroup


class TargetAccuracy(BaseMetric):
    """
    Fraction of question groups where the model's target answer is correct.

    This is the vanilla accuracy metric — it ignores sub-questions entirely.
    """

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        if not groups:
            return 0.0
        correct = sum(1 for g in groups if g.is_target_correct())
        return correct / len(groups)


class SubQuestionAccuracy(BaseMetric):
    """
    Flat accuracy over all sub-questions (target questions excluded).

    Useful as a standalone diagnostic to see how well the model handles
    atomic, decomposed questions.
    """

    @property
    def name(self) -> str:
        return "sub_accuracy"

    def compute(self, groups: List[QuestionGroup]) -> float:
        correct = total = 0
        for g in groups:
            # Skip the first element (target) — only sub-questions
            cv = g.correctness_vector()[1:]
            correct += sum(cv)
            total += len(cv)
        if total == 0:
            return 0.0
        return correct / total