"""
src/metrics/base.py
--------------------
Shared data structures and the abstract BaseMetric contract.

QuestionGroup
-------------
One group G_j = {Q_target, Q_sub1, ..., Q_subK} together with both
the ground-truth answers and the model's predicted answers.

All metric implementations receive a list of QuestionGroup objects so
they are independent of how the benchmark data was loaded or how
inference was run.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.answer_processing import answers_match as _default_match


def _normalise_answer(answer: str) -> str:
    """
    Legacy normaliser kept for backward compatibility with code that
    calls ``correctness_vector(normalise=some_fn)``.

    Internally, QuestionGroup uses ``src.answer_processing.answers_match``
    by default, which correctly handles verbose model outputs by extracting
    the final yes/no token before comparing.
    """
    return answer.strip().lower()


@dataclass
class QuestionGroup:
    """
    One question group: a target question paired with its sub-questions.

    Attributes
    ----------
    target_question : str
        The main question being evaluated.
    sub_questions : List[str]
        Sub-questions that test necessary sub-conditions.
    target_gt : str
        Ground-truth answer for the target question.
    sub_gts : List[str]
        Ground-truth answers for sub-questions (same order).
    target_pred : Optional[str]
        Model's predicted answer for the target question.
        None means the model has not been run yet.
    sub_preds : List[Optional[str]]
        Model's predicted answers for sub-questions.
        None entries mean not yet predicted.
    group_id : Optional[str]
        Optional identifier (e.g. video name + question index) for debugging.
    """

    target_question: str
    sub_questions: List[str]
    target_gt: str
    sub_gts: List[str]
    target_pred: Optional[str] = None
    sub_preds: List[Optional[str]] = field(default_factory=list)
    group_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def all_questions(self) -> List[str]:
        """[target_question] + sub_questions"""
        return [self.target_question] + self.sub_questions

    @property
    def all_gts(self) -> List[str]:
        """[target_gt] + sub_gts"""
        return [self.target_gt] + self.sub_gts

    @property
    def all_preds(self) -> List[Optional[str]]:
        """[target_pred] + sub_preds"""
        return [self.target_pred] + list(self.sub_preds)

    def is_target_correct(self, match_fn: Callable = None) -> bool:
        """
        Return True if the model's target prediction is correct.

        Parameters
        ----------
        match_fn : Callable(pred, gt) -> bool, optional
            Custom comparator.  Defaults to ``answers_match`` from
            ``src.answer_processing``, which extracts the final yes/no
            token from both strings before comparing — correctly handling
            verbose model outputs like "Yes, the man drops the cup."
        """
        if match_fn is None:
            match_fn = _default_match
        if self.target_pred is None:
            return False
        return match_fn(self.target_pred, self.target_gt)

    def correctness_vector(self, match_fn: Callable = None) -> List[bool]:
        """
        Boolean correct/wrong for each question in the group
        (target first, then subs).

        Parameters
        ----------
        match_fn : Callable(pred, gt) -> bool, optional
            Same semantics as in ``is_target_correct``.
        """
        if match_fn is None:
            match_fn = _default_match
        return [
            (pred is not None) and match_fn(pred, gt)
            for pred, gt in zip(self.all_preds, self.all_gts)
        ]

    def group_consistency(self) -> float:
        """
        cons(G_j) = fraction of questions in the group that are correct.
        Returns 0.0 if no predictions have been set.
        """
        vec = self.correctness_vector()
        if not vec:
            return 0.0
        return sum(vec) / len(vec)





# ---------------------------------------------------------------------------
# Abstract metric
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """
    A metric that operates on a list of QuestionGroup objects.

    Subclasses implement ``compute`` and optionally ``name``.
    """

    @property
    def name(self) -> str:
        """Human-readable metric name, used in result dicts."""
        return self.__class__.__name__

    @abstractmethod
    def compute(self, groups: List[QuestionGroup]) -> float:
        """
        Compute the metric over all provided groups.

        Parameters
        ----------
        groups : List[QuestionGroup]
            Must all have predictions filled in.

        Returns
        -------
        float
            Scalar metric value.
        """
        ...

    def compute_with_details(
        self, groups: List[QuestionGroup]
    ) -> Dict[str, float]:
        """
        Return a dict with at least ``{self.name: value}``.
        Override in subclasses to add per-group breakdowns.
        """
        return {self.name: self.compute(groups)}