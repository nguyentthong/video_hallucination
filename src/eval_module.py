"""
src/eval_module.py
-------------------
Evaluation pipeline: processors, evaluators, and the EvaluationMaster.

Public API is unchanged from the original — existing call-sites in
benchmark_sub.py continue to work without modification.

The only internal change: SimpleAnswerProcessor now delegates its yes/no
extraction to src.answer_processing.extract_yes_no so the logic is not
duplicated between this module and the metrics layer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.answer_processing import extract_yes_no, UNKNOWN
from src.metrics import BaseMetric, QuestionGroup
from src.cache import AnswerCache, get_video_id


# ---------------------------------------------------------------------------
# Processors — raw string → canonical int
# ---------------------------------------------------------------------------

class AnswerProcessor(ABC):
    @abstractmethod
    def __call__(self, answer: str) -> int:
        pass


class SimpleAnswerProcessor(AnswerProcessor):
    """
    Maps a free-form answer string to an integer label.

    Returns
    -------
    1   → "yes"
    0   → "no"
    -1  → no yes/no token found ("unknown")
    """

    _TOKEN_TO_INT = {"yes": 1, "no": 0, UNKNOWN: -1}

    def __call__(self, answer: str) -> int:
        return self._TOKEN_TO_INT[extract_yes_no(answer)]


# ---------------------------------------------------------------------------
# Evaluators — List[int] × List[int] → float
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    @abstractmethod
    def __call__(self, predictions: List[int], truths: List[int]) -> float:
        pass


class AccuracyEvaluator(Evaluator):
    def __call__(self, predictions: List[int], truths: List[int]) -> float:
        if not predictions:
            return 0.0
        return sum(p == t for p, t in zip(predictions, truths)) / len(predictions)


# ---------------------------------------------------------------------------
# EvalModule — named (processor, evaluator) pair
# ---------------------------------------------------------------------------

class EvalModule:
    def __init__(self, name: str, processor: AnswerProcessor, evaluator: Evaluator):
        self.name = name
        self.processor = processor
        self.evaluator = evaluator

    def __call__(self, predictions: List[str], truths: List[str]) -> float:
        processed_predictions = [self.processor(p) for p in predictions]
        processed_truths = [self.processor(t) for t in truths]
        return self.evaluator(processed_predictions, processed_truths)


# ---------------------------------------------------------------------------
# EvaluationMaster — accumulates predictions then computes all modules
# ---------------------------------------------------------------------------

class EvaluationMaster:
    def __init__(self, eval_modules: List[EvalModule]):
        self.eval_modules = eval_modules
        self.predictions: List[str] = []
        self.truths: List[str] = []

    def batch_push(self, predictions: List[str], truths: List[str]) -> None:
        self.predictions.extend(predictions)
        self.truths.extend(truths)

    def push(self, prediction: Any, truth: Any) -> None:
        self.predictions.append(prediction)
        self.truths.append(truth)

    def compute_result(self) -> Dict[str, float]:
        result = {}
        for module in self.eval_modules:
            result[module.name] = module(self.predictions, self.truths)
        return result


# ---------------------------------------------------------------------------
# New pipeline helpers (used by benchmark_sub.py → eval_module functions)
# ---------------------------------------------------------------------------

def build_question_groups(sample: Dict) -> List[QuestionGroup]:
    """
    Convert one benchmark sample dict into a list of QuestionGroup objects.
    """
    groups = []
    video_name = sample.get("video_name", "unknown")
    for idx, (q, a, sqs, sas) in enumerate(
        zip(
            sample["questions"],
            sample["answers"],
            sample["sub-questions"],
            sample["sub-answers"],
        )
    ):
        groups.append(
            QuestionGroup(
                target_question=q,
                sub_questions=list(sqs),
                target_gt=a,
                sub_gts=list(sas),
                group_id=f"{video_name}__q{idx}",
            )
        )
    return groups


def fill_predictions(groups, sample, model, cache, max_new_tokens=256):
    """
    Populate target_pred and sub_preds on each group in-place,
    fetching from cache where possible and calling the model otherwise.
    """
    video_id = get_video_id(sample["video_name"])
    video_path = sample["video_path"]

    all_questions: List[str] = []
    for g in groups:
        all_questions.append(g.target_question)
        all_questions.extend(g.sub_questions)

    seen = set()
    unique_questions = []
    for q in all_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    answers = cache.resolve(
        video_id, unique_questions, model, video_path, max_new_tokens
    )
    answer_map = dict(zip(unique_questions, answers))

    for g in groups:
        g.target_pred = answer_map[g.target_question]
        g.sub_preds = [answer_map[sq] for sq in g.sub_questions]


def evaluate(groups: List[QuestionGroup], metrics: List[BaseMetric]) -> Dict[str, float]:
    """Run all metrics over the provided groups and return a flat result dict."""
    result: Dict[str, float] = {}
    for metric in metrics:
        result.update(metric.compute_with_details(groups))
    return result