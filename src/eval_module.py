from typing import Any, Dict, List
from abc import ABC, abstractmethod

class AnswerProcessor(ABC):
    @abstractmethod
    def __call__(self, answer):
        pass

class Evaluator(ABC):
    @abstractmethod
    def __call__(self, prediction, truths):
        pass

class SimpleAnswerProcessor(AnswerProcessor):
    def __call__(self, answer: str) -> int:
        _answer = answer.lower()
        if "Yes" in answer:
            return 1
        elif "No" in answer:
            return 0

class AccuracyEvaluator(Evaluator):
    def __call__(self, predictions: List[int], truths: List[int]):
        total = len(predictions)
        acc = 0
        for p, t in zip(predictions, truths):
            if p == t:
                acc += 1
        return acc / total

class EvalModule:
    def __init__(self, name: str, processor: AnswerProcessor, evaluator: Evaluator):
        self.name = name
        self.processor = processor
        self.evaluator = evaluator

    def __call__(self, predictions, truths):
        processed_predictions = [self.processor(prediction) for prediction in predictions]
        processed_truths = [self.processor(truth) for truth in truths]
        return self.evaluator(processed_predictions, processed_truths)

class EvaluationMaster:
    def __init__(self, eval_modules: List[EvalModule]):
        self.eval_modules = eval_modules
        self.predictions = []
        self.truths = []

    def batch_push(self, predictions: List[str], truths: List[str]):
        self.predictions.extend(predictions)
        self.truths.extend(truths)

    def push(self, prediction: Any, truth: Any):
        self.predictions.append(prediction)
        self.truths.append(truth)

    def compute_result(self) -> Dict[str, float]:
        result = {}
        for eval_module in self.eval_modules:
            result[eval_module.name] = eval_module(self.predictions, self.truths)
        return result