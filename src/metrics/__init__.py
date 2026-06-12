"""
src/metrics/__init__.py
------------------------
Metric registry.

Available metric keys
---------------------
``"accuracy"``          TargetAccuracy      — fraction of correct target answers
``"sub_accuracy"``      SubQuestionAccuracy — accuracy on sub-questions only
``"consistency_all"``   ConsistencyAll      — Cons@All from the PDF
``"consistency_tc"``    ConsistencyTargetCorrect  — Cons@TC
``"consistency_tw"``    ConsistencyTargetWrong    — Cons@TW
``"consistency"``       AllConsistencyMetrics — all three Cons metrics at once

Special sentinel
----------------
``"all"``               Resolves to every metric in the registry.

Usage
-----
>>> from src.metrics import build_metrics
>>> metrics = build_metrics(["accuracy", "consistency"])
>>> for m in metrics:
...     result = m.compute_with_details(groups)
"""

from typing import Dict, List, Type

from .accuracy import SubQuestionAccuracy, TargetAccuracy
from .base import BaseMetric, QuestionGroup
from .consistency import (
    AllConsistencyMetrics,
    ConsistencyAll,
    ConsistencyTargetCorrect,
    ConsistencyTargetWrong,
)

__all__ = [
    "BaseMetric",
    "QuestionGroup",
    "TargetAccuracy",
    "SubQuestionAccuracy",
    "ConsistencyAll",
    "ConsistencyTargetCorrect",
    "ConsistencyTargetWrong",
    "AllConsistencyMetrics",
    "METRIC_REGISTRY",
    "build_metrics",
]

# ---------------------------------------------------------------------------
# Registry: short name → class
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Type[BaseMetric]] = {
    "accuracy": TargetAccuracy,
    "sub_accuracy": SubQuestionAccuracy,
    "consistency_all": ConsistencyAll,
    "consistency_tc": ConsistencyTargetCorrect,
    "consistency_tw": ConsistencyTargetWrong,
    # Convenience alias — computes all three consistency metrics
    "consistency": AllConsistencyMetrics,
}


def build_metrics(keys: List[str]) -> List[BaseMetric]:
    """
    Instantiate metrics from a list of registry keys.

    Pass ``["all"]`` to get every metric in the registry.

    Parameters
    ----------
    keys : List[str]
        Any combination of keys from METRIC_REGISTRY, or the special
        sentinel ``"all"``.

    Returns
    -------
    List[BaseMetric]
        Instantiated metric objects, deduplicated and in a stable order.

    Raises
    ------
    ValueError
        If an unrecognised key is requested.

    Examples
    --------
    >>> build_metrics(["accuracy", "consistency"])
    [TargetAccuracy(), AllConsistencyMetrics()]

    >>> build_metrics(["all"])
    [TargetAccuracy(), SubQuestionAccuracy(), ConsistencyAll(), ...]
    """
    if "all" in keys:
        # Expand to all individual metrics.  Exclude the "consistency" alias
        # because consistency_all/tc/tw are already present individually,
        # and the alias would duplicate the "consistency_all" key in results.
        selected_keys = [k for k in METRIC_REGISTRY if k != "consistency"]
    else:
        unknown = [k for k in keys if k not in METRIC_REGISTRY]
        if unknown:
            valid = ", ".join(sorted(METRIC_REGISTRY.keys()))
            raise ValueError(
                f"Unknown metric key(s): {unknown}.  Valid keys: {valid}"
            )
        selected_keys = keys

    # Deduplicate while preserving order
    seen: set = set()
    result: List[BaseMetric] = []
    for key in selected_keys:
        if key not in seen:
            seen.add(key)
            result.append(METRIC_REGISTRY[key]())
    return result