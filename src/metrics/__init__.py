"""
src/metrics/__init__.py
------------------------
Metric registry.

Available metric keys
---------------------
``"faithful_accuracy"`` FaithfulAccuracy    — A_faith, the PRIMARY STRAND metric
``"accuracy"``          TargetAccuracy      — A_target, correct target answers
``"sub_accuracy"``      SubQuestionAccuracy — A_sub, accuracy on sub-questions only
``"conditional_consistency"`` ConditionalConsistency — A_cons, reference only
``"strand"``            StrandMetrics       — all four headline numbers at once
``"consistency_all"``   ConsistencyAll      — Cons@All from the PDF
``"consistency_tc"``    ConsistencyTargetCorrect  — Cons@TC
``"consistency_tw"``    ConsistencyTargetWrong    — Cons@TW
``"consistency"``       AllConsistencyMetrics — all three Cons metrics at once

Deprecated
----------
The ``Cons@All``/``Cons@TC``/``Cons@TW`` family predates Faithful Accuracy and
averages over the whole question group, target included.  ``Cons@TC`` is
therefore NOT the paper's ``A_cons`` and reads systematically higher.  Use
``"strand"`` for anything you intend to report.

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
from .faithful import (
    ConditionalConsistency,
    FaithfulAccuracy,
    StrandMetrics,
    check_invariants,
)
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
    "FaithfulAccuracy",
    "ConditionalConsistency",
    "StrandMetrics",
    "check_invariants",
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
    # Primary metric first — this is what STRAND is scored on.
    "faithful_accuracy": FaithfulAccuracy,
    "conditional_consistency": ConditionalConsistency,
    "accuracy": TargetAccuracy,
    "sub_accuracy": SubQuestionAccuracy,
    "consistency_all": ConsistencyAll,
    "consistency_tc": ConsistencyTargetCorrect,
    "consistency_tw": ConsistencyTargetWrong,
    # Convenience alias — computes all three consistency metrics
    "consistency": AllConsistencyMetrics,
    # Convenience alias — the four headline STRAND numbers in reporting order
    "strand": StrandMetrics,
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
        selected_keys = [
            k for k in METRIC_REGISTRY if k not in ("consistency", "strand")
        ]
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