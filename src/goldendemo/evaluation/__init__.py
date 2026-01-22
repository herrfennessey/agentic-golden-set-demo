"""Evaluation module for comparing agent vs WANDS judgments."""

from goldendemo.evaluation.comparator import (
    ComparisonResult,
    compare,
    list_golden_sets,
    load_golden_set,
)

__all__ = [
    "ComparisonResult",
    "compare",
    "list_golden_sets",
    "load_golden_set",
]
