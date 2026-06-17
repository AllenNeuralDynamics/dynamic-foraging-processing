"""Shared QC infrastructure: the stage interface, schema helpers, the per-check
result type, and the ``QualityControl`` assembler.

These pieces are stage-agnostic; the raw and processed stages build on them.
"""

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc._core.builder import (
    DEFAULT_GROUPING,
    build_quality_control,
)
from dynamic_foraging_processing.qc._core.result import QCResult, to_metrics
from dynamic_foraging_processing.qc._core.schema import (
    STATUS_CONVERTER,
    bool_to_status,
    make_metric,
    now_seattle,
    now_utc,
    to_builtin,
)

__all__ = [
    "DEFAULT_GROUPING",
    "STATUS_CONVERTER",
    "BaseQC",
    "QCResult",
    "bool_to_status",
    "build_quality_control",
    "make_metric",
    "now_seattle",
    "now_utc",
    "to_builtin",
    "to_metrics",
]
