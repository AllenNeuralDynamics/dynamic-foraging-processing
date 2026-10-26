"""Shared QC infrastructure: the stage interface, the per-check result type, and
the ``QualityControl`` assembler.

These pieces are stage-agnostic; the raw and processed stages build on them.
The ``QualityControl`` assembler, status mapping, and value conversion are
shared with VR foraging via ``aind_contraqctor_to_schema_export``.
"""

from aind_contraqctor_to_schema_export import (
    DEFAULT_GROUPING,
    STATUS_CONVERTER,
    build_quality_control,
    to_builtin,
)

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc._core.result import QCResult, to_metrics

__all__ = [
    "DEFAULT_GROUPING",
    "STATUS_CONVERTER",
    "BaseQC",
    "QCResult",
    "build_quality_control",
    "to_builtin",
    "to_metrics",
]
