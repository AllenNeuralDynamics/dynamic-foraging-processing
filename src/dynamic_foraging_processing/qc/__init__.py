"""Quality control for dynamic foraging datasets.

Builds an ``aind_data_schema`` ``QualityControl`` object from primitive behavior
data (lick times, per-trial choices) plus the contraqctor-based contract QA.

The module is organized by QC stage:

- :mod:`~dynamic_foraging_processing.qc._core` -- the shared stage interface,
  per-check result type, and ``QualityControl`` assembler.
- :mod:`~dynamic_foraging_processing.qc.raw` -- the raw-data (contract QA) stage.
- :mod:`~dynamic_foraging_processing.qc.processed` -- the processed-data
  (behavior metrics) stage.
"""

from dynamic_foraging_processing.qc._core import (
    DEFAULT_GROUPING,
    STATUS_CONVERTER,
    BaseQC,
    QCResult,
    build_quality_control,
    to_builtin,
    to_metrics,
)
from dynamic_foraging_processing.qc.processed import (
    ProcessedQC,
    behavior_qc_results,
    calculate_lick_intervals,
    lick_interval_results,
    plot_lick_intervals,
    plot_side_bias,
    side_bias_result,
)
from dynamic_foraging_processing.qc.raw import (
    RawQC,
    contract_qc_metrics,
    results_to_metrics,
)

__all__ = [
    "DEFAULT_GROUPING",
    "STATUS_CONVERTER",
    "BaseQC",
    "ProcessedQC",
    "QCResult",
    "RawQC",
    "behavior_qc_results",
    "build_quality_control",
    "calculate_lick_intervals",
    "contract_qc_metrics",
    "lick_interval_results",
    "plot_lick_intervals",
    "plot_side_bias",
    "results_to_metrics",
    "side_bias_result",
    "to_builtin",
    "to_metrics",
]
