"""Quality control for dynamic foraging datasets.

Builds an ``aind_data_schema`` ``QualityControl`` object from primitive behavior
data (lick times, per-trial choices) plus the contraqctor-based contract QA.
"""

from dynamic_foraging_processing.qc._behavior import (
    calculate_lick_intervals,
    compute_rolling_bias,
    compute_side_bias,
    lick_interval_results,
    side_bias_result,
)
from dynamic_foraging_processing.qc._builder import (
    DEFAULT_GROUPING,
    behavior_qc_results,
    build_quality_control,
)
from dynamic_foraging_processing.qc._contract_qa import (
    contract_qc_metrics,
    results_to_metrics,
)
from dynamic_foraging_processing.qc._plots import plot_lick_intervals, plot_side_bias
from dynamic_foraging_processing.qc._result import QCResult, to_metrics
from dynamic_foraging_processing.qc._schema import (
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
    "QCResult",
    "behavior_qc_results",
    "bool_to_status",
    "build_quality_control",
    "calculate_lick_intervals",
    "compute_rolling_bias",
    "compute_side_bias",
    "contract_qc_metrics",
    "lick_interval_results",
    "make_metric",
    "now_seattle",
    "now_utc",
    "plot_lick_intervals",
    "plot_side_bias",
    "results_to_metrics",
    "side_bias_result",
    "to_builtin",
    "to_metrics",
]
