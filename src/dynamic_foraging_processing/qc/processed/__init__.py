"""Processed-data QC stage: behavior metrics computed from per-trial arrays."""

from dynamic_foraging_processing.qc.processed.behavior import (
    calculate_lick_intervals,
    compute_rolling_bias,
    compute_side_bias,
    lick_interval_results,
    side_bias_result,
)
from dynamic_foraging_processing.qc.processed.plots import (
    plot_lick_intervals,
    plot_side_bias,
)
from dynamic_foraging_processing.qc.processed.results import behavior_qc_results
from dynamic_foraging_processing.qc.processed.stage import ProcessedQC

__all__ = [
    "ProcessedQC",
    "behavior_qc_results",
    "calculate_lick_intervals",
    "compute_rolling_bias",
    "compute_side_bias",
    "lick_interval_results",
    "plot_lick_intervals",
    "plot_side_bias",
    "side_bias_result",
]
