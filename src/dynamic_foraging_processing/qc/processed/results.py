"""Build the behavior QC results from the trials table and lick-time arrays.

Combines the side-bias and lick-interval checks into the ordered list of
``QCResult`` objects the processed stage returns, optionally writing the
supporting plots so the result references resolve. The per-trial inputs are
pulled from the trials table by column; the lick and manual-water timestamps
are event-time arrays (variable length, not per-trial) and stay explicit.
Convert the results to schema metrics with ``to_metrics`` /
``QCResult.to_metric`` when assembling a ``QualityControl``.
"""

import typing as t

import numpy as np
import pandas as pd

from dynamic_foraging_processing.qc._core.result import QCResult
from dynamic_foraging_processing.qc.processed.behavior import (
    lick_interval_results,
    side_bias_result,
)
from dynamic_foraging_processing.qc.processed.plots import plot_lick_intervals, plot_side_bias

# Logical input -> trials-table column name. Centralized so the mapping is easy
# to correct against the trial-table builder; ``side_bias`` and the
# ``lickspout_*`` arrays are not yet pinned down in trials_table_mapping.md.
_COLUMNS = {
    "animal_response": "animal_response",
    "side_bias": "side_bias",
    "lickspout_x": "lickspout_x",
    "lickspout_y1": "lickspout_y1",
    "lickspout_y2": "lickspout_y2",
    "lickspout_z": "lickspout_z",
    "rewarded_left": "rewarded_historyL",
    "rewarded_right": "rewarded_historyR",
    "reward_probability_left": "reward_probabilityL",
    "reward_probability_right": "reward_probabilityR",
    "autowater_left": "auto_waterL",
    "autowater_right": "auto_waterR",
    "go_cue_times": "goCue_start_time",
}


def _column(trials: pd.DataFrame, key: str) -> t.Optional[np.ndarray]:
    """Return the trials-table column for ``key`` as an array, or ``None``.

    Missing columns return ``None`` so optional plot inputs degrade gracefully,
    matching the previous per-array signature.
    """
    column = _COLUMNS[key]
    if column in trials.columns:
        return trials[column].to_numpy()
    return None


def behavior_qc_results(
    trials: pd.DataFrame,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    results_folder: t.Optional[str] = None,
    *,
    manual_left_times: t.Optional[np.ndarray] = None,
    manual_right_times: t.Optional[np.ndarray] = None,
) -> t.List[QCResult]:
    """Build the behavior QC results (side bias + lick intervals).

    When ``results_folder`` is provided, the supporting ``side_bias.png`` and
    ``lick_intervals.png`` plots are written there so the result references
    resolve. Convert the returned results to schema metrics with
    ``to_metrics`` / ``QCResult.to_metric`` when assembling a ``QualityControl``.

    Parameters
    ----------
    trials : pandas.DataFrame
        Trials table. The per-trial inputs are read by column (see
        ``_COLUMNS``): ``animal_response``, ``side_bias``, the ``lickspout_*``
        positions, the earned-reward / autowater / reward-probability left/right
        columns, and the go-cue start times. Columns that are absent are treated
        as unavailable and skipped in the side-bias figure.
    left_lick_times, right_lick_times : numpy.ndarray
        Timestamps (s) of left/right-port licks. Event-time arrays, not
        per-trial, so they are passed explicitly rather than read from ``trials``.
    results_folder : str, optional
        Directory to write the plots into. If ``None``, plots are skipped.
    manual_left_times, manual_right_times : numpy.ndarray, optional
        Manual-water delivery timestamps (s); event-time arrays passed through
        to the side-bias figure.

    Returns
    -------
    list of QCResult
        The average-side-bias result followed by the four lick-interval results.
    """
    side_bias = _column(trials, "side_bias")
    results = [
        side_bias_result(side_bias),
        *lick_interval_results(left_lick_times, right_lick_times),
    ]
    if results_folder is not None:
        plot_side_bias(
            _column(trials, "animal_response"),
            side_bias,
            results_folder,
            lickspout_x=_column(trials, "lickspout_x"),
            lickspout_y1=_column(trials, "lickspout_y1"),
            lickspout_y2=_column(trials, "lickspout_y2"),
            lickspout_z=_column(trials, "lickspout_z"),
            rewarded_left=_column(trials, "rewarded_left"),
            rewarded_right=_column(trials, "rewarded_right"),
            reward_probability_left=_column(trials, "reward_probability_left"),
            reward_probability_right=_column(trials, "reward_probability_right"),
            go_cue_times=_column(trials, "go_cue_times"),
            autowater_left=_column(trials, "autowater_left"),
            autowater_right=_column(trials, "autowater_right"),
            manual_left_times=manual_left_times,
            manual_right_times=manual_right_times,
        )
        plot_lick_intervals(left_lick_times, right_lick_times, results_folder)
    return results
