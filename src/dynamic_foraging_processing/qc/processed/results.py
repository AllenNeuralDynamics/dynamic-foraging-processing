"""Build the behavior QC results from per-trial and event-time arrays.

Combines the side-bias and lick-interval checks into the ordered list of
``QCResult`` objects the processed stage returns, optionally writing the
supporting plots so the result references resolve. Convert the results to
schema metrics with ``to_metrics`` / ``QCResult.to_metric`` when assembling a
``QualityControl``.
"""

import typing as t

import numpy as np

from dynamic_foraging_processing.qc._core.result import QCResult
from dynamic_foraging_processing.qc.processed.behavior import (
    lick_interval_results,
    side_bias_result,
)
from dynamic_foraging_processing.qc.processed.plots import plot_lick_intervals, plot_side_bias


def behavior_qc_results(
    animal_response: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    results_folder: t.Optional[str] = None,
    *,
    lickspout_x: t.Optional[np.ndarray] = None,
    lickspout_y1: t.Optional[np.ndarray] = None,
    lickspout_y2: t.Optional[np.ndarray] = None,
    lickspout_z: t.Optional[np.ndarray] = None,
    rewarded_left: t.Optional[np.ndarray] = None,
    rewarded_right: t.Optional[np.ndarray] = None,
    reward_probability_left: t.Optional[np.ndarray] = None,
    reward_probability_right: t.Optional[np.ndarray] = None,
    go_cue_times: t.Optional[np.ndarray] = None,
    autowater_left: t.Optional[np.ndarray] = None,
    autowater_right: t.Optional[np.ndarray] = None,
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
    animal_response : numpy.ndarray
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).
    left_lick_times, right_lick_times : numpy.ndarray
        Timestamps (s) of left/right-port licks.
    results_folder : str, optional
        Directory to write the plots into. If ``None``, plots are skipped.
    lickspout_x, lickspout_y1, lickspout_y2, lickspout_z : numpy.ndarray, optional
        Per-trial lickspout position arrays passed through to the side-bias figure.
    rewarded_left, rewarded_right, autowater_left, autowater_right, \
reward_probability_left, reward_probability_right : numpy.ndarray, optional
        Per-trial arrays passed through to the side-bias figure.
    go_cue_times, manual_left_times, manual_right_times : numpy.ndarray, optional
        Event timestamps (s) passed through to the side-bias figure.

    Returns
    -------
    list of QCResult
        The average-side-bias result followed by the four lick-interval results.
    """
    results = [
        side_bias_result(animal_response),
        *lick_interval_results(left_lick_times, right_lick_times),
    ]
    if results_folder is not None:
        plot_side_bias(
            animal_response,
            results_folder,
            lickspout_x=lickspout_x,
            lickspout_y1=lickspout_y1,
            lickspout_y2=lickspout_y2,
            lickspout_z=lickspout_z,
            rewarded_left=rewarded_left,
            rewarded_right=rewarded_right,
            reward_probability_left=reward_probability_left,
            reward_probability_right=reward_probability_right,
            go_cue_times=go_cue_times,
            autowater_left=autowater_left,
            autowater_right=autowater_right,
            manual_left_times=manual_left_times,
            manual_right_times=manual_right_times,
        )
        plot_lick_intervals(left_lick_times, right_lick_times, results_folder)
    return results
