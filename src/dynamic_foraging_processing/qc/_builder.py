"""Assemble the dynamic foraging ``QualityControl`` object.

Combines the behavior metrics (computed from primitive arrays) with the
contraqctor-based contract QA metrics into a single ``QualityControl``, wiring
up ``default_grouping`` so the QC portal lays out ``behavior`` and
``test_suite`` as sibling top-level groups.
"""

import typing as t

import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import QCMetric, QualityControl

from dynamic_foraging_processing.qc._behavior import lick_interval_results, side_bias_result
from dynamic_foraging_processing.qc._plots import plot_lick_intervals, plot_side_bias
from dynamic_foraging_processing.qc._result import QCResult

#: Tag keys laid out as siblings at the top level of the QC portal.
DEFAULT_GROUPING = ["behavior", "test_suite"]


def behavior_qc_results(
    animal_response: np.ndarray,
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    results_folder: t.Optional[str] = None,
    *,
    stage_positions: t.Optional[pd.DataFrame] = None,
    rewarded_history: t.Optional[pd.DataFrame] = None,
    reward_probability_left: t.Optional[np.ndarray] = None,
    reward_probability_right: t.Optional[np.ndarray] = None,
    go_cue_times: t.Optional[np.ndarray] = None,
    auto_water: t.Optional[pd.DataFrame] = None,
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
    stage_positions, rewarded_history, auto_water : pandas.DataFrame, optional
        Optional per-trial data passed through to the side-bias figure.
    reward_probability_left, reward_probability_right, go_cue_times, \
manual_left_times, manual_right_times : numpy.ndarray, optional
        Optional per-trial / event data passed through to the side-bias figure.

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
            stage_positions=stage_positions,
            rewarded_history=rewarded_history,
            reward_probability_left=reward_probability_left,
            reward_probability_right=reward_probability_right,
            go_cue_times=go_cue_times,
            auto_water=auto_water,
            manual_left_times=manual_left_times,
            manual_right_times=manual_right_times,
        )
        plot_lick_intervals(left_lick_times, right_lick_times, results_folder)
    return results


def build_quality_control(
    metrics: t.List[QCMetric],
    *,
    default_grouping: t.Optional[t.List[str]] = None,
    allow_tag_failures: t.Optional[t.List[str]] = None,
    key_experimenters: t.Optional[t.List[str]] = None,
    notes: t.Optional[str] = None,
) -> QualityControl:
    """Wrap a flat list of metrics into a ``QualityControl`` object.

    Parameters
    ----------
    metrics : list of QCMetric
        All metrics (behavior + contract QA).
    default_grouping : list of str, optional
        Tag keys the portal groups by. Defaults to ``["behavior", "test_suite"]``.
    allow_tag_failures : list of str, optional
        Tag values whose metric failures should not fail the overall QC.
    key_experimenters : list of str, optional
        Experimenters associated with the session.
    notes : str, optional
        Free-text notes.

    Returns
    -------
    QualityControl
        The assembled quality-control object.
    """
    return QualityControl(
        metrics=metrics,
        default_grouping=default_grouping if default_grouping is not None else DEFAULT_GROUPING,
        allow_tag_failures=allow_tag_failures if allow_tag_failures is not None else [],
        key_experimenters=key_experimenters,
        notes=notes,
    )
