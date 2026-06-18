"""Processed-data QC stage.

``ProcessedQC`` wraps the behavior metrics computed from per-trial arrays
(side bias, lick intervals) into the ``BaseQC`` interface. It is a thin wrapper
over :func:`behavior_qc_results`; the metric computation lives there.
"""

import typing as t

import numpy as np
from aind_data_schema.core.quality_control import QCMetric

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc._core.result import to_metrics
from dynamic_foraging_processing.qc.processed.results import behavior_qc_results


class ProcessedQC(BaseQC):
    """Behavior QC over the processed per-trial / event arrays."""

    def run(
        self,
        animal_response: np.ndarray,
        side_bias: np.ndarray,
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
    ) -> t.List[QCMetric]:
        """Compute the behavior QC checks and return them as metrics.

        Parameters mirror :func:`behavior_qc_results`. When ``results_folder``
        is provided, the supporting plots are written there so the metric
        references resolve.

        Parameters
        ----------
        animal_response : numpy.ndarray
            Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).
        side_bias : numpy.ndarray
            Per-trial side bias from the trial table (right minus left).
        left_lick_times, right_lick_times : numpy.ndarray
            Timestamps (s) of left/right-port licks.
        results_folder : str, optional
            Directory to write the plots into. If ``None``, plots are skipped.
        lickspout_x, lickspout_y1, lickspout_y2, lickspout_z, rewarded_left, \
rewarded_right, autowater_left, autowater_right, reward_probability_left, \
reward_probability_right : numpy.ndarray, optional
            Optional per-trial arrays passed through to the side-bias figure.
        go_cue_times, manual_left_times, manual_right_times : numpy.ndarray, optional
            Optional event timestamps passed through to the side-bias figure.

        Returns
        -------
        list of QCMetric
            The side-bias metric followed by the four lick-interval metrics.
        """
        results = behavior_qc_results(
            animal_response,
            side_bias,
            left_lick_times,
            right_lick_times,
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
        return to_metrics(results)
