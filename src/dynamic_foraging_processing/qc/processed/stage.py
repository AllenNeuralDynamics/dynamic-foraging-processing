"""Processed-data QC stage.

``ProcessedQC`` wraps the behavior metrics computed from the trials table
(side bias, lick intervals) into the ``BaseQC`` interface. It is a thin wrapper
over :func:`behavior_qc_results`; the metric computation lives there.
"""

import typing as t

import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import QCMetric

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc._core.result import to_metrics
from dynamic_foraging_processing.qc.processed.results import behavior_qc_results


class ProcessedQC(BaseQC):
    """Behavior QC over the trials table and lick-time arrays."""

    def run(
        self,
        trials: pd.DataFrame,
        left_lick_times: np.ndarray,
        right_lick_times: np.ndarray,
        results_folder: t.Optional[str] = None,
        *,
        manual_left_times: t.Optional[np.ndarray] = None,
        manual_right_times: t.Optional[np.ndarray] = None,
    ) -> t.List[QCMetric]:
        """Compute the behavior QC checks and return them as metrics.

        Parameters mirror :func:`behavior_qc_results`. When ``results_folder``
        is provided, the supporting plots are written there so the metric
        references resolve.

        Parameters
        ----------
        trials : pandas.DataFrame
            Trials table; the per-trial inputs (side bias, animal response,
            lickspout positions, reward / autowater columns, go-cue times) are
            read from it by column.
        left_lick_times, right_lick_times : numpy.ndarray
            Timestamps (s) of left/right-port licks.
        results_folder : str, optional
            Directory to write the plots into. If ``None``, plots are skipped.
        manual_left_times, manual_right_times : numpy.ndarray, optional
            Manual-water delivery timestamps passed through to the side-bias figure.

        Returns
        -------
        list of QCMetric
            The side-bias metric followed by the four lick-interval metrics.
        """
        results = behavior_qc_results(
            trials,
            left_lick_times,
            right_lick_times,
            results_folder,
            manual_left_times=manual_left_times,
            manual_right_times=manual_right_times,
        )
        return to_metrics(results)
