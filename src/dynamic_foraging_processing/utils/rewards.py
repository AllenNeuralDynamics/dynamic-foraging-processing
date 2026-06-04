"""Reward-related processing helpers for dynamic foraging data."""

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial

from dynamic_foraging_processing.utils.timestamps import find_closest_timestamps


def get_annotated_rewards(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    sample_global_auto_water_state: pd.DataFrame,
) -> np.ndarray:
    """Annotate each reward delivery with information from the matching trial.

    Parameters
    ----------
    reward_delivery_times : numpy.ndarray
        Hardware timestamps of reward deliveries.
    trial_outcome_df : pandas.DataFrame
        Trial outcome table indexed by trial timestamp.
    sample_global_auto_water_state : pandas.DataFrame
        Auto-water state event table indexed by event timestamp.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` containing the
        annotation for each reward delivery.
    """
    reward_times = np.asarray(reward_delivery_times)
    trial_indices = find_closest_timestamps(
        reward_times,
        trial_outcome_df.index.to_numpy(),
    )
    auto_water_indices = find_closest_timestamps(
        reward_times,
        sample_global_auto_water_state.index.to_numpy(),
    )

    annotated_rewards = []
    for i, trial_index in enumerate(trial_indices):
        # TODO: use auto_water_indices[i] together with trial_outcome_df to
        # determine manual and automatic rewards.
        _ = auto_water_indices[i]
        trial_data_at_index = trial_outcome_df.iloc[trial_index]["data"]
        trial_model = Trial(**trial_data_at_index)

        if trial_model.is_auto_response_right is None:
            annotated_rewards.append("earned")

    return np.array(annotated_rewards)
