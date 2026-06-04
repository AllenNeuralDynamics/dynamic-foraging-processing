"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import numpy as np
import pandas as pd

from dynamic_foraging_processing.utils.rewards import get_annotated_rewards


def _trial_payload() -> dict:
    """Return a minimal ``Trial`` payload using model defaults."""
    return {
        "p_reward_left": 1.0,
        "p_reward_right": 1.0,
        "reward_consumption_duration": 5.0,
        "reward_delay_duration": 0.0,
        "secondary_reinforcer": None,
        "response_deadline_duration": 5.0,
        "enable_fast_retract": False,
        "quiescence_period_duration": 0.5,
        "inter_trial_interval_duration": 5.0,
        "is_auto_response_right": None,
        "lickspout_offset_delta": 0.0,
        "extra_metadata": None,
    }


def _trial_outcome_df(trial_times: np.ndarray) -> pd.DataFrame:
    """Build a trial outcome DataFrame indexed by ``trial_times``."""
    return pd.DataFrame(
        {"data": [_trial_payload() for _ in trial_times]},
        index=pd.Index(trial_times, name="time"),
    )


def _auto_water_state_df(state_times: np.ndarray) -> pd.DataFrame:
    """Build an auto-water state DataFrame indexed by ``state_times``."""
    return pd.DataFrame(
        {"state": [True for _ in state_times]},
        index=pd.Index(state_times, name="time"),
    )


def test_get_annotated_rewards_marks_default_trials_as_earned():
    """Trials with no auto-response setting are annotated as ``earned``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4, 0.9]))
    auto_water_state_df = _auto_water_state_df(np.array([0.0, 1.0]))

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, auto_water_state_df)

    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_annotated_rewards_skips_trials_with_auto_response_set():
    """Trials with ``is_auto_response_right`` set are not annotated as ``earned``."""
    reward_times = np.array([0.15, 0.42])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4]))
    trial_outcome_df.iloc[0]["data"]["is_auto_response_right"] = True
    auto_water_state_df = _auto_water_state_df(np.array([0.0, 1.0]))

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, auto_water_state_df)

    np.testing.assert_array_equal(annotations, np.array(["earned"]))


def test_get_annotated_rewards_returns_ndarray():
    """The return value is a ``numpy.ndarray``."""
    reward_times = np.array([0.1])
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))
    auto_water_state_df = _auto_water_state_df(np.array([0.0]))

    result = get_annotated_rewards(reward_times, trial_outcome_df, auto_water_state_df)

    assert isinstance(result, np.ndarray)
