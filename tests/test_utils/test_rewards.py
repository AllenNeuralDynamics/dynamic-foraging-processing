"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

from dynamic_foraging_processing.utils.rewards import get_annotated_rewards


def _outcome_payload(auto=None) -> dict:
    """Return a serialized ``TrialOutcome`` payload with the given auto-response."""
    return {
        "trial": {
            "p_reward_left": 1.0,
            "p_reward_right": 1.0,
            "response_deadline_duration": 3.0,
            "reward_consumption_duration": 1.0,
            "quiescence_period_duration": 0.5,
            "inter_trial_interval_duration": 4.0,
            "is_auto_reward_right": auto,
        },
        "is_right_choice": True,
        "is_rewarded": True,
    }


def _trial_outcome_df(trial_times: np.ndarray, autos=None) -> pd.DataFrame:
    """Build a trial outcome DataFrame indexed by ``trial_times``."""
    autos = autos if autos is not None else [None] * len(trial_times)
    return pd.DataFrame(
        {"data": [_outcome_payload(auto) for auto in autos]},
        index=pd.Index(trial_times, name="time"),
    )


def test_get_annotated_rewards_marks_default_trials_as_earned():
    """Trials with no auto-response setting and no manual water are ``earned``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4, 0.9]))

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, np.array([]))

    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_annotated_rewards_marks_auto_response_trials_as_automatic():
    """Trials with ``is_auto_reward_right`` set (either side) are ``automatic``."""
    reward_times = np.array([0.15, 0.42])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4]), autos=[True, False])

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, np.array([]))

    np.testing.assert_array_equal(annotations, np.array(["automatic", "automatic"]))


def test_get_annotated_rewards_marks_manual_water_as_manual():
    """Deliveries closest to a manual-water event are annotated as ``manual``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4, 0.9]))
    # Software event near the second delivery (0.42).
    manual_water_times = np.array([0.43])

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, manual_water_times)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual", "earned"]))


def test_get_annotated_rewards_manual_takes_precedence_over_automatic():
    """A manual delivery is ``manual`` even when the trial has auto-response set."""
    reward_times = np.array([0.15, 0.42])
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4]), autos=[None, True])
    manual_water_times = np.array([0.42])

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, manual_water_times)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_annotated_rewards_empty_deliveries_returns_empty():
    """No reward deliveries yields an empty annotation array."""
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))

    result = get_annotated_rewards(np.array([]), trial_outcome_df, np.array([]))

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_get_annotated_rewards_accepts_json_and_model_payloads():
    """``data`` payloads may be JSON strings or already-parsed ``TrialOutcome``."""
    reward_times = np.array([0.15, 0.42])
    payload = _outcome_payload(True)
    trial_outcome_df = pd.DataFrame(
        {"data": [json.dumps(payload), TrialOutcome.model_validate(payload)]},
        index=pd.Index([0.1, 0.4], name="time"),
    )

    annotations = get_annotated_rewards(reward_times, trial_outcome_df, np.array([]))

    np.testing.assert_array_equal(annotations, np.array(["automatic", "automatic"]))


def test_get_annotated_rewards_returns_ndarray():
    """The return value is a ``numpy.ndarray``."""
    reward_times = np.array([0.1])
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))

    result = get_annotated_rewards(reward_times, trial_outcome_df, np.array([]))

    assert isinstance(result, np.ndarray)
