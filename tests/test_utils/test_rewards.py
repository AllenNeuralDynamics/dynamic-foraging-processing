"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
import pytest
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
    """Build a trial outcome DataFrame with one row per entry of ``trial_times``."""
    autos = autos if autos is not None else [None] * len(trial_times)
    return pd.DataFrame(
        {"data": [_outcome_payload(auto) for auto in autos]},
        index=pd.Index(trial_times, name="time"),
    )


def test_get_annotated_rewards_marks_default_trials_as_earned():
    """Trials with no auto-response setting and no manual water are ``earned``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(response_times)

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, np.array([])
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_annotated_rewards_marks_auto_response_trials_as_auto():
    """Trials with ``is_auto_reward_right`` set (either side) are ``auto``."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(response_times, autos=[True, False])

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, np.array([])
    )

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_annotated_rewards_matches_response_times_not_outcome_times():
    """Deliveries are matched to trials by ``Response`` time, not outcome time."""
    # The TrialOutcome timestamps here trail the deliveries; only the Response
    # times sit next to them, so matching on the outcome index would pick the
    # wrong trial and annotate the second delivery as earned.
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(np.array([0.4, 5.0]), autos=[None, True])

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, np.array([])
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "auto"]))


def test_get_annotated_rewards_marks_manual_water_as_manual():
    """Deliveries closest to a manual-water event are annotated as ``manual``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(response_times)
    # Software event near the second delivery (0.42).
    manual_water_times = np.array([0.43])

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, manual_water_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual", "earned"]))


def test_get_annotated_rewards_manual_takes_precedence_over_auto():
    """A manual delivery is ``manual`` even when the trial has auto-response set."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(response_times, autos=[None, True])
    manual_water_times = np.array([0.42])

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, manual_water_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_annotated_rewards_empty_deliveries_returns_empty():
    """No reward deliveries yields an empty annotation array."""
    response_times = np.array([0.0])
    trial_outcome_df = _trial_outcome_df(response_times)

    result = get_annotated_rewards(np.array([]), trial_outcome_df, response_times, np.array([]))

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_get_annotated_rewards_misaligned_responses_raises():
    """A ``Response`` count that disagrees with the trial count is an error."""
    trial_outcome_df = _trial_outcome_df(np.array([0.1, 0.4]))

    with pytest.raises(ValueError, match="misaligned"):
        get_annotated_rewards(np.array([0.15]), trial_outcome_df, np.array([0.1]), np.array([]))


def test_get_annotated_rewards_accepts_json_and_model_payloads():
    """``data`` payloads may be JSON strings or already-parsed ``TrialOutcome``."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    payload = _outcome_payload(True)
    trial_outcome_df = pd.DataFrame(
        {"data": [json.dumps(payload), TrialOutcome.model_validate(payload)]},
        index=pd.Index(response_times, name="time"),
    )

    annotations = get_annotated_rewards(
        reward_times, trial_outcome_df, response_times, np.array([])
    )

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_annotated_rewards_returns_ndarray():
    """The return value is a ``numpy.ndarray``."""
    reward_times = np.array([0.1])
    response_times = np.array([0.0])
    trial_outcome_df = _trial_outcome_df(response_times)

    result = get_annotated_rewards(reward_times, trial_outcome_df, response_times, np.array([]))

    assert isinstance(result, np.ndarray)
