"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
import pytest
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

from dynamic_foraging_processing.utils.rewards import get_reward_deliveries


def _outcome_payload(auto=None, is_rewarded: bool = True) -> dict:
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
        "is_rewarded": is_rewarded,
    }


def _trial_outcome_df(trial_times: np.ndarray, autos=None, rewarded=None) -> pd.DataFrame:
    """Build a trial outcome DataFrame with one row per entry of ``trial_times``."""
    autos = autos if autos is not None else [None] * len(trial_times)
    rewarded = rewarded if rewarded is not None else [True] * len(trial_times)
    return pd.DataFrame(
        {"data": [_outcome_payload(a, r) for a, r in zip(autos, rewarded)]},
        index=pd.Index(trial_times, name="time"),
    )


def test_get_reward_deliveries_marks_default_trials_as_earned():
    """Trials with no auto-response setting and no manual water are ``earned``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4, 1.9]))

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_reward_deliveries_marks_auto_response_trials_as_auto():
    """Trials with ``is_auto_reward_right`` set (either side) are ``auto``."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]), autos=[True, False])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_matches_closest_response_time():
    """Each delivery takes the annotation of the trial whose response is closest.

    The trial-outcome timestamps deliberately disagree with the response times:
    matching on the outcome would pick the first (earned) trial, so this pins the
    match to the ``Response`` stream.
    """
    reward_times = np.array([0.95, 1.05])
    response_times = np.array([0.1, 1.0])
    # Outcome events fire at the end of each trial, far from the deliveries.
    trial_outcome_df = _trial_outcome_df(np.array([0.9, 5.0]), autos=[None, True])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_drops_uncollected_auto_water():
    """Autowater on a trial reporting ``is_rewarded=False`` is dropped, not annotated.

    The water is delivered before the response, so a trial the animal answered
    the other way leaves it uncollected; it is not reward the animal received.
    """
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(
        np.array([1.1, 1.4, 1.9]),
        autos=[None, True, True],
        rewarded=[True, False, True],
    )

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, np.array([0.15, 0.95]))
    np.testing.assert_array_equal(annotations, np.array(["earned", "auto"]))


def test_get_reward_deliveries_keeps_unrewarded_non_auto_deliveries():
    """A delivery on an unrewarded trial that is not autowater is kept."""
    reward_times = np.array([0.15])
    response_times = np.array([0.1])
    trial_outcome_df = _trial_outcome_df(np.array([1.1]), autos=[None], rewarded=[False])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["earned"]))


def test_get_reward_deliveries_marks_manual_water_as_manual():
    """Deliveries closest to a manual-water event are annotated as ``manual``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4, 1.9]))
    # Software event near the second delivery (0.42).
    manual_water_times = np.array([0.43])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, manual_water_times, response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["earned", "manual", "earned"]))


def test_get_reward_deliveries_manual_takes_precedence_over_auto():
    """A manual delivery is ``manual`` even when the trial has auto-response set."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]), autos=[None, True])
    manual_water_times = np.array([0.42])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, manual_water_times, response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_reward_deliveries_manual_water_survives_the_auto_drop():
    """Manual water on an unrewarded auto trial is kept, not dropped."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(
        np.array([1.1, 1.4]), autos=[None, True], rewarded=[True, False]
    )
    manual_water_times = np.array([0.42])

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, manual_water_times, response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_reward_deliveries_empty_deliveries_returns_empty():
    """No reward deliveries yields empty timestamp and annotation arrays."""
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))

    times, annotations = get_reward_deliveries(
        np.array([]), trial_outcome_df, np.array([]), np.array([0.0])
    )

    assert isinstance(annotations, np.ndarray)
    assert times.size == 0
    assert annotations.size == 0


def test_get_reward_deliveries_accepts_json_and_model_payloads():
    """``data`` payloads may be JSON strings or already-parsed ``TrialOutcome``."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    payload = _outcome_payload(True)
    trial_outcome_df = pd.DataFrame(
        {"data": [json.dumps(payload), TrialOutcome.model_validate(payload)]},
        index=pd.Index([1.1, 1.4], name="time"),
    )

    times, annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(times, reward_times)
    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_rejects_misaligned_response_times():
    """``response_times`` must have one entry per trial; they pair by position."""
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]))

    with pytest.raises(ValueError, match="paired by position"):
        get_reward_deliveries(np.array([0.15]), trial_outcome_df, np.array([]), np.array([0.1]))


def test_get_reward_deliveries_returns_ndarray():
    """Both return values are :class:`numpy.ndarray`."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0]))

    times, annotations = get_reward_deliveries(
        np.array([0.1]), trial_outcome_df, np.array([]), np.array([0.0])
    )

    assert isinstance(times, np.ndarray)
    assert isinstance(annotations, np.ndarray)
