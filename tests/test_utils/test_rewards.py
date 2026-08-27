"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
import pytest
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

from dynamic_foraging_processing.utils.rewards import get_reward_deliveries


def _outcome_payload(auto=None, is_rewarded: bool = True, mechanism: str = "autowater") -> dict:
    """Return a serialized ``TrialOutcome`` payload with the given auto-response.

    ``mechanism`` names which free-water flag ``metadata.extra`` carries when
    ``auto`` is set: ``"autowater"`` for scheduled autowater, ``"anti_bias"`` for
    an anti-bias intervention, or ``None`` for neither. The annotation labels all
    free water ``auto`` regardless, so this only matters to the trials table.
    """
    trial = {
        "p_reward_left": 1.0,
        "p_reward_right": 1.0,
        "response_deadline_duration": 3.0,
        "reward_consumption_duration": 1.0,
        "quiescence_period_duration": 0.5,
        "inter_trial_interval_duration": 4.0,
        "is_auto_reward_right": auto,
    }
    if mechanism is not None:
        trial["metadata"] = {
            "extra": {
                "is_autowater": mechanism == "autowater",
                "is_bias_water_intervention": mechanism == "anti_bias",
            }
        }
    return {
        "trial": trial,
        "is_right_choice": True,
        "is_rewarded": is_rewarded,
    }


def _trial_outcome_df(
    trial_times: np.ndarray, autos=None, rewarded=None, mechanism: str = "autowater"
) -> pd.DataFrame:
    """Build a trial outcome DataFrame with one row per entry of ``trial_times``."""
    autos = autos if autos is not None else [None] * len(trial_times)
    rewarded = rewarded if rewarded is not None else [True] * len(trial_times)
    return pd.DataFrame(
        {"data": [_outcome_payload(a, r, mechanism) for a, r in zip(autos, rewarded)]},
        index=pd.Index(trial_times, name="time"),
    )


def test_get_reward_deliveries_marks_default_trials_as_earned():
    """Trials with no auto-response setting and no manual water are ``earned``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4, 1.9]))

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_reward_deliveries_marks_auto_response_trials_as_auto():
    """Trials with ``is_auto_reward_right`` set (either side) are ``auto``."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]), autos=[True, False])

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


@pytest.mark.parametrize("mechanism", ["autowater", "anti_bias", None])
def test_get_reward_deliveries_marks_all_free_water_as_auto(mechanism):
    """Every free-water delivery is ``auto``, whatever mechanism gave it.

    ``is_auto_reward_right`` is the delivery channel, shared by scheduled
    autowater and the anti-bias intervention, and the series does not split them:
    ``auto_waterL``/``auto_waterR`` and
    ``anti_bias_left_water``/``anti_bias_right_water`` record the mechanism per
    trial instead.
    """
    reward_times = np.array([0.15])
    response_times = np.array([0.1])
    trial_outcome_df = pd.DataFrame(
        {"data": [_outcome_payload(True, mechanism=mechanism)]},
        index=pd.Index([1.1], name="time"),
    )

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


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

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_keeps_deliveries_on_unrewarded_trials():
    """A delivery on a trial reporting ``is_rewarded=False`` is still annotated.

    Free water fires at the go cue and the trial then continues normally, so
    ``is_rewarded`` describes the animal's own choice rather than the water. The
    series records every valve opening, so nothing is filtered out.
    """
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(
        np.array([1.1, 1.4, 1.9]),
        autos=[None, True, True],
        rewarded=[True, False, True],
    )

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "auto", "auto"]))


def test_get_reward_deliveries_labels_both_sides_of_a_split_trial():
    """One trial can water the task's side and the animal's side independently.

    When free water goes to one port and the animal earns reward at the other,
    the trial contributes an ``auto`` delivery and an ``earned`` delivery. The
    label follows the matched trial, so both deliveries on that trial read
    ``auto`` from this port's perspective; the sides are separate series.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.1]), autos=[False], rewarded=[True])
    response_times = np.array([0.1])

    # This port saw one opening on that trial; the trial gave free water.
    annotations = get_reward_deliveries(
        np.array([0.15]), trial_outcome_df, np.array([]), response_times
    )
    np.testing.assert_array_equal(annotations, np.array(["auto"]))

    # A trial with no free water at all yields ``earned`` on whichever port opened.
    earned_only = _trial_outcome_df(np.array([1.1]), autos=[None], rewarded=[True])
    annotations = get_reward_deliveries(np.array([0.15]), earned_only, np.array([]), response_times)
    np.testing.assert_array_equal(annotations, np.array(["earned"]))


def test_get_reward_deliveries_marks_manual_water_as_manual():
    """Deliveries closest to a manual-water event are annotated as ``manual``."""
    reward_times = np.array([0.15, 0.42, 0.95])
    response_times = np.array([0.1, 0.4, 0.9])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4, 1.9]))
    # Software event near the second delivery (0.42).
    manual_water_times = np.array([0.43])

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, manual_water_times, response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual", "earned"]))


def test_get_reward_deliveries_manual_takes_precedence_over_auto():
    """A manual delivery is ``manual`` even when the trial has auto-response set."""
    reward_times = np.array([0.15, 0.42])
    response_times = np.array([0.1, 0.4])
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]), autos=[None, True])
    manual_water_times = np.array([0.42])

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, manual_water_times, response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_reward_deliveries_empty_deliveries_returns_empty():
    """No reward deliveries yields an empty annotation array."""
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))

    annotations = get_reward_deliveries(
        np.array([]), trial_outcome_df, np.array([]), np.array([0.0])
    )

    assert isinstance(annotations, np.ndarray)
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

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), response_times
    )

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_rejects_misaligned_response_times():
    """``response_times`` must have one entry per trial; they pair by position."""
    trial_outcome_df = _trial_outcome_df(np.array([1.1, 1.4]))

    with pytest.raises(ValueError, match="paired by position"):
        get_reward_deliveries(np.array([0.15]), trial_outcome_df, np.array([]), np.array([0.1]))


def test_get_reward_deliveries_returns_one_annotation_per_delivery():
    """The result is a :class:`numpy.ndarray` aligned with the input deliveries."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 1.5]))
    reward_times = np.array([0.1, 0.2, 0.6])

    annotations = get_reward_deliveries(
        reward_times, trial_outcome_df, np.array([]), np.array([0.0, 0.5])
    )

    assert isinstance(annotations, np.ndarray)
    assert annotations.shape == reward_times.shape
