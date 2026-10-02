"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
import pytest
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

from dynamic_foraging_processing.utils.rewards import (
    FREE_WATER_TOLERANCE,
    get_manual_go_cue_aligned_trials,
    get_reward_deliveries,
)

# Trials end at 10, 20, 30 s; each trial's go cue sits at +2 s into its window.
EDGES = np.array([10.0, 20.0, 30.0])
GO_CUES = np.array([2.0, 12.0, 22.0])
NO_REQUESTS = np.array([])


def _payload(auto=None, *, choice=True, rewarded=True, mechanism="autowater") -> dict:
    """Serialized ``TrialOutcome``: ``auto`` is ``is_auto_reward_right``, ``choice`` the side."""
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
    return {"trial": trial, "is_right_choice": choice, "is_rewarded": rewarded}


def _trials(*payloads, edges=EDGES) -> pd.DataFrame:
    """Trial outcome table: one row per payload, indexed by trial end time."""
    return pd.DataFrame(
        {"data": list(payloads)}, index=pd.Index(edges[: len(payloads)], name="time")
    )


def _label(times, trials, *, is_right=True, requests=NO_REQUESTS, go_cues=GO_CUES):
    """Run ``get_reward_deliveries`` on one port and return plain string labels."""
    labels = get_reward_deliveries(
        np.asarray(times, dtype=float), trials, go_cues, requests, is_right=is_right
    )
    return [str(label) for label in labels]


# --------------------------------------------------------------------------- #
# Rule 1 -- free water at the go cue is auto
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mechanism", ["autowater", "anti_bias", None])
def test_free_water_at_the_go_cue_is_auto(mechanism):
    """Free water is ``auto`` whatever mechanism the task used."""
    trials = _trials(_payload(True, mechanism=mechanism))
    assert _label([2.0], trials) == ["auto"]


def test_free_water_within_the_tolerance_is_auto():
    """Valve-to-go-cue latency within the tolerance still counts as the go cue."""
    trials = _trials(_payload(True))
    assert _label([2.0 + FREE_WATER_TOLERANCE / 2], trials) == ["auto"]


def test_water_beyond_the_tolerance_is_not_free_water():
    """A delivery too far from the go cue is not the task's free water."""
    trials = _trials(_payload(True))
    assert _label([2.0 + 2 * FREE_WATER_TOLERANCE], trials) == ["manual"]


def test_only_the_delivery_nearest_the_go_cue_is_auto():
    """A trial has one free-water delivery; any extra opening is manual."""
    trials = _trials(_payload(True))
    assert _label([2.0, 2.301], trials) == ["auto", "manual"]


def test_free_water_is_only_on_the_watered_port():
    """Water on the port the task did not water is not free water."""
    trials = _trials(_payload(True))
    assert _label([2.0], trials, is_right=False) == ["manual"]


def test_autowater_on_an_ignore_trial_is_auto():
    """Regression: autowater right after the previous trial's end stays in its own trial.

    On 872547_2026-09-11 trial 499, matching by the nearest ``Response`` charged
    this delivery to the previous trial and labelled it ``earned``.
    """
    trials = _trials(_payload(None, rewarded=False), _payload(True, choice=None, rewarded=False))
    assert _label([10.001], trials, go_cues=np.array([2.0, 10.001])) == ["auto"]


# --------------------------------------------------------------------------- #
# Rule 2 -- the first delivery on the chosen port after the go cue is earned
# --------------------------------------------------------------------------- #
def test_reward_on_the_chosen_port_is_earned():
    """A rewarded choice's delivery is ``earned``."""
    trials = _trials(_payload(None, choice=True))
    assert _label([2.4], trials) == ["earned"]


def test_a_fast_response_near_the_go_cue_is_still_earned():
    """Earned water can land within the free-water tolerance of the go cue.

    The trial's flags, not the timing, separate earned from free water, so a fast
    lick is not mistaken for autowater.
    """
    trials = _trials(_payload(None, choice=True))
    assert _label([2.004], trials) == ["earned"]


def test_only_the_first_delivery_on_the_chosen_port_is_earned():
    """A trial pays out once; later openings are manual."""
    trials = _trials(_payload(None, choice=True))
    assert _label([2.4, 3.0], trials) == ["earned", "manual"]


def test_water_before_the_go_cue_is_not_earned():
    """Water before the go cue cannot be the reward for this trial's choice."""
    trials = _trials(_payload(None, choice=True))
    assert _label([1.0, 2.4], trials) == ["manual", "earned"]


def test_a_rewarded_trial_with_water_only_before_the_go_cue_earns_nothing():
    """With no delivery after the go cue there is no earned water."""
    trials = _trials(_payload(None, choice=True))
    assert _label([1.0], trials) == ["manual"]


def test_water_on_the_other_port_is_not_earned():
    """Only the chosen port's water is earned."""
    trials = _trials(_payload(None, choice=True))
    assert _label([2.4], trials, is_right=False) == ["manual"]


def test_earned_follows_the_choice_not_is_rewarded():
    """A valve opening on the chosen port is earned whatever ``is_rewarded`` says.

    The valve opening is the hardware record that water was delivered, so the
    software flag does not override it.
    """
    trials = _trials(_payload(None, choice=True, rewarded=False))
    assert _label([2.4], trials) == ["earned"]


def test_ignored_trials_earn_nothing():
    """With no choice there is no chosen port, so no earned water."""
    trials = _trials(_payload(None, choice=None, rewarded=False))
    assert _label([2.4], trials) == ["manual"]


# --------------------------------------------------------------------------- #
# Rule 3 -- everything else is manual
# --------------------------------------------------------------------------- #
def test_water_on_a_trial_without_a_go_cue_is_manual():
    """A trial with no go cue in its window has no task delivery."""
    trials = _trials(_payload(True), _payload(None, choice=True))
    assert _label([12.0, 12.4], trials, go_cues=np.array([2.0])) == ["manual", "manual"]


def test_water_on_a_trial_with_no_outcome_payload_is_manual():
    """A missing payload leaves nothing to attribute the water to."""
    trials = _trials(None)
    assert _label([2.0], trials) == ["manual"]


# --------------------------------------------------------------------------- #
# Rule 4 -- a ManualAutoReward request claims the next auto delivery
# --------------------------------------------------------------------------- #
def test_a_request_claims_the_next_auto_delivery():
    """The operator's request turns the next free-water delivery into manual water."""
    trials = _trials(_payload(None, rewarded=False), _payload(True))
    assert _label([12.0], trials, requests=np.array([5.0])) == ["manual_go_cue_aligned"]


def test_a_request_claims_the_next_delivery_not_the_nearest():
    """The request is the button press, so free water just before it is not its water."""
    trials = _trials(_payload(True), _payload(True))
    labels = _label([2.0, 12.0], trials, requests=np.array([2.5]))
    assert labels == ["auto", "manual_go_cue_aligned"]


def test_repeated_requests_collapse_onto_one_delivery():
    """Several presses before the water lands produce one manual delivery."""
    trials = _trials(_payload(None, rewarded=False), _payload(True), _payload(True))
    labels = _label([12.0, 22.0], trials, requests=np.array([5.0, 5.2, 5.4]))
    assert labels == ["manual_go_cue_aligned", "auto"]


def test_a_request_with_no_later_free_water_changes_nothing():
    """A press with no free water after it, e.g. at the end of a session, is ignored."""
    trials = _trials(_payload(True))
    assert _label([2.0], trials, requests=np.array([5.0])) == ["auto"]


def test_a_request_never_claims_earned_or_manual_water():
    """Only free water can be manual go-cue-aligned water."""
    trials = _trials(_payload(None, choice=True))
    assert _label([2.4, 3.0], trials, requests=np.array([1.0])) == ["earned", "manual"]


# --------------------------------------------------------------------------- #
# Trial assignment and validation
# --------------------------------------------------------------------------- #
def test_a_delivery_on_a_trial_boundary_belongs_to_that_trial():
    """A delivery exactly on a trial's end is in that trial, not the next."""
    trials = _trials(_payload(None, choice=True), _payload(True))
    assert _label([10.0], trials, go_cues=np.array([9.0, 12.0])) == ["earned"]


def test_water_after_the_last_trial_is_charged_to_it():
    """Water past the final outcome is attributed to the last trial."""
    trials = _trials(_payload(None, choice=True))
    assert _label([99.0], trials) == ["earned"]


def test_payloads_may_be_json_or_parsed_models():
    """``data`` payloads may be JSON strings or already-parsed ``TrialOutcome``."""
    payload = _payload(True)
    trials = _trials(json.dumps(payload), TrialOutcome.model_validate(payload))
    assert _label([2.0, 12.0], trials) == ["auto", "auto"]


def test_no_deliveries_returns_an_empty_array():
    """No valve openings yield an empty annotation array."""
    labels = get_reward_deliveries(
        np.array([]), _trials(_payload(True)), GO_CUES, NO_REQUESTS, is_right=True
    )
    assert isinstance(labels, np.ndarray) and labels.size == 0


def test_returns_one_label_per_delivery():
    """The result is an object array aligned with the input deliveries."""
    times = np.array([1.0, 2.0, 2.4])
    labels = get_reward_deliveries(
        times, _trials(_payload(True)), GO_CUES, NO_REQUESTS, is_right=True
    )
    assert labels.shape == times.shape and labels.dtype == object


def test_deliveries_without_go_cues_raise():
    """Without go cues nothing can be classified, so fail loudly."""
    with pytest.raises(ValueError, match="go_cue_times is empty"):
        get_reward_deliveries(
            np.array([2.0]), _trials(_payload(True)), np.array([]), NO_REQUESTS, is_right=True
        )


@pytest.mark.parametrize(
    "edges, match",
    [
        (np.array([]), "is empty"),
        (np.array([10.0, np.nan]), "contains NaN"),
        (np.array([20.0, 10.0]), "must be sorted"),
    ],
)
def test_invalid_trial_windows_raise(edges, match):
    """An empty, NaN-bearing, or unsorted trial index cannot bound trials."""
    trials = _trials(*[_payload(True)] * edges.size, edges=edges)
    with pytest.raises(ValueError, match=match):
        get_reward_deliveries(np.array([2.0]), trials, GO_CUES, NO_REQUESTS, is_right=True)


# --------------------------------------------------------------------------- #
# get_manual_go_cue_aligned_trials
# --------------------------------------------------------------------------- #
def test_manual_go_cue_aligned_trials_are_the_claimed_deliveries_trials():
    """The trials returned are those whose delivery a request claimed."""
    trials = _trials(_payload(None, rewarded=False), _payload(True), _payload(True))
    flagged = get_manual_go_cue_aligned_trials(
        np.array([12.0, 22.0]), trials, GO_CUES, np.array([5.0]), is_right=True
    )
    assert flagged == {1}


@pytest.mark.parametrize(
    "times, requests",
    [
        (np.array([]), np.array([5.0])),
        (np.array([12.0]), np.array([])),
        (np.array([2.0]), np.array([5.0])),
    ],
)
def test_manual_go_cue_aligned_trials_empty_when_nothing_is_claimed(times, requests):
    """No deliveries, no requests, or a request with no later free water flags nothing."""
    trials = _trials(_payload(True), _payload(True))
    assert (
        get_manual_go_cue_aligned_trials(times, trials, GO_CUES, requests, is_right=True) == set()
    )
