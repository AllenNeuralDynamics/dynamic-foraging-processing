"""Tests for ``dynamic_foraging_processing.utils.rewards``."""

import json

import numpy as np
import pandas as pd
import pytest
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

from dynamic_foraging_processing.utils.rewards import (
    ManualWaterTimes,
    get_manual_go_cue_aligned_trials,
    get_reward_deliveries,
)


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
    """Build a trial outcome DataFrame with one row per entry of ``trial_times``.

    ``trial_times`` are the ``TrialOutcome`` event timestamps, which fire at the
    *end* of each trial and so bound it: trial ``i`` spans
    ``(trial_times[i - 1], trial_times[i]]``.
    """
    autos = autos if autos is not None else [None] * len(trial_times)
    rewarded = rewarded if rewarded is not None else [True] * len(trial_times)
    return pd.DataFrame(
        {"data": [_outcome_payload(a, r, mechanism) for a, r in zip(autos, rewarded)]},
        index=pd.Index(trial_times, name="time"),
    )


def test_get_reward_deliveries_marks_default_trials_as_earned():
    """Trials with no auto-response setting and no manual water are ``earned``."""
    reward_times = np.array([0.15, 1.42, 2.95])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0, 3.0]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["earned", "earned", "earned"]))


def test_get_reward_deliveries_marks_auto_response_trials_as_auto():
    """Trials with ``is_auto_reward_right`` set (either side) are ``auto``."""
    reward_times = np.array([0.15, 1.42])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]), autos=[True, False])

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, ManualWaterTimes())

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
    trial_outcome_df = pd.DataFrame(
        {"data": [_outcome_payload(True, mechanism=mechanism)]},
        index=pd.Index([1.0], name="time"),
    )

    annotations = get_reward_deliveries(np.array([0.15]), trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


def test_get_reward_deliveries_matches_the_containing_trial_not_the_nearest():
    """A delivery takes the trial whose window contains it, however near others are.

    The first delivery sits just past trial 0's outcome, so it belongs to trial
    1 even though it is far closer to trial 0's boundary than to trial 1's.
    Proximity matching would charge it to trial 0 and read the wrong
    ``is_auto_reward_right``.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 9.0]), autos=[None, True])

    annotations = get_reward_deliveries(np.array([1.05]), trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


def test_get_reward_deliveries_labels_autowater_on_an_ignore_trial_as_auto():
    """Autowater at the go cue of an ignore trial is ``auto``, not ``earned``.

    Regression for the mislabel seen on 872547_2026-09-11_13-05-03 trial 499. On
    an ignore trial the ``Response`` event waits out the full response deadline
    while autowater lands at the go cue, so the *previous* trial's response is
    the nearer one. Matching by proximity therefore read the previous trial's
    ``is_auto_reward_right`` (unset) and annotated task-given free water as water
    the animal worked for. Containment is immune: the delivery falls inside its
    own trial's window whatever the response latency.
    """
    # Trial 0 has no free water and ends at t=10; trial 1 is an autowater trial
    # whose water lands immediately after that boundary, at its own go cue.
    trial_outcome_df = _trial_outcome_df(np.array([10.0, 20.0]), autos=[None, True])
    autowater_at_go_cue = np.array([10.001])

    annotations = get_reward_deliveries(autowater_at_go_cue, trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


def test_get_reward_deliveries_assigns_a_delivery_on_the_boundary_to_that_trial():
    """A delivery landing exactly on a trial's outcome belongs to that trial."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]), autos=[True, None])

    annotations = get_reward_deliveries(np.array([1.0]), trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


def test_get_reward_deliveries_charges_water_after_the_last_trial_to_that_trial():
    """Water delivered past the final outcome has no trial of its own.

    End-of-session experimenter water can land after the last ``TrialOutcome``.
    It is charged to the last trial rather than indexing off the end; in practice
    the manual labels overwrite it.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]), autos=[None, True])

    annotations = get_reward_deliveries(np.array([99.0]), trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto"]))


def test_get_reward_deliveries_keeps_deliveries_on_unrewarded_trials():
    """A delivery on a trial reporting ``is_rewarded=False`` is still annotated.

    Free water fires at the go cue and the trial then continues normally, so
    ``is_rewarded`` describes the animal's own choice rather than the water. The
    series records every valve opening, so nothing is filtered out.
    """
    reward_times = np.array([0.15, 1.42, 2.95])
    trial_outcome_df = _trial_outcome_df(
        np.array([1.0, 2.0, 3.0]),
        autos=[None, True, True],
        rewarded=[True, False, True],
    )

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["earned", "auto", "auto"]))


def test_get_reward_deliveries_labels_both_sides_of_a_split_trial():
    """One trial can water the task's side and the animal's side independently.

    When free water goes to one port and the animal earns reward at the other,
    the trial contributes an ``auto`` delivery and an ``earned`` delivery. The
    label follows the containing trial, so both deliveries on that trial read
    ``auto`` from this port's perspective; the sides are separate series.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.0]), autos=[False], rewarded=[True])

    # This port saw one opening on that trial; the trial gave free water.
    annotations = get_reward_deliveries(np.array([0.15]), trial_outcome_df, ManualWaterTimes())
    np.testing.assert_array_equal(annotations, np.array(["auto"]))

    # A trial with no free water at all yields ``earned`` on whichever port opened.
    earned_only = _trial_outcome_df(np.array([1.0]), autos=[None], rewarded=[True])
    annotations = get_reward_deliveries(np.array([0.15]), earned_only, ManualWaterTimes())
    np.testing.assert_array_equal(annotations, np.array(["earned"]))


def test_get_reward_deliveries_marks_manual_water_as_manual():
    """Deliveries closest to an unaligned manual-water event are ``manual``."""
    reward_times = np.array([0.15, 1.42, 2.95])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0, 3.0]))
    # Software event near the second delivery (1.42).
    manual_water = ManualWaterTimes(unaligned=np.array([1.43]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual", "earned"]))


def test_get_reward_deliveries_marks_manual_auto_reward_as_go_cue_aligned():
    """Deliveries closest to a manual auto-reward event get their own label.

    The experimenter triggered this water at the go cue, so the task logic never
    set ``is_auto_reward_right`` and the trial-derived label would read
    ``earned`` -- water the animal never worked for.
    """
    reward_times = np.array([0.15, 1.42, 2.95])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0, 3.0]))
    manual_water = ManualWaterTimes(go_cue_aligned=np.array([1.43]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(
        annotations, np.array(["earned", "manual_go_cue_aligned", "earned"])
    )


def test_get_reward_deliveries_manual_takes_precedence_over_auto():
    """A manual delivery is ``manual`` even when the trial has auto-response set."""
    reward_times = np.array([0.15, 1.42])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]), autos=[None, True])
    manual_water = ManualWaterTimes(unaligned=np.array([1.42]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_reward_deliveries_go_cue_aligned_takes_precedence_over_auto():
    """Manual go-cue-aligned water outranks the trial's own free-water label.

    Free water can fire on the same trial the experimenter watered; the label
    reports who caused *this* delivery, and the experimenter is the narrower fact.
    """
    reward_times = np.array([0.15, 1.42])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]), autos=[None, True])
    manual_water = ManualWaterTimes(go_cue_aligned=np.array([1.42]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual_go_cue_aligned"]))


def test_get_reward_deliveries_unaligned_manual_outranks_go_cue_aligned():
    """When both kinds match one delivery, the unaligned label wins.

    Unaligned manual water is the stronger claim: it says the delivery is tied to
    no go cue at all, so it is written last and overwrites the aligned label.
    """
    reward_times = np.array([0.15, 1.42])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]))
    manual_water = ManualWaterTimes(unaligned=np.array([1.42]), go_cue_aligned=np.array([1.42]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(annotations, np.array(["earned", "manual"]))


def test_get_reward_deliveries_labels_both_manual_kinds_in_one_session():
    """A session can carry both kinds of experimenter water on the same port."""
    reward_times = np.array([0.15, 1.42, 2.95])
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0, 3.0]))
    manual_water = ManualWaterTimes(unaligned=np.array([0.15]), go_cue_aligned=np.array([2.95]))

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, manual_water)

    np.testing.assert_array_equal(
        annotations, np.array(["manual", "earned", "manual_go_cue_aligned"])
    )


def test_get_reward_deliveries_empty_deliveries_returns_empty():
    """No reward deliveries yields an empty annotation array."""
    trial_outcome_df = _trial_outcome_df(np.array([0.0]))

    annotations = get_reward_deliveries(np.array([]), trial_outcome_df, ManualWaterTimes())

    assert isinstance(annotations, np.ndarray)
    assert annotations.size == 0


def test_get_reward_deliveries_accepts_json_and_model_payloads():
    """``data`` payloads may be JSON strings or already-parsed ``TrialOutcome``."""
    reward_times = np.array([0.15, 1.42])
    payload = _outcome_payload(True)
    trial_outcome_df = pd.DataFrame(
        {"data": [json.dumps(payload), TrialOutcome.model_validate(payload)]},
        index=pd.Index([1.0, 2.0], name="time"),
    )

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, ManualWaterTimes())

    np.testing.assert_array_equal(annotations, np.array(["auto", "auto"]))


def test_get_reward_deliveries_rejects_an_empty_trial_table():
    """Deliveries cannot be matched when there are no trials to match them to."""
    empty = _trial_outcome_df(np.array([]))

    with pytest.raises(ValueError, match="trial_outcome_df is empty"):
        get_reward_deliveries(np.array([0.15]), empty, ManualWaterTimes())


def test_get_reward_deliveries_rejects_a_nan_trial_index():
    """A ``NaN`` trial boundary is rejected rather than silently swallowing deliveries.

    ``searchsorted`` against a ``NaN``-bearing index would quietly charge every
    delivery to the last trial, which is how an upstream stream failure turns
    into wrong annotations instead of an error.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.0, np.nan]))

    with pytest.raises(ValueError, match="contains NaN"):
        get_reward_deliveries(np.array([0.15]), trial_outcome_df, ManualWaterTimes())


def test_get_reward_deliveries_rejects_an_unsorted_trial_index():
    """Trial boundaries must be ordered for containment to be meaningful."""
    trial_outcome_df = _trial_outcome_df(np.array([2.0, 1.0]))

    with pytest.raises(ValueError, match="must be sorted"):
        get_reward_deliveries(np.array([0.15]), trial_outcome_df, ManualWaterTimes())


def test_get_manual_go_cue_aligned_trials_flags_the_delivery_trial():
    """A go-cue-aligned delivery flags the trial it landed in."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0, 3.0]))
    deliveries = np.array([0.15, 1.42, 2.95])
    manual_water = ManualWaterTimes(go_cue_aligned=np.array([1.43]))

    assert get_manual_go_cue_aligned_trials(deliveries, manual_water, trial_outcome_df) == {1}


def test_get_manual_go_cue_aligned_trials_skips_deliveries_unaligned_water_claims():
    """A delivery both kinds match is unaligned water, so it flags no trial.

    ``get_reward_deliveries`` gives unaligned manual water the last word, so
    flagging that trial here would contradict the label the series records.
    """
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 2.0]))
    deliveries = np.array([0.15, 1.42])
    both = ManualWaterTimes(unaligned=np.array([1.42]), go_cue_aligned=np.array([1.42]))

    assert get_manual_go_cue_aligned_trials(deliveries, both, trial_outcome_df) == set()
    # ...and that is exactly the delivery the series calls "manual".
    np.testing.assert_array_equal(
        get_reward_deliveries(deliveries, trial_outcome_df, both),
        np.array(["earned", "manual"]),
    )


def test_get_manual_go_cue_aligned_trials_empty_without_deliveries_or_events():
    """No deliveries, or no go-cue-aligned water, flags no trials."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0]))
    assert get_manual_go_cue_aligned_trials(np.array([]), ManualWaterTimes(), trial_outcome_df) == (
        set()
    )
    assert (
        get_manual_go_cue_aligned_trials(
            np.array([0.15]), ManualWaterTimes(unaligned=np.array([0.15])), trial_outcome_df
        )
        == set()
    )


def test_get_reward_deliveries_returns_one_annotation_per_delivery():
    """The result is a :class:`numpy.ndarray` aligned with the input deliveries."""
    trial_outcome_df = _trial_outcome_df(np.array([1.0, 1.5]))
    reward_times = np.array([0.1, 0.2, 1.2])

    annotations = get_reward_deliveries(reward_times, trial_outcome_df, ManualWaterTimes())

    assert isinstance(annotations, np.ndarray)
    assert annotations.shape == reward_times.shape
