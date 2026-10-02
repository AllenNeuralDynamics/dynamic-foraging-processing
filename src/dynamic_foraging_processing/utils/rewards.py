"""Reward-related processing helpers for dynamic foraging data."""

import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

#: Reward-delivery annotation for water the animal worked for.
EARNED = "earned"
#: Reward-delivery annotation for task-triggered free water (autowater, anti-bias).
AUTO = "auto"
#: Reward-delivery annotation for manual water not aligned to a go cue.
MANUAL = "manual"
#: Reward-delivery annotation for manual water delivered at the go cue.
MANUAL_GO_CUE_ALIGNED = "manual_go_cue_aligned"

#: Largest gap (s) between a go cue and the valve opening that delivers free water
#: on it. Empirical: free water lands -2 to +5 ms from the go cue.
FREE_WATER_TOLERANCE = 0.02

#: Shared empty default for the ``ManualWaterTimes`` fields. Never mutated.
_NO_TIMES = np.array([])


class ManualWaterTimes(t.NamedTuple):
    """One lick port's manual water times, split by alignment.

    The acquisition software emits the two kinds on separate software-event
    streams, so they are kept apart here rather than conflated into a single
    "manual" bucket: only ``unaligned`` water is independent of the trial
    structure, while ``go_cue_aligned`` water fires at the go cue like autowater
    but was requested by the operator, not the task.

    Attributes
    ----------
    unaligned : numpy.ndarray
        Timestamps of the ``LeftManualWater`` / ``RightManualWater`` events:
        water given at an arbitrary moment, not tied to a go cue.
    go_cue_aligned : numpy.ndarray
        Timestamps of the ``LeftManualAutoReward`` / ``RightManualAutoReward``
        events: water the operator requested to land on the go cue.
    """

    unaligned: np.ndarray = _NO_TIMES
    go_cue_aligned: np.ndarray = _NO_TIMES


def _parse_outcome(payload: t.Any) -> t.Optional[TrialOutcome]:
    """Parse a ``TrialOutcome`` software-event payload into its domain model.

    Parameters
    ----------
    payload : Any
        The stream's ``data`` value: a serialized JSON string, a dict, an
        already-parsed ``TrialOutcome``, or ``None``.

    Returns
    -------
    TrialOutcome or None
        The parsed outcome, or ``None`` when ``payload`` is ``None``.
    """
    if payload is None or isinstance(payload, TrialOutcome):
        return payload
    if isinstance(payload, (str, bytes, bytearray)):
        return TrialOutcome.model_validate_json(payload)
    return TrialOutcome.model_validate(payload)


def _trial_end_times(trial_outcome_df: pd.DataFrame) -> np.ndarray:
    """Return each trial's end time (its ``TrialOutcome`` timestamp), validated.

    Raises
    ------
    ValueError
        If the index is empty, unsorted, or contains ``NaN``.
    """
    # Guard what searchsorted assumes: an all-NaN index would silently charge
    # every delivery to the last trial instead of failing.
    trial_ends = np.asarray(trial_outcome_df.index, dtype=float)
    if trial_ends.size == 0:
        raise ValueError("trial_outcome_df is empty; deliveries cannot be matched to a trial.")
    if np.isnan(trial_ends).any():
        raise ValueError("trial_outcome_df index contains NaN; trial windows are undefined.")
    if np.any(np.diff(trial_ends) < 0):
        raise ValueError("trial_outcome_df index must be sorted to bound trials.")
    return trial_ends


def trial_index_of(times: np.ndarray, trial_ends: np.ndarray) -> np.ndarray:
    """Return the index of the trial whose window contains each time.

    Trial ``i`` spans ``(trial_ends[i - 1], trial_ends[i]]``. Times past the last
    trial's end are charged to the last trial.
    """
    # side="left" so a time landing exactly on a trial's end belongs to it.
    return np.minimum(np.searchsorted(trial_ends, times, side="left"), trial_ends.size - 1)


def _trial_go_cues(go_cue_times: np.ndarray, trial_ends: np.ndarray) -> np.ndarray:
    """Return each trial's go cue: the first one inside its window, NaN if none."""
    sorted_go_cues = np.sort(np.asarray(go_cue_times, dtype=float))
    trial_go_cues = np.full(trial_ends.size, np.nan)
    # Go cues are sorted, so each trial's first occurrence is its earliest go cue.
    cue_trials, first_cue = np.unique(trial_index_of(sorted_go_cues, trial_ends), return_index=True)
    trial_go_cues[cue_trials] = sorted_go_cues[first_cue]
    return trial_go_cues


def _label_trial(
    outcome: t.Optional[TrialOutcome],
    delivery_times: np.ndarray,
    go_cue: float,
    *,
    is_right: bool,
) -> np.ndarray:
    """Label one trial's deliveries on this port (rules 1-3).

    At most one delivery is the task's: free water at the go cue on the watered
    port, or earned water on the chosen port after it -- never both. Every other
    delivery is manual.
    """
    # Object dtype, so short labels do not fix a string width that truncates longer ones.
    labels = np.full(delivery_times.size, MANUAL, dtype=object)
    if outcome is None or np.isnan(go_cue):
        return labels

    # True: free water to the right; False: to the left; None: no free water.
    free_water_right = outcome.trial.is_auto_reward_right
    if free_water_right is not None:
        # Rule 1: the delivery nearest the go cue on the watered port, if close enough.
        if free_water_right is is_right:
            gaps_to_go_cue = np.abs(delivery_times - go_cue)
            nearest = np.argmin(gaps_to_go_cue)
            if gaps_to_go_cue[nearest] <= FREE_WATER_TOLERANCE:
                labels[nearest] = AUTO
    elif outcome.is_right_choice is is_right:
        # Rule 2: the first delivery on the chosen port at or after the go cue.
        after_go_cue = np.flatnonzero(delivery_times >= go_cue)
        if after_go_cue.size:
            labels[after_go_cue[np.argmin(delivery_times[after_go_cue])]] = EARNED
    return labels


def get_reward_deliveries(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    go_cue_times: np.ndarray,
    manual_go_cue_aligned_times: np.ndarray,
    *,
    is_right: bool,
) -> np.ndarray:
    """Classify one lick port's reward deliveries by how the water was given.

    Each delivery belongs to the trial whose ``TrialOutcome`` window contains it,
    and each trial's go cue is the first hardware go cue in that window. Then:

    1. ``auto`` -- on a free-water trial (``is_auto_reward_right`` set to this
       port), the delivery closest to the go cue, if within
       :data:`FREE_WATER_TOLERANCE`.
    2. ``earned`` -- on a trial with no free water, the first delivery on
       the chosen port at or after the go cue.
    3. ``manual`` -- every other delivery.
    4. ``manual_go_cue_aligned`` -- each ``ManualAutoReward`` request relabels the
       next ``auto`` delivery after it.

    So a trial has at most one task delivery. Precise timing compares valve
    openings with go cues, both hardware times. The ``ManualAutoReward`` timestamp
    is when the operator pressed the button, seconds before the water, so it is
    only used for ordering. Unaligned manual water needs no event: it is whatever
    the task did not deliver.

    Parameters
    ----------
    reward_delivery_times : numpy.ndarray
        Hardware (harp) timestamps of this port's valve openings.
    trial_outcome_df : pandas.DataFrame
        Trial outcome table indexed by trial timestamp; each row's ``data`` field
        is a :class:`TrialOutcome` payload.
    go_cue_times : numpy.ndarray
        Hardware go-cue timestamps (the sound card's ``PlaySoundOrFrequency``
        writes).
    manual_go_cue_aligned_times : numpy.ndarray
        This port's ``ManualAutoReward`` event timestamps; empty when the session
        has none.
    is_right : bool
        ``True`` for the right lick port, ``False`` for the left.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"auto"``, ``"manual_go_cue_aligned"``, or ``"manual"``.

    Raises
    ------
    ValueError
        If there are deliveries but no go cues, or ``trial_outcome_df`` is empty,
        unsorted, or has a ``NaN`` index.
    """
    delivery_times = np.asarray(reward_delivery_times, dtype=float)
    if delivery_times.size == 0:
        return np.array([], dtype=object)
    if np.size(go_cue_times) == 0:
        raise ValueError("go_cue_times is empty; deliveries cannot be classified.")

    # Which trial each delivery falls in, and each trial's go cue.
    trial_ends = _trial_end_times(trial_outcome_df)
    delivery_trials = trial_index_of(delivery_times, trial_ends)
    trial_go_cues = _trial_go_cues(go_cue_times, trial_ends)

    # Rules 1-3, one trial at a time.
    labels = np.empty(delivery_times.size, dtype=object)
    for trial in np.unique(delivery_trials):
        in_trial = delivery_trials == trial
        outcome = _parse_outcome(trial_outcome_df.iloc[trial]["data"])
        labels[in_trial] = _label_trial(
            outcome, delivery_times[in_trial], trial_go_cues[trial], is_right=is_right
        )

    # Rule 4: each request claims the next auto delivery on this port. Repeated
    # requests before the water lands collapse onto the same delivery.
    auto_indices = np.flatnonzero(labels == AUTO)
    auto_indices = auto_indices[np.argsort(delivery_times[auto_indices])]
    for request in np.asarray(manual_go_cue_aligned_times, dtype=float):
        auto_after_request = auto_indices[delivery_times[auto_indices] >= request]
        if auto_after_request.size:
            labels[auto_after_request[0]] = MANUAL_GO_CUE_ALIGNED
    return labels


def get_manual_go_cue_aligned_trials(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    go_cue_times: np.ndarray,
    manual_go_cue_aligned_times: np.ndarray,
    *,
    is_right: bool,
) -> t.Set[int]:
    """Return the trials given manual go-cue-aligned water, for one port.

    Read from the labels :func:`get_reward_deliveries` assigns, so the trials
    table and the reward-delivery series cannot disagree.

    Parameters
    ----------
    reward_delivery_times, trial_outcome_df, go_cue_times, manual_go_cue_aligned_times
        As for :func:`get_reward_deliveries`.
    is_right : bool
        ``True`` for the right lick port, ``False`` for the left.

    Returns
    -------
    set of int
        Trial indices whose delivery on this port was manual go-cue-aligned water.
    """
    delivery_times = np.asarray(reward_delivery_times, dtype=float)
    if delivery_times.size == 0 or np.size(manual_go_cue_aligned_times) == 0:
        return set()
    labels = get_reward_deliveries(
        delivery_times,
        trial_outcome_df,
        go_cue_times,
        manual_go_cue_aligned_times,
        is_right=is_right,
    )
    aligned_times = delivery_times[labels == MANUAL_GO_CUE_ALIGNED]
    trial_ends = _trial_end_times(trial_outcome_df)
    return {int(trial) for trial in trial_index_of(aligned_times, trial_ends)}
