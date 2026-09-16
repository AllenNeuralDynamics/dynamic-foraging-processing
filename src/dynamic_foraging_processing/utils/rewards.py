"""Reward-related processing helpers for dynamic foraging data."""

import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

from dynamic_foraging_processing.utils.timestamps import find_closest_timestamps

#: Reward-delivery annotation for water the animal worked for.
EARNED = "earned"
#: Reward-delivery annotation for task-triggered free water (autowater, anti-bias).
AUTO = "auto"
#: Reward-delivery annotation for experimenter water not aligned to a go cue.
MANUAL = "manual"
#: Reward-delivery annotation for experimenter water delivered at the go cue.
MANUAL_GO_CUE_ALIGNED = "manual_go_cue_aligned"

#: Shared empty default for the ``ManualWaterTimes`` fields. Never mutated.
_NO_TIMES = np.array([])


class ManualWaterTimes(t.NamedTuple):
    """One lick port's experimenter-triggered water times, split by alignment.

    The acquisition software emits the two kinds on separate software-event
    streams, so they are kept apart here rather than conflated into a single
    "manual" bucket: only ``unaligned`` water is independent of the trial
    structure, while ``go_cue_aligned`` water fires at the go cue like autowater
    but was triggered by the experimenter, not the task.

    Attributes
    ----------
    unaligned : numpy.ndarray
        Timestamps of the ``LeftManualWater`` / ``RightManualWater`` events:
        water given at an arbitrary moment, not tied to a go cue.
    go_cue_aligned : numpy.ndarray
        Timestamps of the ``LeftManualAutoReward`` / ``RightManualAutoReward``
        events: water the experimenter triggered to land on the go cue.
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


def _free_water_label(trial: t.Optional[Trial]) -> str:
    """Classify a delivery's trial as ``auto`` (free water) or ``earned``.

    ``is_auto_reward_right`` triggers an immediate reward to one side, so any
    trial with it set gave free water rather than water the animal worked for.
    Scheduled autowater and the anti-bias water intervention share that channel
    and are both ``auto`` here; which mechanism gave the water is recorded per
    trial by ``auto_waterL``/``auto_waterR`` and
    ``anti_bias_left_water``/``anti_bias_right_water`` in the trials table.

    Parameters
    ----------
    trial : Trial or None
        The per-trial task-logic model, or ``None`` when the outcome payload was
        missing.

    Returns
    -------
    str
        ``"auto"`` when the trial delivered free water, else ``"earned"``.
    """
    if trial is None or trial.is_auto_reward_right is None:
        return EARNED
    return AUTO


def get_reward_deliveries(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    manual_water: ManualWaterTimes,
) -> np.ndarray:
    """Classify one lick port's reward deliveries by how the water was given.

    Annotates the deliveries of a single lick port. Each delivery is classified
    as follows, with the experimenter-triggered labels taking precedence over the
    trial-derived ones because the experimenter acts outside the task logic:

    - ``manual`` -- the delivery is the closest hardware (harp) timestamp to a
      ``LeftManualWater``/``RightManualWater`` software event for this port:
      water given at an arbitrary moment, not aligned to a go cue. Highest
      precedence.
    - ``manual_go_cue_aligned`` -- the delivery is the closest timestamp to a
      ``LeftManualAutoReward``/``RightManualAutoReward`` event: water the
      experimenter triggered to land on the go cue. It fires at the go cue like
      autowater but the task did not schedule it, so it is neither ``auto`` nor
      ``earned``.
    - ``auto`` -- otherwise, when the trial delivered free water
      (``is_auto_reward_right is not None``). Scheduled autowater and the
      anti-bias water intervention are both delivered through that channel, so
      both are ``auto`` here; which mechanism gave the water is recorded per
      trial by the trials table's ``auto_waterL``/``auto_waterR`` and
      ``anti_bias_left_water``/``anti_bias_right_water``.
    - ``earned`` -- otherwise: water the animal worked for.

    The two manual labels are kept apart rather than collapsed into ``manual``
    because they differ in kind: only unaligned manual water is independent of
    the trial structure. Collapsing them would also make the go-cue-aligned
    deliveries indistinguishable from task-scheduled ``auto`` water in the QC
    figure, which is what the separate rows there exist to show.

    The software-event timestamps are correlated to the reward-delivery
    timestamps with :func:`find_closest_timestamps`.

    Every valve opening is annotated and none is filtered out, so the series is a
    complete record of the water this port delivered. In particular a trial
    reporting ``is_rewarded=False`` keeps its delivery: free water is triggered
    immediately at the go cue and the trial then continues normally, so
    ``is_rewarded`` reports the outcome of the animal's *own choice* -- a
    separate event from the water being classified here. A trial can therefore
    contribute an ``auto`` delivery on the side the task watered and an
    ``earned`` delivery on the side the animal chose.

    Deliveries are matched to trials by *containment*, not proximity: a
    ``TrialOutcome`` fires at the end of a trial, after the reward-consumption
    and ITI periods, so trial ``i`` spans ``(outcome[i - 1], outcome[i]]`` and a
    delivery belongs to the first trial whose outcome it precedes.

    Proximity to the ``Response`` event was used previously and is unsound: it
    has no notion of a trial boundary, so a delivery near the start of its trial
    can be closer to the *previous* trial's response and inherit that trial's
    ``is_auto_reward_right``, flipping ``auto`` to ``earned``. Autowater on an
    ignore trial is the worst case -- the water lands at the go cue while the
    trial's own ``Response`` waits out the full response deadline -- but the
    boundary is what makes containment correct, whatever the response latency.

    Parameters
    ----------
    reward_delivery_times : numpy.ndarray
        Hardware (harp) timestamps of this port's reward deliveries.
    trial_outcome_df : pandas.DataFrame
        Trial outcome table indexed by trial timestamp; each row's ``data``
        field is a :class:`TrialOutcome` payload.
    manual_water : ManualWaterTimes
        This port's experimenter-triggered water times, split into ``unaligned``
        and ``go_cue_aligned``. Either field may be empty when the session has
        no water of that kind.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"auto"``, ``"manual_go_cue_aligned"``, or ``"manual"``.

    Raises
    ------
    ValueError
        If ``trial_outcome_df`` is empty, or its index is unsorted or contains
        ``NaN``, since the index is used as the trial window boundaries.
    """
    reward_times = np.asarray(reward_delivery_times)
    if reward_times.size == 0:
        return np.array([], dtype=object)

    # The trial's own end time bounds it, so the index doubles as the window
    # edges. Guard the assumptions searchsorted makes rather than letting a bad
    # index silently push every delivery onto one trial: an all-NaN index would
    # otherwise assign them all to the last trial.
    trial_end_times = np.asarray(trial_outcome_df.index, dtype=float)
    if trial_end_times.size == 0:
        raise ValueError("trial_outcome_df is empty; deliveries cannot be matched to a trial.")
    if np.isnan(trial_end_times).any():
        raise ValueError("trial_outcome_df index contains NaN; trial windows are undefined.")
    if np.any(np.diff(trial_end_times) < 0):
        raise ValueError("trial_outcome_df index must be sorted to bound trials.")

    # side="left" so a delivery landing exactly on a trial's outcome belongs to
    # that trial rather than the next.
    trial_indices_in_reward_times = np.minimum(
        np.searchsorted(trial_end_times, reward_times, side="left"),
        trial_end_times.size - 1,
    )

    trial_labels = []
    for trial_index in trial_indices_in_reward_times:
        outcome = _parse_outcome(trial_outcome_df.iloc[trial_index]["data"])
        trial_labels.append(_free_water_label(outcome.trial if outcome is not None else None))

    # Object dtype, not the inferred fixed-width string dtype: a run of only "auto"
    # and "earned" entries would be too narrow to hold the manual labels and would
    # truncate them.
    annotated_rewards = np.array(trial_labels, dtype=object)

    # Experimenter water is independent of trials (multiple deliveries can occur
    # within one trial), so annotate those deliveries directly rather than through
    # the trial they fall in. Correlate each software event to its closest reward
    # delivery; the returned positions index into reward_times, i.e. the deliveries
    # the experimenter caused. Unaligned manual water is written last so it wins
    # where both kinds land on the same delivery.
    for times, label in (
        (manual_water.go_cue_aligned, MANUAL_GO_CUE_ALIGNED),
        (manual_water.unaligned, MANUAL),
    ):
        times = np.asarray(times)
        if times.size:
            annotated_rewards[find_closest_timestamps(times, reward_times)] = label

    return annotated_rewards
