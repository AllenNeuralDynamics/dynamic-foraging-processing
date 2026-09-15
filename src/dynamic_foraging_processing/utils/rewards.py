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
    response_times: np.ndarray,
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

    Deliveries are matched to trials by the ``Response`` software-event
    timestamp: each delivery takes the annotation of the trial whose response is
    closest. The response is used rather than the ``TrialOutcome`` timestamp
    because ``TrialOutcome`` fires at the *end* of a trial, after the
    reward-consumption and ITI periods, while the valve opens within
    milliseconds of the response. Matching on trial end lets a delivery land
    nearer the *previous* trial's outcome and inherit its
    ``is_auto_reward_right``, flipping ``earned`` and ``auto``.

    ``response_times`` is aligned to ``trial_outcome_df`` positionally: entry
    ``i`` is the response of the trial in row ``i``.

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
    response_times : numpy.ndarray
        ``Response`` software-event timestamps, one per trial, positionally
        aligned with the rows of ``trial_outcome_df``.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"auto"``, ``"manual_go_cue_aligned"``, or ``"manual"``.

    Raises
    ------
    ValueError
        If ``response_times`` has a different length than ``trial_outcome_df``,
        since the two are paired by position.
    """
    response_times = np.asarray(response_times)
    if response_times.size != len(trial_outcome_df):
        raise ValueError(
            f"response_times has {response_times.size} entries but there are "
            f"{len(trial_outcome_df)} trials; the two are paired by position."
        )

    reward_times = np.asarray(reward_delivery_times)
    if reward_times.size == 0:
        return np.array([], dtype=object)

    # Annotate each delivery from its originating trial: query with reward_times so we
    # get one trial position per reward delivery.
    trial_indices_in_reward_times = find_closest_timestamps(reward_times, response_times)

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
