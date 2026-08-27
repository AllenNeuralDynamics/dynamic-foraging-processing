"""Reward-related processing helpers for dynamic foraging data."""

import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

from dynamic_foraging_processing.utils.timestamps import find_closest_timestamps


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
        return "earned"
    return "auto"


def get_reward_deliveries(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    manual_water_times: np.ndarray,
    response_times: np.ndarray,
) -> np.ndarray:
    """Classify one lick port's reward deliveries by how the water was given.

    Annotates the deliveries of a single lick port. Each delivery is classified
    as follows, with ``manual`` taking precedence because manual water is not
    aligned to a go cue:

    - ``manual`` -- the delivery is the closest hardware (harp) timestamp to a
      ``GiveManualWater`` software event for this port. The software-event
      timestamps are correlated to the reward-delivery timestamps with
      :func:`find_closest_timestamps`.
    - ``auto`` -- otherwise, when the trial delivered free water
      (``is_auto_reward_right is not None``). Scheduled autowater and the
      anti-bias water intervention are both delivered through that channel, so
      both are ``auto`` here; which mechanism gave the water is recorded per
      trial by the trials table's ``auto_waterL``/``auto_waterR`` and
      ``anti_bias_left_water``/``anti_bias_right_water``.
    - ``earned`` -- otherwise: water the animal worked for.

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
    manual_water_times : numpy.ndarray
        Software-event timestamps of this port's manual water deliveries
        (``GiveManualWaterLeft`` / ``GiveManualWaterRight``).
    response_times : numpy.ndarray
        ``Response`` software-event timestamps, one per trial, positionally
        aligned with the rows of ``trial_outcome_df``.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"auto"``, or ``"manual"``.

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

    annotated_rewards = []
    for trial_index in trial_indices_in_reward_times:
        outcome = _parse_outcome(trial_outcome_df.iloc[trial_index]["data"])
        annotated_rewards.append(_free_water_label(outcome.trial if outcome is not None else None))

    # Object dtype, not the inferred fixed-width string dtype: a run of only "auto"
    # and "earned" entries would be too narrow to hold "manual" and would truncate it.
    annotated_rewards = np.array(annotated_rewards, dtype=object)

    # Manual water is independent of trials (multiple can occur within a trial) and
    # takes precedence, so annotate the manual deliveries directly. Correlate each
    # manual-water software event to its closest reward delivery; the returned
    # positions index into reward_times, i.e. the deliveries that are manual.
    manual_water_times = np.asarray(manual_water_times)
    if manual_water_times.size:
        manual_indices_in_reward_times = find_closest_timestamps(manual_water_times, reward_times)
        annotated_rewards[manual_indices_in_reward_times] = "manual"

    return annotated_rewards
