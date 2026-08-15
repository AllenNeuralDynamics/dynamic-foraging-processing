"""Reward-related processing helpers for dynamic foraging data."""

import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome

from dynamic_foraging_processing.utils.timestamps import find_closest_timestamps
from dynamic_foraging_processing.utils.trial_metadata import get_bias_metadata


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
    """Classify a delivery's trial as ``auto``, ``anti_bias``, or ``earned``.

    ``is_auto_reward_right`` marks only that free water was triggered and on
    which side; scheduled autowater and the anti-bias intervention share that
    channel and are told apart by the block-based metadata flags. A trial with no
    free water, or free water flagged as neither mechanism, is ``earned``.

    Parameters
    ----------
    trial : Trial or None
        The per-trial task-logic model, or ``None`` when the outcome payload was
        missing.

    Returns
    -------
    str
        ``"auto"``, ``"anti_bias"``, or ``"earned"``.
    """
    if trial is None or trial.is_auto_reward_right is None:
        return "earned"
    metadata = get_bias_metadata(trial)
    if metadata.is_bias_water_intervention:
        return "anti_bias"
    if metadata.is_autowater:
        return "auto"
    return "earned"


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
    - ``anti_bias`` -- otherwise, when the trial's free water came from the
      anti-bias algorithm (``is_bias_water_intervention``).
    - ``auto`` -- otherwise, when the trial's free water was scheduled autowater
      (``is_autowater``).
    - ``earned`` -- otherwise: water the animal worked for.

    ``is_auto_reward_right`` is only the delivery *channel* -- it says free water
    was triggered and on which side, not what kind -- so the ``auto`` versus
    ``anti_bias`` split comes from the block-based metadata flags (see
    :func:`get_bias_metadata`), mirroring ``auto_waterL``/``auto_waterR`` and
    ``anti_bias_left_water``/``anti_bias_right_water`` in the trials table.

    Every delivery is annotated and none is filtered out. A trial reporting
    ``is_rewarded=False`` keeps its delivery: free water is triggered immediately
    at the go cue and the trial then continues normally, so ``is_rewarded``
    reports the outcome of the animal's own choice -- a separate event from the
    water being classified here.

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
        ``"earned"``, ``"auto"``, ``"anti_bias"``, or ``"manual"``.

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
    # and "earned" entries would be too narrow to hold "anti_bias" and would truncate it.
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
