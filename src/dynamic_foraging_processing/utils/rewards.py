"""Reward-related processing helpers for dynamic foraging data."""

import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome

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


def get_annotated_rewards(
    reward_delivery_times: np.ndarray,
    trial_outcome_df: pd.DataFrame,
    manual_water_times: np.ndarray,
    response_times: np.ndarray,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Annotate each reward delivery as ``earned``, ``auto``, or ``manual``.

    Annotates the deliveries of a single lick port. Each delivery is classified
    as follows, with ``manual`` taking precedence because manual water is not
    aligned to a go cue:

    - ``manual`` -- the delivery is the closest hardware (harp) timestamp to a
      ``GiveManualWater`` software event for this port. The software-event
      timestamps are correlated to the reward-delivery timestamps with
      :func:`find_closest_timestamps`.
    - ``auto`` -- otherwise, when the matching trial auto-responded
      (``is_auto_reward_right is not None``).
    - ``earned`` -- otherwise (no matching trial, or no auto-response).

    Deliveries the animal never received are dropped rather than annotated:
    autowater is delivered *before* the response, so on an auto trial the animal
    answered the other way the water goes uncollected and the trial reports
    ``is_rewarded=False``. Manual water keeps its precedence and is never
    dropped. The surviving timestamps are returned alongside their annotations
    so the two stay aligned.

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
        The retained reward-delivery timestamps: ``reward_delivery_times`` less
        the uncollected autowater deliveries.
    numpy.ndarray
        The matching annotations, one per retained timestamp, each
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
        return reward_times, np.array([], dtype=object)

    # Annotate each delivery from its originating trial: query with reward_times so we
    # get one trial position per reward delivery.
    trial_indices_in_reward_times = find_closest_timestamps(reward_times, response_times)

    annotated_rewards = []
    is_unrewarded_auto = []
    for trial_index in trial_indices_in_reward_times:
        outcome = _parse_outcome(trial_outcome_df.iloc[trial_index]["data"])
        trial = outcome.trial if outcome is not None else None
        is_auto = trial is not None and trial.is_auto_reward_right is not None
        annotated_rewards.append("auto" if is_auto else "earned")
        is_unrewarded_auto.append(is_auto and outcome is not None and not outcome.is_rewarded)

    # Object dtype, not the inferred fixed-width string dtype: a run of only "auto"
    # and "earned" entries would be too narrow to hold "manual" and would truncate it.
    annotated_rewards = np.array(annotated_rewards, dtype=object)

    # Manual water is independent of trials (multiple can occur within a trial) and
    # takes precedence, so annotate the manual deliveries directly. Correlate each
    # manual-water software event to its closest reward delivery; the returned
    # positions index into reward_times, i.e. the deliveries that are manual.
    manual_water_times = np.asarray(manual_water_times)
    manual_mask = np.zeros(reward_times.size, dtype=bool)
    if manual_water_times.size:
        manual_indices_in_reward_times = find_closest_timestamps(manual_water_times, reward_times)
        manual_mask[manual_indices_in_reward_times] = True
        annotated_rewards[manual_mask] = "manual"

    # Autowater is delivered before the response, so on a trial the animal answered the
    # other way the water is never collected and the trial reports is_rewarded=False.
    # Those deliveries are dropped: they are not reward the animal received. Manual
    # water keeps its precedence and is never dropped.
    keep = ~(np.array(is_unrewarded_auto, dtype=bool) & ~manual_mask)
    return reward_times[keep], annotated_rewards[keep]
