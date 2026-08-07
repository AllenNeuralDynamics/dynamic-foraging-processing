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
    response_times: np.ndarray,
    manual_water_times: np.ndarray,
) -> np.ndarray:
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

    Deliveries are matched to trials by the ``Response`` software event, which
    is emitted when the animal responds and so sits next to the delivery it
    caused. The ``Response`` and ``TrialOutcome`` streams are emitted once per
    trial and are positionally aligned, so the matched ``Response`` position
    indexes the corresponding ``TrialOutcome`` row.

    Parameters
    ----------
    reward_delivery_times : numpy.ndarray
        Hardware (harp) timestamps of this port's reward deliveries.
    trial_outcome_df : pandas.DataFrame
        Trial outcome table with one row per trial; each row's ``data`` field
        is a :class:`TrialOutcome` payload.
    response_times : numpy.ndarray
        Software-event timestamps of the per-trial ``Response`` events, in
        trial order (one per row of ``trial_outcome_df``).
    manual_water_times : numpy.ndarray
        Software-event timestamps of this port's manual water deliveries
        (``GiveManualWaterLeft`` / ``GiveManualWaterRight``).

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"auto"``, or ``"manual"``.

    Raises
    ------
    ValueError
        If ``response_times`` and ``trial_outcome_df`` have different lengths,
        i.e. the per-trial streams are misaligned.
    """
    reward_times = np.asarray(reward_delivery_times)
    response_times = np.asarray(response_times)
    if response_times.size != len(trial_outcome_df):
        raise ValueError(
            "Response and TrialOutcome streams are misaligned: "
            f"{response_times.size} responses vs {len(trial_outcome_df)} trial outcomes."
        )
    if reward_times.size == 0:
        return np.array([], dtype=object)

    # Annotate each delivery from its originating trial: query with reward_times so we
    # get one trial position per reward delivery.
    trial_indices_in_reward_times = find_closest_timestamps(reward_times, response_times)

    annotated_rewards = []
    for trial_index in trial_indices_in_reward_times:
        outcome = _parse_outcome(trial_outcome_df.iloc[trial_index]["data"])
        trial = outcome.trial if outcome is not None else None
        if trial is None or trial.is_auto_reward_right is None:
            annotated_rewards.append("earned")
        else:
            annotated_rewards.append("auto")

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
