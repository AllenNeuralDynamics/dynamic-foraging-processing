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
) -> np.ndarray:
    """Annotate each reward delivery as ``earned``, ``automatic``, or ``manual``.

    Annotates the deliveries of a single lick port. Each delivery is classified
    as follows, with ``manual`` taking precedence because manual water is not
    aligned to a go cue:

    - ``manual`` -- the delivery is the closest hardware (harp) timestamp to a
      ``GiveManualWater`` software event for this port. The software-event
      timestamps are correlated to the reward-delivery timestamps with
      :func:`find_closest_timestamps`.
    - ``automatic`` -- otherwise, when the matching trial auto-responded to this
      same port (``is_auto_response_right is is_right``: ``True`` for the right
      port, ``False`` for the left port).
    - ``earned`` -- otherwise (no auto-response, or auto-response to the other
      port).

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

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``reward_delivery_times`` whose entries are
        ``"earned"``, ``"automatic"``, or ``"manual"``.
    """
    reward_times = np.asarray(reward_delivery_times)
    if reward_times.size == 0:
        return np.array([], dtype=object)

    # Correlate each manual-water software event to its closest reward delivery;
    # those deliveries are the manual ones. We query with manual_water_times so the
    # returned positions index into reward_times -- i.e. the reward deliveries that
    # are manual -- giving the set we test reward indices against below.
    manual_water_times = np.asarray(manual_water_times)
    if manual_water_times.size:
        manual_indices_in_reward_times = set(
            find_closest_timestamps(manual_water_times, reward_times).tolist()
        )
    else:
        manual_indices_in_reward_times = set()

    # The opposite direction: query with reward_times so we get one trial position
    # per reward delivery (used to look up each delivery's originating trial below).
    trial_indices_in_reward_times = find_closest_timestamps(
        reward_times, trial_outcome_df.index.to_numpy()
    )

    annotated_rewards = []
    for i, trial_index in enumerate(trial_indices_in_reward_times):
        if i in manual_indices_in_reward_times:
            annotated_rewards.append("manual")
            continue

        outcome = _parse_outcome(trial_outcome_df.iloc[trial_index]["data"])
        trial = outcome.trial if outcome is not None else None
        if trial is None or trial.is_auto_response_right is None:
            annotated_rewards.append("earned")
        else:
            annotated_rewards.append("automatic")

    return np.array(annotated_rewards)
