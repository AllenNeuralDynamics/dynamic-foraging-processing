"""Helpers for reading a trial's block-based extra metadata."""

from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    BlockBasedTrialMetadata,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial


def get_bias_metadata(trial: Trial) -> BlockBasedTrialMetadata:
    """Return the block-based extra metadata naming a trial's free-water mechanism.

    ``trial.is_auto_reward_right`` is only the delivery *channel*: it says free
    water was triggered and on which side, not what kind. Scheduled autowater and
    the anti-bias water intervention are told apart here, by ``is_autowater`` and
    ``is_bias_water_intervention``. (``is_bias_stage_intervention`` marks the
    anti-bias algorithm's other lever, moving the lickspouts.)

    The field is schema-typed ``Any``, so it deserializes off the stream as a
    plain ``dict`` rather than a model; a ``BlockBasedTrialMetadata`` instance is
    also accepted. When metadata or extra is missing (e.g. a non-block-based
    generator), the model's all-``False`` default is returned, so a trial whose
    mechanism the data does not record is reported as neither kind rather than
    guessed at.

    Parameters
    ----------
    trial : Trial
        The per-trial task-logic model.

    Returns
    -------
    BlockBasedTrialMetadata
        The parsed extra metadata, or an all-``False`` default when absent
        or unrecognized.
    """
    metadata = trial.metadata
    extra = metadata.extra if metadata is not None else None
    if isinstance(extra, BlockBasedTrialMetadata):
        return extra
    if isinstance(extra, dict):
        return BlockBasedTrialMetadata.model_validate(extra)
    return BlockBasedTrialMetadata()
