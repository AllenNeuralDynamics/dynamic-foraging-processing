"""Helpers for reading task parameters from the curriculum's trainer state."""

import typing as t

from aind_behavior_curriculum import TrainerState
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskParameters


def get_bias_threshold(trainer_state: t.Optional[TrainerState]) -> t.Optional[float]:
    """Return the side-bias magnitude at which the anti-bias algorithm intervenes.

    Read from
    ``stage.task.task_parameters.trial_generator.bias_intervention_parameters.threshold.upper``.
    ``TrainerState`` types the task generically, so the task parameters
    deserialize with the trial generator as untyped extra fields; they are
    re-validated here as the dynamic foraging task parameters.

    Parameters
    ----------
    trainer_state : TrainerState or None
        The session's trainer state, or ``None`` if it is unavailable.

    Returns
    -------
    float or None
        The upper bias threshold, or ``None`` when there is no trainer state or
        stage, or the trial generator has no bias intervention configured.
    """
    if trainer_state is None or trainer_state.stage is None:
        return None
    task_parameters = AindDynamicForagingTaskParameters.model_validate_json(
        trainer_state.stage.task.task_parameters.model_dump_json()
    )
    bias_parameters = getattr(task_parameters.trial_generator, "bias_intervention_parameters", None)
    if bias_parameters is None:
        return None
    return bias_parameters.threshold.upper
