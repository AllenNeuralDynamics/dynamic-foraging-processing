"""Tests for ``dynamic_foraging_processing.utils.curriculum``."""

from aind_behavior_curriculum import Stage, TrainerState
from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
    AindDynamicForagingTaskParameters,
)
from aind_behavior_dynamic_foraging.task_logic.interventions.bias_intervention import (
    BiasInterventionParameters,
    BiasThreshold,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators import (
    CoupledTrialGeneratorSpec,
    IntegrationTestTrialGeneratorSpec,
)

from dynamic_foraging_processing.utils.curriculum import get_bias_threshold


def _loaded_trainer_state(trial_generator) -> TrainerState:
    """Build a trainer state and reload it from JSON, as the data contract does.

    Reloading drops the concrete task type, so the trial generator comes back as
    untyped extra fields, which is the shape ``get_bias_threshold`` must handle.
    """
    task = AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(trial_generator=trial_generator)
    )
    state = TrainerState(
        stage=Stage(name="stage", task=task), curriculum=None, is_on_curriculum=False
    )
    return TrainerState.model_validate_json(state.model_dump_json())


def test_get_bias_threshold_reads_upper_threshold():
    """The upper threshold is read from the trial generator's bias parameters."""
    generator = CoupledTrialGeneratorSpec(
        bias_intervention_parameters=BiasInterventionParameters(threshold=BiasThreshold(upper=0.5))
    )
    assert get_bias_threshold(_loaded_trainer_state(generator)) == 0.5


def test_get_bias_threshold_none_without_bias_intervention():
    """A generator with no bias intervention configured has no threshold."""
    no_bias = CoupledTrialGeneratorSpec(bias_intervention_parameters=None)
    assert get_bias_threshold(_loaded_trainer_state(no_bias)) is None
    assert get_bias_threshold(_loaded_trainer_state(IntegrationTestTrialGeneratorSpec())) is None


def test_get_bias_threshold_none_without_trainer_state_or_stage():
    """A missing trainer state, or one with no stage, has no threshold."""
    assert get_bias_threshold(None) is None
    state = TrainerState(stage=None, curriculum=None, is_on_curriculum=False)
    assert get_bias_threshold(state) is None
