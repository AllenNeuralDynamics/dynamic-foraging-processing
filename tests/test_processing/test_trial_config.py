"""Tests for ``dynamic_foraging_processing.processing.models.trial_config``."""

from dynamic_foraging_processing.processing import TrialConfig


def test_column_descriptions_covers_every_field():
    """``column_descriptions`` maps every field to a non-empty description."""
    descriptions = TrialConfig.column_descriptions()
    assert set(descriptions) == set(TrialConfig.model_fields)
    assert all(descriptions.values())
    assert descriptions["animal_response"].startswith("The response of the animal")
