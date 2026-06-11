"""Tests for ``dynamic_foraging_processing.processing.models.trial_row``."""

from dynamic_foraging_processing.processing import TrialRow


def test_column_descriptions_covers_every_field():
    """``column_descriptions`` maps every field to a non-empty description."""
    descriptions = TrialRow.column_descriptions()
    assert set(descriptions) == set(TrialRow.model_fields)
    assert all(descriptions.values())
    assert descriptions["animal_response"].startswith("The response of the animal")
