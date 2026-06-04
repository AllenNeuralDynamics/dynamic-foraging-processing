"""Tests for ``dynamic_foraging_processing.utils.timestamps``."""

import numpy as np
import pytest

from dynamic_foraging_processing.utils.timestamps import find_closest_timestamps


def test_find_closest_timestamps_returns_nearest_position():
    """Each query maps to the nearest reference position."""
    query = np.array([0.42, 0.05, 1.55, 0.95, 0.85])
    reference = np.array([0.9, 0.1, 2.0, 0.4])

    positions = find_closest_timestamps(query, reference)

    np.testing.assert_array_equal(positions, np.array([3, 1, 2, 0, 0]))
    np.testing.assert_array_equal(reference[positions], np.array([0.4, 0.1, 2.0, 0.9, 0.9]))


def test_find_closest_timestamps_ties_resolve_to_earlier_reference():
    """When a query is equidistant between two references, the earlier one wins."""
    query = np.array([0.5])
    reference = np.array([0.0, 1.0])

    positions = find_closest_timestamps(query, reference)

    np.testing.assert_array_equal(positions, np.array([0]))


def test_find_closest_timestamps_accepts_lists():
    """Inputs are coerced via ``np.asarray``."""
    positions = find_closest_timestamps([0.1, 0.9], [0.0, 1.0])

    np.testing.assert_array_equal(positions, np.array([0, 1]))


def test_find_closest_timestamps_empty_reference_raises():
    """An empty reference array raises ``ValueError``."""
    with pytest.raises(ValueError, match="reference_timestamps must not be empty"):
        find_closest_timestamps(np.array([0.0]), np.array([]))
