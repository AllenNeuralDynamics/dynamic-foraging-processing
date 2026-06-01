"""Tests for ``dynamic_foraging_processing.process.acquisition.models``."""

import numpy as np
import pytest

from dynamic_foraging_processing.process.acquisition.models import (
    AcquisitionSeries,
    NWBAcquisition,
)


def _make_series(name: str = "series", n: int = 3) -> AcquisitionSeries:
    """Build a valid ``AcquisitionSeries`` for tests."""
    return AcquisitionSeries(
        name=name,
        data=np.arange(n),
        timestamps=np.arange(n, dtype=float),
        unit="second",
        description="desc",
    )


# ---------------------------------------------------------------------------
# AcquisitionSeries
# ---------------------------------------------------------------------------


def test_acquisition_series_valid():
    """Equal-length data and timestamps construct successfully."""
    series = _make_series()
    assert series.name == "series"
    assert series.unit == "second"
    assert series.description == "desc"
    assert series.data.shape[0] == series.timestamps.shape[0]


def test_acquisition_series_length_mismatch_raises():
    """Mismatched data and timestamps lengths raise a validation error."""
    with pytest.raises(ValueError, match="must have the same length"):
        AcquisitionSeries(
            name="bad",
            data=np.arange(3),
            timestamps=np.arange(2, dtype=float),
            unit="second",
            description="desc",
        )


def test_acquisition_series_is_frozen():
    """Instances are immutable."""
    series = _make_series()
    with pytest.raises(ValueError):
        series.name = "other"


# ---------------------------------------------------------------------------
# NWBAcquisition
# ---------------------------------------------------------------------------


def test_nwb_acquisition_holds_series():
    """``NWBAcquisition`` stores both reward delivery series."""
    left = _make_series("reward_delivery_left")
    right = _make_series("reward_delivery_right")
    acquisition = NWBAcquisition(
        reward_delivery_left=left,
        reward_delivery_right=right,
    )
    assert acquisition.reward_delivery_left is left
    assert acquisition.reward_delivery_right is right


def test_nwb_acquisition_is_frozen():
    """``NWBAcquisition`` instances are immutable."""
    acquisition = NWBAcquisition(
        reward_delivery_left=_make_series("reward_delivery_left"),
        reward_delivery_right=_make_series("reward_delivery_right"),
    )
    with pytest.raises(ValueError):
        acquisition.reward_delivery_left = _make_series("other")
