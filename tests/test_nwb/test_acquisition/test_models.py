"""Tests for ``dynamic_foraging_processing.nwb.acquisition.models``."""

import numpy as np
import pandas as pd
import pytest

from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
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


def _make_table(name: str = "table") -> AcquisitionTable:
    """Build a valid ``AcquisitionTable`` for tests."""
    return AcquisitionTable(
        name=name,
        data=pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        description="desc",
    )


def test_acquisition_table_valid():
    """``AcquisitionTable`` constructs successfully with a DataFrame."""
    table = _make_table()
    assert table.name == "table"
    assert table.description == "desc"
    assert list(table.data.columns) == ["a", "b"]
    assert len(table.data) == 2


def test_acquisition_table_is_frozen():
    """``AcquisitionTable`` instances are immutable."""
    table = _make_table()
    with pytest.raises(ValueError):
        table.name = "other"
