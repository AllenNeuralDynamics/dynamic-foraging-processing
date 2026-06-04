"""Tests for ``dynamic_foraging_processing.nwb.acquisition.acquisition_builder``."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from dynamic_foraging_processing.nwb.acquisition import AcquisitionBuilder
from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)


def _make_output_set_frame() -> pd.DataFrame:
    """Build an OutputSet-like DataFrame with WRITE and non-WRITE rows."""
    return pd.DataFrame(
        {
            "MessageType": ["WRITE", "READ", "WRITE"],
            "SupplyPort0": [1, 0, 0],
            "SupplyPort1": [0, 0, 1],
        },
        index=pd.Index([0.1, 0.2, 0.3], name="time"),
    )


def _make_loader(frame: pd.DataFrame) -> MagicMock:
    """Build a mock loader whose ``dataset.at`` chain resolves to ``frame``."""
    loader = MagicMock()
    stream = MagicMock()
    stream.data = frame
    loader.dataset.at.return_value.at.return_value.at.return_value.load.return_value = stream
    loader.get_all_raw_data.return_value = {
        "Behavior.RawStream": pd.DataFrame({"x": [1, 2, 3]}),
    }
    loader.raw_data_stream_descriptions = {"Behavior.RawStream": "raw stream desc"}
    return loader


def test_init_stores_loader():
    """The provided loader is stored on the instance."""
    loader = _make_loader(_make_output_set_frame())
    builder = AcquisitionBuilder(loader=loader)
    assert builder.loader is loader


def test_get_reward_delivery_filters_to_write_messages():
    """Only ``MessageType == 'WRITE'`` rows are returned."""
    frame = _make_output_set_frame()
    loader = _make_loader(frame)
    builder = AcquisitionBuilder(loader=loader)

    result = builder.get_reward_delivery()

    assert list(result["MessageType"]) == ["WRITE", "WRITE"]
    assert list(result.index) == [0.1, 0.3]
    ds = loader.dataset
    ds.at.assert_called_once_with("Behavior")
    ds.at.return_value.at.assert_called_once_with("HarpBehavior")
    ds.at.return_value.at.return_value.at.assert_called_once_with("OutputSet")


def test_build_acquisition_returns_populated_list():
    """``build_acquisition`` returns table entries plus both reward port series."""
    frame = _make_output_set_frame()
    loader = _make_loader(frame)
    builder = AcquisitionBuilder(loader=loader)

    acquisition = builder.build_acquisition()

    assert isinstance(acquisition, list)
    assert len(acquisition) == 3

    table, left, right = acquisition
    assert isinstance(table, AcquisitionTable)
    assert table.name == "Behavior.RawStream"
    assert table.description == "raw stream desc"
    assert list(table.data.columns) == ["x"]

    assert isinstance(left, AcquisitionSeries)
    assert isinstance(right, AcquisitionSeries)

    expected_timestamps = np.array([0.1, 0.3])
    np.testing.assert_array_equal(left.data, np.array([1, 0]))
    np.testing.assert_array_equal(left.timestamps, expected_timestamps)
    assert left.name == "left_reward_delivery_time"
    assert left.unit == "second"
    assert "left lick port" in left.description

    np.testing.assert_array_equal(right.data, np.array([0, 1]))
    np.testing.assert_array_equal(right.timestamps, expected_timestamps)
    assert right.name == "right_reward_delivery_time"
    assert right.unit == "second"
    assert "right lick port" in right.description
