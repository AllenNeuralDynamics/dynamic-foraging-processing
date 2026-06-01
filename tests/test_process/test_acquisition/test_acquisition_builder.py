"""Tests for ``dynamic_foraging_processing.process.acquisition.acquisition_builder``."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from dynamic_foraging_processing.process.acquisition import AcqusitionBuilder
from dynamic_foraging_processing.process.acquisition.models import (
    AcquisitionSeries,
    NWBAcquisition,
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


def _make_dataset(frame: pd.DataFrame) -> MagicMock:
    """Build a mock dataset whose ``at`` chain resolves to ``frame``."""
    dataset = MagicMock()
    stream = MagicMock()
    stream.data = frame
    dataset.at.return_value.at.return_value.at.return_value.load.return_value = stream
    return dataset


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_dataset():
    """The provided dataset is stored on the instance."""
    ds = _make_dataset(_make_output_set_frame())
    builder = AcqusitionBuilder(dataset=ds)
    assert builder.dataset is ds


# ---------------------------------------------------------------------------
# get_reward_delivery
# ---------------------------------------------------------------------------


def test_get_reward_delivery_filters_to_write_messages():
    """Only ``MessageType == 'WRITE'`` rows are returned."""
    frame = _make_output_set_frame()
    ds = _make_dataset(frame)
    builder = AcqusitionBuilder(dataset=ds)

    result = builder.get_reward_delivery()

    assert list(result["MessageType"]) == ["WRITE", "WRITE"]
    assert list(result.index) == [0.1, 0.3]
    ds.at.assert_called_once_with("Behavior")
    ds.at.return_value.at.assert_called_once_with("HarpBehavior")
    ds.at.return_value.at.return_value.at.assert_called_once_with("OutputSet")


# ---------------------------------------------------------------------------
# build_acquisition
# ---------------------------------------------------------------------------


def test_build_acquisition_returns_populated_nwb_acquisition():
    """``build_acquisition`` returns an ``NWBAcquisition`` with both ports populated."""
    frame = _make_output_set_frame()
    ds = _make_dataset(frame)
    builder = AcqusitionBuilder(dataset=ds)

    acquisition = builder.build_acquisition()

    assert isinstance(acquisition, NWBAcquisition)

    left = acquisition.reward_delivery_left
    right = acquisition.reward_delivery_right
    assert isinstance(left, AcquisitionSeries)
    assert isinstance(right, AcquisitionSeries)

    expected_timestamps = np.array([0.1, 0.3])
    np.testing.assert_array_equal(left.data, np.array([1, 0]))
    np.testing.assert_array_equal(left.timestamps, expected_timestamps)
    assert left.name == "reward_delivery_left"
    assert left.unit == "second"
    assert "left lick port" in left.description

    np.testing.assert_array_equal(right.data, np.array([0, 1]))
    np.testing.assert_array_equal(right.timestamps, expected_timestamps)
    assert right.name == "reward_delivery_right"
    assert right.unit == "second"
    assert "right lick port" in right.description
