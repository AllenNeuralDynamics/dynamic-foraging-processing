"""Tests for ``dynamic_foraging_processing.nwb.acquisition.acquisition_builder``."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from dynamic_foraging_processing.nwb.acquisition import AcquisitionBuilder
from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)


class _FakeStream:
    """Leaf stream exposing ``load()``/``data`` like the contract."""

    def __init__(self, data):
        """Store the stream's data."""
        self._data = data

    def load(self):
        """Return self (data is already in memory)."""
        return self

    @property
    def data(self):
        """Return the stored data."""
        return self._data


class _FakeNode:
    """Inner dataset node whose ``at`` navigates to named children."""

    def __init__(self, children):
        """Store the child nodes by name."""
        self._children = children

    def at(self, name):
        """Return the named child, raising ``KeyError`` if absent."""
        return self._children[name]


def _make_output_set_frame() -> pd.DataFrame:
    """Build an OutputSet-like DataFrame with WRITE and non-WRITE rows.

    Left valve opens once (SupplyPort0 high at 0.1); the right valve opens twice
    (SupplyPort1 high at 0.3 and 0.5).
    """
    return pd.DataFrame(
        {
            "MessageType": ["WRITE", "READ", "WRITE", "WRITE"],
            "SupplyPort0": [1, 0, 0, 0],
            "SupplyPort1": [0, 0, 1, 1],
        },
        index=pd.Index([0.1, 0.2, 0.3, 0.5], name="time"),
    )


def _outcome_payload(auto) -> dict:
    """Return a serialized ``TrialOutcome`` payload with the given auto-response."""
    return {
        "trial": {
            "p_reward_left": 1.0,
            "p_reward_right": 1.0,
            "response_deadline_duration": 3.0,
            "reward_consumption_duration": 1.0,
            "quiescence_period_duration": 0.5,
            "inter_trial_interval_duration": 4.0,
            "is_auto_response_right": auto,
        },
        "is_right_choice": True,
        "is_rewarded": True,
    }


def _make_trial_outcome_frame() -> pd.DataFrame:
    """Two trials: an earned trial at 0.1 and an auto-response trial at 0.4."""
    return pd.DataFrame(
        {"data": [_outcome_payload(None), _outcome_payload(True)]},
        index=pd.Index([0.1, 0.4], name="time"),
    )


def _empty_manual_water_frame() -> pd.DataFrame:
    """Build an empty manual-water stream with the ``data`` side column."""
    return pd.DataFrame({"data": []}, index=pd.Index([], name="time"))


def _make_dataset(manual_water=None):
    """Build a path-aware fake dataset rooted at ``Behavior``."""
    if manual_water is None:
        manual_water = _empty_manual_water_frame()
    return _FakeNode(
        {
            "Behavior": _FakeNode(
                {
                    "HarpBehavior": _FakeNode({"OutputSet": _FakeStream(_make_output_set_frame())}),
                    "SoftwareEvents": _FakeNode(
                        {
                            "TrialOutcome": _FakeStream(_make_trial_outcome_frame()),
                            "GiveManualWaterRight": _FakeStream(manual_water),
                        }
                    ),
                }
            )
        }
    )


def _make_loader(dataset=None) -> MagicMock:
    """Build a mock loader backed by a path-aware fake dataset."""
    loader = MagicMock()
    loader.dataset = dataset if dataset is not None else _make_dataset()
    loader.get_all_raw_data.return_value = {
        "Behavior.RawStream": pd.DataFrame({"x": [1, 2, 3]}),
    }
    loader.raw_data_stream_descriptions = {"Behavior.RawStream": "raw stream desc"}
    return loader


def test_init_stores_loader():
    """The provided loader is stored on the instance."""
    loader = _make_loader()
    builder = AcquisitionBuilder(loader=loader)
    assert builder.loader is loader


def test_get_reward_delivery_filters_to_write_messages():
    """Only ``MessageType == 'WRITE'`` rows are returned."""
    builder = AcquisitionBuilder(loader=_make_loader())

    result = builder.get_reward_delivery()

    assert list(result["MessageType"]) == ["WRITE", "WRITE", "WRITE"]
    assert list(result.index) == [0.1, 0.3, 0.5]


def test_get_manual_water_times_returns_stream():
    """``get_manual_water_times`` returns the GiveManualWaterRight stream."""
    manual = pd.DataFrame({"data": [True]}, index=pd.Index([0.49], name="time"))
    builder = AcquisitionBuilder(loader=_make_loader(_make_dataset(manual)))

    result = builder.get_manual_water_times()

    pd.testing.assert_frame_equal(result, manual)


def test_get_manual_water_times_returns_empty_when_absent():
    """A missing manual-water stream yields an empty frame with a ``data`` column."""
    dataset = _FakeNode(
        {
            "Behavior": _FakeNode(
                {
                    "HarpBehavior": _FakeNode({"OutputSet": _FakeStream(_make_output_set_frame())}),
                    "SoftwareEvents": _FakeNode(
                        {"TrialOutcome": _FakeStream(_make_trial_outcome_frame())}
                    ),
                }
            )
        }
    )
    builder = AcquisitionBuilder(loader=_make_loader(dataset))

    result = builder.get_manual_water_times()

    assert list(result.columns) == ["data"]
    assert result.empty


def test_build_acquisition_returns_populated_list():
    """``build_acquisition`` returns table entries plus both reward port series."""
    # A right-side manual-water event (data=True) near the second right delivery.
    manual = pd.DataFrame({"data": [True]}, index=pd.Index([0.49], name="time"))
    builder = AcquisitionBuilder(loader=_make_loader(_make_dataset(manual)))

    acquisition = builder.build_acquisition()

    assert isinstance(acquisition, list)
    assert len(acquisition) == 3

    table, left, right = acquisition
    assert isinstance(table, AcquisitionTable)
    assert table.name == "Behavior.RawStream"
    assert table.description == "raw stream desc"

    # Left: one valve-open delivery, earned (no auto-response, no left manual).
    assert isinstance(left, AcquisitionSeries)
    np.testing.assert_array_equal(left.timestamps, np.array([0.1]))
    np.testing.assert_array_equal(left.data, np.array(["earned"]))
    assert left.name == "left_reward_delivery_time"
    assert left.unit == "second"
    assert "left lick port" in left.description

    # Right: two deliveries. Both nearest the auto-response trial (0.4); the
    # second is overridden to manual by the right-side manual-water event.
    assert isinstance(right, AcquisitionSeries)
    np.testing.assert_array_equal(right.timestamps, np.array([0.3, 0.5]))
    np.testing.assert_array_equal(right.data, np.array(["automatic", "manual"]))
    assert right.name == "right_reward_delivery_time"
    assert "right lick port" in right.description
