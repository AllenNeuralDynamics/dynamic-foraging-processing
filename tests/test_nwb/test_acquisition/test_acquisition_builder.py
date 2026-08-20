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
    """Return a serialized ``TrialOutcome`` payload with the given auto-response.

    Free water is flagged as scheduled autowater in ``metadata.extra`` so the
    annotation can attribute it; the channel alone does not name the mechanism.
    """
    return {
        "trial": {
            "p_reward_left": 1.0,
            "p_reward_right": 1.0,
            "response_deadline_duration": 3.0,
            "reward_consumption_duration": 1.0,
            "quiescence_period_duration": 0.5,
            "inter_trial_interval_duration": 4.0,
            "is_auto_reward_right": auto,
            "metadata": {"extra": {"is_autowater": auto is not None}},
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


def _make_response_frame() -> pd.DataFrame:
    """One ``Response`` event per trial, just before each trial's outcome."""
    return pd.DataFrame(
        {"data": [{"Item1": 0.05, "Item2": False}, {"Item1": 0.35, "Item2": True}]},
        index=pd.Index([0.05, 0.35], name="time"),
    )


def _empty_manual_water_frame() -> pd.DataFrame:
    """Build an empty manual-water stream with the ``data`` side column."""
    return pd.DataFrame({"data": []}, index=pd.Index([], name="time"))


def _make_digital_input_frame() -> pd.DataFrame:
    """Build a DigitalInputState frame: left licks high at 1.0, right at 2.0/2.5.

    Includes a non-EVENT ``READ`` row at 0.5 with ``DIPort0`` latched high; it
    reports register state rather than a lick and must be filtered out.
    """
    return pd.DataFrame(
        {
            "MessageType": ["READ", "EVENT", "EVENT", "EVENT"],
            "DIPort0": [True, True, False, False],
            "DIPort1": [False, False, True, True],
        },
        index=pd.Index([0.5, 1.0, 2.0, 2.5], name="time"),
    )


def _make_dataset(manual_water=None):
    """Build a path-aware fake dataset rooted at ``Behavior``."""
    if manual_water is None:
        manual_water = _empty_manual_water_frame()
    return _FakeNode(
        {
            "Behavior": _FakeNode(
                {
                    "HarpBehavior": _FakeNode(
                        {
                            "OutputSet": _FakeStream(_make_output_set_frame()),
                            "DigitalInputState": _FakeStream(_make_digital_input_frame()),
                        }
                    ),
                    "SoftwareEvents": _FakeNode(
                        {
                            "TrialOutcome": _FakeStream(_make_trial_outcome_frame()),
                            "Response": _FakeStream(_make_response_frame()),
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


def test_get_valve_writes_filters_to_write_messages():
    """Only ``MessageType == 'WRITE'`` rows are returned."""
    builder = AcquisitionBuilder(loader=_make_loader())

    result = builder.get_valve_writes()

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
                        {
                            "TrialOutcome": _FakeStream(_make_trial_outcome_frame()),
                            "Response": _FakeStream(_make_response_frame()),
                        }
                    ),
                }
            )
        }
    )
    builder = AcquisitionBuilder(loader=_make_loader(dataset))

    result = builder.get_manual_water_times()

    assert list(result.columns) == ["data"]
    assert result.empty


def test_get_lick_times_selects_di_port_by_side():
    """Left licks come from DIPort0 and right licks from DIPort1."""
    builder = AcquisitionBuilder(loader=_make_loader())

    np.testing.assert_array_equal(
        builder.get_lick_times("HarpBehavior", "DigitalInputState", "DIPort0"), np.array([1.0])
    )
    np.testing.assert_array_equal(
        builder.get_lick_times("HarpBehavior", "DigitalInputState", "DIPort1"), np.array([2.0, 2.5])
    )


def test_get_lick_times_ignores_non_event_rows():
    """Only EVENT rows are licks; latched-high READ/WRITE rows are ignored."""
    builder = AcquisitionBuilder(loader=_make_loader())

    # DIPort0 is high in the READ row at 0.5 and the EVENT row at 1.0; only the
    # EVENT-row timestamp is returned.
    np.testing.assert_array_equal(
        builder.get_lick_times("HarpBehavior", "DigitalInputState", "DIPort0"), np.array([1.0])
    )


def test_get_lick_times_returns_empty_when_absent():
    """A missing DigitalInputState stream yields an empty array."""
    dataset = _FakeNode(
        {
            "Behavior": _FakeNode(
                {
                    "HarpBehavior": _FakeNode({"OutputSet": _FakeStream(_make_output_set_frame())}),
                    "SoftwareEvents": _FakeNode(
                        {
                            "TrialOutcome": _FakeStream(_make_trial_outcome_frame()),
                            "Response": _FakeStream(_make_response_frame()),
                            "GiveManualWaterRight": _FakeStream(_empty_manual_water_frame()),
                        }
                    ),
                }
            )
        }
    )
    builder = AcquisitionBuilder(loader=_make_loader(dataset))

    result = builder.get_lick_times("HarpBehavior", "DigitalInputState", "DIPort1")

    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_build_acquisition_returns_populated_list():
    """``build_acquisition`` returns the table plus reward and lick port series."""
    # A right-side manual-water event (data=True) near the second right delivery.
    manual = pd.DataFrame({"data": [True]}, index=pd.Index([0.49], name="time"))
    builder = AcquisitionBuilder(loader=_make_loader(_make_dataset(manual)))

    acquisition = builder.build_acquisition()

    assert isinstance(acquisition, list)
    assert len(acquisition) == 5

    table, left_reward, right_reward, left_lick, right_lick = acquisition
    assert isinstance(table, AcquisitionTable)
    assert table.name == "Behavior.RawStream"
    assert table.description == "raw stream desc"

    # Left reward: one valve-open delivery, earned (no auto-response, no left manual).
    assert isinstance(left_reward, AcquisitionSeries)
    np.testing.assert_array_equal(left_reward.timestamps, np.array([0.1]))
    np.testing.assert_array_equal(left_reward.data, np.array(["earned"]))
    assert left_reward.name == "left_reward_delivery_time"
    assert left_reward.unit == "second"
    assert "left lick port" in left_reward.description

    # Right reward: two deliveries. Both nearest the auto-response trial (0.4);
    # the second is overridden to manual by the right-side manual-water event.
    assert isinstance(right_reward, AcquisitionSeries)
    np.testing.assert_array_equal(right_reward.timestamps, np.array([0.3, 0.5]))
    np.testing.assert_array_equal(right_reward.data, np.array(["auto", "manual"]))
    assert right_reward.name == "right_reward_delivery_time"
    assert "right lick port" in right_reward.description

    # Lick series: timestamps from the DI ports, data marks each as a lick.
    assert left_lick.name == "left_lick_time"
    np.testing.assert_array_equal(left_lick.timestamps, np.array([1.0]))
    np.testing.assert_array_equal(left_lick.data, np.array([True]))
    assert "DIPort0" in left_lick.description

    assert right_lick.name == "right_lick_time"
    np.testing.assert_array_equal(right_lick.timestamps, np.array([2.0, 2.5]))
    np.testing.assert_array_equal(right_lick.data, np.array([True, True]))
    assert "DIPort1" in right_lick.description


def test_build_acquisition_defaults_none_description_to_empty_string():
    """A None stream description falls back to "" so the table validates."""
    loader = _make_loader()
    loader.raw_data_stream_descriptions = {"Behavior.RawStream": None}
    builder = AcquisitionBuilder(loader=loader)

    table = builder.build_acquisition()[0]

    assert table.description == ""
