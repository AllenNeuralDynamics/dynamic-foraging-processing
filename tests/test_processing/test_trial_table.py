"""Tests for ``dynamic_foraging_processing.processing._trial_table``."""

import numpy as np
import pandas as pd
import pytest
from aind_behavior_dynamic_foraging.task_logic import (
    AindDynamicForagingTaskLogic,
    AindDynamicForagingTaskParameters,
)
from aind_behavior_dynamic_foraging.task_logic.trial_generators import (
    CoupledTrialGeneratorSpec,
    CoupledWarmupTrialGeneratorSpec,
    TrialGeneratorCompositeSpec,
    UncoupledTrialGeneratorSpec,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialOutcome
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    Scalar,
    ScalarDistributionParameter,
    TruncationParameters,
)

from dynamic_foraging_processing.processing import TrialConfig, TrialTableBuilder


# --------------------------------------------------------------------------- #
# Fake dataset (tree-like contract stand-in)
# --------------------------------------------------------------------------- #
class _Stream:
    """Leaf node exposing ``load().data`` like a contraqctor stream."""

    def __init__(self, data):
        """Store the stream's payload."""
        self._data = data

    def load(self):
        """Return self, mirroring the contraqctor stream ``load()`` call."""
        return self

    @property
    def has_data(self):
        """Report data as available, mirroring a successfully loaded stream."""
        return True

    @property
    def data(self):
        """Return the stored payload."""
        return self._data


class _FailedStream:
    """Leaf node that loads but reports no data, like a stream that failed to read."""

    def load(self):
        """Return self, mirroring the contraqctor stream ``load()`` call."""
        return self

    @property
    def has_data(self):
        """Report no data, mirroring a stream whose read failed."""
        return False


class _Node:
    """Branch node exposing ``at(name)`` navigation."""

    def __init__(self, children):
        """Store the node's child streams/nodes by name."""
        self.children = children

    def at(self, name):
        """Return the child registered under ``name``."""
        return self.children[name]


def _events(timestamps, payloads):
    """Build a SoftwareEvents-style frame indexed by timestamp with a ``data`` column."""
    return pd.DataFrame({"data": payloads}, index=pd.Index(timestamps, name="timestamp"))


def _outcome(p_left, p_right, is_right_choice, is_rewarded, auto=None):
    """Build a serialized ``TrialOutcome`` payload (dict, as delivered by the reader)."""
    return {
        "trial": {
            "p_reward_left": p_left,
            "p_reward_right": p_right,
            "response_deadline_duration": 3.0,
            "reward_consumption_duration": 1.0,
            "quiescence_period_duration": 0.5,
            "inter_trial_interval_duration": 4.0,
            "is_auto_response_right": auto,
        },
        "is_right_choice": is_right_choice,
        "is_rewarded": is_rewarded,
    }


def _task_logic(quiescent_scalar=True):
    """Build a coupled-generator task logic with known distribution parameters."""
    quiescent = (
        Scalar(distribution_parameters=ScalarDistributionParameter(value=0.0))
        if quiescent_scalar
        else ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=1.0),
            truncation_parameters=TruncationParameters(min=0.0, max=1.0),
        )
    )
    spec = CoupledTrialGeneratorSpec(
        quiescent_duration=quiescent,
        inter_trial_interval_duration=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=0.2),
            truncation_parameters=TruncationParameters(min=1.0, max=10.0),
        ),
        block_length=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=0.05),
            truncation_parameters=TruncationParameters(min=20.0, max=60.0),
        ),
        min_block_reward=2,
    )
    return AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(trial_generator=spec)
    )


def _output_set():
    """Build a HarpBehavior OutputSet frame with left/right valve WRITE events."""
    return pd.DataFrame(
        {
            "SupplyPort0": [True, False, True],
            "SupplyPort1": [False, True, False],
            "MessageType": ["WRITE", "WRITE", "READ"],
        },
        index=pd.Index([12.0, 22.0, 12.5], name="Time"),
    )


def _full_dataset():
    """Assemble a two-trial fake dataset covering the common path."""
    software_events = _Node(
        {
            "TrialOutcome": _Stream(
                _events(
                    [10.1, 20.1],
                    [
                        _outcome(1.0, 0.2, is_right_choice=True, is_rewarded=True),
                        _outcome(0.5, 0.5, is_right_choice=None, is_rewarded=False),
                    ],
                )
            ),
            "QuiescentPeriod": _Stream(_events([10.0, 20.0], [None, None])),
            "ItiPeriod": _Stream(_events([20.0, 30.0], [None, None])),
            "Response": _Stream(_events([10.5, 20.5], [True, None])),
            "InitialManipulatorPosition": _Stream(
                _events([5.0], [{"x": 1.0, "y1": 2.0, "y2": 3.0, "z": 4.0}])
            ),
        }
    )
    behavior = _Node(
        {
            "SoftwareEvents": software_events,
            "HarpBehavior": _Node({"OutputSet": _Stream(_output_set())}),
            "HarpSoundCard": _Node(
                {
                    "PlaySoundOrFrequency": _Stream(
                        pd.DataFrame(
                            {"MessageType": ["WRITE", "WRITE"]},
                            index=pd.Index([11.0, 21.0], name="Time"),
                        )
                    )
                }
            ),
            "InputSchemas": _Node({"TaskLogic": _Stream(_task_logic())}),
        }
    )
    return _Node({"Behavior": behavior})


def _misaligned_dataset():
    """A two-trial dataset whose QuiescentPeriod (start) stream is one event short."""
    dataset = _full_dataset()
    software_events = dataset.children["Behavior"].children["SoftwareEvents"]
    # One start time for two TrialOutcome events -> positional misalignment.
    software_events.children["QuiescentPeriod"] = _Stream(_events([10.0], [None]))
    return dataset


# --------------------------------------------------------------------------- #
# build() — full happy path
# --------------------------------------------------------------------------- #
def test_build_full_dataset():
    """Two trials are assembled with the expected per-trial and session values."""
    table = TrialTableBuilder(_full_dataset()).build()

    assert list(table.columns) == list(TrialConfig.model_fields)
    assert len(table) == 2

    first, second = table.iloc[0], table.iloc[1]

    # Trial windows.
    assert first["start_time"] == 10.0 and first["stop_time"] == 20.0
    assert first["delay_start_time"] == 10.0

    # Response encoding: True -> right (1), None -> no response (2).
    assert first["animal_response"] == 1
    assert second["animal_response"] == 2

    # Hardware times resolved within each trial window.
    assert first["goCue_start_time"] == 11.0
    assert second["goCue_start_time"] == 21.0
    assert first["left_valve_open_time"] == 12.0
    assert np.isnan(first["right_valve_open_time"])
    assert second["right_valve_open_time"] == 22.0

    # Reward history split by side.
    assert bool(first["rewarded_historyR"]) is True
    assert bool(first["rewarded_historyL"]) is False
    # Second trial was ignored (no choice) -> not rewarded on either side.
    assert bool(second["rewarded_historyL"]) is False
    assert bool(second["rewarded_historyR"]) is False

    # Bait derived from p_reward and auto-response.
    assert bool(first["bait_left"]) is True
    assert bool(first["bait_right"]) is False

    # Session-level distribution summaries.
    assert first["ITI_beta"] == pytest.approx(5.0)
    assert first["ITI_min"] == 1.0 and first["ITI_max"] == 10.0
    assert first["block_beta"] == pytest.approx(20.0)
    assert pd.isna(first["delay_beta"])  # scalar quiescent distribution
    assert first["min_reward_each_block"] == 2
    assert first["base_reward_probability_sum"] == pytest.approx(0.8)

    # Lickspout positions from InitialManipulatorPosition.
    assert first["lickspout_position_x"] == 1.0
    assert first["lickspout_position_z"] == 4.0


def test_build_empty_dataset_returns_empty_frame():
    """Missing streams yield an empty table with the full column set."""
    table = TrialTableBuilder(_Node({}), raise_on_error=False).build()
    assert len(table) == 0
    assert list(table.columns) == list(TrialConfig.model_fields)


def test_build_exponential_quiescent_sets_delay_beta():
    """A non-scalar quiescent distribution populates delay beta/min/max."""
    behavior = _full_dataset().children["Behavior"]
    behavior.children["InputSchemas"].children["TaskLogic"] = _Stream(
        _task_logic(quiescent_scalar=False)
    )
    table = TrialTableBuilder(_Node({"Behavior": behavior})).build()
    assert table.iloc[0]["delay_beta"] == pytest.approx(1.0)
    assert table.iloc[0]["delay_min"] == 0.0
    assert table.iloc[0]["delay_max"] == 1.0


def test_build_warns_on_misaligned_streams(caplog):
    """A per-trial stream shorter than TrialOutcome warns but still builds."""
    table = TrialTableBuilder(_misaligned_dataset()).build()
    assert len(table) == 2
    assert "misaligned" in caplog.text.lower()


def test_build_raises_on_misaligned_streams_when_configured():
    """Misaligned per-trial streams raise ``ValueError`` when ``raise_on_error`` is True."""
    with pytest.raises(ValueError, match="misaligned"):
        TrialTableBuilder(_misaligned_dataset(), raise_on_error=True).build()


# --------------------------------------------------------------------------- #
# _summary_generator — composite trial generators
# --------------------------------------------------------------------------- #
def test_summary_generator_returns_single_generator_unchanged():
    """A non-composite generator (no ``.generators``) is returned as-is."""
    spec = _task_logic().task_parameters.trial_generator
    assert TrialTableBuilder._summary_generator(spec) is spec


def test_summary_generator_prefers_coupled_over_uncoupled():
    """A coupled sub-generator is preferred even when an uncoupled one precedes it."""
    coupled = _task_logic().task_parameters.trial_generator
    composite = TrialGeneratorCompositeSpec(generators=[UncoupledTrialGeneratorSpec(), coupled])
    resolved = TrialTableBuilder._summary_generator(composite)
    assert resolved.type == "CoupledTrialGenerator"
    assert resolved.min_block_reward == coupled.min_block_reward


def test_summary_generator_falls_back_to_uncoupled():
    """An uncoupled sub-generator is used when no coupled generator is present."""
    composite = TrialGeneratorCompositeSpec(
        generators=[CoupledWarmupTrialGeneratorSpec(), UncoupledTrialGeneratorSpec()]
    )
    resolved = TrialTableBuilder._summary_generator(composite)
    assert resolved.type == "UncoupledTrialGenerator"


def test_summary_generator_none_when_no_coupled_or_uncoupled(caplog):
    """A composite with neither coupled nor uncoupled generators yields ``None`` and warns."""
    composite = TrialGeneratorCompositeSpec(generators=[CoupledWarmupTrialGeneratorSpec()])
    assert TrialTableBuilder._summary_generator(composite) is None
    assert "No coupled or uncoupled generator" in caplog.text


def test_session_columns_empty_when_no_summary_generator():
    """A composite task logic with no coupled/uncoupled generator yields no session columns."""
    task_logic = AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            trial_generator=TrialGeneratorCompositeSpec(
                generators=[CoupledWarmupTrialGeneratorSpec()]
            )
        )
    )
    assert TrialTableBuilder(_Node({}))._session_columns(task_logic) == {}


def test_session_columns_uncoupled_has_null_reward_sum():
    """An uncoupled generator populates distribution columns but no coupled-only fields."""
    task_logic = AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            trial_generator=UncoupledTrialGeneratorSpec()
        )
    )
    columns = TrialTableBuilder(_Node({}))._session_columns(task_logic)
    # Distribution summaries are common to all block-based generators.
    assert "ITI_beta" in columns
    # Coupled-only fields are absent / null for an uncoupled generator.
    assert columns["base_reward_probability_sum"] is None
    assert "min_reward_each_block" not in columns


# --------------------------------------------------------------------------- #
# _load — error handling
# --------------------------------------------------------------------------- #
def test_load_missing_stream_returns_none_when_not_raising():
    """A missing stream returns ``None`` when ``raise_on_error`` is False."""
    builder = TrialTableBuilder(_Node({}), raise_on_error=False)
    assert builder._load("Behavior", "Missing") is None


def test_load_missing_stream_raises_when_configured():
    """A missing stream raises ``ValueError`` when ``raise_on_error`` is True."""
    builder = TrialTableBuilder(_Node({}), raise_on_error=True)
    with pytest.raises(ValueError, match="Failed to load stream"):
        builder._load("Behavior", "Missing")


def test_load_stream_failed_to_load_returns_none():
    """A stream that loads without data returns ``None`` when not raising."""
    builder = TrialTableBuilder(_Node({"Broken": _FailedStream()}), raise_on_error=False)
    assert builder._load("Broken") is None


def test_load_stream_failed_to_load_raises_when_configured():
    """A stream that loads without data raises ``ValueError`` when configured."""
    builder = TrialTableBuilder(_Node({"Broken": _FailedStream()}), raise_on_error=True)
    with pytest.raises(ValueError, match="stream failed to load"):
        builder._load("Broken")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def test_closest_time_in_window_picks_nearest_to_start():
    """Among events in the window, the one nearest the start is returned."""
    times = np.array([5.0, 11.0, 12.0, 25.0])
    assert TrialTableBuilder._closest_time_in_window(times, 10.0, 20.0) == 11.0


def test_closest_time_in_window_none_when_empty_or_outside():
    """Returns ``None`` for empty input or when no event falls in the window."""
    assert TrialTableBuilder._closest_time_in_window(np.empty(0), 0.0, 1.0) is None
    assert TrialTableBuilder._closest_time_in_window(np.array([5.0, 25.0]), 10.0, 20.0) is None


def test_valve_open_times_missing_column_returns_empty():
    """A missing supply-port column yields no valve times."""
    df = pd.DataFrame({"MessageType": ["WRITE"]}, index=[1.0])
    assert TrialTableBuilder._valve_open_times(df, "SupplyPort0").size == 0


def test_write_times_without_message_type_uses_all_rows():
    """A register without a ``MessageType`` column returns every timestamp."""
    df = pd.DataFrame({"value": [1, 2]}, index=[3.0, 1.0])
    np.testing.assert_array_equal(TrialTableBuilder._write_times(df), np.array([1.0, 3.0]))


def test_animal_response_encoding():
    """``Response`` payloads map to the documented 0/1/2 codes."""
    assert TrialTableBuilder._animal_response(None) == 2
    assert TrialTableBuilder._animal_response(True) == 1
    assert TrialTableBuilder._animal_response(False) == 0


def test_is_baited_none_trial_returns_false():
    """A missing trial yields ``False`` bait for both sides."""
    assert TrialTableBuilder._is_baited(None, is_right=True) is False
    assert TrialTableBuilder._is_baited(None, is_right=False) is False


def test_is_baited_forfeited_by_auto_response_on_same_side():
    """A side with guaranteed reward stays baited unless auto-responded to that side."""
    trial = TrialOutcome.model_validate(
        _outcome(0.0, 1.0, is_right_choice=True, is_rewarded=True, auto=True)
    ).trial
    # Right is guaranteed (p=1) but auto-responded right -> bait collected.
    assert TrialTableBuilder._is_baited(trial, is_right=True) is False


def test_auto_water_encodes_side_from_auto_response():
    """A non-null auto response encodes ``1`` on its side and ``0`` on the other."""
    trial = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, auto=True)
    ).trial
    assert TrialTableBuilder._auto_water(trial, is_right=True) == 1
    assert TrialTableBuilder._auto_water(trial, is_right=False) == 0
    assert TrialTableBuilder._auto_water(None, is_right=True) is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True),
        TrialOutcome.model_validate(_outcome(1.0, 1.0, is_right_choice=False, is_rewarded=False)),
    ],
)
def test_parse_outcome_accepts_dict_model_and_none(payload):
    """``_parse_outcome`` accepts dicts, model instances, and ``None``."""
    result = TrialTableBuilder._parse_outcome(payload)
    assert result is None or isinstance(result, TrialOutcome)


def test_parse_outcome_accepts_json_string():
    """``_parse_outcome`` parses a JSON-string payload."""
    payload = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True)
    ).model_dump_json()
    assert isinstance(TrialTableBuilder._parse_outcome(payload), TrialOutcome)


def test_lickspout_columns_from_list_payload():
    """A list payload is mapped positionally to x/y1/y2/z."""
    builder = TrialTableBuilder(_Node({}))
    columns = builder._lickspout_columns(_events([1.0], [[10.0, 20.0, 30.0, 40.0]]))
    assert columns["lickspout_position_x"] == 10.0
    assert columns["lickspout_position_y2"] == 30.0


def test_lickspout_columns_unrecognized_payload_is_null():
    """An unrecognized payload leaves all lickspout positions ``None``."""
    builder = TrialTableBuilder(_Node({}))
    columns = builder._lickspout_columns(_events([1.0], ["unexpected"]))
    assert all(value is None for value in columns.values())


def test_lickspout_columns_missing_stream_is_null():
    """A missing manipulator stream leaves all lickspout positions ``None``."""
    builder = TrialTableBuilder(_Node({}))
    columns = builder._lickspout_columns(None)
    assert all(value is None for value in columns.values())
