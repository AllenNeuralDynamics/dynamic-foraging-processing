"""Tests for ``dynamic_foraging_processing.processing._trial_table``."""

from types import SimpleNamespace

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
from aind_behavior_services.rig.aind_manipulator import Axis, MicrostepResolution
from aind_behavior_services.task.distributions import (
    ExponentialDistribution,
    ExponentialDistributionParameters,
    Scalar,
    ScalarDistributionParameter,
    ScalingParameters,
    TruncationParameters,
    UniformDistribution,
    UniformDistributionParameters,
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


def _outcome(
    p_left,
    p_right,
    is_right_choice,
    is_rewarded,
    auto=None,
    block_p_left=None,
    block_p_right=None,
    reward_size_left=None,
    reward_size_right=None,
    lickspout_offset_delta=None,
    is_autowater=None,
    is_bias_water_intervention=None,
    is_bias_stage_intervention=None,
    is_left_baited=None,
    is_right_baited=None,
):
    """Build a serialized ``TrialOutcome`` payload (dict, as delivered by the reader).

    ``p_left`` / ``p_right`` are the per-trial probabilities (top-level ``trial``
    fields); ``block_p_left`` / ``block_p_right``, when given, are the block
    probabilities stored under ``trial.metadata`` (the source of the
    ``reward_probability`` columns). ``reward_size_left`` / ``reward_size_right``
    override the default per-trial reward volumes (uL). ``lickspout_offset_delta``
    sets the per-trial horizontal spout displacement (mm), and ``is_autowater``
    plus the ``is_bias_*_intervention`` flags populate the ``metadata.extra``
    (``BlockBasedTrialMetadata``) block naming the free-water mechanism.
    ``is_left_baited`` / ``is_right_baited`` set the per-side bait flags in that
    same block (the source of the ``bait_*`` columns).
    """
    trial = {
        "p_reward_left": p_left,
        "p_reward_right": p_right,
        "response_deadline_duration": 3.0,
        "reward_consumption_duration": 1.0,
        "quiescence_period_duration": 0.5,
        "inter_trial_interval_duration": 4.0,
        "is_auto_reward_right": auto,
    }
    if lickspout_offset_delta is not None:
        trial["lickspout_offset_delta"] = lickspout_offset_delta
    if reward_size_left is not None or reward_size_right is not None:
        trial["reward_size"] = {
            "left": reward_size_left if reward_size_left is not None else 2.0,
            "right": reward_size_right if reward_size_right is not None else 2.0,
        }
    if block_p_left is not None or block_p_right is not None:
        trial["metadata"] = {"p_reward_left": block_p_left, "p_reward_right": block_p_right}
    if (
        is_autowater is not None
        or is_bias_water_intervention is not None
        or is_bias_stage_intervention is not None
        or is_left_baited is not None
        or is_right_baited is not None
    ):
        metadata = trial.setdefault("metadata", {})
        metadata["extra"] = {
            "is_autowater": bool(is_autowater),
            "is_bias_water_intervention": bool(is_bias_water_intervention),
            "is_bias_stage_intervention": bool(is_bias_stage_intervention),
            "is_left_baited": bool(is_left_baited),
            "is_right_baited": bool(is_right_baited),
        }
    return {
        "trial": trial,
        "is_right_choice": is_right_choice,
        "is_rewarded": is_rewarded,
    }


def _quiescent_distribution(kind):
    """Build the quiescent-duration distribution for the requested family.

    ``"uniform"`` deliberately also carries truncation parameters that differ
    from its own bounds, so tests can pin down which pair is reported.
    """
    if kind == "scalar":
        return Scalar(distribution_parameters=ScalarDistributionParameter(value=0.0))
    if kind == "uniform":
        return UniformDistribution(
            distribution_parameters=UniformDistributionParameters(min=0.25, max=0.75),
            truncation_parameters=TruncationParameters(min=9.0, max=99.0),
        )
    return ExponentialDistribution(
        distribution_parameters=ExponentialDistributionParameters(rate=1.0),
        truncation_parameters=TruncationParameters(min=0.0, max=1.0),
    )


def _task_logic(quiescent="scalar"):
    """Build a coupled-generator task logic with known distribution parameters."""
    quiescent = _quiescent_distribution(quiescent)
    spec = CoupledTrialGeneratorSpec(
        quiescent_duration=quiescent,
        inter_trial_interval_duration=ExponentialDistribution(
            distribution_parameters=ExponentialDistributionParameters(rate=0.2),
            truncation_parameters=TruncationParameters(min=1.0, max=10.0),
            scaling_parameters=ScalingParameters(offset=0.5),
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


def _pulse_supply(column, value_ms):
    """Build a HarpBehavior PulseSupplyPort register frame with one config value.

    The register reports its pulse width (ms) as a single ``READ`` row, mirroring
    a real session where the valve open time is configured once.
    """
    return pd.DataFrame(
        {column: [value_ms], "MessageType": ["READ"]},
        index=pd.Index([12.0], name="Time"),
    )


def _rig(step_mm=0.01, resolution=MicrostepResolution.MICROSTEP8):
    """Build a rig stub exposing the manipulator calibration the builder reads.

    A ``SimpleNamespace`` mirrors the ``AindDynamicForagingRig`` attribute path
    (``manipulator.calibration.full_step_to_mm`` / ``axis_configuration``) used by
    the builder, with real ``Axis`` / ``MicrostepResolution`` enums.
    """
    full_step_to_mm = SimpleNamespace(x=step_mm, y1=step_mm, y2=step_mm, z=step_mm)
    axis_configuration = [
        SimpleNamespace(axis=axis, microstep_resolution=resolution)
        for axis in (Axis.X, Axis.Y1, Axis.Y2, Axis.Z)
    ]
    calibration = SimpleNamespace(
        full_step_to_mm=full_step_to_mm, axis_configuration=axis_configuration
    )
    return SimpleNamespace(manipulator=SimpleNamespace(calibration=calibration))


def _accumulated_steps(times, motor_steps):
    """Build an ``AccumulatedSteps`` frame (Motor0..Motor3 microstep counts).

    ``motor_steps`` is a sequence of ``(m0, m1, m2, m3)`` per timestamp; all rows
    are ``EVENT`` messages, mirroring the HarpManipulator stream.
    """
    columns = {f"Motor{i}": [row[i] for row in motor_steps] for i in range(4)}
    columns["MessageType"] = ["EVENT"] * len(times)
    return pd.DataFrame(columns, index=pd.Index(times, name="Time"))


def _full_dataset():
    """Assemble a two-trial fake dataset covering the common path."""
    software_events = _Node(
        {
            "TrialOutcome": _Stream(
                _events(
                    [10.1, 20.1],
                    [
                        _outcome(
                            1.0,
                            0.2,
                            is_right_choice=True,
                            is_rewarded=True,
                            block_p_left=0.7,
                            block_p_right=0.1,
                            reward_size_left=2.0,
                            reward_size_right=4.0,
                            is_left_baited=True,
                        ),
                        _outcome(0.5, 0.5, is_right_choice=None, is_rewarded=False),
                    ],
                )
            ),
            # The four period streams, each emitted at its period's start:
            # quiescent -> response -> reward consumption -> ITI, per trial.
            "QuiescentPeriod": _Stream(_events([10.0, 20.0], [None, None])),
            "ResponsePeriod": _Stream(_events([11.0, 21.0], [None, None])),
            "RewardConsumptionPeriod": _Stream(_events([12.0, 22.0], [None, None])),
            "ItiPeriod": _Stream(_events([15.0, 25.0], [None, None])),
            "Response": _Stream(
                _events(
                    [10.5, 20.5], [{"Item1": 10.5, "Item2": True}, {"Item1": 20.5, "Item2": None}]
                )
            ),
            "TrialMetrics": _Stream(_events([10.2, 20.2], [{"bias": 0.3}, {"bias": None}])),
            # Closes the last trial's ITI, which has no following QuiescentPeriod.
            "EndSession": _Stream(_events([30.0], [None])),
        }
    )
    behavior = _Node(
        {
            "SoftwareEvents": software_events,
            "HarpBehavior": _Node(
                {
                    "PulseSupplyPort0": _Stream(_pulse_supply("PulseSupplyPort0", 20)),
                    "PulseSupplyPort1": _Stream(_pulse_supply("PulseSupplyPort1", 30)),
                }
            ),
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
            "HarpManipulator": _Node(
                {
                    # 0.00125 mm/microstep (0.01 / MICROSTEP8). One sample falls
                    # inside each trial window ([10, 20) and [20, 30)); x moves
                    # 10.0 -> 15.0 between the two trials.
                    "AccumulatedSteps": _Stream(
                        _accumulated_steps(
                            [10.5, 20.5],
                            [(8000, 1600, 2400, 4000), (12000, 1600, 2400, 4000)],
                        )
                    )
                }
            ),
            "InputSchemas": _Node({"TaskLogic": _Stream(_task_logic()), "Rig": _Stream(_rig())}),
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

    # Period bounds: each period ends where the next one starts, and the ITI
    # ends at the next trial's quiescent period — or, on the last trial, at the
    # EndSession timestamp.
    assert first["quiescent_start_time"] == 10.0 and first["quiescent_stop_time"] == 11.0
    assert first["response_start_time"] == 11.0 and first["response_stop_time"] == 12.0
    assert first["reward_consumption_start_time"] == 12.0
    assert first["reward_consumption_stop_time"] == 15.0
    assert first["ITI_start_time"] == 15.0 and first["ITI_stop_time"] == 20.0
    assert second["ITI_start_time"] == 25.0 and second["ITI_stop_time"] == 30.0
    # delay_start_time is the legacy name for the quiescent period start.
    assert first["delay_start_time"] == first["quiescent_start_time"] == 10.0

    # Response encoding: True -> right (1), None -> no response (2).
    assert first["animal_response"] == 1
    assert second["animal_response"] == 2

    # Go cue resolved within each trial window.
    assert first["goCue_start_time"] == 11.0
    assert second["goCue_start_time"] == 21.0
    # Valve columns hold the configured pulse width (ms -> s), constant per trial.
    assert first["left_valve_open_time"] == 0.02 and second["left_valve_open_time"] == 0.02
    assert first["right_valve_open_time"] == 0.03 and second["right_valve_open_time"] == 0.03

    # Reward history split by side.
    assert bool(first["rewarded_historyR"]) is True
    assert bool(first["rewarded_historyL"]) is False
    # Second trial was ignored (no choice) -> not rewarded on either side.
    assert bool(second["rewarded_historyL"]) is False
    assert bool(second["rewarded_historyR"]) is False

    # Bait read from the acquisition metadata's per-side flags.
    assert bool(first["bait_left"]) is True
    assert bool(first["bait_right"]) is False
    # The second trial carries no extra metadata -> not baited on either side.
    assert bool(second["bait_left"]) is False
    assert bool(second["bait_right"]) is False

    # reward_probability columns are the block probability from trial.metadata,
    # not the top-level per-trial p_reward (1.0 / 0.2 here).
    assert first["reward_probabilityL"] == pytest.approx(0.7)
    assert first["reward_probabilityR"] == pytest.approx(0.1)
    # The second trial has no metadata -> null block probabilities.
    assert pd.isna(second["reward_probabilityL"])
    assert pd.isna(second["reward_probabilityR"])

    # Session-level distribution summaries.
    assert first["ITI_beta"] == pytest.approx(5.0)
    # ``ITI_min`` comes from the scaling offset, not the truncation minimum.
    assert first["ITI_min"] == 0.5 and first["ITI_max"] == 10.0
    assert first["block_beta"] == pytest.approx(20.0)
    # Only block_max takes the floor adjustment: the configured 20/60 truncation
    # yields a longest realizable block of 59 trials.
    assert first["block_min"] == 20.0 and first["block_max"] == 59.0
    assert pd.isna(first["delay_beta"])  # scalar quiescent distribution
    # Scalar has neither a scale nor truncation parameters -> null bounds.
    assert pd.isna(first["delay_min"])
    assert pd.isna(first["delay_max"])
    # No per-block reward minimum on this generator -> a floor of 0, not null.
    assert first["min_reward_each_block"] == 0
    assert first["base_reward_probability_sum"] == pytest.approx(0.8)

    # Lickspout positions from AccumulatedSteps (microsteps * 0.00125 mm),
    # sampled at each trial start and re-referenced to session start. The first
    # sample is the baseline (all zero); x then moves +5.0 mm for the second trial.
    assert first["lickspout_position_x"] == 0.0
    assert first["lickspout_position_y1"] == 0.0
    assert first["lickspout_position_y2"] == 0.0
    assert first["lickspout_position_z"] == 0.0
    assert second["lickspout_position_x"] == 5.0

    # Per-trial reward volumes (uL) from trial.reward_size; second trial uses the default.
    assert first["reward_size_left"] == 2.0
    assert first["reward_size_right"] == 4.0
    assert second["reward_size_left"] == 2.0
    assert second["reward_size_right"] == 2.0

    # Per-trial side bias from the TrialMetrics event; null when not recorded.
    assert first["side_bias"] == pytest.approx(0.3)
    assert pd.isna(second["side_bias"])

    # No anti-bias interventions in the base fixture -> inert defaults.
    assert bool(first["anti_bias_left_water"]) is False
    assert bool(first["anti_bias_right_water"]) is False
    assert first["anti_bias_lickspout_movement"] == 0.0
    assert bool(second["anti_bias_left_water"]) is False
    assert second["anti_bias_lickspout_movement"] == 0.0


def test_build_missing_task_logic_leaves_session_columns_null():
    """A missing TaskLogic stream still builds; session distribution columns are null."""
    dataset = _full_dataset()
    input_schemas = dataset.children["Behavior"].children["InputSchemas"]
    input_schemas.children["TaskLogic"] = _FailedStream()
    table = TrialTableBuilder(dataset).build()
    assert len(table) == 2
    assert table["ITI_beta"].isna().all()
    assert table["block_beta"].isna().all()


def test_build_raises_when_rig_missing_with_trials():
    """A missing Rig stream is an error when there are trials (lickspout position needs it)."""
    dataset = _full_dataset()
    input_schemas = dataset.children["Behavior"].children["InputSchemas"]
    input_schemas.children["Rig"] = _FailedStream()
    with pytest.raises(ValueError, match="Rig stream is required"):
        TrialTableBuilder(dataset).build()


def test_build_raises_when_accumulated_steps_missing_with_trials():
    """A missing AccumulatedSteps stream is an error when there are trials."""
    dataset = _full_dataset()
    harp_manipulator = dataset.children["Behavior"].children["HarpManipulator"]
    harp_manipulator.children["AccumulatedSteps"] = _FailedStream()
    with pytest.raises(ValueError, match="AccumulatedSteps stream is required"):
        TrialTableBuilder(dataset).build()


def test_build_empty_dataset_returns_empty_frame():
    """Missing streams yield an empty table with the full column set."""
    table = TrialTableBuilder(_Node({}), raise_on_error=False).build()
    assert len(table) == 0
    assert list(table.columns) == list(TrialConfig.model_fields)


def test_build_exponential_quiescent_sets_delay_beta():
    """A non-scalar quiescent distribution populates delay beta/min/max."""
    behavior = _full_dataset().children["Behavior"]
    behavior.children["InputSchemas"].children["TaskLogic"] = _Stream(
        _task_logic(quiescent="exponential")
    )
    table = TrialTableBuilder(_Node({"Behavior": behavior})).build()
    assert table.iloc[0]["delay_beta"] == pytest.approx(1.0)
    assert table.iloc[0]["delay_min"] == 0.0
    assert table.iloc[0]["delay_max"] == 1.0


def test_build_uniform_quiescent_sets_delay_bounds_from_parameters():
    """A uniform quiescent distribution reports its own bounds and no beta."""
    behavior = _full_dataset().children["Behavior"]
    behavior.children["InputSchemas"].children["TaskLogic"] = _Stream(
        _task_logic(quiescent="uniform")
    )
    table = TrialTableBuilder(_Node({"Behavior": behavior})).build()
    first = table.iloc[0]
    # Uniform has no scale parameter, so beta stays null.
    assert pd.isna(first["delay_beta"])
    # The bounds come from the distribution parameters (0.25/0.75), not from the
    # truncation parameters the fixture also sets (9.0/99.0).
    assert first["delay_min"] == pytest.approx(0.25)
    assert first["delay_max"] == pytest.approx(0.75)


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
# ITI_stop_time on the last trial — the EndSession timestamp
# --------------------------------------------------------------------------- #
def test_build_last_iti_stop_is_nan_without_end_session():
    """Without an ``EndSession`` stream the last trial's ITI end stays unknown."""
    dataset = _full_dataset()
    del dataset.children["Behavior"].children["SoftwareEvents"].children["EndSession"]

    table = TrialTableBuilder(dataset).build()

    # Earlier trials are unaffected; only the last one lacks a closing event.
    assert table.iloc[0]["ITI_stop_time"] == 20.0
    assert np.isnan(table.iloc[1]["ITI_stop_time"])


def test_build_last_iti_stop_is_nan_when_end_session_is_empty():
    """An ``EndSession`` stream carrying no events leaves the last ITI end unknown."""
    dataset = _full_dataset()
    dataset.children["Behavior"].children["SoftwareEvents"].children["EndSession"] = _Stream(
        _events([], [])
    )

    table = TrialTableBuilder(dataset).build()

    assert np.isnan(table.iloc[1]["ITI_stop_time"])


def test_build_end_session_does_not_close_a_mid_session_gap():
    """Only the last trial falls back to ``EndSession``.

    A short ``QuiescentPeriod`` stream leaves an earlier trial without a closing
    event too, but attributing the session end to it would invent a trial
    spanning the rest of the session, so it stays ``NaN``.
    """
    table = TrialTableBuilder(_misaligned_dataset()).build()

    assert np.isnan(table.iloc[0]["ITI_stop_time"])
    assert table.iloc[1]["ITI_stop_time"] == 30.0


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
    assert resolved is coupled


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
    """With no coupled/uncoupled generator, the session columns dict is empty."""
    task_logic = AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            trial_generator=TrialGeneratorCompositeSpec(
                generators=[CoupledWarmupTrialGeneratorSpec()]
            ),
        )
    )
    assert TrialTableBuilder(_Node({}))._session_columns(task_logic) == {}


def test_side_bias_parses_dict_model_json_and_none():
    """``_side_bias`` extracts ``bias`` from a dict, model, JSON, or ``None``."""
    from aind_behavior_dynamic_foraging.task_logic.trial_models import TrialMetrics

    assert TrialTableBuilder._side_bias(None) is None
    assert TrialTableBuilder._side_bias({"bias": -0.4}) == pytest.approx(-0.4)
    assert TrialTableBuilder._side_bias({"bias": None}) is None
    assert TrialTableBuilder._side_bias(TrialMetrics(bias=0.5)) == pytest.approx(0.5)
    assert TrialTableBuilder._side_bias(TrialMetrics(bias=0.5).model_dump_json()) == pytest.approx(
        0.5
    )


def test_build_missing_trial_metrics_leaves_side_bias_null():
    """A missing ``TrialMetrics`` stream leaves ``side_bias`` null on every trial."""
    dataset = _full_dataset()
    software_events = dataset.children["Behavior"].children["SoftwareEvents"]
    del software_events.children["TrialMetrics"]
    table = TrialTableBuilder(dataset).build()
    assert table["side_bias"].isna().all()


def test_session_columns_warmup_generator_populates_min_reward_each_block():
    """A warmup generator exposes ``min_block_reward``, populating ``min_reward_each_block``."""
    task_logic = AindDynamicForagingTaskLogic(
        task_parameters=AindDynamicForagingTaskParameters(
            trial_generator=CoupledWarmupTrialGeneratorSpec()
        )
    )
    columns = TrialTableBuilder(_Node({}))._session_columns(task_logic)
    assert columns["min_reward_each_block"] == CoupledWarmupTrialGeneratorSpec().min_block_reward


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
    # No ``min_block_reward`` on this generator -> no per-block minimum (0).
    assert columns["min_reward_each_block"] == 0


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


def test_pulse_duration_converts_ms_to_seconds():
    """The pulse-width register value (ms) is returned in seconds."""
    df = pd.DataFrame(
        {"PulseSupplyPort0": [20], "MessageType": ["READ"]},
        index=pd.Index([12.0], name="Time"),
    )
    assert TrialTableBuilder._pulse_duration(df, "PulseSupplyPort0") == 0.02


def test_pulse_duration_uses_most_recent_value():
    """When the pulse width is reconfigured, the latest value wins."""
    df = pd.DataFrame(
        {"PulseSupplyPort0": [20, 50], "MessageType": ["READ", "WRITE"]},
        index=pd.Index([12.0, 30.0], name="Time"),
    )
    assert TrialTableBuilder._pulse_duration(df, "PulseSupplyPort0") == 0.05


def test_pulse_duration_missing_or_empty_returns_none():
    """An absent register/column or an all-null value column yields ``None``."""
    assert TrialTableBuilder._pulse_duration(None, "PulseSupplyPort0") is None
    no_col = pd.DataFrame({"MessageType": ["READ"]}, index=[1.0])
    assert TrialTableBuilder._pulse_duration(no_col, "PulseSupplyPort0") is None
    all_null = pd.DataFrame({"PulseSupplyPort0": [np.nan], "MessageType": ["READ"]}, index=[1.0])
    assert TrialTableBuilder._pulse_duration(all_null, "PulseSupplyPort0") is None


def test_write_times_without_message_type_uses_all_rows():
    """A register without a ``MessageType`` column returns every timestamp."""
    df = pd.DataFrame({"value": [1, 2]}, index=[3.0, 1.0])
    np.testing.assert_array_equal(TrialTableBuilder._write_times(df), np.array([1.0, 3.0]))


def test_animal_response_encoding():
    """``Response`` ``{"Item1", "Item2"}`` payloads map to the 0/1/2 codes."""
    assert TrialTableBuilder._animal_response(None) == 2
    assert TrialTableBuilder._animal_response({"Item1": 1.0, "Item2": True}) == 1
    assert TrialTableBuilder._animal_response({"Item1": 1.0, "Item2": False}) == 0
    assert TrialTableBuilder._animal_response({"Item1": 1.0, "Item2": None}) == 2
    # A payload missing Item2 entirely is treated as no choice.
    assert TrialTableBuilder._animal_response({"Item1": 1.0}) == 2


def test_is_baited_reads_the_per_side_metadata_flags():
    """Bait comes from the metadata flags, independent of p_reward and auto-response."""
    trial = TrialOutcome.model_validate(
        _outcome(
            0.0,
            1.0,
            is_right_choice=True,
            is_rewarded=True,
            auto=True,
            is_left_baited=True,
            is_right_baited=False,
        )
    ).trial
    metadata = TrialTableBuilder._bias_metadata(trial)
    # Right has p_reward 1 but the software reports it unbaited; left is the mirror.
    assert TrialTableBuilder._is_baited(metadata, is_right=True) is False
    assert TrialTableBuilder._is_baited(metadata, is_right=False) is True


def test_is_baited_defaults_to_false_without_metadata():
    """A trial carrying no extra metadata is reported as unbaited on both sides."""
    trial = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True)
    ).trial
    metadata = TrialTableBuilder._bias_metadata(trial)
    assert TrialTableBuilder._is_baited(metadata, is_right=True) is False
    assert TrialTableBuilder._is_baited(metadata, is_right=False) is False


def test_rewarded_history_is_earned_reward_only():
    """Rewarded history is the choice side on earned trials and False otherwise."""
    earned = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, auto=None)
    ).trial
    assert TrialTableBuilder._rewarded_history(earned, True, True, is_right=True) is True
    assert TrialTableBuilder._rewarded_history(earned, True, True, is_right=False) is False
    # An unrewarded trial is False on both sides.
    assert TrialTableBuilder._rewarded_history(earned, False, True, is_right=True) is False
    # An ignored trial (no choice) is False on both sides.
    assert TrialTableBuilder._rewarded_history(earned, True, None, is_right=True) is False
    assert TrialTableBuilder._rewarded_history(earned, True, None, is_right=False) is False


def test_rewarded_history_false_on_every_auto_reward_trial():
    """Autowater is not earned: an auto-reward trial is False on both sides."""
    for auto in (True, False):
        trial = TrialOutcome.model_validate(
            _outcome(1.0, 1.0, is_right_choice=auto, is_rewarded=True, auto=auto)
        ).trial
        assert TrialTableBuilder._rewarded_history(trial, True, auto, is_right=True) is False
        assert TrialTableBuilder._rewarded_history(trial, True, auto, is_right=False) is False


def test_auto_water_encodes_side_from_auto_response():
    """Scheduled autowater encodes ``1`` on its side and ``0`` on the other."""
    trial = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, auto=True, is_autowater=True)
    ).trial
    meta = TrialTableBuilder._bias_metadata(trial)
    assert TrialTableBuilder._auto_water(trial, meta, is_right=True) == 1
    assert TrialTableBuilder._auto_water(trial, meta, is_right=False) == 0
    # No free water at all counts as no autowater (0).
    no_auto = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, auto=None, is_autowater=True)
    ).trial
    assert (
        TrialTableBuilder._auto_water(
            no_auto, TrialTableBuilder._bias_metadata(no_auto), is_right=True
        )
        == 0
    )


def test_auto_water_includes_unrewarded_trials():
    """Autowater on a trial that did not pay out still counts.

    The column records what the task did, and free water fires at the go cue
    regardless of how the animal's own choice resolves. The reward-delivery
    series is reward-keyed and drops those deliveries, so this column can exceed
    that series' ``auto`` count.
    """
    unrewarded = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=False, is_rewarded=False, auto=True, is_autowater=True)
    ).trial
    meta = TrialTableBuilder._bias_metadata(unrewarded)
    assert TrialTableBuilder._auto_water(unrewarded, meta, is_right=True) == 1


def test_auto_water_excludes_anti_bias_water():
    """Anti-bias water is reported by ``anti_bias_*_water``, not ``auto_water*``.

    ``is_auto_reward_right`` is only the delivery channel, shared by scheduled
    autowater and the anti-bias intervention, so the mechanism comes from the
    metadata flags. The two columns are mutually exclusive.
    """
    bias_water = TrialOutcome.model_validate(
        _outcome(
            1.0,
            1.0,
            is_right_choice=True,
            is_rewarded=True,
            auto=True,
            is_bias_water_intervention=True,
        )
    ).trial
    meta = TrialTableBuilder._bias_metadata(bias_water)
    assert TrialTableBuilder._auto_water(bias_water, meta, is_right=True) == 0
    assert TrialTableBuilder._anti_bias_water(bias_water, meta, is_right=True) is True


def test_auto_water_excludes_free_water_with_no_mechanism_flag():
    """Free water the metadata flags as neither mechanism is ``0`` on both columns."""
    unflagged = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, auto=True)
    ).trial
    meta = TrialTableBuilder._bias_metadata(unflagged)
    assert TrialTableBuilder._auto_water(unflagged, meta, is_right=True) == 0
    assert TrialTableBuilder._anti_bias_water(unflagged, meta, is_right=True) is False


def test_bias_metadata_parses_dict_model_and_default():
    """``_bias_metadata`` handles a dict extra, a model extra, and missing metadata."""
    from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
        BlockBasedTrialMetadata,
    )

    # Dict extra (as delivered off the stream) is validated into the model.
    from_dict = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, is_bias_water_intervention=True)
    ).trial
    assert TrialTableBuilder._bias_metadata(from_dict).is_bias_water_intervention is True

    # A ``BlockBasedTrialMetadata`` instance is returned as-is.
    trial = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, block_p_left=0.5)
    ).trial
    trial.metadata.extra = BlockBasedTrialMetadata(is_bias_stage_intervention=True)
    assert TrialTableBuilder._bias_metadata(trial).is_bias_stage_intervention is True

    # No metadata -> all-False default.
    no_meta = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True)
    ).trial
    assert no_meta.metadata is None
    default = TrialTableBuilder._bias_metadata(no_meta)
    assert default.is_bias_water_intervention is False
    assert default.is_bias_stage_intervention is False

    # Metadata present but a non-dict / non-model extra -> default.
    other = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, block_p_left=0.5)
    ).trial
    other.metadata.extra = "unexpected"
    assert TrialTableBuilder._bias_metadata(other).is_bias_water_intervention is False


def test_anti_bias_water_gated_on_intervention_flag_and_side():
    """Anti-bias water is True only for a bias-water intervention on the matching side."""
    # Right-side bias-water intervention.
    right = TrialOutcome.model_validate(
        _outcome(
            1.0,
            1.0,
            is_right_choice=True,
            is_rewarded=True,
            auto=True,
            is_bias_water_intervention=True,
        )
    )
    meta = TrialTableBuilder._bias_metadata(right.trial)
    assert TrialTableBuilder._anti_bias_water(right.trial, meta, is_right=True) is True
    assert TrialTableBuilder._anti_bias_water(right.trial, meta, is_right=False) is False

    # Auto-response to the left without the bias flag is ordinary autowater, not
    # an anti-bias intervention.
    autowater = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=False, is_rewarded=True, auto=False)
    )
    auto_meta = TrialTableBuilder._bias_metadata(autowater.trial)
    assert TrialTableBuilder._anti_bias_water(autowater.trial, auto_meta, is_right=False) is False


def test_anti_bias_water_includes_unrewarded_trials():
    """An intervention on a trial that did not pay out still counts.

    The column records what the anti-bias algorithm did, and the intervention
    fires at the go cue regardless of how the animal's own choice resolves. The
    reward-delivery series is reward-keyed and drops that delivery, so this
    column can exceed the series' ``anti_bias`` count.
    """
    unrewarded = TrialOutcome.model_validate(
        _outcome(
            1.0,
            1.0,
            is_right_choice=False,
            is_rewarded=False,
            auto=True,
            is_bias_water_intervention=True,
        )
    ).trial
    meta = TrialTableBuilder._bias_metadata(unrewarded)
    assert TrialTableBuilder._anti_bias_water(unrewarded, meta, is_right=True) is True


def test_anti_bias_lickspout_movement_gated_on_stage_flag():
    """Movement is the offset delta only when flagged a bias-stage intervention."""
    moved = TrialOutcome.model_validate(
        _outcome(
            1.0,
            1.0,
            is_right_choice=True,
            is_rewarded=True,
            lickspout_offset_delta=1.5,
            is_bias_stage_intervention=True,
        )
    ).trial
    assert TrialTableBuilder._anti_bias_lickspout_movement(
        moved, TrialTableBuilder._bias_metadata(moved)
    ) == pytest.approx(1.5)

    # Offset present but not flagged as a stage intervention -> 0.0.
    unflagged = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True, lickspout_offset_delta=1.5)
    ).trial
    assert (
        TrialTableBuilder._anti_bias_lickspout_movement(
            unflagged, TrialTableBuilder._bias_metadata(unflagged)
        )
        == 0.0
    )


def test_build_populates_anti_bias_columns():
    """A dataset with anti-bias interventions populates the three anti-bias columns."""
    dataset = _full_dataset()
    software_events = dataset.children["Behavior"].children["SoftwareEvents"]
    software_events.children["TrialOutcome"] = _Stream(
        _events(
            [10.1, 20.1],
            [
                _outcome(
                    1.0,
                    1.0,
                    is_right_choice=True,
                    is_rewarded=True,
                    auto=True,
                    is_bias_water_intervention=True,
                ),
                _outcome(
                    1.0,
                    1.0,
                    is_right_choice=True,
                    is_rewarded=True,
                    lickspout_offset_delta=-0.8,
                    is_bias_stage_intervention=True,
                ),
            ],
        )
    )
    table = TrialTableBuilder(dataset).build()
    first, second = table.iloc[0], table.iloc[1]
    assert bool(first["anti_bias_right_water"]) is True
    assert bool(first["anti_bias_left_water"]) is False
    assert first["anti_bias_lickspout_movement"] == 0.0
    assert bool(second["anti_bias_right_water"]) is False
    assert second["anti_bias_lickspout_movement"] == pytest.approx(-0.8)


def test_block_reward_probability_reads_metadata_not_trial():
    """The block probability comes from ``trial.metadata``, not the top-level p_reward."""
    trial = TrialOutcome.model_validate(
        _outcome(
            1.0, 0.2, is_right_choice=True, is_rewarded=True, block_p_left=0.7, block_p_right=0.1
        )
    ).trial
    assert TrialTableBuilder._block_reward_probability(trial, is_right=False) == pytest.approx(0.7)
    assert TrialTableBuilder._block_reward_probability(trial, is_right=True) == pytest.approx(0.1)


def test_block_reward_probability_none_without_metadata():
    """Absent metadata yields ``None``."""
    no_meta = TrialOutcome.model_validate(
        _outcome(1.0, 0.2, is_right_choice=True, is_rewarded=True)
    ).trial
    assert no_meta.metadata is None
    assert TrialTableBuilder._block_reward_probability(no_meta, is_right=False) is None


@pytest.mark.parametrize(
    "payload",
    [
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True),
        TrialOutcome.model_validate(_outcome(1.0, 1.0, is_right_choice=False, is_rewarded=False)),
    ],
)
def test_parse_outcome_accepts_dict_and_model(payload):
    """``_parse_outcome`` accepts dicts and model instances."""
    assert isinstance(TrialTableBuilder._parse_outcome(payload), TrialOutcome)


def test_parse_outcome_raises_on_none():
    """``_parse_outcome`` raises ``ValueError`` for a ``None`` payload."""
    with pytest.raises(ValueError, match="required"):
        TrialTableBuilder._parse_outcome(None)


def test_parse_outcome_accepts_json_string():
    """``_parse_outcome`` parses a JSON-string payload."""
    payload = TrialOutcome.model_validate(
        _outcome(1.0, 1.0, is_right_choice=True, is_rewarded=True)
    ).model_dump_json()
    assert isinstance(TrialTableBuilder._parse_outcome(payload), TrialOutcome)


def test_manipulator_mm_per_step_from_calibration():
    """mm-per-microstep is full_step_to_mm / microstep resolution, per axis."""
    builder = TrialTableBuilder(_Node({}))
    mm_per_step = builder._manipulator_mm_per_step(_rig())
    assert mm_per_step == {"x": 0.00125, "y1": 0.00125, "y2": 0.00125, "z": 0.00125}


def test_manipulator_mm_per_step_respects_resolution():
    """A finer microstep resolution shrinks the distance per step."""
    builder = TrialTableBuilder(_Node({}))
    mm_per_step = builder._manipulator_mm_per_step(
        _rig(step_mm=0.01, resolution=MicrostepResolution.MICROSTEP16)
    )
    assert mm_per_step["x"] == pytest.approx(0.01 / 16)


def test_manipulator_mm_per_step_missing_axis_raises():
    """A calibration lacking one of the four axes raises ``ValueError``."""
    builder = TrialTableBuilder(_Node({}))
    rig = _rig()
    # Drop the Z axis configuration.
    rig.manipulator.calibration.axis_configuration = [
        config
        for config in rig.manipulator.calibration.axis_configuration
        if config.axis is not Axis.Z
    ]
    with pytest.raises(ValueError, match="missing axis 'z'"):
        builder._manipulator_mm_per_step(rig)


def test_manipulator_positions_converts_steps_to_mm_relative_to_start():
    """EVENT rows convert to mm and are re-referenced to the session-start sample."""
    builder = TrialTableBuilder(_Node({}))
    steps = _accumulated_steps([8.0, 15.0], [(8000, 1600, 2400, 4000), (12000, 1600, 2400, 4000)])
    positions = builder._manipulator_positions(steps, _rig())
    assert list(positions.index) == [8.0, 15.0]
    # First sample is the baseline (zeroed); x then moves +5.0 mm (4000 steps).
    assert positions.loc[8.0, "lickspout_position_x"] == 0.0
    assert positions.loc[15.0, "lickspout_position_x"] == 5.0
    assert positions.loc[8.0, "lickspout_position_z"] == 0.0


def test_manipulator_positions_filters_non_event_rows():
    """Only ``EVENT`` rows contribute to the position frame."""
    builder = TrialTableBuilder(_Node({}))
    steps = _accumulated_steps([8.0, 15.0], [(8000, 1600, 2400, 4000), (12000, 1600, 2400, 4000)])
    steps.loc[15.0, "MessageType"] = "WRITE"
    positions = builder._manipulator_positions(steps, _rig())
    assert list(positions.index) == [8.0]


def test_manipulator_positions_all_non_event_raises():
    """A stream with no ``EVENT`` rows raises ``ValueError``."""
    builder = TrialTableBuilder(_Node({}))
    steps = _accumulated_steps([8.0], [(8000, 1600, 2400, 4000)])
    steps["MessageType"] = "WRITE"
    with pytest.raises(ValueError, match="no EVENT rows"):
        builder._manipulator_positions(steps, _rig())


def test_manipulator_positions_missing_motor_column_raises():
    """A stream missing a Motor column raises ``ValueError``."""
    builder = TrialTableBuilder(_Node({}))
    steps = _accumulated_steps([8.0], [(8000, 1600, 2400, 4000)]).drop(columns=["Motor3"])
    with pytest.raises(ValueError, match="missing column Motor3"):
        builder._manipulator_positions(steps, _rig())


def test_manipulator_positions_incomplete_calibration_raises():
    """Steps present but a calibration axis missing raises ``ValueError``."""
    builder = TrialTableBuilder(_Node({}))
    steps = _accumulated_steps([8.0], [(8000, 1600, 2400, 4000)])
    rig = _rig()
    rig.manipulator.calibration.axis_configuration = [
        config
        for config in rig.manipulator.calibration.axis_configuration
        if config.axis is not Axis.Z
    ]
    with pytest.raises(ValueError, match="missing axis 'z'"):
        builder._manipulator_positions(steps, rig)


def _two_sample_positions(builder):
    """Build a position frame relative to session start: t=8.0 (x=0.0), t=15.0 (x=5.0)."""
    return builder._manipulator_positions(
        _accumulated_steps([8.0, 15.0], [(8000, 1600, 2400, 4000), (12000, 1600, 2400, 4000)]),
        _rig(),
    )


def test_sample_lickspout_picks_sample_nearest_start_in_window():
    """A trial takes the in-window sample nearest its start."""
    builder = TrialTableBuilder(_Node({}))
    positions = _two_sample_positions(builder)
    # Window [7.0, 12.0) contains only the t=8.0 baseline sample (0.0).
    assert builder._sample_lickspout(positions, 7.0, 12.0)["lickspout_position_x"] == 0.0
    # Window [14.0, 20.0) contains only the t=15.0 sample (+5.0 mm).
    assert builder._sample_lickspout(positions, 14.0, 20.0)["lickspout_position_x"] == 5.0


def test_sample_lickspout_no_sample_in_window_is_null():
    """A trial whose window contains no sample yields ``None`` columns."""
    builder = TrialTableBuilder(_Node({}))
    positions = _two_sample_positions(builder)
    columns = builder._sample_lickspout(positions, 20.0, 30.0)
    assert all(value is None for value in columns.values())
