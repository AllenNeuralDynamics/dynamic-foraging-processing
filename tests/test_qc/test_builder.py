"""Tests for ``dynamic_foraging_processing.qc._core.builder`` and
``dynamic_foraging_processing.qc.processed.results``."""

import os

import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import QualityControl

from dynamic_foraging_processing.qc._core import builder as _builder
from dynamic_foraging_processing.qc._core.result import to_metrics
from dynamic_foraging_processing.qc.processed import results as _results


def test_behavior_qc_results_without_plots():
    """Five behavior results are produced and no plots are written."""
    trials = pd.DataFrame(
        {
            "animal_response": [0, 1, 2, 1],
            "side_bias": [-0.1, 0.0, np.nan, 0.1],
        }
    )
    results = _results.behavior_qc_results(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
    )
    assert len(results) == 5
    assert results[0].name == "average side bias"


def test_lickspout_columns_map_to_trial_table_names():
    """The lickspout mapping resolves to the ``lickspout_position_*`` trial columns."""
    trials = pd.DataFrame(
        {
            "lickspout_position_x": [16.0, 16.0],
            "lickspout_position_y1": [9.6, 9.6],
            "lickspout_position_y2": [9.7, 9.7],
            "lickspout_position_z": [17.25, 17.25],
        }
    )
    for key in ("lickspout_x", "lickspout_y1", "lickspout_y2", "lickspout_z"):
        column = _results._column(trials, key)
        assert column is not None, key
        assert column[0] == trials[_results._COLUMNS[key]][0]


def test_behavior_qc_results_writes_plots(tmp_path):
    """Supplying a results folder writes both behavior plots."""
    trials = pd.DataFrame(
        {
            "animal_response": [0, 1, 2, 1],
            "side_bias": [-0.1, 0.0, np.nan, 0.1],
        }
    )
    results = _results.behavior_qc_results(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
        str(tmp_path),
    )
    assert len(results) == 5
    assert os.path.exists(tmp_path / "side_bias.png")
    assert os.path.exists(tmp_path / "lick_intervals.png")


def test_build_quality_control_defaults():
    """Defaults fill in the standard grouping and an empty failure allowlist."""
    metrics = to_metrics(
        _results.behavior_qc_results(
            pd.DataFrame({"animal_response": [0, 1], "side_bias": [0.1, -0.1]}),
            np.array([1.0]),
            np.array([2.0]),
        )
    )
    qc = _builder.build_quality_control(metrics)
    assert isinstance(qc, QualityControl)
    assert qc.default_grouping == _builder.DEFAULT_GROUPING
    assert qc.allow_tag_failures == []


def test_build_quality_control_overrides():
    """Explicit grouping / allowlist / metadata are passed through."""
    metrics = to_metrics(
        _results.behavior_qc_results(
            pd.DataFrame({"animal_response": [0, 1], "side_bias": [0.1, -0.1]}),
            np.array([1.0]),
            np.array([2.0]),
        )
    )
    qc = _builder.build_quality_control(
        metrics,
        default_grouping=["behavior"],
        allow_tag_failures=["behavior"],
        key_experimenters=["Alex"],
        notes="hello",
    )
    assert qc.default_grouping == ["behavior"]
    assert qc.allow_tag_failures == ["behavior"]
    assert qc.notes == "hello"
