"""Tests for ``dynamic_foraging_processing.qc.processed.results`` and the shared
``QualityControl`` assembly."""

import os

import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import Status
from contraqctor import qc as cqc

from dynamic_foraging_processing.qc import build_quality_control, results_to_metrics
from dynamic_foraging_processing.qc._core.result import to_metrics
from dynamic_foraging_processing.qc.processed import results as _results


def test_behavior_qc_results_without_plots():
    """Six behavior results are produced and no plots are written."""
    trials = pd.DataFrame(
        {
            "animal_response": [0, 1, 2, 1],
            "side_bias": [-0.1, 0.0, np.nan, 0.1],
            "goCue_start_time": [0.5, 1.5, 2.5, 3.5],
        }
    )
    results = _results.behavior_qc_results(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
    )
    assert len(results) == 6
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
    """Supplying a results folder writes all three behavior plots."""
    trials = pd.DataFrame(
        {
            "animal_response": [0, 1, 2, 1],
            "side_bias": [-0.1, 0.0, np.nan, 0.1],
            "goCue_start_time": [0.5, 1.5, 2.5, 3.5],
        }
    )
    results = _results.behavior_qc_results(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
        str(tmp_path),
    )
    assert len(results) == 6
    assert os.path.exists(tmp_path / "side_bias.png")
    assert os.path.exists(tmp_path / "lick_intervals.png")
    assert os.path.exists(tmp_path / "lick_latency.png")


def test_build_quality_control_excuses_contract_failures_only():
    """A failing contract check does not fail the session; a failing behavior metric does."""
    contract = results_to_metrics(
        {"HarpBehavior": [cqc.Result(cqc.Status.FAILED, False, "test_x", "Suite")]}
    )
    behavior = to_metrics(
        _results.behavior_qc_results(
            pd.DataFrame({"animal_response": [0, 1], "side_bias": [0.1, -0.1]}),
            np.array([1.0]),
            np.array([2.0]),
        )
    )
    assert build_quality_control(contract).evaluate_status() == Status.PASS
    # The behavior results include failing checks (too few licks), which still count.
    assert build_quality_control(contract + behavior).evaluate_status() == Status.FAIL
