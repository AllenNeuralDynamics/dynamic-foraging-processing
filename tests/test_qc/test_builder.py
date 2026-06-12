"""Tests for ``dynamic_foraging_processing.qc._builder``."""

import os

import numpy as np
from aind_data_schema.core.quality_control import QualityControl

from dynamic_foraging_processing.qc import _builder
from dynamic_foraging_processing.qc._result import to_metrics


def test_behavior_qc_results_without_plots():
    """Five behavior results are produced and no plots are written."""
    results = _builder.behavior_qc_results(
        np.array([0, 1, 2, 1]), np.array([1.0, 1.01]), np.array([2.0, 2.01])
    )
    assert len(results) == 5
    assert results[0].name == "average side bias"


def test_behavior_qc_results_writes_plots(tmp_path):
    """Supplying a results folder writes both behavior plots."""
    results = _builder.behavior_qc_results(
        np.array([0, 1, 2, 1]),
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
        _builder.behavior_qc_results(np.array([0, 1]), np.array([1.0]), np.array([2.0]))
    )
    qc = _builder.build_quality_control(metrics)
    assert isinstance(qc, QualityControl)
    assert qc.default_grouping == _builder.DEFAULT_GROUPING
    assert qc.allow_tag_failures == []


def test_build_quality_control_overrides():
    """Explicit grouping / allowlist / metadata are passed through."""
    metrics = to_metrics(
        _builder.behavior_qc_results(np.array([0, 1]), np.array([1.0]), np.array([2.0]))
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
