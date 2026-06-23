"""Tests for the QC stage wrappers (``BaseQC``, ``RawQC``, ``ProcessedQC``)."""

import os

import numpy as np
import pandas as pd
import pytest
from aind_data_schema.core.quality_control import QCMetric

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc.processed.stage import ProcessedQC
from dynamic_foraging_processing.qc.raw import stage as raw_stage
from dynamic_foraging_processing.qc.raw.stage import RawQC


def test_base_qc_run_raises_not_implemented():
    """The base ``run`` body raises ``NotImplementedError`` when delegated to."""

    class _Stage(BaseQC):
        """Concrete stage that defers to the base implementation."""

        def run(self, *args, **kwargs):
            """Delegate to ``BaseQC.run`` so its body executes."""
            return super().run(*args, **kwargs)

    with pytest.raises(NotImplementedError):
        _Stage().run()


def test_raw_qc_run_delegates_to_contract_metrics(monkeypatch):
    """``RawQC.run`` forwards the dataset and folder to ``contract_qc_metrics``."""
    captured = {}

    def _fake_contract_qc_metrics(dataset, results_folder):
        """Record arguments and return a sentinel metric list."""
        captured["dataset"] = dataset
        captured["results_folder"] = results_folder
        return ["sentinel"]

    monkeypatch.setattr(raw_stage, "contract_qc_metrics", _fake_contract_qc_metrics)
    dataset = object()
    result = RawQC().run(dataset, "out")
    assert result == ["sentinel"]
    assert captured == {"dataset": dataset, "results_folder": "out"}


def test_processed_qc_run_returns_metrics():
    """``ProcessedQC.run`` produces the five behavior metrics."""
    trials = pd.DataFrame(
        {"animal_response": [0, 1, 2, 1], "side_bias": [-0.1, 0.0, np.nan, 0.1]}
    )
    metrics = ProcessedQC().run(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
    )
    assert len(metrics) == 5
    assert all(isinstance(m, QCMetric) for m in metrics)
    assert metrics[0].name == "average side bias"


def test_processed_qc_run_writes_plots(tmp_path):
    """Supplying a results folder writes the supporting plots."""
    trials = pd.DataFrame(
        {"animal_response": [0, 1, 2, 1], "side_bias": [-0.1, 0.0, np.nan, 0.1]}
    )
    metrics = ProcessedQC().run(
        trials,
        np.array([1.0, 1.01]),
        np.array([2.0, 2.01]),
        str(tmp_path),
    )
    assert len(metrics) == 5
    assert os.path.exists(tmp_path / "side_bias.png")
    assert os.path.exists(tmp_path / "lick_intervals.png")
