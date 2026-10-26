"""Smoke tests for ``dynamic_foraging_processing.qc._core.result``."""

import dataclasses
import datetime

import numpy as np
import pytest
from aind_data_schema.core.quality_control import Stage
from aind_data_schema_models.modalities import Modality

from dynamic_foraging_processing.qc._core.result import QCResult, to_metrics


def test_to_metrics_converts_fields_and_statuses():
    """Results become behavior/raw metrics; ``passed`` maps to Pass / Fail / Pending."""
    results = [
        QCResult("m", np.float64(0.25), True, "d", "r.png", {"type": "T", "group": "G"}),
        QCResult("f", 1, False),
        QCResult("p", 2, None),
    ]
    passed, failed, pending = to_metrics(results)

    assert (passed.name, passed.value, passed.description, passed.reference) == (
        "m",
        0.25,
        "d",
        "r.png",
    )
    assert passed.tags == {"type": "T", "group": "G"}
    assert passed.modality == Modality.BEHAVIOR and passed.stage == Stage.RAW
    assert passed.status_history[0].timestamp.utcoffset() == datetime.timedelta(0)
    assert [m.status_history[0].status for m in (passed, failed, pending)] == [
        "Pass",
        "Fail",
        "Pending",
    ]
    assert failed.tags == {}


def test_qc_result_is_frozen():
    """``QCResult`` is immutable."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        QCResult("m", 1, True).passed = False
