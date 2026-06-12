"""Tests for ``dynamic_foraging_processing.qc._schema``."""

import datetime

import numpy as np
from aind_data_schema.core.quality_control import Stage, Status
from aind_data_schema_models.modalities import Modality
from contraqctor import qc as cqc

from dynamic_foraging_processing.qc import _schema


def test_status_converter_maps_all_contraqctor_statuses():
    """Every contraqctor status maps to a schema status (warnings -> pending)."""
    assert _schema.STATUS_CONVERTER[cqc.Status.PASSED] == Status.PASS
    assert _schema.STATUS_CONVERTER[cqc.Status.SKIPPED] == Status.PASS
    assert _schema.STATUS_CONVERTER[cqc.Status.WARNING] == Status.PENDING
    assert _schema.STATUS_CONVERTER[cqc.Status.FAILED] == Status.FAIL
    assert _schema.STATUS_CONVERTER[cqc.Status.ERROR] == Status.FAIL


def test_now_helpers_are_timezone_aware():
    """``now_seattle`` and ``now_utc`` return aware datetimes."""
    assert _schema.now_seattle().tzinfo is not None
    assert _schema.now_utc().utcoffset() == datetime.timedelta(0)


def test_bool_to_status_pass_and_fail_with_default_timestamp():
    """A truthy value yields PASS, falsy yields FAIL; timestamp defaults to now."""
    passed = _schema.bool_to_status(True)
    failed = _schema.bool_to_status(False)
    assert passed.status == Status.PASS
    assert failed.status == Status.FAIL
    assert passed.evaluator == "Automated"
    assert passed.timestamp.tzinfo is not None


def test_bool_to_status_uses_supplied_timestamp():
    """An explicit timestamp is passed through unchanged."""
    ts = datetime.datetime(2026, 6, 11, tzinfo=datetime.timezone.utc)
    assert _schema.bool_to_status(True, ts).timestamp == ts


def test_to_builtin_converts_numpy_types():
    """Arrays become lists, numpy scalars become builtins, others pass through."""
    assert _schema.to_builtin(np.array([1, 2])) == [1, 2]
    assert _schema.to_builtin(np.float64(1.5)) == 1.5
    assert isinstance(_schema.to_builtin(np.int64(3)), int)
    assert _schema.to_builtin("text") == "text"


def test_make_metric_defaults_and_overrides():
    """``make_metric`` stamps modality/stage and defaults tags to ``{}``."""
    status = _schema.bool_to_status(True)
    default = _schema.make_metric(name="m", value=np.float64(2.0), status=status)
    assert default.modality == Modality.BEHAVIOR
    assert default.stage == Stage.RAW
    assert default.tags == {}
    assert default.value == 2.0

    tagged = _schema.make_metric(
        name="m",
        value=1,
        status=status,
        description="d",
        reference="r.png",
        tags={"behavior": "m"},
    )
    assert tagged.tags == {"behavior": "m"}
    assert tagged.reference == "r.png"
