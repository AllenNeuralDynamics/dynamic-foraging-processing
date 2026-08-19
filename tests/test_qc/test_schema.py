"""Tests for ``dynamic_foraging_processing.qc._core.schema``."""

import datetime

import numpy as np
from aind_data_schema.core.quality_control import Stage, Status
from aind_data_schema_models.modalities import Modality
from contraqctor import qc as cqc

from dynamic_foraging_processing.qc._core import schema as _schema


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


def test_bool_to_status_none_is_pending():
    """``None`` (no automated pass/fail) yields a PENDING status."""
    pending = _schema.bool_to_status(None)
    assert pending.status == Status.PENDING
    assert pending.evaluator == "Automated"


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


def test_to_builtin_recurses_into_containers():
    """Numpy values nested in dicts/sequences are converted too."""
    nested = {
        "scalar": np.float32(1.5),
        "array": np.array([1, 2]),
        "rows": [(np.int64(3), {"deep": np.float32(0.25)})],
    }
    converted = _schema.to_builtin(nested)
    assert converted == {"scalar": 1.5, "array": [1, 2], "rows": [[3, {"deep": 0.25}]]}
    assert isinstance(converted["scalar"], float)
    assert isinstance(converted["rows"][0][0], int)
    assert isinstance(converted["rows"][0][1]["deep"], float)
    assert _schema.to_builtin({np.int64(1)}) == [1]
    assert _schema.to_builtin(frozenset({np.int64(2)})) == [2]
    assert list(_schema.to_builtin({np.int64(4): "v"})) == [4]


def test_to_builtin_rounds_floats():
    """Floats are rounded to three decimals, at any depth; ints are untouched."""
    assert _schema.to_builtin(np.float32(39.563472747802734)) == 39.563
    assert _schema.to_builtin(1.23456) == 1.235
    assert _schema.to_builtin(np.array([1.23456, 2.0])) == [1.235, 2.0]
    assert _schema.to_builtin({"mean": np.float32(26.0916690826416)}) == {"mean": 26.092}
    assert _schema.to_builtin(np.int64(123456)) == 123456


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
