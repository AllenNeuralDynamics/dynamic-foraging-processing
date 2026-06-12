"""Tests for ``dynamic_foraging_processing.qc._result``."""

from dynamic_foraging_processing.qc import _result


def _result_obj(passed=True):
    """Build a simple QCResult for conversion tests."""
    return _result.QCResult(
        name="m",
        value=0.25,
        passed=passed,
        description="d",
        reference="r.png",
        tags={"behavior": "m"},
    )


def test_to_metric_carries_fields_and_status():
    """``to_metric`` produces a QCMetric with the result's fields and status."""
    metric = _result_obj(passed=True).to_metric()
    assert metric.name == "m"
    assert metric.value == 0.25
    assert metric.reference == "r.png"
    assert metric.tags == {"behavior": "m"}
    assert metric.status_history[0].status == "Pass"


def test_to_metric_failing_status():
    """A failing result converts to a failing metric."""
    assert _result_obj(passed=False).to_metric().status_history[0].status == "Fail"


def test_to_metrics_converts_each_result():
    """``to_metrics`` converts a sequence of results into metrics."""
    metrics = _result.to_metrics([_result_obj(True), _result_obj(False)])
    assert [m.status_history[0].status for m in metrics] == ["Pass", "Fail"]


def test_qc_result_is_frozen():
    """``QCResult`` is immutable."""
    result = _result_obj()
    try:
        result.passed = False
    except Exception as exc:  # FrozenInstanceError
        assert "cannot assign" in str(exc) or "frozen" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("QCResult should be frozen")
