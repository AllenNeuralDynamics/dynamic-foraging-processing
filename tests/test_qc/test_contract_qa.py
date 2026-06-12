"""Tests for ``dynamic_foraging_processing.qc._contract_qa``."""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from contraqctor import qc as cqc

from dynamic_foraging_processing.qc import _contract_qa


def _result(
    status, *, suite="SuiteA", test="test_x", context=None, value=1, message="m", description="d"
):
    """Build a contraqctor ``Result`` for testing."""
    return cqc.Result(
        status=status,
        result=value,
        test_name=test,
        suite_name=suite,
        message=message,
        description=description,
        context=context,
    )


def test_sanitize_replaces_unsafe_characters():
    """Non-filename-safe characters collapse to underscores."""
    assert _contract_qa._sanitize("Harp Hub::test 1") == "Harp_Hub_test_1"


def test_save_asset_non_dict_context_returns_none(tmp_path):
    """A result with no dict context saves nothing."""
    assert _contract_qa._save_asset(_result(cqc.Status.PASSED), str(tmp_path)) is None


def test_save_asset_non_figure_asset_returns_none(tmp_path):
    """A non-figure asset saves nothing."""
    result = _result(cqc.Status.PASSED, context={"asset": "not-a-figure"})
    assert _contract_qa._save_asset(result, str(tmp_path)) is None


def test_save_asset_figure_without_folder_returns_none():
    """A figure asset with no results folder saves nothing."""
    fig = plt.figure()
    result = _result(cqc.Status.PASSED, context={"asset": fig})
    assert _contract_qa._save_asset(result, None) is None
    plt.close(fig)


def test_save_asset_figure_is_saved(tmp_path):
    """A figure asset is written and its filename returned."""
    fig = plt.figure()
    result = _result(cqc.Status.PASSED, suite="S", test="t", context={"asset": fig})
    name = _contract_qa._save_asset(result, str(tmp_path))
    assert name == "S_t.png"
    assert os.path.exists(tmp_path / name)
    plt.close(fig)


def test_results_to_metrics_grouping_and_status(tmp_path):
    """Results convert to tagged metrics; missing group becomes ``NoGroup``."""
    fig = plt.figure()
    results = {
        "Data contract": [
            _result(cqc.Status.WARNING, suite="HubSuite", test="t1", context={"asset": fig})
        ],
        None: [_result(cqc.Status.FAILED, suite="CamSuite", test="t2")],
    }
    metrics = _contract_qa.results_to_metrics(results, str(tmp_path))
    plt.close(fig)

    by_name = {m.name: m for m in metrics}
    warn = by_name["HubSuite::t1"]
    assert warn.status_history[0].status == "Pending"
    assert warn.tags == {"test_suite": "HubSuite", "HubSuite": "Data contract"}
    assert warn.reference == "HubSuite_t1.png"

    fail = by_name["CamSuite::t2"]
    assert fail.status_history[0].status == "Fail"
    assert fail.tags == {"test_suite": "CamSuite", "CamSuite": _contract_qa.NO_GROUP}
    assert fail.reference is None


def test_contract_qc_metrics_uses_runner(monkeypatch, tmp_path):
    """``contract_qc_metrics`` runs the runner and converts its results."""

    class _FakeRunner:
        """Stand-in runner returning a fixed grouped-results dict."""

        def run_all(self):
            """Return one passing result under a single group."""
            return {"grp": [_result(cqc.Status.PASSED, suite="S", test="t")]}

    monkeypatch.setattr(_contract_qa, "make_qc_runner", lambda dataset: _FakeRunner())
    metrics = _contract_qa.contract_qc_metrics(dataset=object(), results_folder=str(tmp_path))
    assert [m.name for m in metrics] == ["S::t"]
    assert metrics[0].status_history[0].status == "Pass"
