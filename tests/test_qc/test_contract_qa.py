"""Tests for ``dynamic_foraging_processing.qc.raw.contract_qa``."""

from contraqctor import qc as cqc

from dynamic_foraging_processing.qc.raw import contract_qa as _contract_qa


def test_contract_qc_metrics_uses_runner(monkeypatch, tmp_path):
    """``contract_qc_metrics`` runs the runner and converts its results."""

    class _FakeRunner:
        """Stand-in runner returning a fixed grouped-results dict."""

        def run_all(self):
            """Return one passing result under a single group."""
            return {"grp": [cqc.Result(cqc.Status.PASSED, True, "t", "S")]}

    monkeypatch.setattr(_contract_qa, "make_qc_runner", lambda dataset: _FakeRunner())
    metrics = _contract_qa.contract_qc_metrics(dataset=object(), results_folder=str(tmp_path))
    assert [m.name for m in metrics] == ["t:grp"]
    assert metrics[0].tags == {"type": "Harp QC Suite", "group": "grp"}
    assert metrics[0].status_history[0].status == "Pass"
