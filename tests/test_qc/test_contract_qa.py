"""Tests for ``dynamic_foraging_processing.qc.raw.contract_qa``."""

from contraqctor import qc as cqc

from dynamic_foraging_processing.qc.raw import contract_qa as _contract_qa


def test_contract_qc_metrics_uses_runner(monkeypatch, tmp_path):
    """``contract_qc_metrics`` runs the runner, drops disabled checks, and converts the rest."""
    fw_test = "test_fw_version_matches_reader"

    class _FakeRunner:
        """Stand-in runner returning a fixed grouped-results dict."""

        def run_all(self):
            """Return a passing result plus firmware checks, one of them disabled."""
            return {
                "grp": [cqc.Result(cqc.Status.PASSED, True, "t", "S")],
                "HarpClockGenerator": [cqc.Result(cqc.Status.WARNING, False, fw_test, "S")],
                "HarpBehavior": [cqc.Result(cqc.Status.PASSED, True, fw_test, "S")],
            }

    monkeypatch.setattr(_contract_qa, "make_qc_runner", lambda dataset: _FakeRunner())
    metrics = _contract_qa.contract_qc_metrics(dataset=object(), results_folder=str(tmp_path))
    assert [m.name for m in metrics] == ["t:grp", f"{fw_test}:HarpBehavior"]
    assert metrics[0].tags == {"type": "Harp QC Suite", "group": "grp"}
    assert metrics[0].status_history[0].status == "Pass"
