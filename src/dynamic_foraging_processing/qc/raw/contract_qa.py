"""Convert ``contraqctor`` QA runner results into ``QCMetric`` objects.

The dynamic foraging contract QA (Harp devices, cameras, CSV streams, the data
contract, and task-specific checks) is provided by
``aind_behavior_dynamic_foraging.data_qc.make_qc_runner``. This module runs that
runner over a dataset and maps each ``contraqctor`` ``Result`` onto a schema
``QCMetric`` named after its test and tagged with exactly the
``default_grouping`` keys: ``type`` :data:`QC_TYPE` (the QC portal's top-level
group) and ``group`` (the device or stream being checked), so each metric is
uniquely identified by name + tags.
"""

import re
import typing as t
from pathlib import Path

import matplotlib.figure
from aind_behavior_dynamic_foraging.data_qc.suite import make_qc_runner
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Stage
from aind_data_schema_models.modalities import Modality
from contraqctor import contract, qc

from dynamic_foraging_processing.qc._core.schema import STATUS_CONVERTER, now_utc, to_builtin

#: ``type`` tag shared by all contract QC metrics (their QC portal group).
QC_TYPE = "Harp QC Suite"

#: Group value used when a runner result has no group.
NO_GROUP = "NoGroup"


def _sanitize(name: str) -> str:
    """Make ``name`` safe to use as a filename component."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _save_asset(result: qc.Result, results_folder: t.Optional[str]) -> t.Optional[str]:
    """Save a result's figure asset (if any) and return its relative path.

    Parameters
    ----------
    result : contraqctor.qc.Result
        A single QA result; its ``context["asset"]`` may be a matplotlib figure.
    results_folder : str or None
        Directory to save figures into. If ``None``, nothing is saved.

    Returns
    -------
    str or None
        The saved figure's reference (``"<results-folder-name>/<file>"``), or
        ``None`` when there is no figure asset (or no ``results_folder``). The
        file name combines the (sanitized) suite and test names, e.g. a
        ``CameraTestSuite`` result for ``test_frame_rate`` in ``.../nwb`` becomes
        ``"nwb/CameraTestSuite_test_frame_rate.png"``.
    """
    context = result.context
    if not isinstance(context, dict):
        return None
    asset = context.get("asset")
    if not isinstance(asset, matplotlib.figure.Figure) or results_folder is None:
        return None
    filename = f"{_sanitize(result.suite_name)}_{_sanitize(result.test_name)}.png"
    asset.savefig(Path(results_folder) / filename, dpi=300, bbox_inches="tight")
    # Reference is relative to the top-level results folder, where the QC JSON is
    # written; the images live in the named subfolder alongside it.
    return f"{Path(results_folder).name}/{filename}"


def results_to_metrics(
    results: t.Dict[t.Optional[str], t.List[qc.Result]],
    results_folder: t.Optional[str] = None,
) -> t.List[QCMetric]:
    """Convert grouped ``contraqctor`` results into ``QCMetric`` objects.

    Parameters
    ----------
    results : dict
        Mapping of group name to list of ``contraqctor`` ``Result`` objects, as
        returned by ``contraqctor.qc.Runner.run_all``. The keys are the group
        names passed to ``runner.add_suite`` (e.g. ``"Data contract"``,
        ``"HarpHub"``, ``"HarpLickometerRight"``, ``"DynamicForaging"``), and
        each ``Result`` carries its suite's class name as ``suite_name`` (e.g.
        ``"ContractTestSuite"``, ``"HarpDeviceTestSuite"``, ``"CameraTestSuite"``,
        ``"DynamicForagingQcSuite"``). A ``None`` key becomes :data:`NO_GROUP`.
    results_folder : str, optional
        Directory to save figure assets into. If ``None``, assets are skipped.

    Returns
    -------
    list of QCMetric
        One metric per result, named after the test and tagged
        ``{"type": QC_TYPE, "group": group}``.

    Examples
    --------
    The dynamic foraging runner produces results grouped roughly like::

        {
            "HarpBehavior": [<HarpDeviceTestSuite results>],
            "HarpLickometerRight": [<HarpDeviceTestSuite results>],
            "HarpHub": [<HarpHubTestSuite results>],
            "DynamicForaging": [<DynamicForagingQcSuite results>],
        }

    The same suite (and so the same test names) runs once per device, so the
    metric name is just the test name and the group goes in the tags (the group
    implies the suite), e.g. a ``HarpDeviceTestSuite`` result in the
    ``"HarpBehavior"`` group yields::

        name = "test_has_whoami"
        tags = {"type": "Harp QC Suite", "group": "HarpBehavior"}
    """
    metrics: t.List[QCMetric] = []
    for group, group_results in results.items():
        group_name = group if group is not None else NO_GROUP
        for result in group_results:
            status = QCStatus(
                evaluator="Automated",
                status=STATUS_CONVERTER[result.status],
                timestamp=now_utc(),
            )
            metrics.append(
                QCMetric(
                    name=result.test_name,
                    modality=Modality.BEHAVIOR,
                    stage=Stage.RAW,
                    value=to_builtin(result.result),
                    status_history=[status],
                    description=f"Test: {result.description} // Message: {result.message}",
                    reference=_save_asset(result, results_folder),
                    tags={"type": QC_TYPE, "group": group_name},
                )
            )
    return metrics


def contract_qc_metrics(
    dataset: contract.Dataset, results_folder: t.Optional[str] = None
) -> t.List[QCMetric]:
    """Run the dynamic foraging contract QA over a dataset and convert results.

    Parameters
    ----------
    dataset : contraqctor.contract.Dataset
        A dynamic foraging dataset (e.g. ``RawDataLoader.dataset``).
    results_folder : str, optional
        Directory to save figure assets into. If ``None``, assets are skipped.

    Returns
    -------
    list of QCMetric
        Converted QA metrics for every suite the runner wires up.
    """
    runner = make_qc_runner(dataset)
    return results_to_metrics(runner.run_all(), results_folder)
