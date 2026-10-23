"""Run the dynamic foraging contract QA and convert its results to ``QCMetric`` objects.

The dynamic foraging contract QA (Harp devices, cameras, CSV streams, the data
contract, and task-specific checks) is provided by
``aind_behavior_dynamic_foraging.data_qc.make_qc_runner``. This module runs that
runner over a dataset and converts each ``contraqctor`` ``Result`` with
``aind_contraqctor_to_schema_export.results_to_metrics``, shared with VR
foraging: metrics are named ``"<test>:<group>"`` (plus ``":<n>"`` for a test that
yields several results), tagged ``{"type": "Harp QC Suite", "group": <group>}``,
and keep any figure and context the test reports.
"""

import typing as t

from aind_behavior_dynamic_foraging.data_qc.suite import make_qc_runner
from aind_contraqctor_to_schema_export import results_to_metrics
from aind_data_schema.core.quality_control import QCMetric
from contraqctor import contract

__all__ = ["contract_qc_metrics", "results_to_metrics"]


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
