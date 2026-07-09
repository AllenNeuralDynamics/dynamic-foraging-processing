"""Raw-data QC stage.

``RawQC`` wraps the ``contraqctor``-based contract QA (Harp devices, cameras,
CSV streams, the data contract, task-specific checks) into the ``BaseQC``
interface. It is a thin wrapper over :func:`contract_qc_metrics`; the metric
computation lives there.
"""

import typing as t

from aind_data_schema.core.quality_control import QCMetric
from contraqctor import contract

from dynamic_foraging_processing.qc._core.base import BaseQC
from dynamic_foraging_processing.qc.raw.contract_qa import contract_qc_metrics


class RawQC(BaseQC):
    """Contract QA over a loaded raw dynamic foraging dataset."""

    def run(
        self,
        acquisition: contract.Dataset,
        results_folder: t.Optional[str] = None,
    ) -> t.List[QCMetric]:
        """Run the contract QA over ``acquisition`` and return its metrics.

        Parameters
        ----------
        acquisition : contraqctor.contract.Dataset
            A loaded dynamic foraging dataset (e.g. ``RawDataLoader.dataset``).
        results_folder : str, optional
            Directory to save figure assets into. If ``None``, assets are
            skipped.

        Returns
        -------
        list of QCMetric
            Converted QA metrics for every suite the runner wires up.
        """
        return contract_qc_metrics(acquisition, results_folder)
