"""Shared interface for the dynamic foraging QC stages.

``BaseQC`` is an optional base class so the QC stages (raw, processed) share a
single ``run`` interface. Each concrete stage computes its checks and returns
them as schema ``QCMetric`` objects, which a caller assembles into one
``QualityControl`` (see :func:`build_quality_control`).
"""

import abc
import typing as t

from aind_data_schema.core.quality_control import QCMetric


class BaseQC(abc.ABC):
    """Common interface for a QC stage.

    A QC stage takes some slice of a session's data and produces a flat list of
    ``QCMetric`` objects. Subclasses define what slice ``run`` consumes (raw
    acquisition data, processed tables, ...); the return type is shared so the
    metrics can be collected uniformly.
    """

    @abc.abstractmethod
    def run(self, *args: t.Any, **kwargs: t.Any) -> t.List[QCMetric]:
        """Run this stage's checks and return them as metrics.

        Returns
        -------
        list of QCMetric
            One metric per check, ready to assemble into a ``QualityControl``.
        """
        raise NotImplementedError
