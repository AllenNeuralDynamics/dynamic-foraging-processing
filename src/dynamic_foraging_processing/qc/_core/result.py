"""The per-check ``QCResult`` and its conversion to a schema ``QCMetric``.

A ``QCResult`` is the raw outcome of one behavior QC check (a name, value, and
pass/fail). It converts into an ``aind_data_schema`` ``QCMetric`` via
``to_metric``; the assembly step collects those metrics into a
``QualityControl``.
"""

import dataclasses
import typing as t

from aind_data_schema.core.quality_control import QCMetric

from dynamic_foraging_processing.qc._core.schema import bool_to_status, make_metric


@dataclasses.dataclass(frozen=True)
class QCResult:
    """The raw outcome of a single QC check.

    Attributes
    ----------
    name : str
        Metric name.
    value : Any
        The computed value.
    passed : bool
        Whether the check passed.
    description : str, optional
        Human-readable description.
    reference : str, optional
        Relative path to a supporting asset (e.g. a plot).
    tags : dict of str to str
        Grouping tags (e.g. ``{"behavior": name}``).
    """

    name: str
    value: t.Any
    passed: bool
    description: t.Optional[str] = None
    reference: t.Optional[str] = None
    tags: t.Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_metric(self) -> QCMetric:
        """Convert this result into a schema ``QCMetric``.

        Returns
        -------
        QCMetric
            A metric carrying this result's value, pass/fail status, and tags.
        """
        return make_metric(
            name=self.name,
            value=self.value,
            status=bool_to_status(self.passed),
            description=self.description,
            reference=self.reference,
            tags=self.tags,
        )


def to_metrics(results: t.Sequence[QCResult]) -> t.List[QCMetric]:
    """Convert a sequence of ``QCResult`` into schema ``QCMetric`` objects.

    Parameters
    ----------
    results : sequence of QCResult
        The per-check results to convert.

    Returns
    -------
    list of QCMetric
        One metric per result.
    """
    return [result.to_metric() for result in results]
