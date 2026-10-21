"""The per-check ``QCResult`` and its conversion to a schema ``QCMetric``.

A ``QCResult`` is the raw outcome of one behavior QC check (a name, value, and
pass/fail). It converts into an ``aind_data_schema`` ``QCMetric`` via
``to_metric``; the assembly step collects those metrics into a
``QualityControl``.
"""

import dataclasses
import datetime
import typing as t
from zoneinfo import ZoneInfo

from aind_contraqctor_to_schema_export import to_builtin
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality

#: Pacific timezone used for behavior-metric timestamps (schema requires
#: timezone-aware datetimes).
SEATTLE_TZ = ZoneInfo("America/Los_Angeles")


@dataclasses.dataclass(frozen=True)
class QCResult:
    """The raw outcome of a single QC check.

    Attributes
    ----------
    name : str
        Metric name.
    value : Any
        The computed value.
    passed : bool or None
        Whether the check passed. ``None`` for a metric with no automated
        pass/fail (its value is reported but the status is left ``PENDING`` for
        manual review).
    description : str, optional
        Human-readable description.
    reference : str, optional
        Relative path to a supporting asset (e.g. a plot).
    tags : dict of str to str
        Grouping tags (e.g. ``{"type": "Licking", "group": "Lick Intervals"}``).
    """

    name: str
    value: t.Any
    passed: t.Optional[bool]
    description: t.Optional[str] = None
    reference: t.Optional[str] = None
    tags: t.Dict[str, str] = dataclasses.field(default_factory=dict)

    def to_metric(self) -> QCMetric:
        """Convert this result into a schema ``QCMetric``.

        The metric is a behavior-modality, raw-stage metric with one automated
        status, timestamped now (Seattle time): ``PASS`` / ``FAIL`` from
        ``passed``, or ``PENDING`` when ``passed`` is ``None``. Numpy values are
        converted to Python builtins.

        Returns
        -------
        QCMetric
            A metric carrying this result's value, pass/fail status, and tags.
        """
        if self.passed is None:
            status = Status.PENDING
        else:
            status = Status.PASS if self.passed else Status.FAIL
        return QCMetric(
            name=self.name,
            modality=Modality.BEHAVIOR,
            stage=Stage.RAW,
            value=to_builtin(self.value),
            status_history=[
                QCStatus(
                    evaluator="Automated",
                    status=status,
                    timestamp=datetime.datetime.now(SEATTLE_TZ),
                )
            ],
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
