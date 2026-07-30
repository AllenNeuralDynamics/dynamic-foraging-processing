"""Schema helpers for building dynamic foraging quality-control objects.

These helpers stamp the boilerplate that every ``QCMetric`` shares (modality,
stage, evaluator, timestamps) and convert ``contraqctor`` statuses onto the
``aind_data_schema`` quality-control schema (v2.4.1+).
"""

import datetime
import typing as t
from zoneinfo import ZoneInfo

import numpy as np
from aind_data_schema.core.quality_control import QCMetric, QCStatus, Stage, Status
from aind_data_schema_models.modalities import Modality
from contraqctor import qc

#: Pacific timezone used for behavior-metric timestamps (schema requires
#: timezone-aware datetimes).
SEATTLE_TZ = ZoneInfo("America/Los_Angeles")

#: Map ``contraqctor`` test statuses onto schema statuses. Warnings become
#: ``PENDING`` (needs review); skips count as passing.
STATUS_CONVERTER: t.Dict[qc.Status, Status] = {
    qc.Status.PASSED: Status.PASS,
    qc.Status.SKIPPED: Status.PASS,
    qc.Status.WARNING: Status.PENDING,
    qc.Status.FAILED: Status.FAIL,
    qc.Status.ERROR: Status.FAIL,
}


def now_seattle() -> datetime.datetime:
    """Return the current timezone-aware time in the Seattle timezone.

    Returns
    -------
    datetime.datetime
        Timezone-aware current time.
    """
    return datetime.datetime.now(SEATTLE_TZ)


def now_utc() -> datetime.datetime:
    """Return the current timezone-aware time in UTC.

    Returns
    -------
    datetime.datetime
        Timezone-aware current UTC time.
    """
    return datetime.datetime.now(datetime.timezone.utc)


def bool_to_status(
    passed: t.Optional[bool], timestamp: t.Optional[datetime.datetime] = None
) -> QCStatus:
    """Convert a boolean pass/fail (or ``None``) into an automated ``QCStatus``.

    Parameters
    ----------
    passed : bool or None
        ``True`` for a passing metric, ``False`` for a failing one, and ``None``
        for a metric with no automated pass/fail — the value is reported but the
        judgment is deferred, so the status is ``PENDING`` (needs manual review).
    timestamp : datetime.datetime, optional
        Timezone-aware evaluation time. Defaults to the current Seattle time.

    Returns
    -------
    QCStatus
        An ``"Automated"`` status with ``PASS``, ``FAIL``, or ``PENDING``.
    """
    timestamp = timestamp if timestamp is not None else now_seattle()
    if passed is None:
        status = Status.PENDING
    else:
        status = Status.PASS if passed else Status.FAIL
    return QCStatus(evaluator="Automated", status=status, timestamp=timestamp)


def to_builtin(value: t.Any) -> t.Any:
    """Convert numpy scalars/arrays to JSON-serializable Python builtins.

    Parameters
    ----------
    value : Any
        A value that may be a numpy scalar or array.

    Returns
    -------
    Any
        The equivalent Python builtin (``list`` for arrays, ``item()`` for
        scalars), or ``value`` unchanged when it is not a numpy type.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_metric(
    *,
    name: str,
    value: t.Any,
    status: QCStatus,
    description: t.Optional[str] = None,
    reference: t.Optional[str] = None,
    tags: t.Optional[t.Dict[str, str]] = None,
    modality: Modality = Modality.BEHAVIOR,
    stage: Stage = Stage.RAW,
) -> QCMetric:
    """Build a ``QCMetric`` with shared modality/stage/tag boilerplate.

    Replaces the old capsule's ``create_evaluation`` + per-metric wiring.

    Parameters
    ----------
    name : str
        Metric name.
    value : Any
        Metric value; numpy types are converted to Python builtins.
    status : QCStatus
        The single status entry for this metric's ``status_history``.
    description : str, optional
        Human-readable description.
    reference : str, optional
        Relative path to a supporting asset (e.g. a plot).
    tags : dict of str to str, optional
        Grouping tags (e.g. ``{"behavior": name}``). Defaults to ``{}``.
    modality : Modality, optional
        Defaults to ``Modality.BEHAVIOR``.
    stage : Stage, optional
        Defaults to ``Stage.RAW``.

    Returns
    -------
    QCMetric
        The assembled metric.
    """
    return QCMetric(
        name=name,
        modality=modality,
        stage=stage,
        value=to_builtin(value),
        status_history=[status],
        description=description,
        reference=reference,
        tags=tags if tags is not None else {},
    )
