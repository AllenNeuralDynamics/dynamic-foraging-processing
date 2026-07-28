"""Assemble the dynamic foraging ``QualityControl`` object.

Collects a flat list of metrics (behavior + contract QC) into a single
``QualityControl``, wiring up ``default_grouping`` so the QC portal groups
metrics by their ``type`` and ``test_suite`` tags.
"""

import typing as t

from aind_data_schema.core.quality_control import QCMetric, QualityControl

#: Tag keys laid out as siblings at the top level of the QC portal.
DEFAULT_GROUPING = [("type", "test_suite")]


def build_quality_control(
    metrics: t.List[QCMetric],
    *,
    default_grouping: t.Optional[t.List[str]] = None,
    allow_tag_failures: t.Optional[t.List[str]] = None,
    key_experimenters: t.Optional[t.List[str]] = None,
    notes: t.Optional[str] = None,
) -> QualityControl:
    """Wrap a flat list of metrics into a ``QualityControl`` object.

    Parameters
    ----------
    metrics : list of QCMetric
        All metrics (behavior + contract QC).
    default_grouping : list of str, optional
        Tag keys the portal groups by. Defaults to ``[("type", "test_suite")]``.
    allow_tag_failures : list of str, optional
        Tag values whose metric failures should not fail the overall QC.
    key_experimenters : list of str, optional
        Experimenters associated with the session.
    notes : str, optional
        Free-text notes.

    Returns
    -------
    QualityControl
        The assembled quality-control object.
    """
    return QualityControl(
        metrics=metrics,
        default_grouping=default_grouping if default_grouping is not None else DEFAULT_GROUPING,
        allow_tag_failures=allow_tag_failures if allow_tag_failures is not None else [],
        key_experimenters=key_experimenters,
        notes=notes,
    )
