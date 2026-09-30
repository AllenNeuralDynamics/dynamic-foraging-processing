"""Assemble the dynamic foraging ``QualityControl`` object.

Collects a flat list of metrics (behavior + contract QC) into a single
``QualityControl``, wiring up ``default_grouping`` so the QC portal groups
metrics by their ``type`` tag ("Harp QC Suite", "Side Bias", "Licking") and
then by ``group``.
"""

import typing as t

from aind_data_schema.core.quality_control import QCMetric, QualityControl

#: Split on ``type`` ("Harp QC Suite", "Side Bias", "Licking"), then ``group``
#: (device/stream for contract QC, plot for behavior QC). Every metric carries
#: exactly these two tag keys, so the portal can build the tree.
DEFAULT_GROUPING = ["type", "group"]


def build_quality_control(
    metrics: t.List[QCMetric],
    *,
    default_grouping: t.Optional[t.List[t.Union[str, t.Tuple[str, ...]]]] = None,
    allow_tag_failures: t.Optional[t.List[str]] = None,
    key_experimenters: t.Optional[t.List[str]] = None,
    notes: t.Optional[str] = None,
) -> QualityControl:
    """Wrap a flat list of metrics into a ``QualityControl`` object.

    Parameters
    ----------
    metrics : list of QCMetric
        All metrics (behavior + contract QC).
    default_grouping : list of str or tuple of str, optional
        Tag keys the portal groups by, one tree level per entry; a tuple splits
        on any of its keys at that level. Defaults to ``["type", "group"]``.
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
