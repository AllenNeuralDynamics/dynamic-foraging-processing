"""Behavior QC metrics computed from primitive arrays.

These functions are agnostic to where the data came from: the caller supplies
plain numpy arrays (lick times, per-trial choice codes), and the functions
return ``QCMetric`` objects. The lick-interval computation is ported from the
old ``aind-dynamic-foraging-qc`` capsule's ``calculate_lick_intervals``,
adapted to take arrays directly instead of a ``behavior.json`` dict.
"""

import typing as t
from pathlib import Path

import numpy as np

from dynamic_foraging_processing.qc._core.result import QCResult

#: Reference plot assets shared by the behavior metrics.
SIDE_BIAS_PLOT = "side_bias.png"
LICK_INTERVALS_PLOT = "lick_intervals.png"
LICK_LATENCY_PLOT = "lick_latency.png"


def _plot_reference(plot_name: str, results_folder: t.Optional[str]) -> str:
    """Build a result's plot reference: ``"<results-folder-name>/<plot>"``.

    Parameters
    ----------
    plot_name : str
        The plot's file name (e.g. ``"side_bias.png"``).
    results_folder : str or None
        Directory the plot is written to. When ``None`` (no plot written), the
        bare plot name is returned.

    Returns
    -------
    str
        ``"<results-folder-name>/<plot_name>"`` when ``results_folder`` is
        given, otherwise ``plot_name``.
    """
    if results_folder is None:
        return plot_name
    return f"{Path(results_folder).name}/{plot_name}"


def calculate_lick_intervals(
    left_lick_times: np.ndarray, right_lick_times: np.ndarray
) -> t.Dict[str, float]:
    """Compute inter-lick-interval percentages from left/right lick times.

    Ported from the old capsule's ``calculate_lick_intervals``; the inputs are
    arrays of lick timestamps (seconds) rather than a ``behavior.json`` dict.

    Parameters
    ----------
    left_lick_times : numpy.ndarray
        Timestamps (s) of left-port licks.
    right_lick_times : numpy.ndarray
        Timestamps (s) of right-port licks.

    Returns
    -------
    dict of str to float
        ``LeftLickIntervalPercent``, ``RightLickIntervalPercent``,
        ``SameSideIntervalPercent``, ``CrossSideIntervalPercent``, and
        ``ArtifactPercent``.
    """
    left = np.asarray(left_lick_times, dtype=float)
    right = np.asarray(right_lick_times, dtype=float)
    same_side_l = np.diff(left)
    same_side_r = np.diff(right)

    threshold = 0.05  # time in s to consider as a fast interval

    if (len(left) == 0) and (len(right) == 0):
        ArtifactPercent = 0.0
    else:
        all_licks = np.sort(np.concatenate([left, right]))
        all_diffs = np.sort(np.diff(all_licks))
        ArtifactPercent = float(np.mean(all_diffs < 0.0005) * 100)

    if len(left) > 1:
        same_side_l_frac = round(float(np.mean(same_side_l <= threshold)), 4)
        LeftLickIntervalPercent = same_side_l_frac * 100
    else:
        LeftLickIntervalPercent = 0.0

    if len(right) > 1:
        same_side_r_frac = round(float(np.mean(same_side_r <= threshold)), 4)
        RightLickIntervalPercent = same_side_r_frac * 100
    else:
        RightLickIntervalPercent = 0.0

    if len(right) > 0 and len(left) > 0:
        same_side_combined = np.concatenate([same_side_l, same_side_r])
        same_side_frac = round(
            float(np.sum(same_side_combined <= threshold) / (len(right) + len(left))), 4
        )
        # Pair each lick time with a direction code (+1 right, -1 left), sort by
        # time, then look at adjacent pairs that switch sides.
        right_dummy = np.ones(np.shape(right))
        left_dummy = np.negative(np.ones(np.shape(left)))
        stacked_right = np.column_stack((right_dummy, right))
        stacked_left = np.column_stack((left_dummy, left))
        merged_sorted = np.array(
            sorted(np.concatenate((stacked_right, stacked_left)), key=lambda x: x[1])
        )
        diffs = np.diff(merged_sorted[:, 0])
        cross_sides = np.array(
            [merged_sorted[i + 1, 1] - merged_sorted[i, 1] for i in np.where(diffs != 0)]
        )[0]
        cross_side_frac = round(
            float(np.sum(cross_sides <= threshold) / (len(left) + len(right))), 4
        )
        CrossSideIntervalPercent = cross_side_frac * 100
        SameSideIntervalPercent = same_side_frac * 100
    else:
        CrossSideIntervalPercent = 0.0
        SameSideIntervalPercent = 0.0

    return {
        "LeftLickIntervalPercent": LeftLickIntervalPercent,
        "RightLickIntervalPercent": RightLickIntervalPercent,
        "SameSideIntervalPercent": SameSideIntervalPercent,
        "CrossSideIntervalPercent": CrossSideIntervalPercent,
        "ArtifactPercent": ArtifactPercent,
    }


def side_bias_result(side_bias: np.ndarray, results_folder: t.Optional[str] = None) -> QCResult:
    """Build the average-side-bias ``QCResult`` from the trial-table column.

    The per-trial side bias is read directly from the trial table rather than
    recomputed; this check averages it over the session.

    Parameters
    ----------
    side_bias : numpy.ndarray
        Per-trial side bias from the trial table (right minus left); positive
        means a rightward bias. Computed over a sliding window, so a single
        no-response trial still has a value; entries are ``nan`` only when the
        mouse has not responded for many consecutive trials.
    results_folder : str, optional
        Directory the side-bias plot is written to; used to build the result's
        reference. When ``None``, the reference is the bare plot name.

    Returns
    -------
    QCResult
        Passes when ``abs(mean_bias) < 0.5``. Fails when the column is empty or
        all ``nan`` (``mean_bias`` is ``nan``). Tagged
        ``{"type": "Average_Side_Bias"}`` and referencing the side-bias plot.
    """
    values = np.asarray(side_bias, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        mean_bias = float("nan")
    else:
        mean_bias = round(float(np.nanmean(values)), 3)
    name = "average side bias"
    return QCResult(
        name=name,
        value=mean_bias,
        passed=bool(abs(mean_bias) < 0.5),  # nan comparisons are False -> fails
        description="Average side bias should be less than 0.5",
        reference=_plot_reference(SIDE_BIAS_PLOT, results_folder),
        tags={"type": "Average_Side_Bias"},
    )


def _first_lick_latency(go_cue: float, licks: np.ndarray) -> float:
    """Return the latency (s) from ``go_cue`` to the first lick after it.

    Parameters
    ----------
    go_cue : float
        The trial's go-cue time (s). ``nan`` yields ``nan`` (no lick compares
        greater than ``nan``).
    licks : numpy.ndarray
        Ascending lick timestamps (s).

    Returns
    -------
    float
        The first-lick latency, or ``nan`` when no lick follows the go cue.
    """
    after = licks[licks > go_cue]
    if after.size:
        return float(after[0] - go_cue)
    return float("nan")


def lick_latency_by_side(
    go_cue_times: t.Optional[np.ndarray],
    animal_response: t.Optional[np.ndarray],
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Return per-trial first-lick latency (s) after the go cue, split by chosen side.

    For each trial the latency is the time from the go cue to the first lick on
    the *chosen* side — left when ``animal_response == 0``, right when ``== 1``.
    Trials with no response (``2``), and any go cue after which the chosen side
    never licks, are ``nan``. Slow or one-sided licking is the diagnostic signal
    (e.g. deafness, or a non-functional lickport on one side).

    Parameters
    ----------
    go_cue_times : numpy.ndarray or None
        Per-trial go-cue times (s). ``None`` (column absent) is treated as no
        trials.
    animal_response : numpy.ndarray or None
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore). ``None``
        is treated as no trials.
    left_lick_times, right_lick_times : numpy.ndarray
        Timestamps (s) of left/right-port licks (need not be sorted).

    Returns
    -------
    tuple of numpy.ndarray
        The ``(left_latency, right_latency)`` per-trial arrays; each trial has a
        latency on at most its chosen side, ``nan`` elsewhere.
    """
    if go_cue_times is None or animal_response is None:
        return np.empty(0), np.empty(0)
    go_cue = np.asarray(go_cue_times, dtype=float)
    response = np.asarray(animal_response)
    left = np.sort(np.asarray(left_lick_times, dtype=float))
    right = np.sort(np.asarray(right_lick_times, dtype=float))
    left_latency = np.full(go_cue.shape, np.nan)
    right_latency = np.full(go_cue.shape, np.nan)
    for i, cue in enumerate(go_cue):
        if response[i] == 0:
            left_latency[i] = _first_lick_latency(cue, left)
        elif response[i] == 1:
            right_latency[i] = _first_lick_latency(cue, right)
    return left_latency, right_latency


def lick_latency_result(results_folder: t.Optional[str] = None) -> QCResult:
    """Build the review-only first-lick-latency ``QCResult``.

    A single review-only metric surfacing the lick-latency plot (per-side
    first-lick latency after the go cue): there is no computed value
    (``value=None``) and no automated pass/fail (``passed=None`` -> ``PENDING``).
    Tagged ``type="Lick_Interval"`` so it groups with the lick-interval metrics.

    Parameters
    ----------
    results_folder : str, optional
        Directory the lick-latency plot is written to; used to build the
        result's reference. When ``None``, the reference is the bare plot name.

    Returns
    -------
    QCResult
        The lick-latency result (``PENDING``, no value or auto pass/fail)
        referencing the lick-latency plot.
    """
    return QCResult(
        name="Lick_Latency",
        value=None,
        passed=None,  # no automated pass/fail -> PENDING for manual review
        description="First-lick latency (s) after the go cue, by side (review-only).",
        reference=_plot_reference(LICK_LATENCY_PLOT, results_folder),
        tags={"metric": "Lick_Latency", "type": "Lick_Interval"},
    )


def lick_interval_results(
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    results_folder: t.Optional[str] = None,
) -> t.List[QCResult]:
    """Build the four inter-lick-interval ``QCResult`` objects.

    Parameters
    ----------
    left_lick_times : numpy.ndarray
        Timestamps (s) of left-port licks.
    right_lick_times : numpy.ndarray
        Timestamps (s) of right-port licks.
    results_folder : str, optional
        Directory the lick-intervals plot is written to; used to build each
        result's reference. When ``None``, the reference is the bare plot name.

    Returns
    -------
    list of QCResult
        ``Left``/``Right``/``Cross Side`` lick-interval results (pass ``< 10``)
        and ``Artifact Percent`` (pass ``< 1``), all referencing the
        lick-intervals plot.
    """
    results = calculate_lick_intervals(left_lick_times, right_lick_times)
    specs = [
        ("Left Lick Interval (%)", round(results["LeftLickIntervalPercent"], 3), 10.0),
        ("Right Lick Interval (%)", round(results["RightLickIntervalPercent"], 3), 10.0),
        ("Cross Side Lick Interval (%)", round(results["CrossSideIntervalPercent"], 3), 10.0),
        ("Artifact Percent (%)", round(results["ArtifactPercent"], 3), 1.0),
    ]
    return [
        QCResult(
            name=name,
            value=value,
            passed=value < limit,
            description=f"{name} of inter-lick intervals; passes when < {limit}.",
            reference=_plot_reference(LICK_INTERVALS_PLOT, results_folder),
            tags={"metric": name, "type": "Lick_Interval"},
        )
        for name, value, limit in specs
    ]
