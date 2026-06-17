"""Behavior QC metrics computed from primitive arrays.

These functions are agnostic to where the data came from: the caller supplies
plain numpy arrays (lick times, per-trial choice codes), and the functions
return ``QCMetric`` objects. The lick-interval computation is ported from the
old ``aind-dynamic-foraging-qc`` capsule's ``calculate_lick_intervals``,
adapted to take arrays directly instead of a ``behavior.json`` dict.
"""

import typing as t

import numpy as np

from dynamic_foraging_processing.qc._core.result import QCResult

#: Reference plot assets shared by the behavior metrics.
SIDE_BIAS_PLOT = "side_bias.png"
LICK_INTERVALS_PLOT = "lick_intervals.png"

#: ``animal_response`` choice codes.
_LEFT, _RIGHT, _IGNORE = 0, 1, 2


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


def compute_side_bias(animal_response: np.ndarray) -> float:
    """Compute the average side bias over responded trials.

    Bias is ``mean(is_right) - mean(is_left)`` across trials where the animal
    responded (``ignore`` trials excluded). The result lies in ``[-1, +1]``;
    positive means a rightward bias.

    Parameters
    ----------
    animal_response : numpy.ndarray
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).

    Returns
    -------
    float
        The average side bias, or ``0.0`` if no trial was responded to.
    """
    responses = np.asarray(animal_response)
    responded = responses != _IGNORE
    if not responded.any():
        return 0.0
    chosen = responses[responded]
    return float(np.mean(chosen == _RIGHT) - np.mean(chosen == _LEFT))


def compute_rolling_bias(
    animal_response: np.ndarray, window: int = 20
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Compute a trailing-window side-bias trace with a normal-approx CI.

    For each trial, the bias is ``p_right - p_left`` over responded trials in
    the trailing ``window``; the confidence band is a 95% normal approximation
    for that difference. Replaces the GUI-precomputed ``B_Bias`` / ``B_Bias_CI``.

    Parameters
    ----------
    animal_response : numpy.ndarray
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).
    window : int, optional
        Number of trailing trials in the rolling window. Defaults to ``20``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(bias, ci)`` where ``bias`` has shape ``(n_trials,)`` and ``ci`` has
        shape ``(n_trials, 2)``. Trials with no responses are ``nan``.
    """
    responses = np.asarray(animal_response)
    n = len(responses)
    bias = np.full(n, np.nan)
    ci = np.full((n, 2), np.nan)
    for i in range(n):
        window_slice = responses[max(0, i - window + 1) : i + 1]
        chosen = window_slice[window_slice != _IGNORE]
        if len(chosen) == 0:
            continue
        p_right = float(np.mean(chosen == _RIGHT))
        b = 2.0 * p_right - 1.0  # p_right - p_left, since p_left + p_right == 1
        se = 2.0 * np.sqrt(p_right * (1.0 - p_right) / len(chosen))
        bias[i] = b
        ci[i] = (b - 1.96 * se, b + 1.96 * se)
    return bias, ci


def side_bias_result(animal_response: np.ndarray) -> QCResult:
    """Build the average-side-bias ``QCResult``.

    Parameters
    ----------
    animal_response : numpy.ndarray
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).

    Returns
    -------
    QCResult
        Passes when ``abs(mean_bias) < 0.5``; tagged ``{"behavior": ...}`` and
        referencing ``side_bias.png``.
    """
    mean_bias = compute_side_bias(animal_response)
    name = "average side bias"
    return QCResult(
        name=name,
        value=mean_bias,
        passed=abs(mean_bias) < 0.5,
        description="Average side bias over responded trials (right minus left).",
        reference=SIDE_BIAS_PLOT,
        tags={"behavior": name},
    )


def lick_interval_results(
    left_lick_times: np.ndarray, right_lick_times: np.ndarray
) -> t.List[QCResult]:
    """Build the four inter-lick-interval ``QCResult`` objects.

    Parameters
    ----------
    left_lick_times : numpy.ndarray
        Timestamps (s) of left-port licks.
    right_lick_times : numpy.ndarray
        Timestamps (s) of right-port licks.

    Returns
    -------
    list of QCResult
        ``Left``/``Right``/``Cross Side`` lick-interval results (pass ``< 10``)
        and ``Artifact Percent`` (pass ``< 1``), all referencing
        ``lick_intervals.png``.
    """
    results = calculate_lick_intervals(left_lick_times, right_lick_times)
    specs = [
        ("Left Lick Interval (%)", results["LeftLickIntervalPercent"], 10.0),
        ("Right Lick Interval (%)", results["RightLickIntervalPercent"], 10.0),
        ("Cross Side Lick Interval (%)", results["CrossSideIntervalPercent"], 10.0),
        ("Artifact Percent (%)", results["ArtifactPercent"], 1.0),
    ]
    return [
        QCResult(
            name=name,
            value=value,
            passed=value < limit,
            description=f"{name} of inter-lick intervals; passes when < {limit}.",
            reference=LICK_INTERVALS_PLOT,
            tags={"behavior": name},
        )
        for name, value, limit in specs
    ]
