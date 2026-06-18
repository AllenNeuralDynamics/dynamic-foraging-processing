"""Tests for ``dynamic_foraging_processing.qc.processed.behavior``."""

import numpy as np
import pytest

from dynamic_foraging_processing.qc.processed import behavior as _behavior


def test_calculate_lick_intervals_both_empty():
    """No licks on either side yields zeroed percentages."""
    result = _behavior.calculate_lick_intervals(np.array([]), np.array([]))
    assert result["ArtifactPercent"] == 0.0
    assert result["LeftLickIntervalPercent"] == 0.0
    assert result["RightLickIntervalPercent"] == 0.0
    assert result["CrossSideIntervalPercent"] == 0.0
    assert result["SameSideIntervalPercent"] == 0.0


def test_calculate_lick_intervals_single_side_only():
    """Licks on one side only skip the cross-side computation."""
    result = _behavior.calculate_lick_intervals(np.array([1.0]), np.array([]))
    # Single left lick: len(left) > 1 is False, right empty.
    assert result["LeftLickIntervalPercent"] == 0.0
    assert result["RightLickIntervalPercent"] == 0.0
    assert result["CrossSideIntervalPercent"] == 0.0


def test_calculate_lick_intervals_fast_and_cross_side():
    """Fast same-side and cross-side intervals are counted as percentages."""
    left = np.array([1.0, 1.01, 5.0])  # one fast left interval
    right = np.array([1.005, 1.02, 6.0])  # interleaved with left -> cross-side
    result = _behavior.calculate_lick_intervals(left, right)
    assert result["LeftLickIntervalPercent"] > 0
    assert result["RightLickIntervalPercent"] > 0
    assert result["CrossSideIntervalPercent"] > 0
    # Sub-millisecond artifacts: none here.
    assert result["ArtifactPercent"] == 0.0


def test_calculate_lick_intervals_detects_artifacts():
    """Sub-0.5ms gaps between licks count toward ArtifactPercent."""
    result = _behavior.calculate_lick_intervals(np.array([1.0, 1.0001]), np.array([2.0]))
    assert result["ArtifactPercent"] > 0


def test_compute_side_bias_balanced_and_all_ignore():
    """Bias is right-minus-left over responses; all-ignore gives nan."""
    assert _behavior.compute_side_bias(np.array([0, 1, 0, 1])) == 0.0
    assert _behavior.compute_side_bias(np.array([1, 1, 1, 0])) == pytest.approx(0.5)
    assert np.isnan(_behavior.compute_side_bias(np.array([2, 2, 2])))


def test_compute_rolling_bias_shapes_and_empty_window():
    """Rolling bias is nan where a window has no responses, set otherwise."""
    responses = np.array([2, 1, 1, 0])  # first trial is ignore -> nan
    bias, ci = _behavior.compute_rolling_bias(responses, window=2)
    assert bias.shape == (4,)
    assert ci.shape == (4, 2)
    assert np.isnan(bias[0])
    assert not np.isnan(bias[1])


def test_side_bias_result_pass_and_fail():
    """Side-bias result passes under 0.5 absolute bias and fails above it."""
    passing = _behavior.side_bias_result(np.array([0, 1, 0, 1]))
    assert passing.passed is True
    assert passing.reference == _behavior.SIDE_BIAS_PLOT
    assert passing.tags == {"behavior": "average side bias"}

    failing = _behavior.side_bias_result(np.array([1, 1, 1, 1]))
    assert failing.passed is False

    no_response = _behavior.side_bias_result(np.array([2, 2, 2]))
    assert no_response.passed is False
    assert np.isnan(no_response.value)


def test_lick_interval_results_names_and_count():
    """Four lick-interval results are produced with the expected names/tags."""
    results = _behavior.lick_interval_results(np.array([1.0, 1.01]), np.array([2.0, 2.01]))
    names = [r.name for r in results]
    assert names == [
        "Left Lick Interval (%)",
        "Right Lick Interval (%)",
        "Cross Side Lick Interval (%)",
        "Artifact Percent (%)",
    ]
    assert all(r.reference == _behavior.LICK_INTERVALS_PLOT for r in results)
    assert all(r.tags == {"behavior": r.name} for r in results)
