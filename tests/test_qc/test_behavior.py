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


def test_side_bias_result_pass_and_fail():
    """Side-bias result averages the column; passes under 0.5 absolute bias."""
    passing = _behavior.side_bias_result(np.array([-0.2, 0.1, np.nan, 0.1]))
    assert passing.passed is True
    assert passing.value == pytest.approx(0.0)
    assert passing.reference == _behavior.SIDE_BIAS_PLOT
    assert passing.tags == {"type": "Average_Side_Bias"}

    failing = _behavior.side_bias_result(np.array([0.8, 0.9, 1.0]))
    assert failing.passed is False
    assert failing.value == pytest.approx(0.9)

    no_response = _behavior.side_bias_result(np.array([np.nan, np.nan]))
    assert no_response.passed is False
    assert np.isnan(no_response.value)

    empty = _behavior.side_bias_result(np.array([]))
    assert empty.passed is False
    assert np.isnan(empty.value)


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
    assert all(r.tags == {"metric": r.name, "type": "Lick_Interval"} for r in results)


def test_first_lick_latency_after_go_cue_and_none():
    """The first lick after the go cue gives the latency; no later lick -> nan."""
    licks = np.array([0.5, 1.2, 2.0])
    assert _behavior._first_lick_latency(1.0, licks) == pytest.approx(0.2)
    # No lick after the cue -> nan.
    assert np.isnan(_behavior._first_lick_latency(2.5, licks))
    # A nan go cue has no lick strictly greater than it -> nan.
    assert np.isnan(_behavior._first_lick_latency(float("nan"), licks))


def test_lick_latency_by_side_splits_on_choice():
    """Latency is measured on the chosen side; other side / ignore trials are nan."""
    go_cue = np.array([0.0, 1.0, 2.0, 3.0])
    response = np.array([0, 1, 2, 1])  # left, right, ignore, right
    left_licks = np.array([0.3])  # after the trial-0 cue
    right_licks = np.array([1.4, 3.2])  # after the trial-1 and trial-3 cues
    left_latency, right_latency = _behavior.lick_latency_by_side(
        go_cue, response, left_licks, right_licks
    )
    assert left_latency[0] == pytest.approx(0.3)
    assert np.isnan(left_latency[1])  # right-choice trial has no left latency
    assert right_latency[1] == pytest.approx(0.4)
    assert right_latency[3] == pytest.approx(0.2)
    assert np.isnan(right_latency[2])  # ignore trial


def test_lick_latency_by_side_none_inputs_return_empty():
    """Absent go-cue / response columns yield empty latency arrays."""
    left, right = _behavior.lick_latency_by_side(None, None, np.array([1.0]), np.array([2.0]))
    assert left.size == 0 and right.size == 0


def test_lick_latency_result_is_pending_review_only():
    """The single latency result is review-only: no value, PENDING, plot ref."""
    result = _behavior.lick_latency_result("/data/my_results")
    assert result.name == "Lick_Latency"
    # No computed value yet, and no automated pass/fail (renders as PENDING).
    assert result.value is None
    assert result.passed is None
    assert result.reference == f"my_results/{_behavior.LICK_LATENCY_PLOT}"
    # Tagged Lick_Interval so it groups with the lick-interval metrics.
    assert result.tags == {"metric": "Lick_Latency", "type": "Lick_Interval"}


def test_reference_includes_results_folder_name():
    """With a results_folder, references are '<folder-name>/<plot>'."""
    side_bias = _behavior.side_bias_result(np.array([0.1]), "/data/my_results")
    assert side_bias.reference == f"my_results/{_behavior.SIDE_BIAS_PLOT}"

    licks = _behavior.lick_interval_results(
        np.array([1.0, 1.01]), np.array([2.0, 2.01]), "/data/my_results"
    )
    assert all(r.reference == f"my_results/{_behavior.LICK_INTERVALS_PLOT}" for r in licks)
