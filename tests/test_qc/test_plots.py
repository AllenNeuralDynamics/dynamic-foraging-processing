"""Tests for ``dynamic_foraging_processing.qc.processed.plots``."""

import os

import matplotlib.pyplot as plt
import numpy as np

from dynamic_foraging_processing.qc.processed import plots as _plots


def test_plot_lick_intervals_writes_file(tmp_path):
    """The lick-interval histogram is written and its filename returned."""
    name = _plots.plot_lick_intervals(
        np.array([1.0, 1.01, 2.0]), np.array([1.5, 1.6]), str(tmp_path)
    )
    assert name == _plots.LICK_INTERVALS_PLOT
    assert os.path.exists(tmp_path / name)


def test_plot_lick_latency_writes_file(tmp_path):
    """The per-side lick-latency histogram is written and its filename returned."""
    go_cue = np.array([0.0, 1.0, 2.0, 3.0])
    animal_response = np.array([0, 1, 2, 1])
    name = _plots.plot_lick_latency(
        go_cue,
        animal_response,
        np.array([0.3]),
        np.array([1.4, 3.2]),
        str(tmp_path),
    )
    assert name == _plots.LICK_LATENCY_PLOT
    assert os.path.exists(tmp_path / name)


def test_plot_lick_latency_no_trials(tmp_path):
    """Absent go-cue / response columns still write an (empty) latency figure."""
    name = _plots.plot_lick_latency(None, None, np.array([1.0]), np.array([2.0]), str(tmp_path))
    assert os.path.exists(tmp_path / name)


def test_time_to_trial_index_covers_all_branches():
    """Empty go cues and early/late event times map to the right indices."""
    # No go cues -> every event maps to -1.
    assert _plots._time_to_trial_index(np.array([]), np.array([1.0])) == [-1]
    # Event before first go cue -> -1; later events -> preceding cue index.
    result = _plots._time_to_trial_index(np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.5]))
    assert result == [-1, 1]


def test_plot_side_bias_full_inputs(tmp_path):
    """All optional panels render when every per-trial array is supplied."""
    animal_response = np.array([0, 1, 2, 1, 0, 1])
    side_bias = np.array([-0.2, 0.0, np.nan, 0.3, 0.1, 0.2])
    name = _plots.plot_side_bias(
        animal_response,
        side_bias,
        str(tmp_path),
        lickspout_x=np.array([1.0, 1.1, 1.0, 1.2, 1.1, 1.0]),
        lickspout_y1=np.array([2.0, 2.0, 2.1, 2.0, 2.0, 2.1]),
        lickspout_y2=np.array([3.0, 3.1, 3.0, 3.1, 3.0, 3.1]),
        lickspout_z=np.array([4.0, 4.0, 4.0, 4.1, 4.0, 4.0]),
        rewarded_left=np.array([True, False, False, False, True, False]),
        rewarded_right=np.array([False, True, False, True, False, True]),
        reward_probability_left=np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.6]),
        reward_probability_right=np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.4]),
        go_cue_times=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        autowater_left=np.array([1, 0, 0, 0, 0, 0]),
        autowater_right=np.array([0, 0, 0, 1, 0, 0]),
        manual_left_times=np.array([0.1, 3.6]),  # 0.1 -> -1, 3.6 -> trial index
        manual_right_times=np.array([5.6]),
        anti_bias_left_water=np.array([False, False, True, False, False, False]),
        anti_bias_right_water=np.array([False, False, False, False, False, True]),
        anti_bias_lickspout_movement=np.array([0.0, 0.5, 0.0, 0.0, -0.3, 0.0]),
    )
    assert name == _plots.SIDE_BIAS_PLOT
    assert os.path.exists(tmp_path / name)


def test_add_bias_plot_movement_with_empty_bias():
    """Lickspout markers fall back to y=0 when the bias trace is empty."""
    fig, ax = plt.subplots()
    # Empty bias but a nonzero movement -> heights come from ``np.zeros``.
    _plots._add_bias_plot(
        ax,
        np.array([]),
        anti_bias_lickspout_movement=np.array([1.0, 0.0]),
    )
    plt.close(fig)


def test_moved_trials_ignores_missing_and_unchanged_axes():
    """Only real per-trial changes count as moves; NaN and short arrays don't."""
    moved = _plots._moved_trials(
        [
            None,
            np.array([1.0]),  # too short to diff
            np.array([2.0, 2.0, 2.5, 2.5]),  # change at trial 2
            np.array([3.0, np.nan, 3.0, 4.0]),  # NaN ignored; change at trial 3
        ]
    )
    assert moved.tolist() == [2, 3]


def test_add_lickspout_position_plot_splits_automatic_and_manual_moves():
    """Anti-bias trials tick as automatic; every other move ticks as manual."""
    fig, ax = plt.subplots()
    _plots._add_lickspout_position_plot(
        ax,
        np.array([0.0, 1.0, 1.0, 2.0]),  # moves at trials 1 and 3
        None,
        None,
        None,
        anti_bias_lickspout_movement=np.array([0.0, 0.5, 0.0, 0.0]),
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "Automatic move" in labels
    assert "Manual move" in labels
    plt.close(fig)


def test_plot_side_bias_minimal_inputs(tmp_path):
    """With only choices supplied, the optional panels are skipped cleanly."""
    name = _plots.plot_side_bias(np.array([]), np.array([]), str(tmp_path))
    assert os.path.exists(tmp_path / name)


def test_plot_side_bias_empty_position_array(tmp_path):
    """An empty lickspout-position array is skipped without plotting."""
    name = _plots.plot_side_bias(
        np.array([0, 1]), np.array([0.1, -0.1]), str(tmp_path), lickspout_x=np.array([])
    )
    assert os.path.exists(tmp_path / name)
