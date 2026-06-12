"""Tests for ``dynamic_foraging_processing.qc._plots``."""

import os

import numpy as np
import pandas as pd

from dynamic_foraging_processing.qc import _plots


def test_plot_lick_intervals_writes_file(tmp_path):
    """The lick-interval histogram is written and its filename returned."""
    name = _plots.plot_lick_intervals(
        np.array([1.0, 1.01, 2.0]), np.array([1.5, 1.6]), str(tmp_path)
    )
    assert name == _plots.LICK_INTERVALS_PLOT
    assert os.path.exists(tmp_path / name)


def test_time_to_trial_index_covers_all_branches():
    """Empty go cues and early/late event times map to the right indices."""
    # No go cues -> every event maps to -1.
    assert _plots._time_to_trial_index(np.array([]), np.array([1.0])) == [-1]
    # Event before first go cue -> -1; later events -> preceding cue index.
    result = _plots._time_to_trial_index(np.array([1.0, 2.0, 3.0]), np.array([0.5, 2.5]))
    assert result == [-1, 1]


def test_plot_side_bias_full_inputs(tmp_path):
    """All optional panels render when every input is supplied."""
    animal_response = np.array([0, 1, 2, 1, 0, 1])
    stage_positions = pd.DataFrame(
        {
            "x": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0],
            "y1": [2.0, 2.0, 2.1, 2.0, 2.0, 2.1],
            "y2": [3.0, 3.1, 3.0, 3.1, 3.0, 3.1],
            "z": [4.0, 4.0, 4.0, 4.1, 4.0, 4.0],
            "ignored_column": [0, 0, 0, 0, 0, 0],  # not in the color map -> skipped
        }
    )
    rewarded_history = pd.DataFrame(
        {
            "left": [True, False, False, False, True, False],
            "right": [False, True, False, True, False, True],
        }
    )
    auto_water = pd.DataFrame({"left": [1, 0, 0, 0, 0, 0], "right": [0, 0, 0, 1, 0, 0]})
    name = _plots.plot_side_bias(
        animal_response,
        str(tmp_path),
        stage_positions=stage_positions,
        rewarded_history=rewarded_history,
        reward_probability_left=np.array([0.5, 0.6, 0.4, 0.5, 0.5, 0.6]),
        reward_probability_right=np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.4]),
        go_cue_times=np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        auto_water=auto_water,
        manual_left_times=np.array([0.1, 3.6]),  # 0.1 -> -1, 3.6 -> trial index
        manual_right_times=np.array([5.6]),
        bias_window=3,
    )
    assert name == _plots.SIDE_BIAS_PLOT
    assert os.path.exists(tmp_path / name)


def test_plot_side_bias_minimal_inputs(tmp_path):
    """With only choices supplied, the optional panels are skipped cleanly."""
    name = _plots.plot_side_bias(np.array([]), str(tmp_path))
    assert os.path.exists(tmp_path / name)


def test_plot_side_bias_empty_stage_positions(tmp_path):
    """An empty stage-positions frame is handled without plotting positions."""
    name = _plots.plot_side_bias(
        np.array([0, 1]), str(tmp_path), stage_positions=pd.DataFrame({"x": []})
    )
    assert os.path.exists(tmp_path / name)
