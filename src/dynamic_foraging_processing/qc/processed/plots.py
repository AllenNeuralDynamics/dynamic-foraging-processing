"""QC plots for the dynamic foraging behavior metrics.

Ported from the old ``aind-dynamic-foraging-qc`` capsule, adapted to take
primitive per-trial and event-time arrays instead of a ``behavior.json`` dict.
The Agg backend is forced so the plots render headlessly (no display required).
"""

import typing as t
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dynamic_foraging_processing.qc.processed.behavior import (
    LICK_INTERVALS_PLOT,
    SIDE_BIAS_PLOT,
)


def plot_lick_intervals(
    left_lick_times: np.ndarray, right_lick_times: np.ndarray, results_folder: str
) -> str:
    """Save the five-panel inter-lick-interval histogram.

    Panels: left licks, right licks, left-to-right, right-to-left, all licks.

    Parameters
    ----------
    left_lick_times : numpy.ndarray
        Timestamps (s) of left-port licks.
    right_lick_times : numpy.ndarray
        Timestamps (s) of right-port licks.
    results_folder : str
        Directory to write ``lick_intervals.png`` into.

    Returns
    -------
    str
        The plot filename (``lick_intervals.png``), for use as a metric
        ``reference``.
    """
    left = np.asarray(left_lick_times, dtype=float)
    right = np.asarray(right_lick_times, dtype=float)

    fig, ax = plt.subplots(1, 5, figsize=(8, 3), sharex=True, sharey=True)
    titles = [
        "left licks",
        "right licks",
        "left to right licks",
        "right to left licks",
        "all licks",
    ]
    for axis, title in zip(ax, titles):
        axis.set_title(title)
        axis.set_xlabel("time (s)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax[0].set_xlim(-0.01, 0.3)
    ax[0].set_ylabel("counts")

    bins = np.linspace(-0.3, 0.3, 100)
    left_index = np.zeros_like(left)
    right_index = np.ones_like(right)
    all_licks = np.concatenate((left, right))
    all_index = np.concatenate((left_index, right_index))
    sort_order = np.argsort(all_licks)
    all_licks_sorted_diff = np.diff(all_licks[sort_order])
    index_sorted_diff = np.diff(all_index[sort_order])
    left_to_right = all_licks_sorted_diff[index_sorted_diff == 1]
    right_to_left = all_licks_sorted_diff[index_sorted_diff == -1]

    ax[0].hist(np.diff(left), bins=bins, color="red", alpha=0.7)
    ax[1].hist(np.diff(right), bins=bins, color="blue", alpha=0.7)
    ax[2].hist(left_to_right, bins=bins, color="black", alpha=0.7)
    ax[3].hist(right_to_left, bins=bins, color="black", alpha=0.7)
    ax[4].hist(all_licks_sorted_diff, bins=bins, color="black", alpha=0.7)

    fig.tight_layout()
    fig.savefig(Path(results_folder) / LICK_INTERVALS_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return LICK_INTERVALS_PLOT


def _add_bias_plot(ax: plt.Axes, side_bias: np.ndarray) -> None:
    """Draw the per-trial side-bias trace from the trial-table column."""
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Side Bias")
    ax.axhline(+0.7, color="r", linestyle="--")
    ax.axhline(-0.7, color="r", linestyle="--")
    ax.axhline(0, color="k", linestyle="--")
    ax.set_ylim([-1, +1])

    bias = np.asarray(side_bias, dtype=float)
    trials = np.arange(len(bias))
    ax.plot(trials, bias, "k", linewidth=2)
    if len(bias):
        ax.set_xlim([0, len(bias)])


def _add_lickspout_position_plot(
    ax: plt.Axes,
    lickspout_x: t.Optional[np.ndarray],
    lickspout_y1: t.Optional[np.ndarray],
    lickspout_y2: t.Optional[np.ndarray],
    lickspout_z: t.Optional[np.ndarray],
) -> None:
    """Draw lickspout x/y/z positions relative to session start (mm)."""
    ax.set_xlabel("Trial #")
    ax.set_ylabel("Lickspout Position \n relative to session start (mm)")
    positions = [
        ("X", lickspout_x, "r"),
        ("Y1", lickspout_y1, "b"),
        ("Y2", lickspout_y2, "lightblue"),
        ("Z", lickspout_z, "m"),
    ]
    plotted = False
    for label, position, color in positions:
        if position is None or len(position) == 0:
            continue
        values = np.asarray(position, dtype=float)
        # Values already come in relative to session start (normalized by the
        # trial-table builder), so plot them directly.
        ax.plot(values, color, label=label)
        plotted = True
    if plotted:
        ax.legend()


def _time_to_trial_index(go_cue_times: np.ndarray, times: np.ndarray) -> t.List[int]:
    """Map event times to the index of the most recent preceding go cue."""
    go_cue_times = np.asarray(go_cue_times, dtype=float)
    trial_index = []
    for event_time in np.asarray(times, dtype=float):
        if len(go_cue_times) == 0 or event_time < go_cue_times[0]:
            trial_index.append(-1)
        else:
            trial_index.append(int(np.where(go_cue_times < event_time)[0][-1]))
    return trial_index


def _add_behavior_plot(
    ax: plt.Axes,
    animal_response: np.ndarray,
    rewarded_left: t.Optional[np.ndarray],
    rewarded_right: t.Optional[np.ndarray],
    autowater_left: t.Optional[np.ndarray],
    autowater_right: t.Optional[np.ndarray],
    manual_left_times: t.Optional[np.ndarray],
    manual_right_times: t.Optional[np.ndarray],
    go_cue_times: t.Optional[np.ndarray],
) -> None:
    """Draw the per-trial behavior raster (choices, rewards, water)."""
    choices = np.asarray(animal_response)
    ax.vlines(np.where(choices == 1)[0], 0.8, 1, linewidth=1, color="gray", label="Choice")
    ax.vlines(np.where(choices == 0)[0], 0, 0.2, linewidth=1, color="gray")
    ax.vlines(np.where(choices == 2)[0], 0.4, 0.6, linewidth=1, color="darkviolet", label="ignore")

    if rewarded_left is not None:
        left_rewards = np.where(np.asarray(rewarded_left))[0]
        ax.vlines(left_rewards, -0.2, 0, linewidth=1, color="black", label="Earned Water")
    if rewarded_right is not None:
        right_rewards = np.where(np.asarray(rewarded_right))[0]
        ax.vlines(right_rewards, 1, 1.2, linewidth=1, color="black")

    if manual_right_times is not None and go_cue_times is not None:
        ax.vlines(
            _time_to_trial_index(go_cue_times, manual_right_times),
            1.2,
            1.4,
            linewidth=1,
            color="blue",
            label="Manual Water",
        )
    if manual_left_times is not None and go_cue_times is not None:
        ax.vlines(
            _time_to_trial_index(go_cue_times, manual_left_times),
            -0.4,
            -0.2,
            linewidth=1,
            color="blue",
        )

    if autowater_right is not None:
        ax.vlines(
            np.where(np.asarray(autowater_right) == 1)[0],
            1.2,
            1.4,
            linewidth=1,
            color="cyan",
            label="Auto Water",
        )
    if autowater_left is not None:
        ax.vlines(
            np.where(np.asarray(autowater_left) == 1)[0],
            -0.4,
            -0.2,
            linewidth=1,
            color="cyan",
        )

    ax.set_ylim([-0.4, 1.4])
    ax.set_xlim([0, len(choices)])
    ax.set_xlabel("Trial #")
    ax.set_yticks(
        [-0.3, -0.1, 0.1, 0.5, 0.9, 1.1, 1.3],
        labels=[
            "L Auto Water",
            "L Reward",
            "L Choice",
            "Ignore",
            "R Choice",
            "R Reward",
            "R Auto Water",
        ],
    )


def _add_reward_probabilities(
    ax: plt.Axes,
    reward_probability_left: t.Optional[np.ndarray],
    reward_probability_right: t.Optional[np.ndarray],
) -> None:
    """Draw the per-trial left/right reward probabilities."""
    ax.set_xlabel("Trial #")
    ax.set_ylim([0, 1])
    if reward_probability_left is not None:
        ax.plot(np.asarray(reward_probability_left, dtype=float), "b", label="Prob. L")
    if reward_probability_right is not None:
        ax.plot(np.asarray(reward_probability_right, dtype=float), "r", label="Prob. R")
    ax.legend()


def plot_side_bias(
    animal_response: np.ndarray,
    side_bias: np.ndarray,
    results_folder: str,
    *,
    lickspout_x: t.Optional[np.ndarray] = None,
    lickspout_y1: t.Optional[np.ndarray] = None,
    lickspout_y2: t.Optional[np.ndarray] = None,
    lickspout_z: t.Optional[np.ndarray] = None,
    rewarded_left: t.Optional[np.ndarray] = None,
    rewarded_right: t.Optional[np.ndarray] = None,
    reward_probability_left: t.Optional[np.ndarray] = None,
    reward_probability_right: t.Optional[np.ndarray] = None,
    go_cue_times: t.Optional[np.ndarray] = None,
    autowater_left: t.Optional[np.ndarray] = None,
    autowater_right: t.Optional[np.ndarray] = None,
    manual_left_times: t.Optional[np.ndarray] = None,
    manual_right_times: t.Optional[np.ndarray] = None,
) -> str:
    """Save the four-panel side-bias figure.

    Panels: per-trial side bias, lickspout position, behavior raster, and
    reward probabilities.

    Parameters
    ----------
    animal_response : numpy.ndarray
        Per-trial choice codes (``0`` left, ``1`` right, ``2`` ignore).
    side_bias : numpy.ndarray
        Per-trial side bias from the trial table (right minus left).
    results_folder : str
        Directory to write ``side_bias.png`` into.
    lickspout_x, lickspout_y1, lickspout_y2, lickspout_z : numpy.ndarray, optional
        Per-trial lickspout positions (one array each).
    rewarded_left, rewarded_right : numpy.ndarray, optional
        Boolean per-trial earned-reward arrays.
    reward_probability_left, reward_probability_right : numpy.ndarray, optional
        Per-trial reward probabilities.
    go_cue_times : numpy.ndarray, optional
        Go-cue timestamps (s), used to map manual-water times to trials.
    autowater_left, autowater_right : numpy.ndarray, optional
        Per-trial autowater indicator arrays.
    manual_left_times, manual_right_times : numpy.ndarray, optional
        Manual-water delivery timestamps (s).

    Returns
    -------
    str
        The plot filename (``side_bias.png``), for use as a metric
        ``reference``.
    """
    fig, ax = plt.subplots(nrows=4, figsize=(10, 12))
    for axis in ax:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    _add_bias_plot(ax[0], side_bias)
    _add_lickspout_position_plot(ax[1], lickspout_x, lickspout_y1, lickspout_y2, lickspout_z)
    _add_behavior_plot(
        ax[2],
        animal_response,
        rewarded_left,
        rewarded_right,
        autowater_left,
        autowater_right,
        manual_left_times,
        manual_right_times,
        go_cue_times,
    )
    _add_reward_probabilities(ax[3], reward_probability_left, reward_probability_right)

    # Align the x-axis across every panel so trials line up vertically. The
    # panels are all indexed by trial, but some auto-scale (adding margins) while
    # others set [0, N]; pin them all to a common [0, n_trials].
    n_trials = max(len(np.asarray(side_bias)), len(np.asarray(animal_response)))
    if n_trials:
        for axis in ax:
            axis.set_xlim([0, n_trials])

    fig.savefig(Path(results_folder) / SIDE_BIAS_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return SIDE_BIAS_PLOT
