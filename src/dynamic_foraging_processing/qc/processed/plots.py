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
    LICK_LATENCY_PLOT,
    SIDE_BIAS_PLOT,
    lick_latency_by_side,
)

#: Vertical offset (in side-bias units) of the anti-bias lickspout-move markers
#: from the zero-bias line: rightward moves sit this far above it, leftward moves
#: this far below, so the pair reads as arrows straddling y=0.
_MOVE_MARKER_OFFSET = 0.06


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


def plot_lick_latency(
    go_cue_times: t.Optional[np.ndarray],
    animal_response: t.Optional[np.ndarray],
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    results_folder: str,
) -> str:
    """Save the per-side first-lick-latency histogram (response to the go cue).

    One overlaid histogram of the time from the go cue to the first lick on the
    chosen side (right and left, density-normalized). It shows how quickly the
    animal licks each side after the go cue; a shifted or absent distribution on
    one side is the diagnostic signal (e.g. deafness or a dead lickport).

    Parameters
    ----------
    go_cue_times, animal_response : numpy.ndarray or None
        Per-trial go-cue times and choice codes (see ``lick_latency_by_side``).
    left_lick_times, right_lick_times : numpy.ndarray
        Timestamps (s) of left/right-port licks.
    results_folder : str
        Directory to write ``lick_latency.png`` into.

    Returns
    -------
    str
        The plot filename (``lick_latency.png``), for use as a metric
        ``reference``.
    """
    left_latency, right_latency = lick_latency_by_side(
        go_cue_times, animal_response, left_lick_times, right_lick_times
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    bins = np.arange(0, 1, 0.05)
    ax.hist(right_latency[~np.isnan(right_latency)], bins=bins, alpha=0.5, label="R", density=True)
    ax.hist(left_latency[~np.isnan(left_latency)], bins=bins, alpha=0.5, label="L", density=True)
    ax.legend()
    ax.set_title("lick latency by lick side")
    ax.set_xlabel("Time from go cue (s)")
    ax.set_ylabel("density %")
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(Path(results_folder) / LICK_LATENCY_PLOT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return LICK_LATENCY_PLOT


def _legend_outside(ax: plt.Axes) -> None:
    """Place the axes legend just outside the right edge of the panel.

    The trial traces span the full width of these panels, so an in-axes legend
    sits on top of the data. Anchoring it outside keeps the trace readable; the
    figure is saved with ``bbox_inches="tight"``, so the legend is not clipped.
    """
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize="x-small")


def _add_bias_plot(
    ax: plt.Axes,
    side_bias: np.ndarray,
    anti_bias_left_water: t.Optional[np.ndarray] = None,
    anti_bias_right_water: t.Optional[np.ndarray] = None,
    anti_bias_lickspout_movement: t.Optional[np.ndarray] = None,
) -> None:
    """Draw the per-trial side-bias trace with anti-bias interventions overlaid.

    The anti-bias algorithm pushes against a developing side bias, so its two
    interventions are drawn on top of the bias trace they respond to: water
    interventions as short ticks at the top (right port) and bottom (left
    port), and lickspout movements as triangles straddling the zero-bias line,
    pointing (and coloured) in the direction the spout was moved.
    """
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

    plotted = False
    if anti_bias_right_water is not None:
        right = np.where(np.asarray(anti_bias_right_water, dtype=bool))[0]
        ax.vlines(right, 0.9, 1.0, color="darkred", linewidth=1, label="Anti-bias water (R)")
        plotted = True
    if anti_bias_left_water is not None:
        left = np.where(np.asarray(anti_bias_left_water, dtype=bool))[0]
        ax.vlines(left, -1.0, -0.9, color="darkblue", linewidth=1, label="Anti-bias water (L)")
        plotted = True
    if anti_bias_lickspout_movement is not None:
        move = np.asarray(anti_bias_lickspout_movement, dtype=float)
        # Direction-coded markers straddling the zero-bias line: a rightward move
        # is a red up-triangle above it, a leftward move a blue down-triangle
        # below it, matching the red/blue sides used elsewhere in the figure.
        for indices, offset, marker, color, side in (
            (np.where(move > 0)[0], _MOVE_MARKER_OFFSET, "^", "red", "R"),
            (np.where(move < 0)[0], -_MOVE_MARKER_OFFSET, "v", "blue", "L"),
        ):
            ax.plot(
                indices,
                np.full(len(indices), offset),
                marker=marker,
                color=color,
                linestyle="none",
                markersize=6,
                label=f"Anti-bias lickspout move ({side})",
            )
        plotted = True
    if plotted:
        _legend_outside(ax)


def _moved_trials(positions: t.Sequence[t.Optional[np.ndarray]]) -> np.ndarray:
    """Return the trial indices where any lickspout axis changed position.

    A move is detected as a change from the previous trial on any of the
    supplied axes, so a trial is flagged once no matter how many axes shifted.
    NaN samples (missing position readings) are ignored rather than counted as
    a change.

    Parameters
    ----------
    positions : sequence of numpy.ndarray or None
        Per-trial position arrays (one per axis); ``None`` and empty arrays are
        skipped.

    Returns
    -------
    numpy.ndarray
        Sorted, unique trial indices at which the lickspouts moved.
    """
    moved: t.Set[int] = set()
    for position in positions:
        if position is None or len(position) < 2:
            continue
        values = np.asarray(position, dtype=float)
        delta = np.diff(values)
        changed = np.where(~np.isnan(delta) & (delta != 0))[0] + 1
        moved.update(int(index) for index in changed)
    return np.array(sorted(moved), dtype=int)


def _add_lickspout_position_plot(
    ax: plt.Axes,
    lickspout_x: t.Optional[np.ndarray],
    lickspout_y1: t.Optional[np.ndarray],
    lickspout_y2: t.Optional[np.ndarray],
    lickspout_z: t.Optional[np.ndarray],
    anti_bias_lickspout_movement: t.Optional[np.ndarray] = None,
) -> None:
    """Draw lickspout x/y/z positions relative to session start (mm).

    Trials where the spouts moved are marked with ticks along the bottom of the
    panel, split by who moved them: the anti-bias algorithm (same flag as the
    markers on the side-bias panel) versus the experimenter, which is any other
    change in position.
    """
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

    # Ticks sit in axes-fraction coordinates on y so they stay pinned to the
    # bottom of the panel whatever the position data's range is.
    transform = ax.get_xaxis_transform()
    automatic = np.array([], dtype=int)
    if anti_bias_lickspout_movement is not None:
        move = np.asarray(anti_bias_lickspout_movement, dtype=float)
        automatic = np.where(move != 0)[0]
        ax.vlines(
            automatic,
            0.0,
            0.08,
            transform=transform,
            color="g",
            linewidth=1,
            label="Automatic move",
        )
        plotted = True
    moved = _moved_trials([lickspout_x, lickspout_y1, lickspout_y2, lickspout_z])
    manual = np.setdiff1d(moved, automatic)
    if len(manual):
        ax.vlines(
            manual,
            0.0,
            0.08,
            transform=transform,
            color="k",
            linewidth=1,
            label="Manual move",
        )
        plotted = True
    if plotted:
        _legend_outside(ax)


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
    anti_bias_left_water: t.Optional[np.ndarray] = None,
    anti_bias_right_water: t.Optional[np.ndarray] = None,
    anti_bias_lickspout_movement: t.Optional[np.ndarray] = None,
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
    anti_bias_left_water, anti_bias_right_water : numpy.ndarray, optional
        Boolean per-trial arrays flagging anti-bias water interventions on each
        side; overlaid on the side-bias trace.
    anti_bias_lickspout_movement : numpy.ndarray, optional
        Per-trial signed lickspout displacement (mm) applied by the anti-bias
        algorithm; nonzero trials are marked on the side-bias trace and ticked
        as automatic moves on the lickspout-position panel, where any other
        change in position is ticked as a manual move.

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

    _add_bias_plot(
        ax[0],
        side_bias,
        anti_bias_left_water=anti_bias_left_water,
        anti_bias_right_water=anti_bias_right_water,
        anti_bias_lickspout_movement=anti_bias_lickspout_movement,
    )
    _add_lickspout_position_plot(
        ax[1],
        lickspout_x,
        lickspout_y1,
        lickspout_y2,
        lickspout_z,
        anti_bias_lickspout_movement=anti_bias_lickspout_movement,
    )
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
