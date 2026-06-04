"""Timestamp matching utilities."""

import numpy as np


def find_closest_timestamps(
    timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
) -> np.ndarray:
    """Find the closest ``reference_timestamps`` for each entry in ``timestamps``.

    Parameters
    ----------
    timestamps : numpy.ndarray
        Query timestamps to match against ``reference_timestamps``.
    reference_timestamps : numpy.ndarray
        Candidate timestamps to search within.

    Returns
    -------
    numpy.ndarray
        Integer positions into ``reference_timestamps`` giving, for each query
        timestamp, the index of the nearest candidate timestamp.

    Examples
    --------
    Match query timestamps to the nearest reference timestamp. Note that
    references are not required to be sorted, multiple queries can map to the
    same reference, and ties resolve to the earlier reference:

    >>> import numpy as np
    >>> from dynamic_foraging_processing.utils import find_closest_timestamps
    >>> query_times = np.array([0.42, 0.05, 1.55, 0.95, 0.85])
    >>> reference_times = np.array([0.9, 0.1, 2.0, 0.4])
    >>> positions = find_closest_timestamps(query_times, reference_times)
    >>> positions
    array([3, 1, 2, 0, 0])
    >>> reference_times[positions]
    array([0.4, 0.1, 2. , 0.9, 0.9])

    Use the returned positions to align rows of a reference-indexed DataFrame:

    >>> import pandas as pd
    >>> trial_outcome_df = pd.DataFrame(
    ...     {"outcome": ["earned", "manual", "automatic", "earned"]},
    ...     index=reference_times,
    ... )
    >>> trial_outcome_df.iloc[positions]["outcome"].tolist()
    ['earned', 'manual', 'automatic', 'earned', 'earned']
    """
    timestamps = np.asarray(timestamps)
    reference_timestamps = np.asarray(reference_timestamps)

    if reference_timestamps.size == 0:
        raise ValueError("reference_timestamps must not be empty")

    sort_order = np.argsort(reference_timestamps)
    sorted_refs = reference_timestamps[sort_order]

    right_idx = np.searchsorted(sorted_refs, timestamps)
    right_idx = np.clip(right_idx, 1, len(sorted_refs) - 1)
    left_idx = right_idx - 1

    pick_left = (timestamps - sorted_refs[left_idx]) <= (sorted_refs[right_idx] - timestamps)
    closest_sorted = np.where(pick_left, left_idx, right_idx)
    return sort_order[closest_sorted]
