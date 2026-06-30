"""Tests for ``dynamic_foraging_processing.nwb.utils``."""

import json
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from dynamic_foraging_processing.nwb.utils import (
    clean_dataframe_for_nwb,
    convert_datetimes_to_iso_string,
    convert_values_in_nested_structure,
)


class _Side(Enum):
    """Sample enum used to check enum-to-value conversion."""

    LEFT = "left"
    RIGHT = "right"


def test_convert_values_in_nested_structure_recurses_dicts_and_lists():
    """Conversion reaches values nested inside dicts and lists, leaving others."""
    data = {"a": [1, 2], "b": {"c": 3, "d": 4}}

    result = convert_values_in_nested_structure(
        data,
        check_fn=lambda x: isinstance(x, int) and x % 2 == 0,
        convert_fn=lambda x: x * 10,
    )

    assert result == {"a": [1, 20], "b": {"c": 3, "d": 40}}


def test_convert_values_in_nested_structure_scalar_passthrough():
    """A scalar that fails the check is returned unchanged."""
    assert (
        convert_values_in_nested_structure(5, check_fn=lambda x: False, convert_fn=lambda x: 0) == 5
    )


def test_convert_datetimes_to_iso_string_nested():
    """Datetimes nested in dicts/lists become ISO strings; other values stay."""
    dt = datetime(2026, 6, 30, 12, 0, 0)
    data = {"when": dt, "tags": [dt, "x"]}

    result = convert_datetimes_to_iso_string(data)

    assert result == {"when": "2026-06-30T12:00:00", "tags": ["2026-06-30T12:00:00", "x"]}


def test_clean_dataframe_replaces_none_with_nan():
    """``None`` values become ``NaN`` for NWB compatibility."""
    df = pd.DataFrame({"x": [1.0, None, 3.0]})

    cleaned = clean_dataframe_for_nwb(df)

    assert cleaned["x"].isna().tolist() == [False, True, False]
    assert cleaned["x"].dropna().tolist() == [1.0, 3.0]


def test_clean_dataframe_unwraps_enums():
    """Enum cells are replaced by their ``.value``."""
    df = pd.DataFrame({"side": [_Side.LEFT, _Side.RIGHT]})

    cleaned = clean_dataframe_for_nwb(df)

    assert cleaned["side"].tolist() == ["left", "right"]


def test_clean_dataframe_serializes_dicts_with_datetimes():
    """Dict cells are JSON-encoded with nested datetimes converted to ISO."""
    df = pd.DataFrame({"payload": [{"t": datetime(2026, 6, 30), "n": 1}]})

    cleaned = clean_dataframe_for_nwb(df)

    assert cleaned["payload"].iloc[0] == json.dumps({"t": "2026-06-30T00:00:00", "n": 1})


def test_clean_dataframe_leaves_plain_columns():
    """Scalar non-dict, non-enum columns are left as-is."""
    df = pd.DataFrame({"n": [1, 2, 3], "s": ["a", "b", "c"]})

    cleaned = clean_dataframe_for_nwb(df)

    assert cleaned["n"].tolist() == [1, 2, 3]
    assert cleaned["s"].tolist() == ["a", "b", "c"]
    assert not any(isinstance(v, np.ndarray) for v in cleaned["s"])
