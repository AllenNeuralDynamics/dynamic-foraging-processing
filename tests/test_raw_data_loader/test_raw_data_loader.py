"""Tests for ``dynamic_foraging_processing.raw_data_loader.raw_data_loader``."""

from unittest.mock import MagicMock, patch

import pytest

from dynamic_foraging_processing.raw_data_loader import RawDataLoader


def _make_stream(
    name,
    parent_name="Parent",
    is_collection=False,
    has_error=False,
    errors=None,
    data="data",
    description="desc",
):
    """Build a mock stream with the attributes accessed by ``RawDataLoader``."""
    stream = MagicMock()
    stream.name = name
    stream.parent.name = parent_name
    stream.is_collection = is_collection
    stream.has_error = has_error
    stream.collect_errors.return_value = errors
    stream.data = data
    stream.description = description
    return stream


def _make_dataset(streams_for_load_all=None, streams_for_iter_all=None):
    """Build a mock dataset whose ``load_all`` and ``iter_all`` return the given streams."""
    dataset = MagicMock()
    loaded = MagicMock()
    loaded.iter_all.return_value = list(streams_for_load_all or [])
    dataset.load_all.return_value = loaded
    dataset.iter_all.return_value = list(streams_for_iter_all or [])
    return dataset


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_with_dataset():
    """Provided dataset is stored and ``_build_dataset`` is not called."""
    ds = _make_dataset()
    loader = RawDataLoader(path="some/path", dataset=ds)
    assert loader.dataset is ds
    assert loader.path == "some/path"
    assert loader.raise_on_error is False


def test_init_without_dataset_builds():
    """When dataset is ``None`` the temp builder is invoked."""
    built = _make_dataset()
    with patch(
        "dynamic_foraging_processing.raw_data_loader.raw_data_loader._build_dataset",
        return_value=built,
    ) as build_mock:
        loader = RawDataLoader(path="root")
    build_mock.assert_called_once_with("root")
    assert loader.dataset is built
    assert loader.raise_on_error is False


def test_init_raise_on_error_true():
    """``raise_on_error`` flag is stored on the instance."""
    loader = RawDataLoader(path="p", dataset=_make_dataset(), raise_on_error=True)
    assert loader.raise_on_error is True


# ---------------------------------------------------------------------------
# get_all_raw_data
# ---------------------------------------------------------------------------


def test_get_all_raw_data_collection_without_errors_is_skipped():
    """Collection streams without errors are skipped; leaves are collected."""
    collection = _make_stream("Col", is_collection=True, errors=None)
    leaf = _make_stream("Leaf", parent_name="Col", data="payload")
    ds = _make_dataset(streams_for_load_all=[collection, leaf])
    loader = RawDataLoader(path="p", dataset=ds)
    assert loader.get_all_raw_data() == {"Col.Leaf": "payload"}


def test_get_all_raw_data_collection_with_errors_no_raise():
    """Collection errors are logged and skipped when ``raise_on_error`` is False."""
    collection = _make_stream("Col", is_collection=True, errors=["boom"])
    ds = _make_dataset(streams_for_load_all=[collection])
    loader = RawDataLoader(path="p", dataset=ds, raise_on_error=False)
    assert loader.get_all_raw_data() == {}


def test_get_all_raw_data_collection_with_errors_raise():
    """Collection errors raise when ``raise_on_error`` is True."""
    collection = _make_stream("Col", is_collection=True, errors=["boom"])
    ds = _make_dataset(streams_for_load_all=[collection])
    loader = RawDataLoader(path="p", dataset=ds, raise_on_error=True)
    with pytest.raises(ValueError):
        loader.get_all_raw_data()


def test_get_all_raw_data_leaf_has_error_no_raise():
    """Leaf errors are tolerated when ``raise_on_error`` is False."""
    leaf = _make_stream(
        "Leaf",
        parent_name="Col",
        has_error=True,
        errors=["bad"],
        data="payload",
    )
    ds = _make_dataset(streams_for_load_all=[leaf])
    loader = RawDataLoader(path="p", dataset=ds, raise_on_error=False)
    assert "Col.Leaf" in loader.get_all_raw_data()


def test_get_all_raw_data_leaf_has_error_raise():
    """Leaf errors raise when ``raise_on_error`` is True."""
    leaf = _make_stream("Leaf", parent_name="Col", has_error=True, errors=["bad"])
    ds = _make_dataset(streams_for_load_all=[leaf])
    loader = RawDataLoader(path="p", dataset=ds, raise_on_error=True)
    with pytest.raises(ValueError):
        loader.get_all_raw_data()


# ---------------------------------------------------------------------------
# raw_data_stream_descriptions
# ---------------------------------------------------------------------------


def test_raw_data_stream_descriptions():
    """Maps ``parent.name`` to each stream description."""
    s1 = _make_stream("A", parent_name="P", description="d1")
    s2 = _make_stream("B", parent_name="P", description="d2")
    ds = _make_dataset(streams_for_iter_all=[s1, s2])
    loader = RawDataLoader(path="p", dataset=ds)
    assert loader.raw_data_stream_descriptions == {"P.A": "d1", "P.B": "d2"}
