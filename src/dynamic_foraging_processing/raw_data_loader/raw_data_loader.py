"""Raw data loader for dynamic foraging datasets."""

import logging
import os
import typing as t

import pandas as pd

from .temp_dataset import dataset as _build_dataset

if t.TYPE_CHECKING:
    from contraqctor.contract import Dataset

logger = logging.getLogger(__name__)


class RawDataLoader:
    """Loads raw dynamic foraging data using the aind-behavior-dynamic-foraging data contract."""

    def __init__(
        self,
        path: os.PathLike,
        dataset: t.Optional["Dataset"] = None,
        raise_on_error: bool = False,
    ):
        """Initialize the raw data loader.

        Parameters
        ----------
        path : os.PathLike
            Path to the dataset root directory.
        dataset : Dataset, optional
            Pre-built ``Dataset`` object. If ``None``, a ``Dataset`` is
            instantiated at the latest version using the data contract.
        raise_on_error : bool, optional
            If ``True``, raise on errors encountered while loading data.
            Defaults to ``False``.
        """
        if dataset is None:
            dataset = _build_dataset(path)
        self.path = path
        self.dataset = dataset
        self.raise_on_error = raise_on_error

    def get_all_raw_data(self) -> t.Dict[str, t.Union[pd.DataFrame, t.Dict]]:
        """Raw data loaded from ``self.dataset``.

        Returns
        -------
        dict of str to pandas.DataFrame or dict
            Mapping of stream name to the stream's loaded data.
        """
        streams = self.dataset.load_all()
        raw_data_dict: t.Dict[str, t.Union[pd.DataFrame, t.Dict]] = {}
        for stream in streams.iter_all():
            if stream.is_collection:  # only process leaf nodes for data
                err = stream.collect_errors()
                if err:
                    logger.warning(f"Collection stream {stream.name} has errors: {err}")
                    if self.raise_on_error:
                        raise ValueError(f"Collection stream {stream.name} has errors: {err}")
                continue

            if stream.has_error:
                logger.warning(f"Stream {stream.name} has error: {stream.collect_errors()}. Skipping")
                if self.raise_on_error:
                    raise ValueError(f"Stream {stream.name} has error: {stream.collect_errors()}")
                continue

            raw_data_dict[f"{stream.parent.name}.{stream.name}"] = stream.data
            logger.debug(f"Stream {stream.name} loaded successfully.")

        return raw_data_dict

    @property
    def raw_data_stream_descriptions(self) -> t.Dict[str, str]:
        """Descriptions for each stream in ``self.dataset``
        Will mainly be used for NWB.

        Returns
        -------
        dict of str to str
            Mapping of stream name to the stream's description.
        """
        return {
            f"{stream.parent.name}.{stream.name}": stream.description
            for stream in self.dataset.iter_all()
        }
