"""Acquisition builder for NWB acquisition module."""

import typing as t

import pandas as pd

from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)
from dynamic_foraging_processing.raw_data_loader import RawDataLoader


class AcquisitionBuilder:
    """Builds the NWB acquisition module from raw dynamic foraging data."""

    def __init__(self, loader: RawDataLoader):
        """Initialize the acquisition builder.

        Parameters
        ----------
        loader : RawDataLoader
            Loader providing access to the dynamic foraging dataset.
        """
        self.loader = loader

    def get_reward_delivery(self) -> pd.DataFrame:
        """Get the reward delivery stream from the dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame from the loaded ``OutputSet`` stream under
            ``Behavior/HarpBehavior``.
        """
        data = self.loader.dataset.at("Behavior").at("HarpBehavior").at("OutputSet").load().data
        data_write_messages = data[data["MessageType"] == "WRITE"]

        return data_write_messages

    def build_acquisition(self) -> t.List[t.Union[AcquisitionSeries, AcquisitionTable]]:
        """Build the NWB acquisition entries.

        Returns
        -------
        list of AcquisitionSeries or AcquisitionTable
            Acquisition entries to write to the NWB acquisition module.
        """
        rewards = self.get_reward_delivery()
        acquisition_streams = self.loader.get_all_raw_data()
        acqusition_streams_descriptions = self.loader.raw_data_stream_descriptions

        acquisiton_entries: t.List[t.Union[AcquisitionSeries, AcquisitionTable]] = []

        for stream_name, stream_data in acquisition_streams.items():
            description = acqusition_streams_descriptions.get(stream_name, "")
            acquisiton_entries.append(
                AcquisitionTable(
                    name=stream_name,
                    data=stream_data,
                    description=description,
                )
            )

        # TODO: fix data so that it is array of annotations of whether the reward was earned, manual, or automatic
        # TODO: add left and right lick times
        acquisiton_entries.append(
            AcquisitionSeries(
                name="left_reward_delivery_time",
                data=rewards["SupplyPort0"].to_numpy(),
                timestamps=rewards.index.to_numpy(),
                unit="second",
                description=(
                    "The reward delivery time of the left lick port. The data field "
                    "annotates whether the reward was earned, manual, or automatic"
                ),
            )
        )
        acquisiton_entries.append(
            AcquisitionSeries(
                name="right_reward_delivery_time",
                data=rewards["SupplyPort1"].to_numpy(),
                timestamps=rewards.index.to_numpy(),
                unit="second",
                description=(
                    "The reward delivery time of the right lick port. The data field "
                    "annotates whether the reward was earned, manual, or automatic"
                ),
            )
        )

        return acquisiton_entries
