"""Acquisition builder for NWB acquisition module."""

import contraqctor
import pandas as pd

from dynamic_foraging_processing.process.acquisition.models import (
    AcquisitionSeries,
    NWBAcquisition,
)


class AcqusitionBuilder:
    """Builds the NWB acquisition module from raw dynamic foraging data."""

    def __init__(self, dataset: contraqctor.contract.Dataset):
        """Initialize the acquisition builder.

        Parameters
        ----------
        dataset : contraqctor.contract.Dataset
            Dataset providing access to the dynamic foraging data.
        """
        self.dataset = dataset

    def get_reward_delivery(self) -> pd.DataFrame:
        """Get the reward delivery stream from the dataset.

        Returns
        -------
        pandas.DataFrame
            DataFrame from the loaded ``OutputSet`` stream under
            ``Behavior/HarpBehavior``.
        """
        data = self.dataset.at("Behavior").at("HarpBehavior").at("OutputSet").load().data
        data_write_messages = data[data["MessageType"] == "WRITE"]

        return data_write_messages

    def build_acquisition(self) -> NWBAcquisition:
        """Build the NWB acquisition collection.

        Returns
        -------
        NWBAcquisition
            Object holding all acquisition series.
        """
        writes = self.get_reward_delivery()
        # TODO: fix data so that it is array of annotations of whether the reward was earned, manual, or automatic
        return NWBAcquisition(
            reward_delivery_left=AcquisitionSeries(
                name="reward_delivery_left",
                data=writes["SupplyPort0"].to_numpy(),
                timestamps=writes.index.to_numpy(),
                unit="second",
                description="The reward delivery time of the left lick port. The data field annotates whether the reward was earned, manual, or automatic",
            ),
            reward_delivery_right=AcquisitionSeries(
                name="reward_delivery_right",
                data=writes["SupplyPort1"].to_numpy(),
                timestamps=writes.index.to_numpy(),
                unit="second",
                description="The reward delivery time of the right lick port. The data field annotates whether the reward was earned, manual, or automatic",
            ),
        )
