"""Acquisition builder for NWB acquisition module."""

import typing as t

import numpy as np
import pandas as pd

from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)
from dynamic_foraging_processing.raw_data_loader import RawDataLoader
from dynamic_foraging_processing.utils.rewards import get_annotated_rewards


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
            ``WRITE`` messages from the loaded ``OutputSet`` stream under
            ``Behavior/HarpBehavior``.
        """
        data = self.loader.dataset.at("Behavior").at("HarpBehavior").at("OutputSet").load().data
        data_write_messages = data[data["MessageType"] == "WRITE"]

        return data_write_messages

    def get_trial_outcomes(self) -> pd.DataFrame:
        """Get the ``TrialOutcome`` software-event stream.

        Returns
        -------
        pandas.DataFrame
            The ``TrialOutcome`` stream under ``Behavior/SoftwareEvents``,
            indexed by trial timestamp with a ``data`` payload column.
        """
        return (
            self.loader.dataset.at("Behavior").at("SoftwareEvents").at("TrialOutcome").load().data
        )

    def get_manual_water_times(self) -> pd.DataFrame:
        """Get the manual-water software-event stream.

        Returns
        -------
        pandas.DataFrame
            The ``GiveManualWaterRight`` stream under ``Behavior/SoftwareEvents``,
            indexed by event timestamp with a ``data`` column that is ``True``
            for right-port manual water and ``False`` for left-port manual water.
            An empty frame (with a ``data`` column) is returned when the stream
            is absent.
        """
        try:
            return (
                self.loader.dataset.at("Behavior")
                .at("SoftwareEvents")
                .at("GiveManualWaterRight")
                .load()
                .data
            )
        except (KeyError, FileNotFoundError):
            return pd.DataFrame({"data": []})

    def get_lick_times(self, stream_name: str, port: str) -> np.ndarray:
        """Get the lick times for one lick port from the behavior board DI ports.

        Licks are read from a HarpBehavior digital-input stream. On the standard
        behavior board this is ``DigitalInputState``; LicketySplit boards expose
        an equivalent stream under a different name. Left licks are on
        ``DIPort0`` and right licks on ``DIPort1``; a lick time is a timestamp at
        which that port's digital input is high.

        Parameters
        ----------
        stream_name : str
            The HarpBehavior digital-input stream to read licks from (e.g.
            ``"DigitalInputState"`` for the standard behavior board).
        port : str
            The DI port column to read licks from (``"DIPort1"`` for the right
            lick port, ``"DIPort0"`` for the left lick port).

        Returns
        -------
        numpy.ndarray
            Sorted lick timestamps for the requested port, or an empty array
            when the stream is absent.
        """
        try:
            data = (
                self.loader.dataset.at("Behavior")
                .at("HarpBehavior")
                .at(stream_name)
                .load()
                .data
            )
        except (KeyError, FileNotFoundError):
            return np.array([])
        licks = data[data[port].fillna(False).astype(bool)]
        return licks.index.to_numpy()

    def _lick_time_series(
        self, *, stream_name: str, port: str, name: str, side_label: str
    ) -> AcquisitionSeries:
        """Build one lick port's lick-time series from the behavior board DI ports.

        Parameters
        ----------
        stream_name : str
            The HarpBehavior digital-input stream to read licks from (e.g.
            ``"DigitalInputState"`` for the standard behavior board).
        port : str
            The DI port column to read licks from (``"DIPort1"`` for the right
            lick port, ``"DIPort0"`` for the left lick port).
        name : str
            Acquisition series name.
        side_label : str
            Human-readable side label used in the description.

        Returns
        -------
        AcquisitionSeries
            The lick-time series for this lick port. The ``data`` array marks
            each timestamp as a detected lick (``True``).
        """
        lick_times = self.get_lick_times(stream_name, port)
        return AcquisitionSeries(
            name=name,
            data=np.ones(lick_times.shape[0], dtype=bool),
            timestamps=lick_times,
            unit="second",
            description=(
                f"The lick times of the {side_label} lick port ({port} on the "
                "behavior board)."
            ),
        )

    def _reward_delivery_series(
        self,
        writes: pd.DataFrame,
        trial_outcomes: pd.DataFrame,
        manual_water: pd.DataFrame,
        *,
        port_column: str,
        is_right: bool,
        name: str,
        side_label: str,
    ) -> AcquisitionSeries:
        """Build one lick port's reward-delivery series with reward annotations.

        Only valve-open events (``port_column`` is truthy) are reward
        deliveries; the ``data`` field annotates each as earned, manual, or
        automatic via :func:`get_annotated_rewards`.

        Parameters
        ----------
        writes : pandas.DataFrame
            ``OutputSet`` ``WRITE`` messages indexed by timestamp.
        trial_outcomes : pandas.DataFrame
            The ``TrialOutcome`` stream, indexed by trial timestamp.
        manual_water : pandas.DataFrame
            The ``GiveManualWaterRight`` stream; the ``data`` column selects the
            side (``True`` right, ``False`` left).
        port_column : str
            Supply-port column for this side (``"SupplyPort0"`` left,
            ``"SupplyPort1"`` right).
        is_right : bool
            ``True`` for the right lick port, ``False`` for the left.
        name : str
            Acquisition series name.
        side_label : str
            Human-readable side label used in the description.

        Returns
        -------
        AcquisitionSeries
            The reward-delivery series for this lick port.
        """
        open_writes = writes[writes[port_column].fillna(False).astype(bool)]
        delivery_times = open_writes.index.to_numpy()
        manual_water_times = manual_water.index[manual_water["data"] == is_right].to_numpy()
        annotations = get_annotated_rewards(
            delivery_times,
            trial_outcomes,
            manual_water_times,
        )
        return AcquisitionSeries(
            name=name,
            data=annotations,
            timestamps=delivery_times,
            unit="second",
            description=(
                f"The reward delivery time of the {side_label} lick port. The data field "
                "annotates whether the reward was earned, manual, or automatic"
            ),
        )

    def build_acquisition(
        self, lick_stream_name: str = "DigitalInputState"
    ) -> t.List[t.Union[AcquisitionSeries, AcquisitionTable]]:
        """Build the NWB acquisition entries.

        Parameters
        ----------
        lick_stream_name : str, optional
            The HarpBehavior digital-input stream to read lick times from.
            Defaults to ``"DigitalInputState"`` for the Janelia boards;
            pass the equivalent stream name for LicketySplit boards.

        Returns
        -------
        list of AcquisitionSeries or AcquisitionTable
            Acquisition entries to write to the NWB acquisition module.
        """
        rewards = self.get_reward_delivery()
        trial_outcomes = self.get_trial_outcomes()
        manual_water = self.get_manual_water_times()

        # acquisition_streams = self.loader.get_all_raw_data()
        # acqusition_streams_descriptions = self.loader.raw_data_stream_descriptions

        acquisiton_entries: t.List[t.Union[AcquisitionSeries, AcquisitionTable]] = []

        # for stream_name, stream_data in acquisition_streams.items():
        #     description = acqusition_streams_descriptions.get(stream_name, "")
        #     acquisiton_entries.append(
        #         AcquisitionTable(
        #             name=stream_name,
        #             data=stream_data,
        #             description=description,
        #         )
        #     )

        acquisiton_entries.append(
            self._reward_delivery_series(
                rewards,
                trial_outcomes,
                manual_water,
                port_column="SupplyPort0",
                is_right=False,
                name="left_reward_delivery_time",
                side_label="left",
            )
        )
        acquisiton_entries.append(
            self._reward_delivery_series(
                rewards,
                trial_outcomes,
                manual_water,
                port_column="SupplyPort1",
                is_right=True,
                name="right_reward_delivery_time",
                side_label="right",
            )
        )
        acquisiton_entries.append(
            self._lick_time_series(
                stream_name=lick_stream_name,
                port="DIPort0",
                name="left_lick_time",
                side_label="left",
            )
        )
        acquisiton_entries.append(
            self._lick_time_series(
                stream_name=lick_stream_name,
                port="DIPort1",
                name="right_lick_time",
                side_label="right",
            )
        )

        return acquisiton_entries
