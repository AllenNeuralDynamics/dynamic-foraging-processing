"""Acquisition builder for NWB acquisition module."""

import typing as t

import numpy as np
import pandas as pd

from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)
from dynamic_foraging_processing.nwb.utils import clean_for_nwb
from dynamic_foraging_processing.raw_data_loader import RawDataLoader
from dynamic_foraging_processing.utils.rewards import get_reward_deliveries


class LickSource(t.NamedTuple):
    """Location of one lick port's signal in the raw dataset.

    Attributes
    ----------
    device : str
        The Harp device node under ``Behavior`` (``"HarpBehavior"`` for the
        standard behavior board, ``"HarpLickometerLeft"`` /
        ``"HarpLickometerRight"`` for the lickometer board).
    stream : str
        The digital-input stream on that device (``"DigitalInputState"`` for
        the behavior board, ``"LickState"`` for the lickometer board).
    port : str
        The column (``"DIPort0"`` / ``"DIPort1"``
        for the behavior board, ``"Channel0"`` for the lickometer board).
    """

    device: str
    stream: str
    port: str


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

    def get_valve_writes(self) -> pd.DataFrame:
        """Get the raw valve command stream.

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

    def get_response_times(self) -> np.ndarray:
        """Get the per-trial ``Response`` software-event timestamps.

        The event fires when the animal's choice is registered, within
        milliseconds of the valve opening, so it anchors a reward delivery to
        its trial. Only the event timestamp is used; the payload's ``Item1``
        field nominally carries a response time but is unreliable (it can lag
        the event by thousands of seconds), so it is ignored.

        Returns
        -------
        numpy.ndarray
            The ``Response`` event timestamps, positionally aligned with the
            ``TrialOutcome`` stream.
        """
        responses = (
            self.loader.dataset.at("Behavior").at("SoftwareEvents").at("Response").load().data
        )
        return responses.index.to_numpy()

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

    def get_lick_times(self, device: str, stream_name: str, port: str) -> np.ndarray:
        """Get the lick times for one lick port from a Harp digital-input stream.

        On the standard behavior board licks are read from
        ``HarpBehavior``/``DigitalInputState``, with left licks on ``DIPort0``
        and right licks on ``DIPort1``. The lickometer board exposes each side
        as its own device (``HarpLickometerLeft`` / ``HarpLickometerRight``)
        with a ``LickState`` stream and a ``Channel0`` column.

        Parameters
        ----------
        device : str
            The Harp device node under ``Behavior`` (e.g. ``"HarpBehavior"`` for
            the standard behavior board).
        stream_name : str
            The digital-input stream to read licks from (e.g.
            ``"DigitalInputState"`` for the standard behavior board).
        port : str
            The column to read licks from (``"DIPort1"`` for the right lick
            port, ``"DIPort0"`` for the left lick port on the behavior board).

        Returns
        -------
        numpy.ndarray
            Sorted lick timestamps for the requested port, or an empty array
            when the stream is absent.
        """
        try:
            data = self.loader.dataset.at("Behavior").at(device).at(stream_name).load().data
        except (KeyError, FileNotFoundError):
            return np.array([])
        data = data[data["MessageType"] == "EVENT"]
        licks = data[data[port].fillna(False).astype(bool)]
        return licks.index.to_numpy()

    def _lick_time_series(
        self, *, source: LickSource, name: str, side_label: str
    ) -> AcquisitionSeries:
        """Build one lick port's lick-time series from a Harp digital-input stream.

        Parameters
        ----------
        source : LickSource
            The device, stream, and column locating this lick port's signal
            (e.g. ``LickSource("HarpBehavior", "DigitalInputState", "DIPort0")``
            for the standard behavior board's left port).
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
        lick_times = self.get_lick_times(source.device, source.stream, source.port)
        return AcquisitionSeries(
            name=name,
            data=np.ones(lick_times.shape[0], dtype=bool),
            timestamps=lick_times,
            unit="second",
            description=(
                f"The lick times of the {side_label} lick port ({source.port} on {source.device})."
            ),
        )

    def _reward_delivery_series(
        self,
        writes: pd.DataFrame,
        trial_outcomes: pd.DataFrame,
        manual_water: pd.DataFrame,
        response_times: np.ndarray,
        *,
        port_column: str,
        is_right: bool,
        name: str,
        side_label: str,
    ) -> AcquisitionSeries:
        """Build one lick port's reward-delivery series with reward annotations.

        Only valve-open events (``port_column`` is truthy) are reward
        deliveries; the ``data`` field annotates each as earned, manual, or auto
        via :func:`get_reward_deliveries`, which also drops deliveries on trials
        that did not pay out, so the series reports reward rather than every
        valve opening.

        Parameters
        ----------
        writes : pandas.DataFrame
            ``OutputSet`` ``WRITE`` messages indexed by timestamp.
        trial_outcomes : pandas.DataFrame
            The ``TrialOutcome`` stream, indexed by trial timestamp.
        manual_water : pandas.DataFrame
            The ``GiveManualWaterRight`` stream; the ``data`` column selects the
            side (``True`` right, ``False`` left).
        response_times : numpy.ndarray
            ``Response`` event timestamps, one per trial, used to match each
            delivery to its trial.
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
        delivery_times, annotations = get_reward_deliveries(
            delivery_times,
            trial_outcomes,
            manual_water_times,
            response_times,
        )
        return AcquisitionSeries(
            name=name,
            data=annotations,
            timestamps=delivery_times,
            unit="second",
            description=(
                f"The reward delivery time of the {side_label} lick port. The data field "
                "annotates whether the reward was earned, manual, or auto"
            ),
        )

    def build_acquisition(
        self,
        left_lick: LickSource = LickSource("HarpBehavior", "DigitalInputState", "DIPort0"),
        right_lick: LickSource = LickSource("HarpBehavior", "DigitalInputState", "DIPort1"),
    ) -> t.List[t.Union[AcquisitionSeries, AcquisitionTable]]:
        """Build the NWB acquisition entries.

        Parameters
        ----------
        left_lick, right_lick : LickSource, optional
            Where to read each side's lick times. Defaults to the standard
            behavior board (``HarpBehavior``/``DigitalInputState`` on
            ``DIPort0``/``DIPort1``). For the lickometer board pass, e.g.,
            ``LickSource("HarpLickometerLeft", "LickState", "Channel0")`` and
            ``LickSource("HarpLickometerRight", "LickState", "Channel0")``.

        Returns
        -------
        list of AcquisitionSeries or AcquisitionTable
            Acquisition entries to write to the NWB acquisition module.
        """
        rewards = self.get_valve_writes()
        trial_outcomes = self.get_trial_outcomes()
        manual_water = self.get_manual_water_times()
        response_times = self.get_response_times()

        acquisition_streams = self.loader.get_all_raw_data()
        acqusition_streams_descriptions = self.loader.raw_data_stream_descriptions

        acquisiton_entries: t.List[t.Union[AcquisitionSeries, AcquisitionTable]] = []

        for stream_name, stream_data in acquisition_streams.items():
            description = acqusition_streams_descriptions.get(stream_name)
            if description is None:
                description = ""
            acquisiton_entries.append(
                AcquisitionTable(
                    name=stream_name,
                    data=clean_for_nwb(stream_data),
                    description=description,
                )
            )

        acquisiton_entries.append(
            self._reward_delivery_series(
                rewards,
                trial_outcomes,
                manual_water,
                response_times,
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
                response_times,
                port_column="SupplyPort1",
                is_right=True,
                name="right_reward_delivery_time",
                side_label="right",
            )
        )
        acquisiton_entries.append(
            self._lick_time_series(
                source=left_lick,
                name="left_lick_time",
                side_label="left",
            )
        )
        acquisiton_entries.append(
            self._lick_time_series(
                source=right_lick,
                name="right_lick_time",
                side_label="right",
            )
        )

        return acquisiton_entries
