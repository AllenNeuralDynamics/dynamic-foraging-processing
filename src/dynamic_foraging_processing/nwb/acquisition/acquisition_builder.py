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
from dynamic_foraging_processing.utils.rewards import ManualWaterTimes, get_reward_deliveries


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

    def _software_event_times(self, stream_name: str) -> np.ndarray:
        """Get one ``Behavior/SoftwareEvents`` stream's event timestamps.

        Parameters
        ----------
        stream_name : str
            The software-event stream to read (e.g. ``"LeftManualWater"``).

        Returns
        -------
        numpy.ndarray
            The stream's event timestamps, or an empty array when the stream is
            absent. Only the timestamps are used; these events carry no payload
            this pipeline reads.
        """
        try:
            data = (
                self.loader.dataset.at("Behavior").at("SoftwareEvents").at(stream_name).load().data
            )
        except (KeyError, FileNotFoundError):
            return np.array([])
        return data.index.to_numpy()

    def get_manual_water_times(self, *, is_right: bool) -> ManualWaterTimes:
        """Get one lick port's experimenter-triggered water times.

        The acquisition software emits four side-specific streams:
        ``{Left,Right}ManualWater`` for water given at an arbitrary moment and
        ``{Left,Right}ManualAutoReward`` for water triggered to land on the go
        cue. The side comes from the stream name, so no payload inspection is
        needed. Each stream is optional -- a session where the experimenter gave
        no water of that kind has no file -- and reads as an empty array.

        Parameters
        ----------
        is_right : bool
            ``True`` for the right lick port, ``False`` for the left.

        Returns
        -------
        ManualWaterTimes
            This port's ``unaligned`` and ``go_cue_aligned`` event timestamps.
        """
        side = "Right" if is_right else "Left"
        return ManualWaterTimes(
            unaligned=self._software_event_times(f"{side}ManualWater"),
            go_cue_aligned=self._software_event_times(f"{side}ManualAutoReward"),
        )

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
        manual_water: ManualWaterTimes,
        *,
        port_column: str,
        name: str,
        side_label: str,
    ) -> AcquisitionSeries:
        """Build one lick port's reward-delivery series with reward annotations.

        Only valve-open events (``port_column`` is truthy) are reward
        deliveries; the ``data`` field annotates each as earned, auto, manual, or
        manual-go-cue-aligned via :func:`get_reward_deliveries`. Every valve
        opening is reported, so the series is a complete record of the water
        delivered at this port.

        Parameters
        ----------
        writes : pandas.DataFrame
            ``OutputSet`` ``WRITE`` messages indexed by timestamp.
        trial_outcomes : pandas.DataFrame
            The ``TrialOutcome`` stream, indexed by trial timestamp.
        manual_water : ManualWaterTimes
            This port's experimenter-triggered water times, already side-specific
            (see :meth:`get_manual_water_times`).
        port_column : str
            Supply-port column for this side (``"SupplyPort0"`` left,
            ``"SupplyPort1"`` right).
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
        annotations = get_reward_deliveries(
            delivery_times,
            trial_outcomes,
            manual_water,
        )
        return AcquisitionSeries(
            name=name,
            data=annotations,
            timestamps=delivery_times,
            unit="second",
            description=(
                f"The reward delivery time of the {side_label} lick port. The data field "
                "annotates whether the reward was earned, auto (task-triggered free water), "
                "manual (experimenter water not aligned to a go cue), or "
                "manual_go_cue_aligned (experimenter water delivered at the go cue)"
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
        left_manual_water = self.get_manual_water_times(is_right=False)
        right_manual_water = self.get_manual_water_times(is_right=True)

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
                left_manual_water,
                port_column="SupplyPort0",
                name="left_reward_delivery_time",
                side_label="left",
            )
        )
        acquisiton_entries.append(
            self._reward_delivery_series(
                rewards,
                trial_outcomes,
                right_manual_water,
                port_column="SupplyPort1",
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
