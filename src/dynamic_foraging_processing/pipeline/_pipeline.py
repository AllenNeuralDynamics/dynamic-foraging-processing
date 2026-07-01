"""End-to-end dynamic foraging pipeline.

Ties the existing building blocks into a single flow: load a raw acquisition,
assemble the NWB file (base metadata + acquisition entries + trials table),
write it to disk, run the raw and processed QC stages, and write the combined
``QualityControl`` to disk.

The individual builders produce NWB-ready pydantic models
(``AcquisitionSeries`` / ``AcquisitionTable``) and a trials ``DataFrame``; this
module translates those into ``pynwb`` objects on top of the base file created
by :func:`aind_nwb_utils.utils.create_base_nwb_file`, then writes the result
with the Zarr backend.
"""

import os
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from aind_data_schema.core.quality_control import QualityControl
from aind_nwb_utils.utils import create_base_nwb_file
from hdmf.common import DynamicTable
from hdmf_zarr.nwb import NWBZarrIO

from dynamic_foraging_processing.nwb.acquisition import (
    AcquisitionBuilder,
    AcquisitionSeries,
    AcquisitionTable,
)
from dynamic_foraging_processing.nwb.acquisition.acquisition_builder import LickSource
from dynamic_foraging_processing.processing import TrialConfig, TrialTableBuilder
from dynamic_foraging_processing.qc import ProcessedQC, RawQC, build_quality_control
from dynamic_foraging_processing.raw_data_loader import RawDataLoader

#: Default lick-port sources on the standard behavior board.
_DEFAULT_LEFT_LICK = LickSource("HarpBehavior", "DigitalInputState", "DIPort0")
_DEFAULT_RIGHT_LICK = LickSource("HarpBehavior", "DigitalInputState", "DIPort1")

#: Trials-table columns NWB models natively; every other column is added as an
#: extra trial column.
_TRIAL_TIME_COLUMNS = ("start_time", "stop_time")


class Pipeline:
    """Package a raw dynamic foraging acquisition to NWB and run QC.

    The pipeline reuses the existing builders: :class:`AcquisitionBuilder` for
    the acquisition entries and lick / manual-water event times,
    :class:`TrialTableBuilder` for the trials table, and the :class:`RawQC` /
    :class:`ProcessedQC` stages assembled into a single ``QualityControl``.
    """

    def __init__(
        self,
        loader: RawDataLoader,
        *,
        left_lick: LickSource = _DEFAULT_LEFT_LICK,
        right_lick: LickSource = _DEFAULT_RIGHT_LICK,
        raise_on_error: bool = False,
    ):
        """Initialize the pipeline.

        Parameters
        ----------
        loader : RawDataLoader
            Loader providing access to the raw dynamic foraging dataset. Its
            ``path`` must hold the acquisition metadata files
            (``acquisition.json`` / ``session.json``, ``subject.json``, ...)
            that seed the base NWB file.
        left_lick, right_lick : LickSource, optional
            Where to read each side's lick times. Defaults to the standard
            behavior board (``HarpBehavior``/``DigitalInputState`` on
            ``DIPort0``/``DIPort1``). For the lickometer board pass, e.g.,
            ``LickSource("HarpLickometerLeft", "LickState", "Channel0")``.
        raise_on_error : bool, optional
            If ``True``, propagate errors raised while loading streams instead
            of logging and continuing. Passed through to the trials-table
            builder. Defaults to ``False``.
        """
        self.loader = loader
        self.left_lick = left_lick
        self.right_lick = right_lick
        self.raise_on_error = raise_on_error
        self._acquisition_builder = AcquisitionBuilder(loader)
        self._trial_table_builder = TrialTableBuilder(
            loader.dataset, raise_on_error=raise_on_error
        )

    def build_acquisition(self) -> t.List[t.Union[AcquisitionSeries, AcquisitionTable]]:
        """Build the NWB acquisition entries.

        Returns
        -------
        list of AcquisitionSeries or AcquisitionTable
            Acquisition entries to write to the NWB acquisition module.
        """
        return self._acquisition_builder.build_acquisition(
            left_lick=self.left_lick,
            right_lick=self.right_lick,
        )

    def build_trials(self) -> pd.DataFrame:
        """Build the trials table.

        Returns
        -------
        pandas.DataFrame
            One row per trial.
        """
        return self._trial_table_builder.build()

    def build_nwb(
        self,
        acquisition: t.Optional[t.List[t.Union[AcquisitionSeries, AcquisitionTable]]] = None,
        trials: t.Optional[pd.DataFrame] = None,
    ) -> pynwb.NWBFile:
        """Assemble the NWB file: base metadata, acquisition entries, and trials.

        The base file (subject / session metadata) is created from the loader's
        metadata directory via :func:`create_base_nwb_file`; the acquisition
        entries and trials table are then added onto it.

        Parameters
        ----------
        acquisition : list of AcquisitionSeries or AcquisitionTable, optional
            Acquisition entries to add. Built via :meth:`build_acquisition` when
            omitted.
        trials : pandas.DataFrame, optional
            Trials table to add. Built via :meth:`build_trials` when omitted.

        Returns
        -------
        pynwb.NWBFile
            The assembled NWB file.
        """
        if acquisition is None:
            acquisition = self.build_acquisition()
        if trials is None:
            trials = self.build_trials()

        nwb_file = create_base_nwb_file(self.loader.path)
        for entry in acquisition:
            if isinstance(entry, AcquisitionSeries):
                self._add_acquisition_series(nwb_file, entry)
            else:
                self._add_acquisition_table(nwb_file, entry)
        self._add_trials(nwb_file, trials)
        return nwb_file

    def write(self, nwb_file: pynwb.NWBFile, output_path: t.Union[str, os.PathLike]) -> None:
        """Write an NWB file to disk using the Zarr backend.

        Parameters
        ----------
        nwb_file : pynwb.NWBFile
            The NWB file to write (e.g. from :meth:`build_nwb`).
        output_path : os.PathLike
            Destination path for the ``.nwb.zarr`` store.
        """
        with NWBZarrIO(output_path, mode="w") as io:
            io.write(nwb_file)

    def run_qc(
        self,
        trials: pd.DataFrame,
        results_folder: t.Optional[str] = None,
    ) -> QualityControl:
        """Run the raw and processed QC stages and assemble one ``QualityControl``.

        Parameters
        ----------
        trials : pandas.DataFrame
            The trials table (e.g. from :meth:`build_trials`), consumed by the
            processed (behavior) QC stage.
        results_folder : str, optional
            Directory to write figure assets into so the metric references
            resolve. If ``None``, assets are skipped.

        Returns
        -------
        QualityControl
            The combined raw (contract QA) + processed (behavior) metrics as a
            single flat metric list.
        """
        left_lick_times, right_lick_times = self._lick_times()
        manual_left_times, manual_right_times = self._manual_water_times()

        raw_metrics = RawQC().run(self.loader.dataset, results_folder)
        processed_metrics = ProcessedQC().run(
            trials,
            left_lick_times,
            right_lick_times,
            results_folder,
            manual_left_times=manual_left_times,
            manual_right_times=manual_right_times,
        )
        return build_quality_control([*raw_metrics, *processed_metrics])

    # ------------------------------------------------------------------ #
    # NWB assembly helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_acquisition_series(nwb_file: pynwb.NWBFile, series: AcquisitionSeries) -> None:
        """Add one ``AcquisitionSeries`` to the NWB acquisition module."""
        nwb_file.add_acquisition(
            pynwb.TimeSeries(
                name=series.name,
                data=np.asarray(series.data),
                timestamps=np.asarray(series.timestamps),
                unit=series.unit,
                description=series.description,
            )
        )

    @staticmethod
    def _add_acquisition_table(nwb_file: pynwb.NWBFile, table: AcquisitionTable) -> None:
        """Add one ``AcquisitionTable`` to the NWB acquisition module."""
        nwb_file.add_acquisition(
            DynamicTable.from_dataframe(
                df=table.data,
                name=table.name,
                table_description=table.description,
            )
        )

    @staticmethod
    def _add_trials(nwb_file: pynwb.NWBFile, trials: pd.DataFrame) -> None:
        """Add the trials table to the NWB file's native ``trials`` table.

        ``start_time`` / ``stop_time`` are modeled natively by NWB; every other
        column is registered as an extra trial column (described by the matching
        :class:`TrialConfig` field) and populated per row. An empty table (or one
        missing the required time columns) is skipped.
        """
        if trials.empty or any(col not in trials.columns for col in _TRIAL_TIME_COLUMNS):
            return
        descriptions = TrialConfig.column_descriptions()
        extra_columns = [col for col in trials.columns if col not in _TRIAL_TIME_COLUMNS]
        for column in extra_columns:
            nwb_file.add_trial_column(name=column, description=descriptions.get(column, column))
        for _, row in trials.iterrows():
            nwb_file.add_trial(**{column: row[column] for column in trials.columns})

    # ------------------------------------------------------------------ #
    # QC input helpers
    # ------------------------------------------------------------------ #
    def _lick_times(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return the ``(left, right)`` lick-time arrays for the processed QC stage."""
        left = self._acquisition_builder.get_lick_times(
            self.left_lick.device, self.left_lick.stream, self.left_lick.port
        )
        right = self._acquisition_builder.get_lick_times(
            self.right_lick.device, self.right_lick.stream, self.right_lick.port
        )
        return left, right

    def _manual_water_times(self) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return the ``(left, right)`` manual-water delivery times.

        The ``GiveManualWaterRight`` stream's ``data`` column is ``True`` for a
        right-port delivery and ``False`` for left; split it by side. An empty
        frame yields two empty arrays.
        """
        manual_water = self._acquisition_builder.get_manual_water_times()
        left = manual_water.index[~manual_water["data"].astype(bool)].to_numpy()
        right = manual_water.index[manual_water["data"].astype(bool)].to_numpy()
        return left, right

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    def run(self, output_path: t.Union[str, os.PathLike]) -> QualityControl:
        """Run the full pipeline: write the NWB file and the QC to disk.

        Assembles and writes the NWB file to ``output_path``, runs the combined
        QC, and writes the QC figure assets and ``quality_control.json`` to the
        same location. (The NWB store and QC outputs share ``output_path`` for
        now; split them into separate destinations if that stops holding.)

        Parameters
        ----------
        output_path : os.PathLike
            Destination for the NWB (Zarr) file and the QC outputs.

        Returns
        -------
        QualityControl
            The combined raw + processed quality-control object. The NWB file
            and the QC JSON are written to disk as a side effect.
        """
        acquisition = self.build_acquisition()
        trials = self.build_trials()
        nwb_file = self.build_nwb(acquisition, trials)
        self.write(nwb_file, output_path)

        quality_control = self.run_qc(trials, os.fspath(output_path))
        quality_control.write_standard_file(output_directory=Path(output_path))
        return quality_control
