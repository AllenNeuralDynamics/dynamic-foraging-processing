"""Dynamic foraging pipeline entry points.

Two Code Ocean capsules drive this module, so it exposes two independent entry
points over the shared building blocks rather than one combined flow:

- :meth:`Pipeline.run_nwb` -- assemble the NWB file (base metadata + acquisition
  entries + trials table), write it to disk, and write the ``processing.json``
  ``aind-data-schema`` metadata alongside it.
- :meth:`Pipeline.run_qc` -- run the raw QC stage (over the loader's raw dataset)
  and the processed QC stage (over a given NWB file) and write the combined
  ``quality_control.json`` to disk.

The individual builders produce NWB-ready pydantic models
(``AcquisitionSeries`` / ``AcquisitionTable``) and a trials ``DataFrame``; this
module translates those into ``pynwb`` objects on top of the base file created
by :func:`aind_nwb_utils.utils.create_base_nwb_file`, then writes the result
with the Zarr backend.
"""

import os
import typing as t
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from aind_data_schema.components.identifiers import DataAsset
from aind_data_schema.core.processing import (
    Code,
    DataProcess,
    Processing,
    ProcessName,
    ProcessStage,
)
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

#: Trials-table column NWB's required native ``start_time`` is taken from. The
#: trials table itself no longer carries trial ``start_time`` / ``stop_time``
#: columns — it carries one start/stop pair per task period instead — but
#: ``TimeIntervals`` requires both, so they are derived here.
_NWB_START_COLUMN = "quiescent_start_time"

#: Trials-table column NWB's required native ``stop_time`` is taken from: the
#: end of the ITI, which on the last trial of the session is the ``EndSession``
#: timestamp. It is ``NaN`` only when that stream is unavailable, and the ``NaN``
#: is propagated rather than substituted, so an unknown trial end reads as
#: unknown instead of as a shortened trial.
_NWB_STOP_COLUMN = "ITI_stop_time"

#: Source repository recorded in the ``processing.json`` data process.
_CODE_URL = "https://github.com/AllenNeuralDynamics/dynamic-foraging-processing"

#: Installed version of this package, recorded on the processing ``Code``.
_PACKAGE_VERSION = version("dynamic-foraging-processing")

#: Default pipeline name linking the data process to the ``processing.json`` pipeline.
_PIPELINE_NAME = "dynamic-foraging-processing-pipeline"

#: Filename of the NWB (Zarr) store written under the output directory.
_NWB_FILENAME = "behavior.nwb.zarr"

#: Acquisition-series names read back for the processed QC stage.
_LEFT_LICK_SERIES = "left_lick_time"
_RIGHT_LICK_SERIES = "right_lick_time"
_LEFT_REWARD_SERIES = "left_reward_delivery_time"
_RIGHT_REWARD_SERIES = "right_reward_delivery_time"

#: Reward-delivery annotation marking a manual-water event.
_MANUAL_ANNOTATION = "manual"


class Pipeline:
    """Package a raw dynamic foraging acquisition to NWB and run QC.

    :meth:`run_nwb` and :meth:`run_qc` are the two entry points (one per Code
    Ocean capsule); the ``build_*`` / :meth:`write` methods are the lower-level
    building blocks they compose from, exposed for notebooks and debugging.

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
        self._trial_table_builder = TrialTableBuilder(loader.dataset, raise_on_error=raise_on_error)

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
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
        """Add one ``AcquisitionTable`` to the NWB acquisition module.

        The stream's timestamp index is reset into an explicit ``timestamp``
        column so it is preserved as data rather than consumed as the table's
        auto-generated ``id``.
        """
        frame = table.data.reset_index(names="timestamp")
        nwb_file.add_acquisition(
            DynamicTable.from_dataframe(
                df=frame,
                name=table.name,
                table_description=table.description,
            )
        )

    @staticmethod
    def _trial_extent(row: pd.Series) -> t.Tuple[float, float]:
        """Return NWB's required native ``(start_time, stop_time)`` for one trial.

        The trials table has no trial start/stop columns of its own, so the
        trial's extent is taken from its period bounds: it starts with the
        quiescent period and ends with the ITI. The last trial of the session has
        no following quiescent period, so its ITI — and therefore its stop time —
        ends at the ``EndSession`` timestamp. Should that be unavailable the stop
        time is ``NaN``: an unknown end is reported as unknown rather than
        substituted with an earlier landmark.

        Parameters
        ----------
        row : pandas.Series
            One row of the trials table.

        Returns
        -------
        tuple of (float, float)
            The trial start and stop time (seconds); the stop time is ``NaN``
            where the ITI end is unknown.
        """
        return float(row[_NWB_START_COLUMN]), float(row[_NWB_STOP_COLUMN])

    @classmethod
    def _add_trials(cls, nwb_file: pynwb.NWBFile, trials: pd.DataFrame) -> None:
        """Add the trials table to the NWB file's native ``trials`` table.

        Every trials-table column is registered as an extra trial column
        (described by the matching :class:`TrialConfig` field) and populated per
        row. NWB additionally requires a native ``start_time`` / ``stop_time`` per
        trial, which are derived from the period columns (see
        :meth:`_trial_extent`) rather than stored as columns. The DataFrame index
        (named ``id``) is replicated as each trial's NWB ``id``. An empty table
        (or one missing the period columns the extent is derived from) is skipped.
        """
        required = (_NWB_START_COLUMN, _NWB_STOP_COLUMN)
        if trials.empty or any(col not in trials.columns for col in required):
            return
        descriptions = TrialConfig.column_descriptions()
        for column in trials.columns:
            nwb_file.add_trial_column(name=column, description=descriptions.get(column, column))
        for row_id, row in trials.iterrows():
            start_time, stop_time = cls._trial_extent(row)
            nwb_file.add_trial(
                id=int(row_id),
                start_time=start_time,
                stop_time=stop_time,
                **{column: row[column] for column in trials.columns},
            )

    # ------------------------------------------------------------------ #
    # Writers
    # ------------------------------------------------------------------ #
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

    def _write_processing(
        self,
        output_path: t.Union[str, os.PathLike],
        start_date_time: datetime,
        end_date_time: datetime,
    ) -> None:
        """Write the ``processing.json`` metadata for the NWB packaging step.

        Parameters
        ----------
        output_path : os.PathLike
            Directory the ``processing.json`` is written into.
        start_date_time, end_date_time : datetime
            When the NWB packaging started and finished.
        """
        pipeline_name = os.getenv("PIPELINE_NAME", _PIPELINE_NAME)
        processing = Processing(
            data_processes=[
                DataProcess(
                    name="Dynamic foraging NWB packaging",
                    process_type=ProcessName.PIPELINE,
                    stage=ProcessStage.PROCESSING,
                    code=Code(url=_CODE_URL, version=_PACKAGE_VERSION),
                    experimenters=["Alex Piet", "Micah Woodard", "Bruno Cruz", "Arjun Sridhar"],
                    start_date_time=start_date_time,
                    end_date_time=end_date_time,
                    pipeline_name=pipeline_name,
                )
            ],
            pipelines=[
                Code(
                    url=os.getenv("PIPELINE_URL", _CODE_URL),
                    version=os.getenv("PIPELINE_VERSION", _PACKAGE_VERSION),
                    name=pipeline_name,
                    input_data=[DataAsset(name=Path(self.loader.path).stem)],
                )
            ],
        )
        processing.write_standard_file(output_directory=Path(output_path))

    # ------------------------------------------------------------------ #
    # QC helpers
    # ------------------------------------------------------------------ #
    def _read_processed_inputs(
        self, nwb_file: pynwb.NWBFile
    ) -> t.Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read the processed-QC inputs from an NWB file.

        Parameters
        ----------
        nwb_file : pynwb.NWBFile
            The NWB file to read the trials table and acquisition series from.

        Returns
        -------
        tuple
            ``(trials, left_lick_times, right_lick_times, manual_left_times,
            manual_right_times)``.
        """
        trials = nwb_file.trials.to_dataframe()
        left_lick_times = np.asarray(nwb_file.acquisition[_LEFT_LICK_SERIES].timestamps)
        right_lick_times = np.asarray(nwb_file.acquisition[_RIGHT_LICK_SERIES].timestamps)
        manual_left_times, manual_right_times = self._manual_water_times(nwb_file)
        return trials, left_lick_times, right_lick_times, manual_left_times, manual_right_times

    @staticmethod
    def _manual_water_times(nwb_file: pynwb.NWBFile) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return the ``(left, right)`` manual-water delivery times from the NWB.

        Manual-water deliveries are the reward-delivery events annotated
        ``"manual"`` on each side's ``*_reward_delivery_time`` acquisition series.
        """
        left = Pipeline._annotated_times(nwb_file, _LEFT_REWARD_SERIES, _MANUAL_ANNOTATION)
        right = Pipeline._annotated_times(nwb_file, _RIGHT_REWARD_SERIES, _MANUAL_ANNOTATION)
        return left, right

    @staticmethod
    def _annotated_times(nwb_file: pynwb.NWBFile, series_name: str, annotation: str) -> np.ndarray:
        """Return timestamps of ``series_name`` whose annotation ``data`` equals ``annotation``."""
        series = nwb_file.acquisition[series_name]
        data = np.asarray(series.data)
        timestamps = np.asarray(series.timestamps)
        return timestamps[data == annotation]

    def _assemble_quality_control(
        self,
        trials: pd.DataFrame,
        left_lick_times: np.ndarray,
        right_lick_times: np.ndarray,
        manual_left_times: np.ndarray,
        manual_right_times: np.ndarray,
        results_folder: t.Optional[str] = None,
    ) -> QualityControl:
        """Run the raw and processed QC stages and assemble one ``QualityControl``.

        Parameters
        ----------
        trials : pandas.DataFrame
            The trials table, consumed by the processed (behavior) QC stage.
        left_lick_times, right_lick_times : numpy.ndarray
            Left/right-port lick times for the processed QC stage.
        manual_left_times, manual_right_times : numpy.ndarray
            Left/right manual-water delivery times passed through to the side-bias
            figure.
        results_folder : str, optional
            Directory to write figure assets into so the metric references
            resolve. If ``None``, assets are skipped.

        Returns
        -------
        QualityControl
            The combined raw (contract QA) + processed (behavior) metrics as a
            single flat metric list.
        """
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
    # Entry points (one per Code Ocean capsule)
    # ------------------------------------------------------------------ #
    def run_nwb(self, output_path: t.Optional[t.Union[str, os.PathLike]] = None) -> pynwb.NWBFile:
        """Assemble the NWB file, optionally writing it plus its ``processing.json``.

        Parameters
        ----------
        output_path : os.PathLike, optional
            Output directory. The NWB store is written to
            ``output_path / "behavior.nwb.zarr"`` and the ``processing.json`` to
            ``output_path``. When ``None`` (the default), the NWB file is built
            and returned without touching disk.

        Returns
        -------
        pynwb.NWBFile
            The assembled NWB file. When ``output_path`` is given, the NWB store
            and ``processing.json`` are written to disk as a side effect.
        """
        start_date_time = datetime.now(timezone.utc)
        acquisition = self.build_acquisition()
        trials = self.build_trials()
        nwb_file = self.build_nwb(acquisition, trials)
        if output_path is not None:
            self.write(nwb_file, Path(output_path) / _NWB_FILENAME)
            self._write_processing(output_path, start_date_time, datetime.now(timezone.utc))
        return nwb_file

    def run_qc(
        self,
        nwb_file: pynwb.NWBFile,
        output_path: t.Optional[t.Union[str, os.PathLike]] = None,
        folder_directory: str = "qc",
    ) -> QualityControl:
        """Run the raw and processed QC stages, optionally writing to disk.

        The raw (contract QA) stage runs over the loader's raw dataset; the
        processed-QC inputs (trials table, lick times, manual-water times) are read
        from ``nwb_file`` (e.g. the output of :meth:`run_nwb`).

        Parameters
        ----------
        nwb_file : pynwb.NWBFile
            The NWB file to run QC against.
        output_path : os.PathLike, optional
            Destination directory for ``quality_control.json``. When ``None`` (the
            default), the QC is assembled and returned without touching disk
            (figure assets are skipped too).
        folder_directory : str, optional
            Subdirectory of ``output_path`` that the QC figure assets (plots, ...)
            are written into, i.e. ``output_path / folder_directory``. Defaults to
            ``"qc"``. Ignored when ``output_path`` is ``None``.

        Returns
        -------
        QualityControl
            The combined raw + processed quality-control object. When
            ``output_path`` is given, the QC JSON and figure assets are written to
            disk as a side effect.
        """
        trials, left_lick_times, right_lick_times, manual_left_times, manual_right_times = (
            self._read_processed_inputs(nwb_file)
        )

        results_folder: t.Optional[str] = None
        if output_path is not None:
            artifacts_dir = Path(output_path) / folder_directory
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            results_folder = os.fspath(artifacts_dir)

        quality_control = self._assemble_quality_control(
            trials,
            left_lick_times,
            right_lick_times,
            manual_left_times,
            manual_right_times,
            results_folder,
        )
        if output_path is not None:
            quality_control.write_standard_file(output_directory=Path(output_path))
        return quality_control
