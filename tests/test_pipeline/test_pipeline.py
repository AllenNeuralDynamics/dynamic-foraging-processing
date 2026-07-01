"""Tests for ``dynamic_foraging_processing.pipeline._pipeline``."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from aind_data_schema.core.processing import Processing, ProcessName

from dynamic_foraging_processing.nwb.acquisition.acquisition_builder import LickSource
from dynamic_foraging_processing.nwb.acquisition.models import (
    AcquisitionSeries,
    AcquisitionTable,
)
from dynamic_foraging_processing.pipeline import _pipeline
from dynamic_foraging_processing.pipeline._pipeline import Pipeline
from dynamic_foraging_processing.processing.models import TrialConfig


def _make_loader() -> MagicMock:
    """Build a mock loader with a dataset and a metadata path."""
    loader = MagicMock()
    loader.dataset = object()
    loader.path = Path("dataset")
    return loader


def _make_pipeline() -> Pipeline:
    """Build a pipeline whose internal builders are replaced with mocks."""
    pipeline = Pipeline(_make_loader())
    pipeline._acquisition_builder = MagicMock()
    pipeline._trial_table_builder = MagicMock()
    return pipeline


def _trials_frame() -> pd.DataFrame:
    """Two-trial table with the time columns plus one modeled, one unmodeled column."""
    return pd.DataFrame(
        {
            "start_time": [0.0, 1.0],
            "stop_time": [0.5, 1.5],
            "animal_response": [0, 1],
            "not_in_model": [7, 8],
        }
    )


def test_init_stores_config_and_builds_helpers():
    """``__init__`` stores the loader/lick sources and constructs the builders."""
    loader = _make_loader()
    left = LickSource("HarpLickometerLeft", "LickState", "Channel0")

    pipeline = Pipeline(loader, left_lick=left, raise_on_error=True)

    assert pipeline.loader is loader
    assert pipeline.left_lick is left
    assert pipeline.right_lick == _pipeline._DEFAULT_RIGHT_LICK
    assert pipeline.raise_on_error is True
    assert pipeline._acquisition_builder.loader is loader
    assert pipeline._trial_table_builder.dataset is loader.dataset
    assert pipeline._trial_table_builder.raise_on_error is True


def test_build_acquisition_delegates_with_lick_sources():
    """``build_acquisition`` forwards the configured lick sources to the builder."""
    pipeline = _make_pipeline()
    pipeline._acquisition_builder.build_acquisition.return_value = ["entry"]

    result = pipeline.build_acquisition()

    assert result == ["entry"]
    pipeline._acquisition_builder.build_acquisition.assert_called_once_with(
        left_lick=pipeline.left_lick,
        right_lick=pipeline.right_lick,
    )


def test_build_trials_delegates_to_builder():
    """``build_trials`` returns the trials-table builder's frame."""
    pipeline = _make_pipeline()
    frame = _trials_frame()
    pipeline._trial_table_builder.build.return_value = frame

    assert pipeline.build_trials() is frame
    pipeline._trial_table_builder.build.assert_called_once_with()


def test_build_nwb_uses_provided_components(monkeypatch):
    """Provided acquisition/trials are added onto the base NWB file."""
    nwb_file = MagicMock()
    monkeypatch.setattr(_pipeline, "create_base_nwb_file", lambda path: nwb_file)
    pipeline = _make_pipeline()

    acquisition = [
        AcquisitionSeries(
            name="left_lick_time",
            data=np.array([True]),
            timestamps=np.array([1.0]),
            unit="second",
            description="left licks",
        ),
        AcquisitionTable(
            name="Behavior.RawStream",
            data=pd.DataFrame({"x": [1, 2]}),
            description="raw",
        ),
    ]
    trials = _trials_frame()

    result = pipeline.build_nwb(acquisition, trials)

    assert result is nwb_file
    # One series + one table added to acquisition; trials rows populated.
    assert nwb_file.add_acquisition.call_count == 2
    assert nwb_file.add_trial.call_count == len(trials)
    # The internal builders are not called when components are supplied.
    pipeline._acquisition_builder.build_acquisition.assert_not_called()
    pipeline._trial_table_builder.build.assert_not_called()


def test_build_nwb_builds_missing_components(monkeypatch):
    """When omitted, acquisition and trials are built from the builders."""
    nwb_file = MagicMock()
    monkeypatch.setattr(_pipeline, "create_base_nwb_file", lambda path: nwb_file)
    pipeline = _make_pipeline()
    pipeline.build_acquisition = MagicMock(return_value=[])
    pipeline.build_trials = MagicMock(return_value=_trials_frame())

    pipeline.build_nwb()

    pipeline.build_acquisition.assert_called_once_with()
    pipeline.build_trials.assert_called_once_with()


def test_write_uses_zarr_io(monkeypatch):
    """``write`` opens ``NWBZarrIO`` in write mode and writes the file."""
    captured = {}

    class _FakeIO:
        """Context-managed stand-in for ``NWBZarrIO``."""

        def __init__(self, path, mode):
            """Record the path and mode."""
            captured["path"] = path
            captured["mode"] = mode

        def __enter__(self):
            """Enter the context."""
            return self

        def __exit__(self, *exc):
            """Exit the context without suppressing exceptions."""
            return False

        def write(self, nwb_file):
            """Record the written file."""
            captured["written"] = nwb_file

    monkeypatch.setattr(_pipeline, "NWBZarrIO", _FakeIO)
    pipeline = _make_pipeline()
    nwb_file = object()

    result = pipeline.write(nwb_file, "out.nwb.zarr")

    assert result is None
    assert captured == {"path": "out.nwb.zarr", "mode": "w", "written": nwb_file}


def test_assemble_quality_control_combines_metrics_into_single_list(monkeypatch):
    """``_assemble_quality_control`` flattens raw and processed metrics into one list."""
    captured = {}

    class _FakeRawQC:
        """Raw QC stub returning two sentinel metrics."""

        def run(self, dataset, results_folder):
            """Record args and return raw metrics."""
            captured["raw"] = (dataset, results_folder)
            return ["raw1", "raw2"]

    class _FakeProcessedQC:
        """Processed QC stub returning one sentinel metric."""

        def run(self, trials, left, right, results_folder, *, manual_left_times, manual_right_times):
            """Record args and return processed metrics."""
            captured["processed"] = {
                "left": left,
                "right": right,
                "results_folder": results_folder,
                "manual_left": manual_left_times,
                "manual_right": manual_right_times,
            }
            return ["proc1"]

    monkeypatch.setattr(_pipeline, "RawQC", _FakeRawQC)
    monkeypatch.setattr(_pipeline, "ProcessedQC", _FakeProcessedQC)
    monkeypatch.setattr(_pipeline, "build_quality_control", lambda metrics: metrics)

    pipeline = _make_pipeline()
    pipeline._acquisition_builder.get_lick_times.side_effect = [
        np.array([1.0]),
        np.array([2.0]),
    ]
    pipeline._acquisition_builder.get_manual_water_times.return_value = pd.DataFrame(
        {"data": [False, True]}, index=pd.Index([0.1, 0.2], name="time")
    )
    trials = _trials_frame()

    result = pipeline._assemble_quality_control(trials, "out")

    assert result == ["raw1", "raw2", "proc1"]
    assert captured["raw"] == (pipeline.loader.dataset, "out")
    np.testing.assert_array_equal(captured["processed"]["left"], np.array([1.0]))
    np.testing.assert_array_equal(captured["processed"]["right"], np.array([2.0]))
    np.testing.assert_array_equal(captured["processed"]["manual_left"], np.array([0.1]))
    np.testing.assert_array_equal(captured["processed"]["manual_right"], np.array([0.2]))


def test_add_acquisition_series_builds_time_series():
    """``_add_acquisition_series`` adds a ``TimeSeries`` mirroring the model."""
    nwb_file = MagicMock()
    series = AcquisitionSeries(
        name="left_lick_time",
        data=np.array([True, False]),
        timestamps=np.array([1.0, 2.0]),
        unit="second",
        description="left licks",
    )

    Pipeline._add_acquisition_series(nwb_file, series)

    (added,), _ = nwb_file.add_acquisition.call_args
    assert added.name == "left_lick_time"
    assert added.unit == "second"
    np.testing.assert_array_equal(added.timestamps, np.array([1.0, 2.0]))


def test_add_acquisition_table_builds_dynamic_table():
    """``_add_acquisition_table`` adds a ``DynamicTable`` from the dataframe."""
    nwb_file = MagicMock()
    table = AcquisitionTable(
        name="Behavior.RawStream",
        data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        description="raw stream",
    )

    Pipeline._add_acquisition_table(nwb_file, table)

    (added,), _ = nwb_file.add_acquisition.call_args
    assert added.name == "Behavior.RawStream"
    assert added.description == "raw stream"
    assert len(added) == 2


def test_add_trials_populates_columns_and_rows():
    """Extra columns are described via ``TrialConfig``; rows are added."""
    nwb_file = MagicMock()
    trials = _trials_frame()

    Pipeline._add_trials(nwb_file, trials)

    added_columns = {
        call.kwargs["name"]: call.kwargs["description"]
        for call in nwb_file.add_trial_column.call_args_list
    }
    # start_time / stop_time are native and not registered as extra columns.
    assert set(added_columns) == {"animal_response", "not_in_model"}
    assert added_columns["animal_response"] == TrialConfig.column_descriptions()["animal_response"]
    # A column absent from TrialConfig falls back to its own name as description.
    assert added_columns["not_in_model"] == "not_in_model"
    assert nwb_file.add_trial.call_count == 2


def test_add_trials_skips_empty_frame():
    """An empty trials table adds nothing."""
    nwb_file = MagicMock()

    Pipeline._add_trials(nwb_file, pd.DataFrame())

    nwb_file.add_trial_column.assert_not_called()
    nwb_file.add_trial.assert_not_called()


def test_add_trials_skips_when_time_columns_missing():
    """A table lacking start/stop time columns is skipped."""
    nwb_file = MagicMock()
    trials = pd.DataFrame({"animal_response": [0, 1]})

    Pipeline._add_trials(nwb_file, trials)

    nwb_file.add_trial_column.assert_not_called()
    nwb_file.add_trial.assert_not_called()


def test_manual_water_times_splits_by_side():
    """Left deliveries are ``data == False`` and right are ``data == True``."""
    pipeline = _make_pipeline()
    pipeline._acquisition_builder.get_manual_water_times.return_value = pd.DataFrame(
        {"data": [False, True, True]}, index=pd.Index([0.1, 0.2, 0.3], name="time")
    )

    left, right = pipeline._manual_water_times()

    np.testing.assert_array_equal(left, np.array([0.1]))
    np.testing.assert_array_equal(right, np.array([0.2, 0.3]))


def test_manual_water_times_empty_frame_yields_empty_arrays():
    """An empty manual-water stream yields two empty arrays."""
    pipeline = _make_pipeline()
    pipeline._acquisition_builder.get_manual_water_times.return_value = pd.DataFrame(
        {"data": []}, index=pd.Index([], name="time")
    )

    left, right = pipeline._manual_water_times()

    assert left.size == 0
    assert right.size == 0


def test_run_nwb_writes_nwb_and_processing(tmp_path):
    """``run_nwb`` writes the NWB store and a valid ``processing.json``."""
    pipeline = _make_pipeline()
    acquisition = ["entry"]
    trials = _trials_frame()
    nwb_file = object()

    pipeline.build_acquisition = MagicMock(return_value=acquisition)
    pipeline.build_trials = MagicMock(return_value=trials)
    pipeline.build_nwb = MagicMock(return_value=nwb_file)
    pipeline.write = MagicMock()

    result = pipeline.run_nwb(str(tmp_path))

    assert result is nwb_file
    pipeline.build_nwb.assert_called_once_with(acquisition, trials)
    pipeline.write.assert_called_once_with(nwb_file, str(tmp_path))
    # processing.json is written and round-trips through the schema.
    processing_json = tmp_path / "processing.json"
    assert processing_json.exists()
    loaded = Processing.model_validate_json(processing_json.read_text())
    assert loaded.data_processes[0].process_type == ProcessName.PIPELINE


def test_run_qc_writes_quality_control():
    """``run_qc`` builds trials, assembles the QC, and writes ``quality_control.json``."""
    pipeline = _make_pipeline()
    trials = _trials_frame()
    quality_control = MagicMock()

    pipeline.build_trials = MagicMock(return_value=trials)
    pipeline._assemble_quality_control = MagicMock(return_value=quality_control)

    result = pipeline.run_qc("out")

    assert result is quality_control
    pipeline.build_trials.assert_called_once_with()
    pipeline._assemble_quality_control.assert_called_once_with(trials, "out")
    quality_control.write_standard_file.assert_called_once_with(output_directory=Path("out"))


def test_run_nwb_without_output_skips_writes():
    """``run_nwb`` returns the NWB file and touches no disk when no path is given."""
    pipeline = _make_pipeline()
    nwb_file = object()

    pipeline.build_acquisition = MagicMock(return_value=["entry"])
    pipeline.build_trials = MagicMock(return_value=_trials_frame())
    pipeline.build_nwb = MagicMock(return_value=nwb_file)
    pipeline.write = MagicMock()
    pipeline._write_processing = MagicMock()

    result = pipeline.run_nwb()

    assert result is nwb_file
    pipeline.write.assert_not_called()
    pipeline._write_processing.assert_not_called()


def test_run_qc_without_output_skips_write():
    """``run_qc`` assembles the QC with no results folder and skips the JSON write."""
    pipeline = _make_pipeline()
    trials = _trials_frame()
    quality_control = MagicMock()

    pipeline.build_trials = MagicMock(return_value=trials)
    pipeline._assemble_quality_control = MagicMock(return_value=quality_control)

    result = pipeline.run_qc()

    assert result is quality_control
    pipeline._assemble_quality_control.assert_called_once_with(trials, None)
    quality_control.write_standard_file.assert_not_called()
