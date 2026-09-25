# dynamic-foraging-processing

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.12-blue?logo=python)

THIS LIBRARY IS CURRENTLY UNDER DEVLOPMENT AND IS SUBJECT TO CHANGE.
Tries to follow this [diagram](https://github.com/AllenNeuralDynamics/aind-software-docs/blob/main/docs/source/diagrams/dynamic_foraging/low_level/dynamic-foraging-low-level-processing.svg). The relationship is a two-way street: the diagram guides the design here, and where the code deliberately diverges (or discovers something the diagram didn't anticipate), the diagram will get updated to match. Equally, now that the foundation has been built, the code could be refactored to match the diagram where that's the better fit. Both are treated as living artifacts kept in sync as this repository develops — see [`docs/architecture.md`](docs/architecture.md) for where they currently differ and why.

This library contains tools for processing raw [dynamic foraging](https://github.com/AllenNeuralDynamics/dynamic-foraging-task)
acquisition data into derived containers for NWB. The package loads raw acquisition
streams through the
[aind-behavior-dynamic-foraging](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging)
data contract and assembles higher-level tables (currently the NWB `trials` table), arrays, and does QC from those streams.

> **Note:** Data must be acquired in the `aind-behavior-dynamic-foraging` data
> contract format to be compatible with these tools.

## Architecture
For the design overview — how the pieces map to the reference diagram, how the stages
compose, how to extend them, and the rationale behind the design choices — see
[`docs/architecture.md`](docs/architecture.md).

## Contributing
Two entry points, depending on what you're doing:

- **How to set up and submit** — environment, linting, tests, commit/PR conventions,
  and releasing: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Where to add code** — the small, self-contained plug-in points (add a trials-table
  column, add a QC check, prototype in a notebook), each with a worked example:
  [`docs/architecture/extending.md`](docs/architecture/extending.md).

## Inputs and outputs
**Input** — a raw acquisition directory in the `aind-behavior-dynamic-foraging`
data contract format (Harp device registers, software events, and the task-logic /
rig / session input schemas), loaded via `RawDataLoader`.

**Outputs** — the NWB-ready containers assembled from those streams:

- **Acquisition.** Built by `AcquisitionBuilder` and written to the NWB acquisition
  module:
  - 4 derived event series:
    - `left_reward_delivery_time` / `right_reward_delivery_time` — reward delivery
      times per lick port, with each event annotated as `earned`, `manual`, or
      `automatic`.
    - `left_lick_time` / `right_lick_time` — detected lick times per lick port.
  - All raw contract streams — every stream returned by the loader
    (`get_all_raw_data()`) is additionally packaged verbatim as a `DynamicTable`,
    so the full raw dataset travels alongside the derived series.
- **Trials table.** Built by `TrialTableBuilder`, one row per trial written to the
  NWB `trials` table (see [`docs/trials_table_mapping.md`](docs/trials_table_mapping.md)).
- **Quality control.** A single `aind-data-schema` `QualityControl` object assembled by
  `build_quality_control` from the raw (contract QC) and processed (behavior metrics)
  QC stages, written to `quality_control.json`. Each check contributes a metric (a value
  and pass/fail), and some attach a reference media (e.g. side bias, lick intervals)
  saved as a PNG alongside the report for the QC portal to render. Unlike the two above,
  this is a sidecar report rather than an NWB container.

## Level of Support
 - [x] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Modules
The package (`src/dynamic_foraging_processing`) is organized into the following
modules:

### `raw_data_loader`
Loads raw acquisition data via the data contract.
 - **`RawDataLoader`** — wraps a `contraqctor` `Dataset` built from an acquisition
   directory. Individual streams are accessed lazily through the tree-like contract
   (`loader.dataset.at("Behavior").at("SoftwareEvents").at("TrialOutcome")`), or all
   at once:
   - `get_all_raw_data()` → `dict` mapping `parent.stream` to its loaded data
     (`pandas.DataFrame` or `dict`).
   - `raw_data_stream_descriptions` → `dict` mapping each stream to its description
     (used when building NWB).

### `nwb`
Builds the NWB acquisition entries (requires the `nwb` extra).
 - **`AcquisitionBuilder`** — assembles the acquisition module from the raw
   streams: the four derived event series (`left`/`right_reward_delivery_time`,
   `left`/`right_lick_time`) plus every raw contract stream packaged as a
   `DynamicTable`. Returns `AcquisitionSeries` / `AcquisitionTable` models that a
   writer translates into `pynwb` objects.

### `processing`
Builds derived NWB containers from the raw streams.
 - **`TrialTableBuilder`** — assembles the NWB `trials` table from a `Dataset`.
   `build()` returns one `pandas.DataFrame` row per trial, aligning the per-trial
   software events, task-logic configuration, and Harp hardware streams. Pass
   `raise_on_error=True` to fail loudly on missing/misaligned streams instead of
   warning and continuing.
 - **`TrialConfig`** (`processing.models.trial_config`) — a Pydantic model
   describing one row of the trials table. Each field's `description` is the source
   of truth for the corresponding NWB trial-column description; use
   `TrialConfig.column_descriptions()` to retrieve them. The column-by-column source
   mapping is documented in [`docs/trials_table_mapping.md`](docs/trials_table_mapping.md).

### `qc`
Assembles an `aind-data-schema` `QualityControl` object (requires the `qc` extra).
 - **`RawQC`** / **`ProcessedQC`** — the raw (contract QA) and processed (behavior
   metrics) QC stages. `build_quality_control(metrics)` collects their metrics into
   a single `QualityControl`.

### `pipeline`
Ties the building blocks into two Code Ocean capsule entry points (requires the
`full` extra).
 - **`Pipeline`** — over a `RawDataLoader`:
   - `run_nwb(output_path)` — assembles the NWB file (base metadata + acquisition
     entries + trials table), writes it to `output_path / "behavior.nwb.zarr"`, and
     writes a `processing.json` alongside it.
   - `run_qc(output_path)` — runs the QC stages and writes `quality_control.json`.

   Both entry points build and return their objects in memory; `output_path` is
   optional and only written to when provided.

## Repository structure
```
src/dynamic_foraging_processing/
├── raw_data_loader/        # RawDataLoader: load raw streams via the data contract
│   └── loader.py
├── nwb/                    # NWB acquisition builder + models
│   ├── acquisition/        # AcquisitionBuilder, AcquisitionSeries/Table
│   └── utils.py            # clean_for_nwb: make frames safe to write to NWB
├── processing/             # Derived NWB containers
│   ├── _trial_table.py     # TrialTableBuilder
│   └── models/
│       └── trial_config.py # TrialConfig (one trials-table row + NWB descriptions)
├── qc/                     # QualityControl assembly (raw + processed stages)
└── pipeline/               # Pipeline: run_nwb / run_qc capsule entry points

docs/                       # Source mapping and NWB notes (trials_table_mapping.md, ...)
examples/                   # Runnable notebooks (raw_data_loader_example, trial_table_example)
tests/                      # Unit tests
```

## Quickstart
```python
from pathlib import Path

from dynamic_foraging_processing.raw_data_loader import RawDataLoader
from dynamic_foraging_processing.pipeline import Pipeline

# Point at the root of an acquisition directory.
loader = RawDataLoader(path=Path("/path/to/acquisition"))
pipeline = Pipeline(loader)

# Write the NWB file (+ processing.json) and the QC (+ quality_control.json).
pipeline.run_nwb("/path/to/output")
pipeline.run_qc("/path/to/output")

# Or use the building blocks directly.
trials = pipeline.build_trials()   # NWB trials table, one row per trial
raw = loader.get_all_raw_data()    # inspect raw streams
```
See [`examples/`](examples) for end-to-end notebooks.

## Installation
This project uses [uv](https://docs.astral.sh/uv/). From the root directory, run
```bash
uv sync
```

> [!IMPORTANT]
> The base install includes **only** the raw data loader. The optional modules
> pull in their own dependencies, so importing them will fail on a base install:
> - **`qc`** — `aind-data-schema`, `matplotlib` (for `dynamic_foraging_processing.qc`).
> - **`nwb`** — `aind-nwb-utils`, `hdmf-zarr`, `pynwb` (for the `nwb` builder).
> - **`full`** — everything above (`qc` + `nwb`); required by the `pipeline` module.
>
> Install the extra you need, e.g.:
> ```bash
> pip install -e ".[full]"
> ```

To develop the code, run
to create the environment and install the package. To include the development
dependencies (linting, tests, docs), run
```bash
uv sync --group dev
```

## Development
Tests and style checks run on every pull request via
`.github/workflows/test_and_lint.yml`. To run them locally with `uv`:
```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run interrogate -v .                               # docstring coverage
uv run coverage run -m pytest && uv run coverage report
```
To include the optional-module dependencies (qc + nwb) with uv, run
```bash
uv sync --extra full
```

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.
