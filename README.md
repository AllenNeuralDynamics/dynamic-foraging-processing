# dynamic-foraging-processing

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/badge/python->=3.12-blue?logo=python)

Tools for processing raw [dynamic foraging](https://github.com/AllenNeuralDynamics/dynamic-foraging-task)
acquisition data into derived containers for NWB. The package loads raw acquisition
streams through the
[aind-behavior-dynamic-foraging](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging)
data contract and assembles higher-level tables (currently the NWB `trials` table)
from those streams.

> **Note:** Data must be acquired in the `aind-behavior-dynamic-foraging` data
> contract format to be compatible with these tools.

## Level of Support
 - [x] Supported: We are releasing this code to the public as a tool we expect others to use. Issues are welcomed, and we expect to address them promptly; pull requests will be vetted by our staff before inclusion.
 - [ ] Occasional updates: We are planning on occasional updating this tool with no fixed schedule. Community involvement is encouraged through both issues and pull requests.
 - [ ] Unsupported: We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Modules
The package (`src/dynamic_foraging_processing`) is organized into two modules:

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

## Repository structure
```
src/dynamic_foraging_processing/
├── raw_data_loader/        # RawDataLoader: load raw streams via the data contract
│   └── loader.py
└── processing/             # Derived NWB containers
    ├── _trial_table.py     # TrialTableBuilder
    └── models/
        └── trial_config.py # TrialConfig (one trials-table row + NWB descriptions)

docs/                       # Source mapping and NWB notes (trials_table_mapping.md, ...)
examples/                   # Runnable notebooks (raw_data_loader_example, trial_table_example)
tests/                      # Unit tests
```

## Quickstart
```python
from pathlib import Path

from dynamic_foraging_processing.raw_data_loader import RawDataLoader
from dynamic_foraging_processing.processing import TrialTableBuilder

# Point at the root of an acquisition directory.
loader = RawDataLoader(path=Path("/path/to/acquisition"))

# Build the NWB trials table (one row per trial).
trials = TrialTableBuilder(loader.dataset).build()

# Or inspect raw streams directly.
raw = loader.get_all_raw_data()
```
See [`examples/`](examples) for end-to-end notebooks.

## Installation
This project uses [uv](https://docs.astral.sh/uv/). From the root directory, run
```bash
uv sync
```
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

## Release Status
GitHub's tags and Release features can be used to indicate a Release status.

 - Stable: v1.0.0 and above. Ready for production.
 - Beta:  v0.x.x or indicated in the tag. Ready for beta testers and early adopters.
 - Alpha: v0.x.x or indicated in the tag. Still in early development.
