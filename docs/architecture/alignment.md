# Alignment with the reference diagram

Back to the [architecture overview](../architecture.md).

This compares the reference diagram (the target design) with what the repo currently
implements, calls out where they differ, and documents the rationale — doubling as a
checklist for updating the reference diagram to match the source code.

## TL;DR
- **Overall shape matches.** Loader → builders → package-to-NWB → QC-from-NWB, with
  `BaseQC` / `RawQC` / `ProcessedQC`, and `aind-nwb-utils` + `aind-data-schema` as
  the external writers. All present.
- **Biggest intentional gap:** the diagram's dedicated **`EventsBuilder` →
  `EventsTable`** stage is **deferred/blocked**. Per-event data (licks, reward
  deliveries) is carried as **acquisition time series** — both because that stage
  isn't built yet *and* because that form **maintains backward compatibility with the
  existing pipeline** (refer to section 4 below).
- **Biggest structural difference:** the diagram models the two entry points as
  **free functions** (`build_df_nwb`, `qc_from_nwb_file`); the repo implements them
  as **`Pipeline` methods** (`run_nwb`, `run_qc`).
- **A necessary difference (and the `RawDataDict` is still present):** `RawQC`
  consumes the `contraqctor` **`Dataset`** (real contract QC), not the plain
  `RawDataDict` the diagram shows. The dict *is* available (`get_all_raw_data()`) —
  the builders (`TrialTableBuilder` included) just build from the `Dataset` because
  it's easier to navigate and access.

## Component-by-component alignment

| Diagram component | Current repo | Status |
| --- | --- | --- |
| `RawDataLoader` (`dfp.raw_data`) → `RawDataDict` | `raw_data_loader.RawDataLoader` → `get_all_raw_data()` dict (+ `.dataset`) | ✅ Aligned |
| `TrialsBuilder` (`dfp.process.trials`) → `TrialsTable` | `processing.TrialTableBuilder` → trials `DataFrame` | ✅ Aligned |
| `EventsBuilder` (`dfp.process.events`) → `EventsTable` | *(none — deferred/blocked)* | ⛔ Not yet built |
| `BaseQC` interface | `qc._core.BaseQC` | ✅ Aligned |
| `RawQC` (`dfp.qc.raw`) | `qc.raw.RawQC` | ✅ Aligned (input differs — refer to section 3 below) |
| `ProcessedQC` (`dfp.qc.processed`) | `qc.processed.ProcessedQC` | ✅ Aligned (input differs — refer to section 4 below) |
| `build_df_nwb` function (`dfp.package`) | `pipeline.Pipeline.run_nwb` | ◐ Same role, method vs function |
| `qc_from_nwb_file` function (`dfp.qc`) | `pipeline.Pipeline.run_qc` | ◐ Same role, method vs function |
| `NWBFile` (Zarr: acquisition / events / intervals) | NWB (Zarr): acquisition + `trials` (no separate events table) | ◐ Partial (events folded into acquisition) |
| `aind-nwb-utils` (base NWB) | `create_base_nwb_file` in `run_nwb` | ✅ Aligned |
| `aind-data-schema` `Processing` / `QualityControl` | `Processing` in `run_nwb`, `QualityControl` in `run_qc` | ✅ Aligned |
| *(implicit: raw streams written to acquisition)* | explicit `nwb.acquisition.AcquisitionBuilder` | ➕ Extra in repo |

## Where the repo differs, and why

### 1. Entry points: class methods vs. free functions
- **Diagram:** `build_df_nwb(...)` and `qc_from_nwb_file(...)` are stand-alone functions.
- **Repo:** `Pipeline(loader).run_nwb(...)` and `.run_qc(nwb_file, ...)`.
- **Assessment — neutral / mild trade-off.** The `Pipeline` object holds the loader
  and per-session config (lick sources) once and exposes composable building blocks
  (`build_acquisition`, `build_trials`, `build_nwb`, `write`), which is convenient
  for notebooks and tests. The diagram's free functions are simpler for a capsule
  author (call one function). Both are fine; if the capsule surface ever feels
  heavy, thin `build_df_nwb` / `qc_from_nwb_file` wrappers around `Pipeline` would
  reconcile the two with no loss.

### 2. QC-from-NWB
- **Diagram:** `qc_from_nwb_file` reads the `NWBFile` and runs QC from it.
- **Repo:** `run_qc(nwb_file, ...)` takes the in-memory `NWBFile` and reads the
  trials table + lick times back from it; raw QC runs over the loader's dataset.
- **Assessment — aligned.** This matches the diagram's key idea (QC is driven by the
  packaged NWB, not by rebuilding from raw). The one nuance is that raw contract-QC
  still needs the raw dataset, so `run_qc` also relies on the loader — inherent, not
  a divergence.

### 3. `RawQC` input: `Dataset` vs `RawDataDict`
- **Diagram:** `RawQC.run(acquisition: RawDataDict)`.
- **Repo:** `RawQC.run(acquisition: contraqctor.Dataset)`.
- **Assessment — the `Dataset` is the required input here (and the dict still
  exists).** The `RawDataDict` is available via `get_all_raw_data()`; it simply isn't
  what `RawQC` consumes. Contract QC (Harp/camera/CSV/data-contract checks) is defined
  over the `contraqctor` `Dataset`, not a plain stream-name→DataFrame dict — a
  `RawDataDict` cannot support those checks — so `RawQC` takes the `Dataset`. See the
  [Design decisions](design-decisions.md) note on why the builders use the `Dataset` too.

### 4. Processed-events representation (the deferred `EventsBuilder`)
- **Diagram:** a first-class `EventsBuilder` → `EventsTable`, written into the NWB as
  a processed events container; `ProcessedQC` takes `(EventsTable, TrialsTable)`.
- **Repo (interim):** no events table. Per-event data lives as **acquisition time
  series** (`left/right_lick_time`, `left/right_reward_delivery_time`), and
  `ProcessedQC.run(trials, left_lick_times, right_lick_times, ...)` takes lick-time
  arrays rather than an `EventsTable`.
- **Assessment — deliberate (backward compatibility), plus an interim piece.** Two
  reasons the event data lives as acquisition series today: (1) the existing pipeline
  expects licks/rewards in that acquisition-series form, so keeping them there
  **maintains backward compatibility** with downstream consumers, and (2) the
  dedicated `EventsBuilder` → `EventsTable` stage is deferred. So this isn't purely a
  stopgap. A first-class `EventsTable` is still the more standard, queryable
  representation and would let `ProcessedQC` take a clean `(events, trials)` pair —
  but even once `EventsBuilder` lands, the acquisition series may need to **remain**
  (or be produced alongside the table) for backward compatibility. Treat the
  `EventsTable` as additive, not necessarily a straight replacement.

### 5. Explicit `AcquisitionBuilder`
- **Diagram:** raw streams are written straight into the NWB acquisition inside
  `build_df_nwb`; there is no separate acquisition builder.
- **Repo:** `nwb.acquisition.AcquisitionBuilder` is a distinct class that both
  packages every raw stream as a `DynamicTable` and derives the lick/reward series.
- **Assessment — required.** Packaging the raw streams into the NWB acquisition was
  an explicit **requirement from the scientist** (the raw data must travel in the
  NWB), so this stage is deliberate, not incidental. It is implemented as an explicit,
  unit-tested `AcquisitionBuilder` rather than inlined in the packaging function. Its
  derived lick/reward series overlap with the deferred `EventsBuilder`, but even once
  that lands, `AcquisitionBuilder` will **keep producing those series in acquisition
  for backward compatibility** — the `EventsTable` is additive, not a replacement
  (refer to section 4 above). So this stage
  stays, packaging both the raw streams and the derived series.

### 6. Module naming
- **Diagram:** `dfp.raw_data`, `dfp.process.events`, `dfp.process.trials`,
  `dfp.qc.raw`, `dfp.qc.processed`, `dfp.package`, `dfp.qc`.
- **Repo:** `raw_data_loader`, `nwb.acquisition`, `processing`, `qc.raw`,
  `qc.processed`, `pipeline`.
- **Assessment — naming only.** The QC layout matches; the rest is renamed. The repo
  has no `process.events` (deferred) and adds `nwb` + `pipeline` packages. No
  functional difference; renaming isn't necessary.
