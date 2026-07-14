# Changelog

Back to the [architecture overview](architecture.md).

## 0.1.0 — initial release
End-to-end pipeline that packages a raw dynamic foraging acquisition into NWB and runs
quality control. Delivered:

- **`RawDataLoader`** — raw streams via the `aind-behavior-dynamic-foraging` data
  contract (`get_all_raw_data()` plus the `contraqctor` `Dataset`).
- **`AcquisitionBuilder`** — NWB acquisition module: the four derived event series
  plus every raw contract stream as a `DynamicTable`.
- **`TrialTableBuilder` / `TrialConfig`** — NWB `trials` table, with `TrialConfig`
  field descriptions as the source of truth for the column descriptions.
- **`RawQC` / `ProcessedQC`** — contract QC and behavior QC assembled into one
  `aind-data-schema` `QualityControl` via `build_quality_control`.
- **`Pipeline`** — the two capsule entry points: `run_nwb` (writes
  `behavior.nwb.zarr` + `processing.json`) and `run_qc` (writes
  `quality_control.json` + figures).
- Release automation (semantic-version bump + tag on merge to `main`) and 100% test
  coverage / docstring gates.

Reasoning for the choices above — including the deferred `EventsBuilder`, the
`Dataset`-over-dict decision, and the backward-compatibility constraint — is in
[Alignment](architecture/alignment.md).
