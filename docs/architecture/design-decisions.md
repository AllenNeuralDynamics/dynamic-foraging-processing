# Design decisions

Back to the [architecture overview](../architecture.md).

Divergences from the diagram were deliberate; the reasoning is recorded here so it
lives with the code (not only in PR descriptions). These are current decisions, not
final — they are expected to evolve as requirements, tools, and the reference diagram
converge, and this doc is updated alongside the code:

- **The `RawDataDict` exists — builders use the `Dataset` by choice.**
  `RawDataLoader.get_all_raw_data()` returns the `RawDataDict` the diagram shows, so
  that representation *is* present. The trials-table builder (and `RawQC`) instead
  consume the `contraqctor` `Dataset` because its lazy tree navigation
  (`dataset.at("Behavior").at("SoftwareEvents").at("TrialOutcome").load()`) and
  typed models make targeted stream access far easier than digging through a fully
  materialized dict. The dict stays available for any consumer that prefers it — this
  is a choice of convenience, not a missing piece.
- **Entry points are `Pipeline` methods, not free functions.** Considered mirroring
  the diagram's `build_df_nwb` / `qc_from_nwb_file` functions. Chose the class so everything is all in one place and the build steps stay
  composable for notebooks/tests. Thin free-function wrappers can be added later
  without breaking this.
- **`run_qc` combines raw + processed, but the two stages stay independently
  callable.** Weighed a processed-only, loader-free QC against a combined run; chose
  combined so a single `quality_control.json` carries both stages (the diagram's
  intent), accepting that the QC capsule needs both the raw asset and the NWB.
  Crucially the raw-vs-processed split already exists at the stage level: `RawQC`,
  `ProcessedQC`, and `build_quality_control` are callable and composable on their own
  (see [Composability & usage](composability.md)). `run_qc` is just the convenience combiner.
- **`RawQC` consumes the `contraqctor` `Dataset`, not `RawDataDict`.** The diagram's
  dict can't support contract checks.
- **Raw data is packaged into the NWB acquisition** — an explicit requirement from
  the scientist, so it's required, not incidental (refer to section 5 of the
  [Alignment](alignment.md) doc).
- **Per-event data (licks/rewards) is kept as acquisition series** for two deliberate
  reasons: **backward compatibility with the existing pipeline** and the deferred
  `EventsBuilder` stage — not merely a stopgap (refer to section 4 of the
  [Alignment](alignment.md) doc).
