# Final NWB Contents

This document describes the contents of the NWB file produced by this
repository. It is a companion to issue
[#12](https://github.com/AllenNeuralDynamics/dynamic-foraging-processing/issues/12),
which serves as the authoritative changelog for content decisions.

When the NWB contents change, update both the [Changelog](#changelog) below
and the relevant section in this document. Each entry should record at minimum
the date, what changed, and why.

## Acquisition

The `acquisition` container holds the HARP streams from the rig (e.g.
VR Foraging) along with four behavior-derived series carried over from the
NWB produced by the combined dynamic foraging + FIP pipeline:

- `left_lick_time`
- `right_lick_time`
- `left_reward_delivery_time`
- `right_reward_delivery_time`

Each series stores both timestamps and a parallel `data` array. For the
reward delivery series, `data` annotates each reward as `earned`, `manual`,
or `automatic`.

See [`trials_table_mapping.md`](trials_table_mapping.md#acquisition-container)
for the raw sources backing each of these four series.

## Events

The `events` container follows the conventions in
[aind-physio-arch#1072](https://github.com/AllenNeuralDynamics/aind-physio-arch/issues/1072).

The events sidecar will be version-controlled in this repository for now so that
changes can be tracked alongside the code.

Events are on pause pending validation by the HED team. See the
[Changelog](#changelog) for details.

## Trials

The `trials` table is built from the raw acquisition streams. The full
column-by-column mapping is documented in
[`trials_table_mapping.md`](trials_table_mapping.md), and the source-of-truth
discussion lives in issue
[#5](https://github.com/AllenNeuralDynamics/dynamic-foraging-processing/issues/5).

## Changelog

| Date | Section | Change | Reason |
| --- | --- | --- | --- |
| 2026-06-03 | acquisition / trials | Initial scope confirmed: HARP streams + `{left,right}_lick_time` and `{left,right}_reward_delivery_time` in `acquisition`; trials mapping per issue #5. | Meeting with Alex. |
| 2026-06-05 | events | Events on pause. | Pending validation by the HED team. |
| 2026-06-08 | acquisition | Documented `data` arrays alongside timestamps; reward delivery series annotate each reward as `earned`, `manual`, or `automatic`. | Clarify acquisition contents. |
