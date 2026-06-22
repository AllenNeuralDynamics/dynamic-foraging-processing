# QC Upgrade Plan

Plan for upgrading
[`aind-dynamic-foraging-qc/code/run_capsule.py`](https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-qc/blob/main/code/run_capsule.py)
to:

1. Conform to the current
   [`aind_data_schema.core.quality_control`](https://github.com/AllenNeuralDynamics/aind-data-schema/blob/dev/src/aind_data_schema/core/quality_control.py)
   schema (v2.4.1).
2. Operate on primitive structures (numpy arrays, pandas DataFrames). QC
   functions are agnostic to where the data came from — the caller is free
   to load from
   [`RawDataLoader`](../src/dynamic_foraging_processing/raw_data_loader/loader.py),
   an NWB file, or anything else, as long as the primitives match the
   expected shape.

This document is a design reference only. Implementation will happen on a
separate branch.

## 1. Schema changes

The schema removed `QCEvaluation`. The new `QualityControl` object holds a flat
`metrics: List[QCMetric | CurationMetric]` and groups metrics via per-metric
`tags`. Each `QCMetric` now requires `modality` and `stage` directly (these
moved off `QCEvaluation`), and `QualityControl` requires `default_grouping`.

### Field-by-field migration

| Old (capsule) | New (schema v2.4.1) |
| --- | --- |
| `QCEvaluation(name, modality, stage, metrics, description, allow_failed_metrics)` | Removed. Replace each evaluation with one or more `QCMetric`s sharing a tag. |
| `QCMetric(name, value, status_history, description?, reference?)` | `QCMetric(name, modality, stage, value, status_history, description?, reference?, tags={}, evaluated_assets?)` |
| `QualityControl(evaluations=[...])` | `QualityControl(metrics=[...], default_grouping=[...], key_experimenters?, notes?, allow_tag_failures?)` |
| n/a | `Status.PENDING` is now a valid third state alongside `PASS` / `FAIL`. |
| `allow_failed=True` on an evaluation | `allow_tag_failures=["<tag value>"]` on the top-level `QualityControl`. |

### Tag convention

Each ported behavior metric is tagged with `{"behavior": "<metric name>"}` —
the key is the group, the value is the metric's name. Contraqctor results
use a fixed `"test_suite"` key plus a dynamic per-suite key (see
[Contraqctor-based QA suites](#contraqctor-based-qa-suites-per-meeting-with-alex-2026-06-03)).

### Helper rewrites

`Bool2Status` keeps its shape but must produce timezone-aware timestamps
(schema uses `AwareDatetimeWithDefault`). The existing `datetime.now(seattle_tz)`
already satisfies this.

`create_evaluation(...)` is deleted. Replace with a small `make_metric(...)`
helper that stamps `modality`, `stage`, and `tags` onto each `QCMetric`.

## 2. Data inputs

The old capsule consumed a single `behavior.json` (e.g. `B_Bias`,
`B_LeftLickTime`, `B_RightLickTime`, `B_StagePositions`, `drop_frames_tag`,
`Experimenter`, `dirty_files`, ...). The new pipeline does not produce this
file.

QC functions now take primitive structures directly. The entry point is
responsible for producing those primitives — whether it pulls them from
`RawDataLoader.get_all_raw_data()`, an NWB file, or any other source is
out of scope for the QC module. This keeps the QC logic testable without
any dataset on disk.

### Primitive inputs per metric

| Primitive | Type | Old `behavior.json` analogue |
| --- | --- | --- |
| `left_lick_times` | `np.ndarray` of seconds | `B_LeftLickTime` |
| `right_lick_times` | `np.ndarray` of seconds | `B_RightLickTime` |
| `animal_response` | `np.ndarray` of `{0,1,2}` per trial | `B_AnimalResponseHistory` |
| `side_bias` | `np.ndarray` per trial (right minus left, `nan` on no-response) | `B_Bias` |
| `go_cue_times` | `np.ndarray` of seconds | `B_GoCueTimeSoundCard` |
| `rewarded_history` | `pd.DataFrame` with `left`/`right` boolean columns | `B_RewardedHistory` |
| `stage_positions` | `pd.DataFrame` with `x`/`y`/`z` columns per trial | `B_StagePositions` |

### Out-of-scope (no equivalent in the new data, drop the check)

- `drop_frames_tag`, `frame_num`, `trigger_length` — dropped-frames check.
- `Experimenter`, `dirty_files`, `repo_dirty_flag` — basic-configuration check.
- `B_Bias_CI` — side-bias confidence interval; dropped (the bias trace plots
  the per-trial `side_bias` column directly, with no CI band).

## 3. Metrics in the new capsule

Keep only what maps cleanly. All metrics get `stage=Stage.RAW` and
`modality=Modality.BEHAVIOR` unless noted.

### Side bias (`tags={"behavior": "average side bias"}`)

- Input: `side_bias: np.ndarray` — the per-trial side bias read directly from
  the trial table (right minus left; `nan` on no-response trials). It is *not*
  recomputed from `animal_response`.
- Average bias = `nanmean(side_bias)` over the session.
- Metric: `"average side bias"`, pass when `abs(mean_bias) < 0.5`. An empty or
  all-`nan` column yields `nan`, which fails.
- `reference="side_bias.png"`.

### Lick intervals

Port `calculate_lick_intervals` verbatim. Inputs are
`left_lick_times: np.ndarray` and `right_lick_times: np.ndarray`, extracted
from the `Behavior.Lickometer` stream at the entry point.

Emit the same four metrics, each tagged with its own name under the
`behavior` key:

| Metric | Tag | Pass rule |
| --- | --- | --- |
| `Left Lick Interval (%)` | `{"behavior": "Left Lick Interval (%)"}` | `< 10` |
| `Right Lick Interval (%)` | `{"behavior": "Right Lick Interval (%)"}` | `< 10` |
| `Cross Side Lick Interval (%)` | `{"behavior": "Cross Side Lick Interval (%)"}` | `< 10` |
| `Artifact Percent (%)` | `{"behavior": "Artifact Percent (%)"}` | `< 1` |

All carry `reference="lick_intervals.png"`.

### Plots to keep

- `lick_intervals.png` — five-panel histogram of inter-lick intervals
  (`left licks`, `right licks`, `left to right licks`, `right to left licks`,
  `all licks`); inputs are `left_lick_times` and `right_lick_times`.
- `side_bias.png` — four-panel figure:
  - Side bias trace — the per-trial `side_bias` column read from the trial
    table (no confidence-interval band).
  - Lickspout position over trials — `stage_positions` (x / y1 / y2 / z,
    relative to session start, in mm).
  - Behavior event raster — `animal_response` (L/R choice, ignore),
    `rewarded_history` (L/R earned water), manual water times, and
    `auto_water` (L/R) per trial.
  - Reward probabilities — `reward_probabilityL` / `reward_probabilityR`
    per trial.

### Contraqctor-based QA suites (per meeting with Alex, 2026-06-03)

Same approach as VR foraging QA.

The runner is provided by
[`aind_behavior_dynamic_foraging.data_qc.suite.make_qc_runner(dataset)`](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/blob/main/src/aind_behavior_dynamic_foraging/data_qc/suite.py),
so just needs to call it on `loader.dataset` and convert
the results. `make_qc_runner` already wires up:

- `ContractTestSuite` (dataset loading errors, excluding Harp command streams)
- `HarpDeviceTestSuite` for every `HarpDevice` under `Behavior`
- `HarpHubTestSuite`
- `HarpLicketySplitTestSuite` for the left and right lickometers
- `HarpSniffDetectorTestSuite` / `HarpEnvironmentSensorTestSuite` (conditional on the rig)
- `CameraTestSuite` for every camera in `BehaviorVideos` (uses `rig.triggered_camera_controller.frame_rate`)
- `CsvTestSuite` for every CSV stream
- `DynamicForagingQcSuite` (currently `test_end_session_exists`)

#### Result → `QCMetric` conversion

Map contraqctor statuses onto schema statuses:

```python
status_converter = {
    qc.Status.PASSED:  Status.PASS,
    qc.Status.SKIPPED: Status.PASS,
    qc.Status.WARNING: Status.PENDING,
    qc.Status.FAILED:  Status.FAIL,
    qc.Status.ERROR:   Status.FAIL,
}
```

For each `qc.Result`:

- `name = f"{result.suite_name}::{result.test_name}"`
- `description = f"Test: {result.description} // Message: {result.message}"`
- `value = convert_numpy_to_python_data_type(result.result)`
- `status_history = [QCStatus(evaluator="Automated", status=..., timestamp=now_utc)]`
- `modality = Modality.BEHAVIOR`, `stage = Stage.RAW`
- `tags = {"test_suite": result.suite_name, result.suite_name: group_name}`
  — one fixed `"test_suite"` key whose value is the suite name, plus a
  dynamic key (the suite name) whose value is the runner group (defaulting
  to `"NoGroup"`).
- `reference`: if `result.context["asset"]` is a `matplotlib.figure.Figure`,
  save it under the results folder and store the relative path.

#### Updated tag / grouping plan

| Tag key | Values |
| --- | --- |
| `behavior` | metric name (e.g. `average side bias`, `Left Lick Interval (%)`) |
| `test_suite` | only on contraqctor metrics; suite name (e.g. `HarpEnvironmentSensorTestSuite`) |

`default_grouping` tells the QC portal which tag *keys* to use when
laying out the metrics hierarchically (see the schema field's
[description](https://github.com/AllenNeuralDynamics/aind-data-schema/blob/dev/src/aind_data_schema/core/quality_control.py)).
Each entry is a tag key (or a list of tag keys at the same level); the
portal walks them in order and groups metrics by the values it finds for
those keys.

So `behavior` and `test_suite` are siblings at the top level; a metric
ends up under whichever one its tags match. They don't overlap because
the two groups of metrics carry disjoint tag keys.

Sample portal layout:

```
behavior
  Metric...
  Metric...

test_suite
  Metric...
  Metric...
```

## Changelog

| Date | Section | Change | Reason |
| --- | --- | --- | --- |
| 2026-06-03 | metrics | Confirmed kept QC metrics: side bias, lick intervals, and Harp/contract QA via `make_qc_runner`. Dropped checks tied to old `behavior.json` (dropped frames, basic configuration). | Meeting with Alex. |
| 2026-06-03 | qa | Adopt contraqctor `qc.Runner` output (`make_qc_runner(dataset)`) as the source for Harp / camera / contract / DynamicForaging QA, converted into `QCMetric`s. | Meeting with Alex. |
| 2026-06-22 | metrics, data inputs, plots | Side bias is read from the precomputed per-trial `side_bias` column (averaged via `nanmean`) instead of being recomputed from `animal_response`; dropped the `B_Bias_CI` confidence-interval band. | Reflect implemented `side_bias_result` / `plot_side_bias`. |
