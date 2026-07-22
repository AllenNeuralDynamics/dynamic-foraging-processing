# Mapping of Raw Acquisition Streams to the NWB Trials Table

This document describes how the NWB `acquisition` container and the `trials` table
are constructed from the raw dynamic foraging acquisition streams.

Reference asset used while mapping:
[behavior_836626_2026-05-20_14-19-10_processed_2026-05-21_17-40-47](https://codeocean.allenneuraldynamics.org/data-assets/49d1b596-c1a0-4c52-a3dd-26181f4b2b55/behavior_836626_2026-05-20_14-19-10_processed_2026-05-21_17-40-47).

Trial column descriptions are derived from
[`nwb_trial_column_info.json`](https://github.com/AllenNeuralDynamics/aind-fip-nwb-base-capsule/blob/main/code/util/nwb_trial_column_info.json)
in the combined pipeline.

> **Note:** Any column related to `autoTrain` can be disregarded (per meeting with
> Alex on June 3rd, 2026).

## Acquisition Container

The NWB `acquisition` container holds four behavior-related time series:

| Acquisition series | Source stream | Notes |
| --- | --- | --- |
| `left_lick_time` | `Behavior/Lickometer` | |
| `right_lick_time` | `Behavior/Lickometer` | |
| `left_reward_delivery_time` | `Behavior/HarpBehavior` `OutputSet` (`SupplyPort0`, `WRITE` messages) | The left valve open timestamp. |
| `right_reward_delivery_time` | `Behavior/HarpBehavior` `OutputSet` (`SupplyPort1`, `WRITE` messages) | The right valve open timestamp. |

Earlier mapping used `Response.json` (`SoftwareEvents`) for lick times (where
`Item1` is the time and `Item2` is `left`/`right`) and `TrialOutcome.json`
(filtered on `is_rewarded`, then `left`/`right`) for reward delivery times.
Lick times now come from the `Behavior/Lickometer` stream, and reward delivery
times use the Harp valve open times.

## Trials Table

Columns are grouped by the raw source they map from.

### From `task_logic_input` (under `Logs`, `trial_generator` key)

| Trials column | Source field |
| --- | --- |
| `ITI_beta`, `ITI_min`, `ITI_max`, `ITI_duration` | `inter_trial_interval_duration` |
| `block_beta`, `block_duration`, `block_min`, `block_max` | `block_length` |
| `delay_beta`, `delay_duration`, `delay_min`, `delay_max` | `quiescent_duration_key` (scalar distribution, so no beta/min/max) |

### From `task_logic_input` (under `task_parameters`)

| Trials column | Source field |
| --- | --- |
| `reward_size_left` | `task_parameters.reward_size.left_value_volume` — the reward volume (uL) at the left port. |
| `reward_size_right` | `task_parameters.reward_size.right_value_volume` — the reward volume (uL) at the right port. |

> **Note:** `reward_size` is read from the task parameters, not the trial
> generator, so it is populated even when no summarising generator is resolved.
> The acquisition system can in principle vary reward size per trial, but the
> current data format only exposes a single session-level value, so these
> columns are constant across trials. They are **required** (non-nullable): a
> missing `TaskLogic` stream raises rather than silently producing null reward
> sizes when there are trials to build.

### From `TrialMetrics.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `side_bias` | Per-trial `bias` field from the `TrialMetrics` event (negative → left bias, positive → right bias). `None` when not recorded. Aligned by position with `TrialOutcome`. |

### From `Response.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `animal_response` | From the event payload `{ "Item1": <time>, "Item2": <choice> }`. `Item2` `True` → right (`1`), `False` → left (`0`); a missing payload or `None` `Item2` → no response (`2`). |

### From `TrialOutcome.json` (`SoftwareEvents` stream)

> **Note:** For `is_auto_reward_right`, `True` means right and `False`
> means left.

| Trials column | Mapping |
| --- | --- |
| `auto_waterL` / `auto_waterR` | From `is_auto_reward_right`. `1` on the auto-responded side; `0` on the other side, when there was no auto-response (`None`), or when the trial is missing. |
| `bait_left` / `bait_right` | Boolean. `bait_right` is `True` if `p_reward_right == 1` and `is_auto_reward_right` is `None` or `False`. `bait_left` is `True` if `p_reward_left == 1` and `is_auto_reward_right` is `None` or `True`. |
| `response_duration` | `response_deadline_duration`. |
| `reward_consumption_duration` | `Trial -> reward_consumption_duration`. |
| `reward_probabilityL` / `reward_probabilityR` | The **block** probability from `Trial -> metadata -> p_reward_left` / `p_reward_right`. The top-level `trial.p_reward_left` / `p_reward_right` is the per-trial probability, not the block probability, so it is not used here. `None` when the trial or its metadata is missing. |
| `rewarded_historyL` / `rewarded_historyR` | Filter `is_rewarded == True`, then on `is_right_choice`. |

### From `TrialGeneratorSpec.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `base_reward_probability_sum` | If `type == "CoupledTrialGenerator"`, look at `reward_probability_parameters`. |
| `min_reward_each_block` | Present when `type == "CoupledTrialGenerator"`; otherwise `None`. |

### From `QuiescentPeriod.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `delay_start_time` | `timestamp`. |
| `start_time` | `timestamp` column. |

### From `ITI_period.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `stop_time` | `timestamp` column. Possible QC check: length should match `QuiescentPeriod.json`. |

### From `HarpBehavior` (`PulseSupplyPort{0,1}`)

| Trials column | Mapping |
| --- | --- |
| `left_valve_open_time` | `PulseSupplyPort0` value (ms -> s). Duration the valve is open. |
| `right_valve_open_time` | `PulseSupplyPort1` value (ms -> s). Duration the valve is open. |

The `PulseSupplyPort{0,1}` register holds the valve-open pulse width in
milliseconds — a reward opens the valve for this fixed duration. It is a
per-session configuration value, so the same duration is written to every trial
(converted to seconds). Note: `OutputSet`'s `SupplyPort` columns are *not* the
reward pulse — they track a sustained left/right state sampled only every few
seconds, far too coarse for the ~tens-of-ms valve pulse.

### From `SoundCard` (`WRITE` messages)

| Trials column | Mapping |
| --- | --- |
| `goCue_start_time` | `PlaySoundOrFrequency` `WRITE` message. |

### From `HarpManipulator` `AccumulatedSteps` (+ `InputSchemas.Rig`)

| Trials column | Mapping |
| --- | --- |
| `lickspout_position_x` / `y1` / `y2` / `z` | Per-motor cumulative microstep count from the `AccumulatedSteps` stream, converted to millimetres via the rig manipulator calibration (`full_step_to_mm / microstep_resolution`) and re-referenced to the session-start position (displacement **relative to session start**, mm). The manipulator is a continuously-sampled hardware value, so — like the go cue — each trial takes the sample within its `[start_time, stop_time)` window nearest the start. `Motor{i}` drives `Axis(i + 1)` (X, Y1, Y2, Z). `None` when no sample falls in the trial window. The rig and `AccumulatedSteps` streams are required inputs (`build` raises if either is missing with trials present). |

### From `trainer_state.json` and `acquisition.json` (autoTrain — can be disregarded)

These were mapped during exploration but are no longer in scope:

- `auto_train_curriculum_name` / `auto_train_curriculum_schema_version` —
  `trainer_state.json` (top level).
- `auto_train_engaged` — Boolean flag in `acquisition.json` indicating whether
  the curriculum is running.
- `auto_train_stage` — `stage` in `trainer_state.json` (should always exist).
- `auto_train_stage_overridden` — `True` when `on_curriculum` in
  `acquisition.json` is `False`.

### Not applicable to this task

| Trials column | Mapping |
| --- | --- |
| `reward_random_L` / `reward_random_R` | None — no task component drives these. |

## Changelog

| Date | Change |
| --- | --- |
| 2026-06-17 | `animal_response` now decodes the `Response` event's `{ "Item1": <time>, "Item2": <choice> }` payload via `Item2` (`True` → right `1`, `False` → left `0`, missing/`None` → no response `2`), rather than treating the whole payload as the choice. |
| 2026-06-17 | `auto_waterL` / `auto_waterR` now encode no auto-response (`is_auto_reward_right` is `None`) and missing trials as `0` instead of `NULL`. The columns are non-nullable (`int`, default `0`). |
| 2026-06-20 | Added `reward_size_left` / `reward_size_right` (reward volume in uL) from `task_parameters.reward_size`, and `side_bias` from the per-trial `TrialMetrics` event (`bias` field). |
| 2026-06-20 | `reward_probabilityL` / `reward_probabilityR` now read the block probability from `trial.metadata.p_reward_left` / `p_reward_right` instead of the top-level per-trial `trial.p_reward_left` / `p_reward_right`. |
| 2026-07-22 | `lickspout_position_x` / `y1` / `y2` / `z` now derive from the `HarpManipulator` `AccumulatedSteps` stream (microsteps → mm via the `InputSchemas.Rig` manipulator calibration, `full_step_to_mm / microstep_resolution`), sampled per trial via the closest sample in the `[start_time, stop_time)` window and re-referenced to the session-start position (displacement relative to session start, mm), replacing the static `InitialManipulatorPosition` software event. `Motor{i}` maps to `Axis(i + 1)` (X, Y1, Y2, Z). The rig and `AccumulatedSteps` streams are required when there are trials (`build` raises if either is missing). Column descriptions corrected from `um` to `mm`. |
