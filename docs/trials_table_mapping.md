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
| `anti_bias_left_water` / `anti_bias_right_water` | Boolean. `True` when the anti-bias algorithm delivered a water intervention to that side — i.e. `trial.metadata.extra.is_bias_water_intervention` is `True` **and** `is_auto_reward_right` points to that side (`False` → left, `True` → right). The anti-bias water uses the same auto-response channel as ordinary autowater, so the `is_bias_water_intervention` flag is what distinguishes it. `False` otherwise. |
| `anti_bias_lickspout_movement` | Signed horizontal displacement (mm, positive is rightward) the anti-bias algorithm moved the lickspouts on this trial: `trial.lickspout_offset_delta` when `trial.metadata.extra.is_bias_stage_intervention` is `True`, else `0.0`. |
| `bait_left` / `bait_right` | Boolean. `bait_right` is `True` if `p_reward_right == 1` and `is_auto_reward_right` is `None` or `False`. `bait_left` is `True` if `p_reward_left == 1` and `is_auto_reward_right` is `None` or `True`. |
| `response_duration` | `response_deadline_duration`. |
| `reward_consumption_duration` | `Trial -> reward_consumption_duration`. |
| `reward_probabilityL` / `reward_probabilityR` | The **block** probability from `Trial -> metadata -> p_reward_left` / `p_reward_right`. The top-level `trial.p_reward_left` / `p_reward_right` is the per-trial probability, not the block probability, so it is not used here. `None` when the trial or its metadata is missing. |
| `reward_size_left` | `Trial -> reward_size.left` — the reward volume (uL) at the left port. Defaults to `2.0` when not set on the trial. `None` when the trial is missing. |
| `reward_size_right` | `Trial -> reward_size.right` — the reward volume (uL) at the right port. Defaults to `2.0` when not set on the trial. `None` when the trial is missing. |
| `rewarded_historyL` / `rewarded_historyR` | **Earned** reward only: filter `is_rewarded == True`, then on `is_right_choice`. `False` on both sides when `is_auto_reward_right` is set (either side) — that trial's water is autowater and is reported by `auto_waterL` / `auto_waterR`. |

### From `TrialGeneratorSpec.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `base_reward_probability_sum` | If `type == "CoupledTrialGenerator"`, look at `reward_probability_parameters`. |
| `min_reward_each_block` | `min_block_reward` when `type == "CoupledWarmupTrialGenerator"`; otherwise `0`, since a generator without that field enforces no per-block reward minimum. |

### Trial period timing (the four period `SoftwareEvents` streams)

Each of `QuiescentPeriod.json`, `ResponsePeriod.json`,
`RewardConsumptionPeriod.json`, and `ItiPeriod.json` emits one event per trial at
the **start** of its period, and the periods run back-to-back (the AIND DF v2
trial structure):

```
              go cue                  response         ITI start      next trial
                 |                    registered           |               |
   quiescent     |      response      |  reward consumption |     ITI       | quiescent
|--------------->|------------------->|------------------->|-------------->|---------->
q[i]             r[i]                 c[i]                 iti[i]          q[i+1]
```

So each period's stop time is the next period's start time. All four streams are
aligned with `TrialOutcome` by position (a length mismatch is reported by
`_check_aligned`; a short stream pads with `NaN`).

Verified on
`864253_2026-07-29_11-50-18` (753 trials): all five streams have equal length,
`q[i] < r[i] < c[i] < iti[i] < q[i+1]` holds for every trial, the `SoundCard` go
cue falls within 1.2 ms of `r[i]` on every trial, and the realized period
durations track the configured ones (reward consumption ≈
`reward_consumption_duration`, `iti[i] → q[i+1]` ≈ `ITI_duration`).

| Trials column | Mapping |
| --- | --- |
| `quiescent_start_time` | `QuiescentPeriod` `timestamp`. |
| `quiescent_stop_time` | `ResponsePeriod` `timestamp` (the quiescent period ends where the response period begins). |
| `response_start_time` | `ResponsePeriod` `timestamp`. |
| `response_stop_time` | `RewardConsumptionPeriod` `timestamp`. |
| `reward_consumption_start_time` | `RewardConsumptionPeriod` `timestamp`. |
| `reward_consumption_stop_time` | `ItiPeriod` `timestamp`. |
| `ITI_start_time` | `ItiPeriod` `timestamp`. |
| `ITI_stop_time` | The **next** trial's `QuiescentPeriod` `timestamp`; `NaN` on the last trial of the session. |
| `delay_start_time` | `QuiescentPeriod` `timestamp` — the legacy name for `quiescent_start_time` (see the note below). |

There are no `start_time` / `stop_time` trial columns. NWB's `TimeIntervals`
requires a native `start_time` / `stop_time` per trial, so the pipeline derives
the trial extent when writing: `start_time` is `quiescent_start_time` and
`stop_time` is `ITI_stop_time`, falling back to `ITI_start_time` on the last
trial.

> **`delay` means `quiescent`.** The legacy `delay_*` columns describe the
> acquisition software's *quiescence period* — the lick-free interval preceding
> the go cue. `delay_start_time` is therefore the `QuiescentPeriod` timestamp and
> always equals `quiescent_start_time`, and `delay_duration` /
> `delay_beta` / `delay_min` / `delay_max` summarize
> `quiescence_period_duration`. Note `delay_duration` is the *configured*
> duration: each lick restarts the quiescent period, so the realized duration
> (`quiescent_stop_time - quiescent_start_time`) can be longer.

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
| 2026-07-24 | `reward_size_left` / `reward_size_right` moved from session-level `task_parameters.reward_size` to per-trial `Trial.reward_size` (fields `.left` / `.right`). The columns are now nullable — `None` when the trial is missing. A missing `TaskLogic` stream no longer raises; session distribution columns are simply null. `min_reward_each_block` moved from `CoupledTrialGenerator` to `CoupledWarmupTrialGenerator`. |
| 2026-07-27 | Added `anti_bias_left_water` / `anti_bias_right_water` (boolean anti-bias water interventions per side) and `anti_bias_lickspout_movement` (mm the anti-bias algorithm shifted the lickspouts) from `TrialOutcome`'s `trial.metadata.extra` (`is_bias_water_intervention` / `is_bias_stage_intervention`), `is_auto_reward_right`, and `lickspout_offset_delta`. These are also overlaid on the QC `side_bias.png` figure. |
| 2026-08-06 | **Breaking:** the trial `start_time` / `stop_time` columns are removed and replaced by one start/stop pair per task period: `quiescent_start_time` / `quiescent_stop_time`, `response_start_time` / `response_stop_time`, `reward_consumption_start_time` / `reward_consumption_stop_time`, and `ITI_start_time` / `ITI_stop_time`, read from the `ResponsePeriod` and `RewardConsumptionPeriod` streams in addition to `QuiescentPeriod` and `ItiPeriod`. Each period event marks its period's start, so each stop is the next period's start; `ITI_stop_time` is the next trial's `QuiescentPeriod` timestamp (`NaN` on the last trial). The two new streams are also checked for positional alignment with `TrialOutcome`. NWB's required native `start_time` / `stop_time` are now derived when writing (`quiescent_start_time` → `ITI_stop_time`, falling back to `ITI_start_time`), so the NWB trials table changes in two ways: the old `start_time` / `stop_time` columns are gone, and the native trial extent now ends at the *end* of the ITI rather than at its start. |
| 2026-08-06 | Confirmed and documented that the legacy `delay_*` columns describe the acquisition software's **quiescence period**: `delay_start_time` is the `QuiescentPeriod` timestamp (always equal to the new `quiescent_start_time`) and `delay_duration` / `delay_beta` / `delay_min` / `delay_max` summarize `quiescence_period_duration`. `delay_duration` is the *configured* duration — each lick restarts the quiescent period, so the realized `quiescent_stop_time - quiescent_start_time` can be longer. Column descriptions updated accordingly. |
| 2026-08-12 | `rewarded_historyL` / `rewarded_historyR` now record **earned** reward only: an auto-reward trial (`is_auto_reward_right` set to either side) is `False` on *both* sides, since `TrialOutcome.is_rewarded` is `True` for autowater too and that water is already reported by `auto_waterL` / `auto_waterR`. This matches the `earned` / `automatic` split used for the NWB reward-delivery annotations. |
| 2026-08-12 | `min_reward_each_block` is now `0` rather than `NULL` when the trial generator exposes no `min_block_reward` — no per-block minimum is a floor of zero, not an unknown. The column is non-nullable (`float`, default `0`). |
