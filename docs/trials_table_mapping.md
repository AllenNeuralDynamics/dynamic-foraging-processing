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

### Reward-delivery annotations

Each reward-delivery timestamp carries a label in the series' `data` field:

| Label | Meaning |
| --- | --- |
| `earned` | Water the animal worked for: the matched trial has no free water (`is_auto_reward_right` is `None`). |
| `auto` | Free water: the matched trial has `is_auto_reward_right` set. Scheduled autowater and the anti-bias intervention share that channel and are **not** split here — `auto_waterL` / `auto_waterR` and `anti_bias_left_water` / `anti_bias_right_water` record the mechanism per trial. |
| `manual` | The delivery is the closest valve opening to a `GiveManualWater` software event for this port. Takes precedence over the other labels, since manual water is not aligned to a go cue. |

Two properties of this series are worth stating explicitly, because both differ
from "every time the valve opened":

**Deliveries are matched to trials by the `Response` timestamp**, not the
`TrialOutcome` timestamp. `TrialOutcome` fires at the *end* of a trial, after the
reward-consumption and ITI periods, so a delivery can sit nearer the *previous*
trial's outcome and inherit its `is_auto_reward_right`. The valve opens within
milliseconds of the response, so the response anchors each delivery to its own
trial. Two independent checks back this: the nearest-`Response` trial agrees
with the trial whose `[quiescent_start_time, ITI_start_time)` window contains the
delivery, and every `earned` delivery follows a lick on that same port within a
few milliseconds.

**The series records every valve opening**: nothing is filtered out, so a
delivery on a trial reporting `is_rewarded=False` is still annotated. Free water
is triggered immediately at the go cue and the trial then continues normally, so
`is_rewarded` reports the outcome of the animal's *own choice* — a separate
event from the free water. A single trial can therefore contribute an `auto`
delivery at the port the task watered and an `earned` delivery at the port the
animal chose. The series is a water record rather than a reward-trial count, so
its length is not expected to equal `sum(is_rewarded)` over the trials.

## Trials Table

Columns are grouped by the raw source they map from.

### From `task_logic_input` (under `Logs`, `trial_generator` key)

| Trials column | Source field |
| --- | --- |
| `ITI_beta`, `ITI_min`, `ITI_max`, `ITI_duration` | `inter_trial_interval_duration`. `ITI_min` is the distribution's scaling `offset` (the sampled value is shifted by it, so the offset is the shortest possible ITI) rather than the truncation minimum. |
| `block_beta`, `block_duration`, `block_min`, `block_max` | `block_length`. `block_max` is one below the configured maximum, which accounts for the floor applied upstream. |
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
| `auto_waterL` / `auto_waterR` | **Scheduled autowater only**: `1` when `trial.metadata.extra.is_autowater` is `True` **and** `is_auto_reward_right` points to that side. `0` otherwise, including when the trial's free water came from the anti-bias algorithm — that is reported by `anti_bias_left_water` / `anti_bias_right_water`. `is_auto_reward_right` is only the delivery *channel* (free water fired, and to which side); the mechanism comes from the metadata, so the two columns are mutually exclusive. Not gated on `is_rewarded`: the column records what the task did, and free water fires at the go cue regardless of how the animal's own choice resolves. Note this is narrower than the legacy `dynamic-foraging-task` column of the same name, which was the ungated channel ("Autowater given at Left", straight from `B_AutoWaterTrial`) and predates anti-bias water. |
| `anti_bias_left_water` / `anti_bias_right_water` | Boolean. `True` when the anti-bias algorithm delivered a water intervention to that side — i.e. `trial.metadata.extra.is_bias_water_intervention` is `True` **and** `is_auto_reward_right` points to that side (`False` → left, `True` → right). The anti-bias water uses the same auto-response channel as scheduled autowater, so the `is_bias_water_intervention` flag is what distinguishes it and the two columns are mutually exclusive. `False` otherwise. Like `auto_water*`, **not** gated on `is_rewarded`: these columns record what the algorithm did, and the intervention fires at the go cue regardless of how the animal's own choice resolves. The reward-delivery series is also ungated, so an intervention on a trial that did not pay out appears in both. |
| `anti_bias_lickspout_movement` | Signed horizontal displacement (mm, positive is rightward) the anti-bias algorithm moved the lickspouts on this trial: `trial.lickspout_offset_delta` when `trial.metadata.extra.is_bias_stage_intervention` is `True`, else `0.0`. |
| `bait_left` / `bait_right` | Boolean, read straight from `trial.metadata.extra.is_left_baited` / `is_right_baited` — the bait state the acquisition software reports for each port. `False` when the trial carries no extra metadata. |
| `response_duration` | `response_deadline_duration`. |
| `reward_consumption_duration` | `Trial -> reward_consumption_duration`. |
| `reward_probabilityL` / `reward_probabilityR` | The **block** probability from `Trial -> metadata -> p_reward_left` / `p_reward_right`. The top-level `trial.p_reward_left` / `p_reward_right` is the per-trial probability, not the block probability, so it is not used here. `None` when the trial or its metadata is missing. |
| `reward_size_left` | `Trial -> reward_size.left` — the reward volume (uL) at the left port. Defaults to `2.0` when not set on the trial. `None` when the trial is missing. |
| `reward_size_right` | `Trial -> reward_size.right` — the reward volume (uL) at the right port. Defaults to `2.0` when not set on the trial. `None` when the trial is missing. |
| `rewarded_historyL` / `rewarded_historyR` | **Earned** reward only: filter `is_rewarded == True`, then on `is_right_choice`. `False` on both sides when `is_auto_reward_right` is set (either side) — that trial's water is free water, reported by `auto_waterL` / `auto_waterR` (scheduled autowater) or `anti_bias_left_water` / `anti_bias_right_water` (anti-bias intervention). This matches the `earned` / `auto` split in the reward-delivery series: per side, `rewarded_history*` equals that series' `earned` count exactly. |

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
| `ITI_stop_time` | The **next** trial's `QuiescentPeriod` `timestamp`. The last trial has no following quiescent period, so it takes the `EndSession` `timestamp`; `NaN` if that stream is unavailable. |
| `delay_start_time` | `QuiescentPeriod` `timestamp` — the legacy name for `quiescent_start_time` (see the note below). |

There are no `start_time` / `stop_time` trial columns. NWB's `TimeIntervals`
requires a native `start_time` / `stop_time` per trial, so the pipeline derives
the trial extent when writing: `start_time` is `quiescent_start_time` and
`stop_time` is `ITI_stop_time` — which on the last trial is the `EndSession`
timestamp.

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
| 2026-08-17 | Reward-delivery annotations now match each delivery to its trial by the `Response` software-event timestamp rather than the `TrialOutcome` timestamp. `TrialOutcome` fires at the *end* of a trial (after the reward-consumption and ITI periods), so a delivery could land nearer the *previous* trial's outcome and inherit its `is_auto_reward_right`, flipping `earned` and `auto`. The valve opens within milliseconds of the response, so the response anchors the delivery to its own trial. Verified 0-mismatch against trial-window containment. |
| 2026-08-17 | The reward-delivery series annotates every valve opening; deliveries are no longer filtered on `is_rewarded`. Free water fires at the go cue and the trial then continues normally, so `is_rewarded` describes the outcome of the animal's own choice, not the water — a trial where the task waters one port and the animal earns reward at the other is valid and contributes one `auto` and one `earned` delivery. The series is therefore a complete water record rather than a reward-trial count, and its total is not expected to equal the metadata mapper's `sum(is_rewarded)`. |
| 2026-08-17 | `auto_waterL` / `auto_waterR` now read `trial.metadata.extra.is_autowater` rather than the `is_auto_reward_right` channel, making them **scheduled autowater only** and mutually exclusive with `anti_bias_left_water` / `anti_bias_right_water`. `is_auto_reward_right` says free water fired and on which side but not what kind; the mechanism is in the metadata. Neither column is gated on `is_rewarded`, since both record what the task did. This is narrower than the legacy `dynamic-foraging-task` column of the same name, which was the ungated channel and predates anti-bias water. |
| 2026-08-17 | The reward-delivery labels stay `earned` / `auto` / `manual`: free water is `auto` whatever mechanism produced it, so the series does not split scheduled autowater from anti-bias water. That split lives in the trials table. Consequence: the series' `auto` count tracks the channel while `auto_waterL` / `auto_waterR` track `is_autowater`, so the two are not expected to be equal. |
| 2026-08-20 | `block_max` is now one below `block_length`'s configured maximum, which accounts for the floor applied upstream: a block is a whole number of trials, so the configured bound is never itself reachable. `block_min`, `block_beta`, and the `ITI_*` / `delay_*` bounds are unchanged — those durations are continuous and take no such adjustment. |
| 2026-08-20 | `ITI_min` now reports `inter_trial_interval_duration`'s scaling `offset` instead of its truncation minimum: the sampled ITI is shifted by the offset, so the offset is the shortest ITI the generator can produce. Falls back to the truncation minimum when no scaling parameters are configured. |
| 2026-08-20 | `bait_left` / `bait_right` now read `trial.metadata.extra.is_left_baited` / `is_right_baited` from the acquisition software instead of being re-derived from `p_reward_left` / `p_reward_right` and the `is_auto_reward_right` channel. The software is the authority on bait state, so the two can disagree — notably a port with `p_reward == 1` is no longer assumed baited. `False` when the trial carries no extra metadata. |
