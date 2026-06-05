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
| `left_reward_delivery_time` | `Behavior/HarpBehavior` `OutputSet` (`SupplyPort0`, `WRITE` messages) | Same as left valve open. |
| `right_reward_delivery_time` | `Behavior/HarpBehavior` `OutputSet` (`SupplyPort1`, `WRITE` messages) | Same as right valve open. |

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

### From `Response.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `animal_response` | `0` = left choice, `1` = right choice, `2` = no response. |

### From `TrialOutcome.json` (`SoftwareEvents` stream)

| Trials column | Mapping |
| --- | --- |
| `auto_waterL` / `auto_waterR` | From `is_auto_response_right`. `NULL` for None, `true` for right, `false` for left. Encoded `0`/`1`. |
| `bait_left` / `bait_right` | Boolean. `bait_right` is `True` if `p_reward_right == 1` and `auto_response_right` is `None` or `False`. `bait_left` is `True` if `p_reward_left == 1` and `auto_response_right` is `None` or `True`. |
| `response_duration` | `response_deadline_duration`. |
| `reward_consumption_duration` | `Trial -> reward_consumption_duration`. |
| `reward_probabilityL` / `reward_probabilityR` | Most likely the block probability: `Trial -> Metadata -> p_reward_left` / `p_reward_right`. Confirm with Alex whether the actual lickspout probability is intended. |
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

### From `HarpBehavior` (`OutputSet`)

| Trials column | Mapping |
| --- | --- |
| `left_valve_open_time` | `SupplyPort0`. |
| `right_valve_open_time` | `SupplyPort1`. |

Cross-correlate with software-event manual-reward times from the UI against
trial `start_time`/`stop_time` to disambiguate manual valve opens. Double-check
this.

### From `SoundCard` (`WRITE` messages)

| Trials column | Mapping |
| --- | --- |
| `goCue_start_time` | `PlaySoundOrFrequency` `WRITE` message. |

### From `InitialManipulatorPosition` (software event)

| Trials column | Mapping |
| --- | --- |
| `lickspout_positions` | `data` field. |

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
