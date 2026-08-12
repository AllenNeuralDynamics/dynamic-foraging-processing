"""Pydantic model for a single row of the NWB trials table.

Field descriptions are copied verbatim from ``nwb_trial_column_info.json`` in the
combined pipeline and become the NWB trial-column descriptions. Only in-scope
columns are modeled; ``autoTrain`` curriculum and optogenetics columns are out of
scope
"""

from typing import Optional

from pydantic import BaseModel, Field


class TrialConfig(BaseModel):
    """One row of the NWB trials table.

    Each field's ``description`` is the source of truth for the corresponding
    NWB trial-column description.
    """

    # --- Trial period timing (one start/stop pair per task period) ---
    # The four periods run back-to-back in this order, each software event marking
    # the *start* of its period, so each period's stop is the next period's start:
    # quiescent -> response -> reward consumption -> ITI -> (next trial's
    # quiescent). No entry in the column-info JSON.
    quiescent_start_time: float = Field(
        description=(
            "Start time of the quiescent period (QuiescentPeriod timestamp). The quiescent period is the lick-free delay preceding the go cue; each lick restarts it, so its realized duration can exceed the configured delay_duration."
        ),
    )
    quiescent_stop_time: float = Field(
        description=(
            "End time of the quiescent period, i.e. the start of the response period (ResponsePeriod timestamp); the go cue is played at this boundary."
        ),
    )
    response_start_time: float = Field(
        description=(
            "Start time of the response period (ResponsePeriod timestamp), when the go cue is played."
        ),
    )
    response_stop_time: float = Field(
        description=(
            "End time of the response period, i.e. the start of the reward consumption period (RewardConsumptionPeriod timestamp). This is when the animal responded, or the response deadline for an ignored trial."
        ),
    )
    reward_consumption_start_time: float = Field(
        description=(
            "Start time of the reward consumption period (RewardConsumptionPeriod timestamp)."
        ),
    )
    reward_consumption_stop_time: float = Field(
        description=(
            "End time of the reward consumption period, i.e. the start of the inter-trial interval (ItiPeriod timestamp)."
        ),
    )
    ITI_start_time: float = Field(
        description="Start time of the inter-trial interval (ItiPeriod timestamp).",
    )
    ITI_stop_time: float = Field(
        description=(
            "End time of the inter-trial interval, i.e. the start of the next trial's quiescent period (the following QuiescentPeriod timestamp); NaN on the last trial of the session."
        ),
    )

    # --- trial_info ---
    animal_response: int = Field(
        ge=0,
        le=2,
        description="The response of the animal. 0, left choice; 1, right choice; 2, no response",
    )
    rewarded_historyL: bool = Field(
        default=False,
        description=(
            "The earned reward history of the left lick port; False on autowater trials, whose water is reported by auto_waterL"
        ),
    )
    rewarded_historyR: bool = Field(
        default=False,
        description=(
            "The earned reward history of the right lick port; False on autowater trials, whose water is reported by auto_waterR"
        ),
    )
    delay_start_time: Optional[float] = Field(
        default=None,
        description=(
            "Legacy name for the start of the quiescent period (QuiescentPeriod timestamp); the 'delay' of the legacy delay_* columns is the acquisition software's quiescent period, so this always equals quiescent_start_time."
        ),
    )
    goCue_start_time: Optional[float] = Field(default=None, description="The go cue start time")

    # --- behavior_structure ---
    bait_left: bool = Field(
        default=False, description="Whether the current left lickport has a bait or not"
    )
    bait_right: bool = Field(
        default=False, description="Whether the current right lickport has a bait or not"
    )
    base_reward_probability_sum: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="The summation of left and right reward probability",
    )
    reward_probabilityL: Optional[float] = Field(
        default=None, ge=0, le=1, description="The block reward probability of the left lick port"
    )
    reward_probabilityR: Optional[float] = Field(
        default=None, ge=0, le=1, description="The block reward probability of the right lick port"
    )
    left_valve_open_time: Optional[float] = Field(
        default=None, description="Duration (s) the left valve was open"
    )
    right_valve_open_time: Optional[float] = Field(
        default=None, description="Duration (s) the right valve was open"
    )
    reward_size_left: float = Field(
        description="The reward volume (uL) delivered at the left lick port if rewarded.",
    )
    reward_size_right: float = Field(
        description="The reward volume (uL) delivered at the right lick port if rewarded.",
    )

    # --- trial_metrics (per-trial metrics emitted by the acquisition system) ---
    side_bias: Optional[float] = Field(
        default=None,
        description=(
            "Per-trial side bias computed by the acquisition system. Negative values correspond to a left bias, positive values to a right bias."
        ),
    )

    # --- block_information ---
    block_beta: Optional[float] = Field(
        default=None,
        description="The beta of exponential distribution to generate the block length",
    )
    block_min: Optional[float] = Field(
        default=None, description="The minimum length allowed for each block"
    )
    block_max: Optional[float] = Field(
        default=None, description="The maximum length allowed for each block"
    )
    min_reward_each_block: Optional[float] = Field(
        default=None, description="The minimum reward allowed for each block"
    )

    # --- delay_duration ---
    delay_beta: Optional[float] = Field(
        default=None,
        description="The beta of exponential distribution to generate the delay duration(s). Can be none depending on distribution type (scalar for example)",
    )
    delay_min: Optional[float] = Field(
        default=None, description="The minimum duration(s) allowed for each delay"
    )
    delay_max: Optional[float] = Field(
        default=None, description="The maximum duration(s) allowed for each delay"
    )
    delay_duration: Optional[float] = Field(
        default=None,
        description=(
            "The configured duration (s) of the delay (quiescent) period between delay start and go cue start. Each lick restarts the quiescent period, so the realized duration (quiescent_stop_time - quiescent_start_time) can be longer."
        ),
    )

    # --- ITI_duration ---
    ITI_beta: Optional[float] = Field(
        default=None,
        description="The beta of exponential distribution to generate the ITI duration(s).",
    )
    ITI_min: Optional[float] = Field(
        default=None, description="The minimum duration(s) allowed for each ITI"
    )
    ITI_max: Optional[float] = Field(
        default=None, description="The maximum duration(s) allowed for each ITI"
    )
    ITI_duration: Optional[float] = Field(
        default=None, description="The expected time duration (s) between trial start and ITI start"
    )

    # --- response_and_reward_duration ---
    response_duration: Optional[float] = Field(
        default=None,
        description="The maximum time that the animal must make a choice in order to get a reward",
    )
    reward_consumption_duration: Optional[float] = Field(
        default=None, description="The duration for the animal to consume the reward"
    )

    # --- auto_waterL/R (autowater per-side; autoTrain curriculum fields out of scope) ---
    auto_waterL: int = Field(default=0, description="Autowater given at Left")
    auto_waterR: int = Field(default=0, description="Autowater given at Right")

    # --- anti_bias (interventions the anti-bias algorithm applies) ---
    anti_bias_left_water: bool = Field(
        default=False,
        description=(
            "Whether the anti-bias algorithm delivered a water intervention to the left lickport on this trial."
        ),
    )
    anti_bias_right_water: bool = Field(
        default=False,
        description=(
            "Whether the anti-bias algorithm delivered a water intervention to the right lickport on this trial."
        ),
    )
    anti_bias_lickspout_movement: float = Field(
        default=0.0,
        description=(
            "Horizontal distance (mm) the lickspouts were moved by the anti-bias algorithm on this trial (positive is rightward); 0 when no lickspout intervention occurred."
        ),
    )

    # --- lickspout_position (mapping's `lickspout_positions` -> these four components) ---
    lickspout_position_x: Optional[float] = Field(
        default=None,
        description="x lickspout position (mm), relative to session start (left-right)",
    )
    lickspout_position_y1: Optional[float] = Field(
        default=None,
        description="y1 left lickspout position (mm), relative to session start (forward-backward)",
    )
    lickspout_position_y2: Optional[float] = Field(
        default=None,
        description="y2 right lickspout position (mm), relative to session start (forward-backward)",
    )
    lickspout_position_z: Optional[float] = Field(
        default=None,
        description="z lickspout position (mm), relative to session start (up-down)",
    )

    @classmethod
    def column_descriptions(cls) -> dict[str, str]:
        """Map each column name to its NWB description.

        Returns
        -------
        dict of str to str
            Column name to description, suitable for adding NWB trial columns.
        """
        return {
            name: field.description
            for name, field in cls.model_fields.items()
            if field.description is not None
        }
