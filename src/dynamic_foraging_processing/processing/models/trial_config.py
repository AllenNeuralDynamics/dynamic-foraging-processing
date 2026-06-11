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

    # --- Trial timing (NWB built-ins; no entry in the column-info JSON) ---
    start_time: float = Field(description="Trial start time (QuiescentPeriod timestamp).")
    stop_time: float = Field(description="Trial stop time (ItiPeriod timestamp).")

    # --- trial_info ---
    animal_response: Optional[int] = Field(
        default=None,
        description="The response of the animal. 0, left choice; 1, right choice; 2, no response",
    )
    rewarded_historyL: Optional[bool] = Field(
        default=None, description="The reward history of left lick port"
    )
    rewarded_historyR: Optional[bool] = Field(
        default=None, description="The reward history of right lick port"
    )
    delay_start_time: Optional[float] = Field(
        default=None,
        description="The delay start time, this is the first delay of the trial",
    )
    goCue_start_time: Optional[float] = Field(default=None, description="The go cue start time")

    # --- behavior_structure ---
    bait_left: Optional[bool] = Field(
        default=None, description="Whether the current left lickport has a bait or not"
    )
    bait_right: Optional[bool] = Field(
        default=None, description="Whether the current right lickport has a bait or not"
    )
    base_reward_probability_sum: Optional[float] = Field(
        default=None, description="The summation of left and right reward probability"
    )
    reward_probabilityL: Optional[float] = Field(
        default=None, description="The reward probability of left lick port"
    )
    reward_probabilityR: Optional[float] = Field(
        default=None, description="The reward probability of right lick port"
    )
    left_valve_open_time: Optional[float] = Field(
        default=None, description="The left valve open time"
    )
    right_valve_open_time: Optional[float] = Field(
        default=None, description="The right valve open time"
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
        description="The beta of exponential distribution to generate the delay duration(s)",
    )
    delay_min: Optional[float] = Field(
        default=None, description="The minimum duration(s) allowed for each delay"
    )
    delay_max: Optional[float] = Field(
        default=None, description="The maximum duration(s) allowed for each delay"
    )
    delay_duration: Optional[float] = Field(
        default=None, description="The duration between delay start and go cue start"
    )

    # --- ITI_duration ---
    ITI_beta: Optional[float] = Field(
        default=None,
        description="The beta of exponential distribution to generate the ITI duration(s)",
    )
    ITI_min: Optional[float] = Field(
        default=None, description="The minimum duration(s) allowed for each ITI"
    )
    ITI_max: Optional[float] = Field(
        default=None, description="The maximum duration(s) allowed for each ITI"
    )
    ITI_duration: Optional[float] = Field(
        default=None, description="The expected time duration between trial start and ITI start"
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
    auto_waterL: Optional[int] = Field(default=None, description="Autowater given at Left")
    auto_waterR: Optional[int] = Field(default=None, description="Autowater given at Right")

    # --- lickspout_position (mapping's `lickspout_positions` -> these four components) ---
    lickspout_position_x: Optional[float] = Field(
        default=None, description="x position (um) of the lickspout position (left-right)"
    )
    lickspout_position_y1: Optional[float] = Field(
        default=None,
        description="y1 position (um) of the left lickspout position (forward-backward)",
    )
    lickspout_position_y2: Optional[float] = Field(
        default=None,
        description="y2 position (um) of the right lickspout position (forward-backward)",
    )
    lickspout_position_z: Optional[float] = Field(
        default=None, description="z position (um) of the lickspout position (up-down)"
    )

    @classmethod
    def column_descriptions(cls) -> dict[str, str]:
        """Map each column name to its NWB description.

        Returns
        -------
        dict of str to str
            Column name to description, suitable for adding NWB trial columns.
        """
        return {name: field.description for name, field in cls.model_fields.items()}
