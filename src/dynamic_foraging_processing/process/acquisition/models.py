"""Pydantic models mirroring the NWB acquisition schema.

These types decouple the processing layer from ``pynwb``: builders produce
plain pydantic models that an NWB writer can later translate into the
corresponding ``pynwb`` objects.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator


class AcquisitionSeries(BaseModel):
    """Single time series destined for an NWB acquisition entry.

    Parameters
    ----------
    name : str
        Name of the series.
    data : numpy.ndarray or pandas.Series
        Sample values.
    timestamps : numpy.ndarray or pandas.Series
        Timestamps aligned with ``data``.
    unit : str
        Unit of the samples in ``data``.
    description : str
        Human-readable description of the series.
    """

    # Allow numpy/pandas fields (no native pydantic schema) and make instances immutable.
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    data: np.ndarray | pd.Series
    timestamps: np.ndarray | pd.Series
    unit: str
    description: str

    @model_validator(mode="after")
    def _check_lengths(self) -> "AcquisitionSeries":
        """Validate that ``data`` and ``timestamps`` have matching lengths."""
        if self.data.shape[0] != self.timestamps.shape[0]:
            raise ValueError(
                f"data and timestamps must have the same length for series "
                f"{self.name!r}: got {self.data.shape[0]} and "
                f"{self.timestamps.shape[0]}."
            )
        return self


class NWBAcquisition(BaseModel):
    """Collection of acquisition series for the NWB acquisition module.

    Parameters
    ----------
    reward_delivery_left : AcquisitionSeries
        Reward delivery events for the left port.
    reward_delivery_right : AcquisitionSeries
        Reward delivery events for the right port.
    """

    # Immutable container; nested AcquisitionSeries already permits arbitrary types.
    model_config = ConfigDict(frozen=True)

    reward_delivery_left: AcquisitionSeries
    reward_delivery_right: AcquisitionSeries
