"""Processing module: derived NWB containers (e.g. the trials table)."""

from ._trial_table import TrialTableBuilder
from .models import TrialConfig

__all__ = ["TrialTableBuilder", "TrialConfig"]
