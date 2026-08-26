"""Builder for the NWB trials table from raw dynamic foraging streams.

The column-by-column source mapping is documented in
``docs/trials_table_mapping.md``
columns.
"""

import logging
import typing as t

import numpy as np
import pandas as pd
from aind_behavior_dynamic_foraging.rig import AindDynamicForagingRig
from aind_behavior_dynamic_foraging.task_logic import AindDynamicForagingTaskLogic
from aind_behavior_dynamic_foraging.task_logic.trial_generators import TrialGeneratorSpec
from aind_behavior_dynamic_foraging.task_logic.trial_generators.block_based_trial_generator import (
    BlockBasedTrialMetadata,
)
from aind_behavior_dynamic_foraging.task_logic.trial_models import (
    Trial,
    TrialMetrics,
    TrialOutcome,
)
from aind_behavior_services.task.distributions import Distribution, DistributionFamily
from contraqctor.contract import Dataset

from dynamic_foraging_processing.processing.models import TrialConfig
from dynamic_foraging_processing.utils.trial_metadata import get_bias_metadata

logger = logging.getLogger(__name__)

# The four manipulator axes, ordered so that index ``i`` is the axis driven
# by ``Motor{i}`` in the ``AccumulatedSteps`` stream: ``Motor{i}`` maps to
# ``Axis(i + 1)`` (Axis.X=1, Y1=2, Y2=3, Z=4), whose names match the
# ``lickspout_position_*`` suffixes and the ``full_step_to_mm`` attributes.
_MANIPULATOR_AXES = ("x", "y1", "y2", "z")


class TrialTableBuilder:
    """Builds the NWB ``trials`` table from a dynamic foraging ``Dataset``.

    The builder reads the per-trial software events, the task-logic
    configuration, and the hardware (Harp) streams, then assembles one
    ``TrialConfig`` per trial.
    """

    def __init__(self, dataset: Dataset, *, raise_on_error: bool = False):
        """Initialize the builder.

        Parameters
        ----------
        dataset : Dataset
            A dynamic foraging ``contraqctor`` ``Dataset`` (e.g.
            ``RawDataLoader.dataset``).
        raise_on_error : bool, optional
            If ``True``, raise when a required stream fails to load. If
            ``False`` (default), log a warning and continue with missing data.
        """
        self.dataset = dataset
        self.raise_on_error = raise_on_error

    # ------------------------------------------------------------------ #
    # Stream access
    # ------------------------------------------------------------------ #
    def _load(self, *path: str) -> t.Optional[t.Any]:
        """Load the ``data`` of the stream at ``path``.

        Navigation and loading are handled separately: ``load`` records a read
        failure on the node rather than raising, so success is checked via the
        contract's ``has_data`` property instead of catching exceptions from the
        load itself. Only the navigation step (a missing node) is guarded with a
        ``try``, since ``at`` raises ``KeyError`` for an absent child.

        Parameters
        ----------
        *path : str
            Sequence of node names navigating the dataset's tree-like
            contract, e.g. ``("Behavior", "SoftwareEvents", "TrialOutcome")``.

        Returns
        -------
        Any or None
            The stream's loaded ``data``, or ``None`` if the node is missing or
            failed to load and ``raise_on_error`` is ``False``.
        """
        node = self.dataset
        for name in path:
            try:
                node = node.at(name)
            except KeyError:
                return self._missing_stream(path, f"node {name!r} not found")
        node.load()
        if node.has_data:
            return node.data
        return self._missing_stream(path, "stream failed to load")

    def _missing_stream(self, path: t.Sequence[str], reason: str) -> None:
        """Warn (or raise under ``raise_on_error``) that a stream is unavailable."""
        msg = f"Failed to load stream {'.'.join(path)}: {reason}"
        logger.warning(msg)
        if self.raise_on_error:
            raise ValueError(msg)
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _event_payloads(df: t.Optional[pd.DataFrame]) -> t.List[t.Any]:
        """Return the ordered ``data`` payloads of a software-event stream.

        Parameters
        ----------
        df : pandas.DataFrame or None
            A ``SoftwareEvents`` stream's data (indexed by timestamp with a
            ``data`` column).

        Returns
        -------
        list
            Payloads sorted by timestamp, or an empty list if ``df`` is empty.
        """
        if df is None or len(df) == 0:
            return []
        return list(df.sort_index()["data"])

    @staticmethod
    def _event_times(df: t.Optional[pd.DataFrame]) -> np.ndarray:
        """Return the sorted timestamps (index) of a software-event stream."""
        if df is None or len(df) == 0:
            return np.empty(0)
        return df.sort_index().index.to_numpy(dtype=float)

    @classmethod
    def _session_end_time(cls, end_session: t.Optional[pd.DataFrame]) -> float:
        """Return the session's end timestamp from the ``EndSession`` stream.

        The stream carries a single event marking the end of the session; its
        timestamp closes the last trial's ITI, which has no following
        ``QuiescentPeriod`` event to end it.

        Parameters
        ----------
        end_session : pandas.DataFrame or None
            The ``EndSession`` software-event stream's data.

        Returns
        -------
        float
            The end-of-session timestamp, or ``NaN`` when the stream is absent
            or empty. The last event is used if more than one is present.
        """
        times = cls._event_times(end_session)
        return float(times[-1]) if times.size else np.nan

    @staticmethod
    def _time_at(times: np.ndarray, index: int) -> float:
        """Return ``times[index]``, or ``NaN`` when the stream is that much shorter.

        The per-trial streams are paired by position, so a stream with fewer
        events than there are trials (already reported by ``_check_aligned``) is
        padded rather than raising.

        Parameters
        ----------
        times : numpy.ndarray
            Sorted per-trial event timestamps.
        index : int
            The trial index to read.

        Returns
        -------
        float
            The timestamp, or ``numpy.nan`` when ``index`` is out of range.
        """
        if index < 0 or index >= times.size:
            return np.nan
        return float(times[index])

    @staticmethod
    def _closest_time_in_window(times: np.ndarray, start: float, stop: float) -> t.Optional[float]:
        """Return the timestamp in ``times`` closest to ``start`` within ``[start, stop)``.

        Hardware streams are not aligned per-trial, so for each trial we select
        the hardware event that falls inside the trial window, taking the one
        nearest the trial start.

        Parameters
        ----------
        times : numpy.ndarray
            Candidate hardware timestamps (need not be sorted).
        start, stop : float
            Trial window bounds. ``start`` is inclusive, ``stop`` exclusive.

        Returns
        -------
        float or None
            The selected timestamp, or ``None`` if no event falls in the window.
        """
        if times.size == 0:
            return None
        mask = (times >= start) & (times < stop)
        if not mask.any():
            return None
        in_window = times[mask]
        return float(in_window[np.argmin(in_window - start)])

    @staticmethod
    def _pulse_duration(pulse_register: t.Optional[pd.DataFrame], column: str) -> t.Optional[float]:
        """Return the configured valve-open pulse width (seconds) for a supply port.

        The Harp ``PulseSupplyPort{0,1}`` register holds the valve-open pulse
        width in milliseconds. It is a per-session configuration value (a reward
        opens the valve for this fixed duration), so the same value applies to
        every trial. Returns the most recent value converted to seconds.

        Parameters
        ----------
        pulse_register : pandas.DataFrame or None
            ``HarpBehavior`` ``PulseSupplyPort{0,1}`` register data.
        column : str
            The register's value column, e.g. ``"PulseSupplyPort0"`` (left) or
            ``"PulseSupplyPort1"`` (right).

        Returns
        -------
        float or None
            The pulse width in seconds, or ``None`` when the register or column
            is absent or has no value.
        """
        if (
            pulse_register is None
            or len(pulse_register) == 0
            or column not in pulse_register.columns
        ):
            return None
        values = pulse_register.sort_index()[column].dropna()
        if values.empty:
            return None
        return float(values.iloc[-1]) / 1000.0

    @staticmethod
    def _write_times(register: t.Optional[pd.DataFrame]) -> np.ndarray:
        """Return sorted timestamps of ``WRITE`` messages in a Harp register."""
        if register is None or len(register) == 0:
            return np.empty(0)
        if "MessageType" in register.columns:
            register = register[register["MessageType"] == "WRITE"]
        return np.sort(register.index.to_numpy(dtype=float))

    @staticmethod
    def _distribution_stats(
        distribution: Distribution,
    ) -> t.Tuple[t.Optional[float], t.Optional[float], t.Optional[float]]:
        """Return ``(beta, min, max)`` for a task-logic distribution.

        ``beta`` is the scale of an exponential distribution (``1 / rate``); it
        is ``None`` for non-exponential families (e.g. the scalar quiescent
        duration). ``min``/``max`` come from the truncation parameters when set,
        except for a uniform distribution, whose bounds are its own ``min`` and
        ``max`` distribution parameters.

        The optional ``scaling_parameters`` apply ``value * scale + offset`` to
        each sample, so they are folded into all three values. Two details
        decide where each one lands:

        * The truncation bounds are applied *after* scaling (per the schema), so
          they are already in output units and are **not** re-transformed. The
          offset does raise the support floor of an exponential, though — its
          samples start at ``0``, hence at ``offset`` once shifted — so the
          reported minimum is whichever of the two constraints binds. This is
          what makes a configured offset visible when the truncation minimum is
          left at its ``0`` default.
        * A uniform's bounds live in the distribution parameters, which *are*
          pre-scaling, so they take the full transform.

        Parameters
        ----------
        distribution : Distribution
            An ``aind_behavior_services`` distribution model.

        Returns
        -------
        tuple of (float or None, float or None, float or None)
            The ``beta``, ``min``, and ``max`` summary values.
        """
        beta: t.Optional[float] = None
        params = distribution.distribution_parameters
        truncation = distribution.truncation_parameters
        scaling = distribution.scaling_parameters
        scale = scaling.scale if scaling is not None else 1.0
        offset = scaling.offset if scaling is not None else 0.0
        minimum = truncation.min if truncation is not None else None
        maximum = truncation.max if truncation is not None else None
        if params.family == DistributionFamily.EXPONENTIAL and params.rate:
            beta = scale / params.rate
            # An exponential's support starts at the offset once shifted.
            minimum = offset if minimum is None else max(minimum, offset)
        elif params.family == DistributionFamily.UNIFORM:
            # A uniform distribution carries its bounds in the distribution
            # parameters rather than the truncation parameters.
            minimum = params.min * scale + offset
            maximum = params.max * scale + offset
        return beta, minimum, maximum

    @staticmethod
    def _animal_response(payload: t.Any) -> int:
        """Encode a ``Response`` payload as ``0`` (left), ``1`` (right), ``2`` (none).

        The ``Response`` event serializes a ``{"Item1": <timestamp>,
        "Item2": <choice>}`` pair, where ``Item2`` is ``True`` for a right choice,
        ``False`` for left, and ``None`` for no choice. A missing payload (or a
        missing/``None`` ``Item2``) is treated as no choice (``2``), so this always
        returns a code and never ``None``.
        """
        choice = payload.get("Item2") if isinstance(payload, dict) else payload
        if choice is None:
            return 2
        return 1 if bool(choice) else 0

    @staticmethod
    def _parse_outcome(payload: t.Any) -> TrialOutcome:
        """Parse a ``TrialOutcome`` software-event payload into its domain model.

        Parameters
        ----------
        payload : Any
            The event's ``data`` payload (a ``dict`` or JSON string), or an
            already-parsed ``TrialOutcome``.

        Returns
        -------
        TrialOutcome
            The parsed model.

        Raises
        ------
        ValueError
            If ``payload`` is ``None``.
        """
        if payload is None:
            raise ValueError("TrialOutcome payload is required but received None.")
        if isinstance(payload, TrialOutcome):
            return payload
        if isinstance(payload, str):
            return TrialOutcome.model_validate_json(payload)
        return TrialOutcome.model_validate(payload)

    @staticmethod
    def _side_bias(payload: t.Any) -> t.Optional[float]:
        """Extract the per-trial side bias from a ``TrialMetrics`` event payload.

        The ``TrialMetrics`` event serializes a model with a ``bias`` field
        (negative for left bias, positive for right). Accepts the parsed model, a
        ``dict``, or a JSON string; returns ``None`` for a missing payload or a
        ``bias`` that was not recorded.

        Parameters
        ----------
        payload : Any
            The event's ``data`` payload (a ``TrialMetrics`` model, ``dict``, or
            JSON string), or ``None``.

        Returns
        -------
        float or None
            The side bias, or ``None`` when unavailable.
        """
        if payload is None:
            return None
        if isinstance(payload, TrialMetrics):
            metrics = payload
        elif isinstance(payload, str):
            metrics = TrialMetrics.model_validate_json(payload)
        else:
            metrics = TrialMetrics.model_validate(payload)
        return metrics.bias

    # ------------------------------------------------------------------ #
    # Per-trial column helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rewarded_history(
        trial: Trial,
        is_rewarded: bool,
        is_right_choice: t.Optional[bool],
        *,
        is_right: bool,
    ) -> bool:
        """Return whether the mouse *earned* reward on the requested side.

        ``rewarded_history`` records earned reward only, i.e. water the animal
        worked for. ``TrialOutcome.is_rewarded`` is ``True`` for any water
        delivered on the trial, autowater included, so an autowater trial
        (``trial.is_auto_reward_right is not None``) is ``False`` on *both*
        sides here — its water is reported by ``auto_waterL``/``auto_waterR``
        instead. This matches the ``earned``/``automatic`` split in
        :func:`~dynamic_foraging_processing.utils.rewards.get_reward_deliveries`.

        A trial with no reward or an ignored trial (no choice) likewise counts
        as not rewarded on either side (``False``).

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model; ``is_auto_reward_right`` being set
            (to either side) marks the trial as an autowater trial.
        is_rewarded : bool
            Whether the trial delivered reward (earned *or* auto).
        is_right_choice : bool or None
            ``True`` for a right choice, ``False`` for left, ``None`` for ignored.
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            ``True`` only when the trial delivered reward with no autowater and
            the choice was on the requested side; ``False`` otherwise.
        """
        if is_right_choice is None or trial.is_auto_reward_right is not None:
            return False
        return is_rewarded and (is_right_choice is is_right)

    @staticmethod
    def _is_baited(bias_metadata: BlockBasedTrialMetadata, *, is_right: bool) -> bool:
        """Return whether the requested lickport is baited on this trial.

        The bait state is reported directly by the acquisition software as
        ``is_left_baited`` / ``is_right_baited`` on the trial's extra metadata,
        so it is read rather than re-derived from the reward probability and the
        auto-response channel. A trial whose metadata does not carry the flags
        falls back to the model's ``False`` default (see ``_bias_metadata``).

        Parameters
        ----------
        bias_metadata : BlockBasedTrialMetadata
            The trial's extra metadata (see ``_bias_metadata``).
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            Whether the requested side is baited.
        """
        if is_right:
            return bias_metadata.is_right_baited
        return bias_metadata.is_left_baited

    @staticmethod
    def _auto_water(trial: Trial, bias_metadata: BlockBasedTrialMetadata, *, is_right: bool) -> int:
        """Return whether scheduled autowater was delivered to the requested side.

        ``is_auto_reward_right`` is only the delivery *channel* -- it says free
        water was triggered and to which side (``True`` right, ``False`` left,
        ``None`` none), not what kind. Scheduled autowater and the anti-bias water
        intervention share that channel, so the mechanism comes from
        ``is_autowater`` and the side from the channel -- the mirror of
        :meth:`_anti_bias_water`. Free water driven by the anti-bias algorithm is
        ``0`` here and is reported by
        ``anti_bias_left_water``/``anti_bias_right_water`` instead.

        Like the anti-bias columns, this records what the *task* did and so is not
        gated on ``is_rewarded``: the water fires at the go cue regardless of how
        the animal's own choice later resolves. The reward-delivery series is
        reward-keyed and drops free water on trials that did not pay out, so this
        column can exceed that series' ``auto`` count.

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.
        bias_metadata : BlockBasedTrialMetadata
            The trial's extra metadata (see ``_bias_metadata``).
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        int
            ``1`` when scheduled autowater targeted the requested side, else ``0``.
        """
        if not bias_metadata.is_autowater:
            return 0
        return int(trial.is_auto_reward_right is is_right)

    @staticmethod
    def _bias_metadata(trial: Trial) -> BlockBasedTrialMetadata:
        """Return the block-based extra metadata carrying the free-water flags.

        Thin wrapper over :func:`get_bias_metadata`, shared with the reward
        annotation so both classify autowater and anti-bias water identically.

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.

        Returns
        -------
        BlockBasedTrialMetadata
            The parsed extra metadata, or an all-``False`` default when absent
            or unrecognized.
        """
        return get_bias_metadata(trial)

    @staticmethod
    def _anti_bias_water(
        trial: Trial, bias_metadata: BlockBasedTrialMetadata, *, is_right: bool
    ) -> bool:
        """Return whether the anti-bias algorithm watered the requested side.

        The anti-bias algorithm delivers its water intervention through the same
        auto-response channel as ordinary autowater (``is_auto_reward_right``:
        ``True`` right, ``False`` left), so the two are distinguished only by the
        ``is_bias_water_intervention`` flag. This is ``True`` only when the trial
        was a bias-water intervention *and* the auto-response was to the
        requested side.

        This records what the *algorithm* did, so it is not gated on
        ``is_rewarded``: the intervention fires at the go cue regardless of how
        the animal's own choice later resolves. It is therefore not a subset of
        ``auto_waterL``/``auto_waterR``, which count only rewarded autowater --
        an intervention on a trial that did not pay out appears here and not
        there.

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.
        bias_metadata : BlockBasedTrialMetadata
            The trial's extra metadata (see ``_bias_metadata``).
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            Whether an anti-bias water intervention targeted the requested side.
        """
        if not bias_metadata.is_bias_water_intervention:
            return False
        return trial.is_auto_reward_right is is_right

    @staticmethod
    def _anti_bias_lickspout_movement(
        trial: Trial, bias_metadata: BlockBasedTrialMetadata
    ) -> float:
        """Return the anti-bias lickspout displacement (mm) for this trial.

        The anti-bias algorithm's other intervention shifts the lickspouts
        horizontally; the per-trial displacement is ``trial.lickspout_offset_delta``
        (positive is rightward). Reported only when the trial is flagged as a
        bias-stage intervention, so a stray offset from another source is not
        attributed to the anti-bias algorithm; ``0.0`` otherwise.

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.
        bias_metadata : BlockBasedTrialMetadata
            The trial's extra metadata (see ``_bias_metadata``).

        Returns
        -------
        float
            The signed displacement (mm), or ``0.0`` when there was no
            lickspout intervention.
        """
        if not bias_metadata.is_bias_stage_intervention:
            return 0.0
        return trial.lickspout_offset_delta

    @staticmethod
    def _block_reward_probability(trial: Trial, *, is_right: bool) -> t.Optional[float]:
        """Return the block reward probability for a side from the trial metadata.

        The top-level ``trial.p_reward_left/right`` is the *per-trial* probability;
        the block probability that the ``reward_probability`` columns represent
        lives on ``trial.metadata.p_reward_left/right``.

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        float or None
            The block reward probability, or ``None`` when metadata is unavailable.
        """
        if trial.metadata is None:
            return None
        return trial.metadata.p_reward_right if is_right else trial.metadata.p_reward_left

    # ------------------------------------------------------------------ #
    # Session-level (constant across trials) columns
    # ------------------------------------------------------------------ #
    @staticmethod
    def _summary_generator(
        generator: TrialGeneratorSpec,
    ) -> t.Optional[TrialGeneratorSpec]:
        """Resolve the generator whose parameters populate the session columns.

        ``task_parameters.trial_generator`` is either a single generator spec or
        a ``TrialGeneratorCompositeSpec`` wrapping several sub-generators in
        ``.generators`` (e.g. a warm-up stage concatenated before the main
        generator). Both coupled and uncoupled generators are supported: the
        block-length, ITI, and delay summaries are common to all block-based
        generators, while coupled-only fields (reward-probability sum, minimum
        reward per block) are read defensively in ``_session_columns``.

        For a composite, a ``CoupledTrialGenerator`` is preferred (it carries the
        richest set of session parameters); an ``UncoupledTrialGenerator`` is
        used as a fallback when no coupled generator is present. Other stages
        (e.g. warm-up generators) are skipped. A non-composite spec is returned
        unchanged.

        Parameters
        ----------
        generator : TrialGeneratorSpec
            The ``trial_generator`` from the task parameters.

        Returns
        -------
        TrialGeneratorSpec or None
            The summarising generator, or ``None`` if a composite contains
            neither a coupled nor an uncoupled sub-generator.
        """
        sub_generators = getattr(generator, "generators", None)
        if sub_generators is None:
            return generator
        uncoupled_fallback: t.Optional[TrialGeneratorSpec] = None
        for sub in sub_generators:
            sub_type = getattr(sub, "type", None)
            if sub_type == "CoupledTrialGenerator":
                return sub
            if sub_type == "UncoupledTrialGenerator" and uncoupled_fallback is None:
                uncoupled_fallback = sub
        if uncoupled_fallback is not None:
            return uncoupled_fallback
        logger.warning(
            "No coupled or uncoupled generator among composite trial generators; "
            "session columns will be null."
        )
        return None

    def _session_columns(self, task_logic: AindDynamicForagingTaskLogic) -> t.Dict[str, t.Any]:
        """Return the per-session trial columns derived from the task logic.

        These (block/ITI/delay distribution summaries and reward structure) are
        constant across trials.
        """
        columns: t.Dict[str, t.Any] = {}
        if task_logic is None:
            return columns

        generator = self._summary_generator(task_logic.task_parameters.trial_generator)
        if generator is None:
            return columns

        block_beta, block_min, block_max = self._distribution_stats(generator.block_length)
        # Account for the floor applied upstream.
        if block_max is not None:
            block_max -= 1
        iti_beta, iti_min, iti_max = self._distribution_stats(
            generator.inter_trial_interval_duration
        )
        delay_beta, delay_min, delay_max = self._distribution_stats(generator.quiescent_duration)

        # Coupled generators normalize both sides to a fixed reward-probability
        # sum; uncoupled generators draw each side independently and expose no
        # such sum, so the column is left null for them.
        reward_params = getattr(generator, "reward_probability_parameters", None)
        base_reward_sum = reward_params.base_reward_sum if reward_params is not None else None

        columns.update(
            block_beta=block_beta,
            block_min=block_min,
            block_max=block_max,
            ITI_beta=iti_beta,
            ITI_min=iti_min,
            ITI_max=iti_max,
            delay_beta=delay_beta,
            delay_min=delay_min,
            delay_max=delay_max,
            base_reward_probability_sum=base_reward_sum,
        )
        # ``min_block_reward`` is warmup-generator-only; a generator that omits it
        # enforces no per-block minimum, which is a floor of 0 rather than unknown.
        columns["min_reward_each_block"] = getattr(generator, "min_block_reward", 0)
        return columns

    def _manipulator_mm_per_step(self, rig: AindDynamicForagingRig) -> t.Dict[str, float]:
        """Return millimetres travelled per accumulated step, keyed by axis.

        ``AccumulatedSteps`` counts *microsteps*. The physical distance per
        microstep is the full-step-to-mm calibration divided by the microstep
        resolution (e.g. ``MICROSTEP8`` → 8 microsteps per full step), both read
        from the rig's manipulator calibration.

        Parameters
        ----------
        rig : AindDynamicForagingRig
            The ``InputSchemas.Rig`` stream data. The rig is a required input
            schema, so ``build`` guarantees it is present before this is called.

        Returns
        -------
        dict of str to float
            ``{axis: mm_per_step}`` for ``x`` / ``y1`` / ``y2`` / ``z``.

        Raises
        ------
        ValueError
            If the manipulator calibration is missing one of the four axes.
        """
        calibration = rig.manipulator.calibration
        resolution_by_axis = {
            config.axis.name.lower(): config.microstep_resolution
            for config in calibration.axis_configuration
        }
        mm_per_step: t.Dict[str, float] = {}
        for axis in _MANIPULATOR_AXES:
            resolution = resolution_by_axis.get(axis)
            if resolution is None:
                raise ValueError(
                    f"Manipulator calibration is missing axis {axis!r}; "
                    "cannot derive lickspout position."
                )
            # Resolution enum names encode the divisor (MICROSTEP8 -> 8); the
            # enum *value* (0..3) does not, so parse from the name.
            microsteps = int(resolution.name.replace("MICROSTEP", ""))
            mm_per_step[axis] = getattr(calibration.full_step_to_mm, axis) / microsteps
        return mm_per_step

    def _manipulator_positions(
        self,
        accumulated_steps: pd.DataFrame,
        rig: AindDynamicForagingRig,
    ) -> pd.DataFrame:
        """Return a time-indexed frame of lickspout position (mm) over the session.

        Built from the ``HarpManipulator`` ``AccumulatedSteps`` stream: each
        ``EVENT`` row's per-motor microstep count is converted to millimetres via
        the rig calibration. ``AccumulatedSteps`` is absolute (zero-referenced at
        the homing position), so the count maps directly to position with no
        offset. The frame is then re-referenced to the session start (the first
        sample subtracted from every row), so the stored values are displacement
        *relative to session start* — the units the QC plot expects.

        Parameters
        ----------
        accumulated_steps : pandas.DataFrame
            The ``AccumulatedSteps`` stream data (``Motor0``..``Motor3`` columns).
            Required; guaranteed present by ``build``.
        rig : AindDynamicForagingRig
            The ``InputSchemas.Rig`` stream data (required; guaranteed present by
            ``build``).

        Returns
        -------
        pandas.DataFrame
            Columns ``lickspout_position_x`` / ``y1`` / ``y2`` / ``z`` indexed by
            time (sorted ascending), relative to the session-start position.

        Raises
        ------
        ValueError
            If the stream has no ``EVENT`` rows, is missing a motor column, or the
            rig calibration is missing an axis.
        """
        mm_per_step = self._manipulator_mm_per_step(rig)
        events = accumulated_steps
        if "MessageType" in events.columns:
            events = events[events["MessageType"] == "EVENT"]
        if len(events) == 0:
            raise ValueError(
                "AccumulatedSteps stream has no EVENT rows; cannot derive lickspout position."
            )
        events = events.sort_index()
        columns: t.Dict[str, np.ndarray] = {}
        for index, axis in enumerate(_MANIPULATOR_AXES):
            motor = f"Motor{index}"
            if motor not in events.columns:
                raise ValueError(
                    f"AccumulatedSteps stream is missing column {motor}; "
                    "cannot derive lickspout position."
                )
            columns[f"lickspout_position_{axis}"] = (
                events[motor].to_numpy(dtype=float) * mm_per_step[axis]
            )
        positions = pd.DataFrame(columns, index=events.index)
        # Re-reference to session start so the stored values are displacement
        # from the first sample (previously done in the plot as ``values[0]``).
        return positions - positions.iloc[0]

    def _sample_lickspout(
        self, positions: pd.DataFrame, start: float, stop: float
    ) -> t.Dict[str, t.Optional[float]]:
        """Return the lickspout position sampled within a trial's window.

        The manipulator position is a continuously-sampled hardware value, so —
        like the go cue — each trial takes the sample within its ``[start, stop)``
        window nearest the start (see :meth:`_closest_time_in_window`).

        Parameters
        ----------
        positions : pandas.DataFrame
            The time-indexed position frame from :meth:`_manipulator_positions`.
        start, stop : float
            The trial window bounds (seconds); ``start`` inclusive, ``stop``
            exclusive. May be ``NaN`` when unaligned.

        Returns
        -------
        dict of str to (float or None)
            The four ``lickspout_position_*`` values, all ``None`` for a trial
            whose window contains no manipulator sample.
        """
        keys = tuple(f"lickspout_position_{axis}" for axis in _MANIPULATOR_AXES)
        times = positions.index.to_numpy(dtype=float)
        sample_time = self._closest_time_in_window(times, start, stop)
        if sample_time is None:
            return {key: None for key in keys}
        row = positions.loc[sample_time]
        return {key: (None if pd.isna(row[key]) else float(row[key])) for key in keys}

    # ------------------------------------------------------------------ #
    # Per-trial assembly
    # ------------------------------------------------------------------ #
    @classmethod
    def _trial_periods(
        cls,
        index: int,
        *,
        quiescent_times: np.ndarray,
        response_period_times: np.ndarray,
        consumption_times: np.ndarray,
        iti_times: np.ndarray,
        session_end_time: float = np.nan,
    ) -> t.Dict[str, float]:
        """Return the start and stop time of every task period for one trial.

        Each software event marks the *start* of its period and the periods run
        back-to-back — quiescent, response, reward consumption, ITI, then the
        next trial's quiescent — so every period's stop time is the following
        period's start time. The last trial's ITI has no following quiescent
        event, so it is closed by ``session_end_time`` instead.

        Parameters
        ----------
        index : int
            The trial index; every per-trial stream is paired by position.
        quiescent_times, response_period_times, consumption_times, iti_times : numpy.ndarray
            Per-trial timestamps of the ``QuiescentPeriod``, ``ResponsePeriod``,
            ``RewardConsumptionPeriod``, and ``ItiPeriod`` streams.
        session_end_time : float, optional
            Fallback ``ITI_stop_time`` for a trial with no following
            ``QuiescentPeriod`` event — the ``EndSession`` timestamp, passed only
            for the last trial. Defaults to ``NaN``, leaving the stop time unset.

        Returns
        -------
        dict of str to float
            The eight ``*_start_time`` / ``*_stop_time`` period columns; an entry
            is ``NaN`` where the corresponding event is missing.
        """
        response_start = cls._time_at(response_period_times, index)
        consumption_start = cls._time_at(consumption_times, index)
        iti_start = cls._time_at(iti_times, index)
        iti_stop = cls._time_at(quiescent_times, index + 1)
        if np.isnan(iti_stop):
            iti_stop = session_end_time
        return {
            "quiescent_start_time": cls._time_at(quiescent_times, index),
            "quiescent_stop_time": response_start,
            "response_start_time": response_start,
            "response_stop_time": consumption_start,
            "reward_consumption_start_time": consumption_start,
            "reward_consumption_stop_time": iti_start,
            "ITI_start_time": iti_start,
            "ITI_stop_time": iti_stop,
        }

    def _build_row(
        self,
        *,
        outcome: TrialOutcome,
        periods: t.Dict[str, float],
        response: t.Any,
        side_bias: t.Optional[float],
        left_valve_open_time: t.Optional[float],
        right_valve_open_time: t.Optional[float],
        go_cue_times: np.ndarray,
        session: t.Dict[str, t.Any],
        lickspout: t.Dict[str, t.Optional[float]],
    ) -> TrialConfig:
        """Assemble a single ``TrialConfig`` from aligned per-trial inputs.

        ``periods`` holds the trial's period bounds (see :meth:`_trial_periods`);
        the quiescent-period start through the ITI start is also the window used
        to pick this trial's go cue out of the unaligned hardware stream.
        """
        trial = outcome.trial
        is_right_choice = outcome.is_right_choice
        is_rewarded = bool(outcome.is_rewarded)
        bias_metadata = self._bias_metadata(trial)
        start = periods["quiescent_start_time"]
        stop = periods["ITI_start_time"]

        return TrialConfig(
            **periods,
            delay_start_time=start,
            animal_response=self._animal_response(response),
            rewarded_historyL=self._rewarded_history(
                trial, is_rewarded, is_right_choice, is_right=False
            ),
            rewarded_historyR=self._rewarded_history(
                trial, is_rewarded, is_right_choice, is_right=True
            ),
            goCue_start_time=self._closest_time_in_window(go_cue_times, start, stop),
            left_valve_open_time=left_valve_open_time,
            right_valve_open_time=right_valve_open_time,
            bait_left=self._is_baited(bias_metadata, is_right=False),
            bait_right=self._is_baited(bias_metadata, is_right=True),
            reward_probabilityL=self._block_reward_probability(trial, is_right=False),
            reward_probabilityR=self._block_reward_probability(trial, is_right=True),
            reward_size_left=trial.reward_size.left,
            reward_size_right=trial.reward_size.right,
            side_bias=side_bias,
            response_duration=trial.response_deadline_duration,
            reward_consumption_duration=trial.reward_consumption_duration,
            ITI_duration=trial.inter_trial_interval_duration,
            delay_duration=trial.quiescence_period_duration,
            auto_waterL=self._auto_water(trial, bias_metadata, is_right=False),
            auto_waterR=self._auto_water(trial, bias_metadata, is_right=True),
            anti_bias_left_water=self._anti_bias_water(trial, bias_metadata, is_right=False),
            anti_bias_right_water=self._anti_bias_water(trial, bias_metadata, is_right=True),
            anti_bias_lickspout_movement=self._anti_bias_lickspout_movement(trial, bias_metadata),
            **session,
            **lickspout,
        )

    @staticmethod
    def _check_aligned(n_trials: int, counts: t.Mapping[str, int]) -> t.List[str]:
        """Return human-readable warnings for per-trial streams that mismatch ``n_trials``.

        The builder aligns the per-trial software-event streams *positionally*:
        the i-th ``TrialOutcome`` is paired with the i-th event of each period
        stream (``QuiescentPeriod``, ``ResponsePeriod``,
        ``RewardConsumptionPeriod``, ``ItiPeriod``) and the i-th ``Response``. If
        any of those streams has a different length than the ``TrialOutcome`` stream,
        the pairing slips and every subsequent row is silently misaligned
        (shorter streams are padded with ``NaN``/``None``; longer streams are
        ignored past ``n_trials``).

        Parameters
        ----------
        n_trials : int
            The number of trials, i.e. the ``TrialOutcome`` count, which defines
            the expected length of every other per-trial stream.
        counts : mapping of str to int
            Stream label to its observed length.

        Returns
        -------
        list of str
            One message per mismatched stream; empty when everything aligns.
        """
        return [
            f"{name} has {count} events but there are {n_trials} trials"
            for name, count in counts.items()
            if count != n_trials
        ]

    # ------------------------------------------------------------------ #
    # Build (main entry point)
    # ------------------------------------------------------------------ #
    def build(self) -> pd.DataFrame:
        """Build the trials table.

        The per-trial software-event streams (``TrialOutcome``,
        ``QuiescentPeriod``, ``ResponsePeriod``, ``RewardConsumptionPeriod``,
        ``ItiPeriod``, ``Response``) are emitted once per trial and are aligned
        here *by position*: row ``i`` draws its outcome, period start times, and
        response from index ``i`` of each stream. The ``TrialOutcome`` stream
        defines the trial count; the lengths of the other streams are checked
        against it before assembly (see ``_check_aligned``) so a slipped stream
        surfaces as a warning (or a ``ValueError`` when ``raise_on_error`` is set)
        rather than silently misaligned rows.

        Each period event marks the start of its period, so the periods' stop
        times come from the next event in sequence (see ``_trial_periods``). The
        last trial's ITI has no following event, so it is closed by the
        ``EndSession`` timestamp.

        Hardware streams are handled per their nature: the go cue is an event
        each trial selects within its ``[quiescent_start_time, ITI_start_time)``
        window, while the valve open duration is a constant session-configured
        supply-port pulse width applied to every trial.

        Returns
        -------
        pandas.DataFrame
            One row per trial with the in-scope trial columns. ``None`` values
            are converted to ``numpy.nan``.

        Raises
        ------
        ValueError
            If ``raise_on_error`` is ``True`` and a per-trial stream length
            disagrees with the ``TrialOutcome`` trial count.
        """
        outcomes = self._load("Behavior", "SoftwareEvents", "TrialOutcome")
        quiescent = self._load("Behavior", "SoftwareEvents", "QuiescentPeriod")
        response_period = self._load("Behavior", "SoftwareEvents", "ResponsePeriod")
        consumption = self._load("Behavior", "SoftwareEvents", "RewardConsumptionPeriod")
        iti = self._load("Behavior", "SoftwareEvents", "ItiPeriod")
        responses = self._load("Behavior", "SoftwareEvents", "Response")
        metrics = self._load("Behavior", "SoftwareEvents", "TrialMetrics")
        end_session = self._load("Behavior", "SoftwareEvents", "EndSession")

        pulse_supply_left = self._load("Behavior", "HarpBehavior", "PulseSupplyPort0")
        pulse_supply_right = self._load("Behavior", "HarpBehavior", "PulseSupplyPort1")
        go_cue = self._load("Behavior", "HarpSoundCard", "PlaySoundOrFrequency")
        accumulated_steps = self._load("Behavior", "HarpManipulator", "AccumulatedSteps")
        rig = self._load("Behavior", "InputSchemas", "Rig")
        task_logic = self._load("Behavior", "InputSchemas", "TaskLogic")

        # Per-trial streams: one payload/timestamp per trial, aligned by index.
        outcome_payloads = self._event_payloads(outcomes)
        quiescent_times = self._event_times(quiescent)
        response_period_times = self._event_times(response_period)
        consumption_times = self._event_times(consumption)
        iti_times = self._event_times(iti)
        response_payloads = self._event_payloads(responses)
        metric_payloads = self._event_payloads(metrics)

        # Closes the last trial's ITI, which has no following QuiescentPeriod.
        session_end_time = self._session_end_time(end_session)

        # Guard the positional alignment before we pair streams by index.
        n_trials = len(outcome_payloads)

        warnings = self._check_aligned(
            n_trials,
            {
                "QuiescentPeriod (quiescent_start_time)": quiescent_times.size,
                "ResponsePeriod (response_start_time)": response_period_times.size,
                "RewardConsumptionPeriod (reward_consumption_start_time)": consumption_times.size,
                "ItiPeriod (ITI_start_time)": iti_times.size,
                "Response": len(response_payloads),
                "TrialMetrics (side_bias)": len(metric_payloads),
            },
        )
        if warnings:
            msg = (
                "Per-trial streams are misaligned with TrialOutcome; rows are "
                "paired by index so the table may be incorrect: " + "; ".join(warnings)
            )
            logger.warning(msg)
            if self.raise_on_error:
                raise ValueError(msg)

        # Hardware streams. The valve open duration is the configured supply-port
        # pulse width (constant per session); the go cue is a per-trial event each
        # trial picks within its window.
        left_valve_open_time = self._pulse_duration(pulse_supply_left, "PulseSupplyPort0")
        right_valve_open_time = self._pulse_duration(pulse_supply_right, "PulseSupplyPort1")
        go_cue_times = self._write_times(go_cue)

        session = self._session_columns(task_logic)
        if n_trials:
            if rig is None:
                raise ValueError("Rig stream is required when there are trials.")
            if accumulated_steps is None:
                raise ValueError("AccumulatedSteps stream is required when there are trials.")
        # With no trials the frame goes unused (the loop does not run).
        lickspout_positions = (
            self._manipulator_positions(accumulated_steps, rig) if n_trials else None
        )

        rows: t.List[TrialConfig] = []
        for i, outcome_payload in enumerate(outcome_payloads):
            outcome = self._parse_outcome(outcome_payload)
            # Pad with NaN/None when a stream is shorter than the trial count;
            # _check_aligned has already warned about any such mismatch.
            periods = self._trial_periods(
                i,
                quiescent_times=quiescent_times,
                response_period_times=response_period_times,
                consumption_times=consumption_times,
                iti_times=iti_times,
                # Only the last trial's ITI is closed by the session end; an
                # earlier gap means a short stream, which _check_aligned reports.
                session_end_time=session_end_time if i == n_trials - 1 else np.nan,
            )
            response = response_payloads[i] if i < len(response_payloads) else None
            side_bias = self._side_bias(metric_payloads[i] if i < len(metric_payloads) else None)
            lickspout = self._sample_lickspout(
                lickspout_positions,
                periods["quiescent_start_time"],
                periods["ITI_start_time"],
            )

            rows.append(
                self._build_row(
                    outcome=outcome,
                    periods=periods,
                    response=response,
                    side_bias=side_bias,
                    left_valve_open_time=left_valve_open_time,
                    right_valve_open_time=right_valve_open_time,
                    go_cue_times=go_cue_times,
                    session=session,
                    lickspout=lickspout,
                )
            )

        frame = pd.DataFrame(
            [row.model_dump() for row in rows], columns=list(TrialConfig.model_fields)
        )
        return frame.where(pd.notnull(frame), np.nan)
