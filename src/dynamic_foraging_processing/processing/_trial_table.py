"""Builder for the NWB trials table from raw dynamic foraging streams.

The column-by-column source mapping is documented in
``docs/trials_table_mapping.md``
columns.
"""

import logging
import typing as t

import numpy as np
import pandas as pd
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

logger = logging.getLogger(__name__)


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
        duration). ``min``/``max`` come from the truncation parameters when set.

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
        if params.family == DistributionFamily.EXPONENTIAL and params.rate:
            beta = 1.0 / params.rate
        truncation = distribution.truncation_parameters
        minimum = truncation.min if truncation is not None else None
        maximum = truncation.max if truncation is not None else None
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
        is_rewarded: bool, is_right_choice: t.Optional[bool], *, is_right: bool
    ) -> bool:
        """Return if mouse was rewarded based on choice.

        A trial with no reward or an ignored trial (no choice) counts as not
        rewarded on either side (``False``).

        Parameters
        ----------
        is_rewarded : bool
            Whether the trial delivered reward.
        is_right_choice : bool or None
            ``True`` for a right choice, ``False`` for left, ``None`` for ignored.
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            ``True`` only when the trial was rewarded and the choice was on the
            requested side; ``False`` otherwise.
        """
        if is_right_choice is None:
            return False
        return is_rewarded and (is_right_choice is is_right)

    @staticmethod
    def _is_baited(trial: Trial, *, is_right: bool) -> bool:
        """Return whether the requested lickport is baited on this trial.

        A port is "baited" when reward is guaranteed there (its reward
        probability is ``1``) *and* the trial was not auto-responded to *that
        same* port.

        ``is_auto_reward_right`` encodes the auto-response: ``True`` means the
        trial was auto-responded to the right, ``False`` to the left, and
        ``None`` means there was no auto-response.

        In plain English, for the right port the conditions are:

        * ``p_reward_right == 1`` — reward is certain on the right, and
        * the trial was *not* auto-responded to the right
          (``is_auto_reward_right`` is ``None`` or ``False``).

        The left port is the mirror image (``p_reward_left == 1`` and not
        auto-responded to the left, i.e. ``is_auto_reward_right`` is ``None``
        or ``True``).

        Parameters
        ----------
        trial : Trial
            The per-trial task-logic model.
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            Whether the requested side is baited.

        Examples
        --------
        Right port guaranteed reward, no auto-response → baited:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_reward_right=None)
        >>> TrialTableBuilder._is_baited(trial, is_right=True)
        True

        Same trial, but auto-responded to the right collects (forfeits) the bait:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_reward_right=True)
        >>> TrialTableBuilder._is_baited(trial, is_right=True)
        False

        Left port without guaranteed reward → not baited:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_reward_right=None)
        >>> TrialTableBuilder._is_baited(trial, is_right=False)
        False
        """
        auto = trial.is_auto_reward_right
        if is_right:
            # Right stays baited unless the animal was auto-responded right.
            return trial.p_reward_right == 1 and auto in (None, False)
        # Left stays baited unless the animal was auto-responded left.
        return trial.p_reward_left == 1 and auto in (None, True)

    @staticmethod
    def _auto_water(trial: Trial, *, is_right: bool) -> int:
        """Encode autowater for a side from ``is_auto_reward_right``.

        Returns ``1`` if the auto response was to the requested side, else ``0``.
        No auto-response (``is_auto_reward_right`` is ``None``) counts as no
        autowater (``0``). ``is_right`` is ``True`` for right.
        """
        if trial.is_auto_reward_right is None:
            return 0
        return int(trial.is_auto_reward_right is is_right)

    @staticmethod
    def _bias_metadata(trial: Trial) -> BlockBasedTrialMetadata:
        """Return the block-based extra metadata carrying the anti-bias flags.

        The anti-bias flags (``is_bias_water_intervention``,
        ``is_bias_stage_intervention``) live on ``trial.metadata.extra``. That
        field is schema-typed ``Any``, so it deserializes off the stream as a
        plain ``dict`` rather than a model; a ``BlockBasedTrialMetadata``
        instance is also accepted. When metadata or extra is missing (e.g. an
        older session, or a non-block-based generator), the model's all-``False``
        default is returned so the anti-bias columns are simply inert.

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
        metadata = trial.metadata
        extra = metadata.extra if metadata is not None else None
        if isinstance(extra, BlockBasedTrialMetadata):
            return extra
        if isinstance(extra, dict):
            return BlockBasedTrialMetadata.model_validate(extra)
        return BlockBasedTrialMetadata()

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
        # ``min_block_reward`` is warmup-generator-only; main coupled generators omit it.
        if hasattr(generator, "min_block_reward"):
            columns["min_reward_each_block"] = generator.min_block_reward
        return columns

    def _lickspout_columns(
        self, manipulator: t.Optional[pd.DataFrame]
    ) -> t.Dict[str, t.Optional[float]]:
        """Return lickspout position columns from ``InitialManipulatorPosition``.

        The event's ``data`` payload carries the x / y1 / y2 / z positions. These
        are the *initial* manipulator coordinates recorded once at experiment
        start, so the columns are currently constant across trials.

        Known limitation: the manipulator can move within a session (e.g. the
        anti-bias intervention shifts the lickspouts horizontally), which this
        static initial position does not capture.

        TODO: think about how to represent
        this if it becomes relevant. The ideal solution would be to use the harp files
        to track the manipulator position over time
        """
        keys = (
            "lickspout_position_x",
            "lickspout_position_y1",
            "lickspout_position_y2",
            "lickspout_position_z",
        )
        empty = {key: None for key in keys}
        payloads = self._event_payloads(manipulator)
        if not payloads:
            return empty
        payload = payloads[0]
        components = ("x", "y1", "y2", "z")
        if isinstance(payload, dict):
            return {key: payload.get(comp) for key, comp in zip(keys, components)}
        if isinstance(payload, (list, tuple)) and len(payload) == len(keys):
            return {key: payload[idx] for idx, key in enumerate(keys)}
        logger.warning("Unrecognized InitialManipulatorPosition payload; leaving lickspout null.")
        return empty

    # ------------------------------------------------------------------ #
    # Per-trial assembly
    # ------------------------------------------------------------------ #
    def _build_row(
        self,
        *,
        outcome: TrialOutcome,
        start: float,
        stop: float,
        response: t.Any,
        side_bias: t.Optional[float],
        left_valve_open_time: t.Optional[float],
        right_valve_open_time: t.Optional[float],
        go_cue_times: np.ndarray,
        session: t.Dict[str, t.Any],
        lickspout: t.Dict[str, t.Optional[float]],
    ) -> TrialConfig:
        """Assemble a single ``TrialConfig`` from aligned per-trial inputs."""
        trial = outcome.trial
        is_right_choice = outcome.is_right_choice
        is_rewarded = bool(outcome.is_rewarded)
        bias_metadata = self._bias_metadata(trial)

        return TrialConfig(
            start_time=start,
            stop_time=stop,
            delay_start_time=start,
            animal_response=self._animal_response(response),
            rewarded_historyL=self._rewarded_history(is_rewarded, is_right_choice, is_right=False),
            rewarded_historyR=self._rewarded_history(is_rewarded, is_right_choice, is_right=True),
            goCue_start_time=self._closest_time_in_window(go_cue_times, start, stop),
            left_valve_open_time=left_valve_open_time,
            right_valve_open_time=right_valve_open_time,
            bait_left=self._is_baited(trial, is_right=False),
            bait_right=self._is_baited(trial, is_right=True),
            reward_probabilityL=self._block_reward_probability(trial, is_right=False),
            reward_probabilityR=self._block_reward_probability(trial, is_right=True),
            reward_size_left=trial.reward_size.left,
            reward_size_right=trial.reward_size.right,
            side_bias=side_bias,
            response_duration=trial.response_deadline_duration,
            reward_consumption_duration=trial.reward_consumption_duration,
            ITI_duration=trial.inter_trial_interval_duration,
            delay_duration=trial.quiescence_period_duration,
            auto_waterL=self._auto_water(trial, is_right=False),
            auto_waterR=self._auto_water(trial, is_right=True),
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
        the i-th ``TrialOutcome`` is paired with the i-th ``QuiescentPeriod``
        (start), the i-th ``ItiPeriod`` (stop), and the i-th ``Response``. If any
        of those streams has a different length than the ``TrialOutcome`` stream,
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
        ``QuiescentPeriod``, ``ItiPeriod``, ``Response``) are emitted once per
        trial and are aligned here *by position*: row ``i`` draws its outcome,
        start time, stop time, and response from index ``i`` of each stream. The
        ``TrialOutcome`` stream defines the trial count; the lengths of the other
        streams are checked against it before assembly (see ``_check_aligned``)
        so a slipped stream surfaces as a warning (or a ``ValueError`` when
        ``raise_on_error`` is set) rather than silently misaligned rows.

        Hardware streams are handled per their nature: the go cue is an event
        each trial selects within its ``[start, stop)`` window, while the valve
        open duration is a constant session-configured supply-port pulse width
        applied to every trial.

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
        iti = self._load("Behavior", "SoftwareEvents", "ItiPeriod")
        responses = self._load("Behavior", "SoftwareEvents", "Response")
        metrics = self._load("Behavior", "SoftwareEvents", "TrialMetrics")

        pulse_supply_left = self._load("Behavior", "HarpBehavior", "PulseSupplyPort0")
        pulse_supply_right = self._load("Behavior", "HarpBehavior", "PulseSupplyPort1")
        go_cue = self._load("Behavior", "HarpSoundCard", "PlaySoundOrFrequency")
        manipulator = self._load("Behavior", "SoftwareEvents", "InitialManipulatorPosition")
        task_logic = self._load("Behavior", "InputSchemas", "TaskLogic")

        # Per-trial streams: one payload/timestamp per trial, aligned by index.
        outcome_payloads = self._event_payloads(outcomes)
        start_times = self._event_times(quiescent)
        stop_times = self._event_times(iti)
        response_payloads = self._event_payloads(responses)
        metric_payloads = self._event_payloads(metrics)

        # Guard the positional alignment before we pair streams by index.
        n_trials = len(outcome_payloads)

        warnings = self._check_aligned(
            n_trials,
            {
                "QuiescentPeriod (start_time)": start_times.size,
                "ItiPeriod (stop_time)": stop_times.size,
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
        lickspout = self._lickspout_columns(manipulator)

        rows: t.List[TrialConfig] = []
        for i, outcome_payload in enumerate(outcome_payloads):
            outcome = self._parse_outcome(outcome_payload)
            # Pad with NaN/None when a stream is shorter than the trial count;
            # _check_aligned has already warned about any such mismatch.
            start = float(start_times[i]) if i < start_times.size else np.nan
            stop = float(stop_times[i]) if i < stop_times.size else np.nan
            response = response_payloads[i] if i < len(response_payloads) else None
            side_bias = self._side_bias(metric_payloads[i] if i < len(metric_payloads) else None)

            rows.append(
                self._build_row(
                    outcome=outcome,
                    start=start,
                    stop=stop,
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
