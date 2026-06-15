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
from aind_behavior_dynamic_foraging.task_logic.trial_models import Trial, TrialOutcome
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
    def _valve_open_times(output_set: t.Optional[pd.DataFrame], port: str) -> np.ndarray:
        """Return Harp valve-open timestamps for ``port`` (``WRITE`` messages only).

        Parameters
        ----------
        output_set : pandas.DataFrame or None
            ``HarpBehavior`` ``OutputSet`` register data.
        port : str
            Supply-port column, e.g. ``"SupplyPort0"`` (left) or
            ``"SupplyPort1"`` (right).

        Returns
        -------
        numpy.ndarray
            Sorted timestamps at which ``port`` was set high via a ``WRITE``.
        """
        if output_set is None or len(output_set) == 0 or port not in output_set.columns:
            return np.empty(0)
        writes = output_set[(output_set["MessageType"] == "WRITE") & output_set[port]]
        return np.sort(writes.index.to_numpy(dtype=float))

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

        The ``Response`` event serializes ``null`` for no choice, ``true`` for
        right, and ``false`` for left. A missing payload is treated as no choice
        (``2``), so this always returns a code and never ``None``.
        """
        if payload is None:
            return 2
        return 1 if bool(payload) else 0

    @staticmethod
    def _parse_outcome(payload: t.Any) -> t.Optional[TrialOutcome]:
        """Parse a ``TrialOutcome`` software-event payload into its domain model.

        Parameters
        ----------
        payload : Any
            The event's ``data`` payload (a ``dict`` or JSON string), or an
            already-parsed ``TrialOutcome``.

        Returns
        -------
        TrialOutcome or None
            The parsed model, or ``None`` if ``payload`` is empty.
        """
        if payload is None:
            return None
        if isinstance(payload, TrialOutcome):
            return payload
        if isinstance(payload, str):
            return TrialOutcome.model_validate_json(payload)
        return TrialOutcome.model_validate(payload)

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
    def _is_baited(trial: t.Optional[Trial], *, is_right: bool) -> bool:
        """Return whether the requested lickport is baited on this trial.

        A port is "baited" when reward is guaranteed there (its reward
        probability is ``1``) *and* the trial was not auto-responded to *that
        same* port.

        ``is_auto_response_right`` encodes the auto-response: ``True`` means the
        trial was auto-responded to the right, ``False`` to the left, and
        ``None`` means there was no auto-response.

        In plain English, for the right port the conditions are:

        * ``p_reward_right == 1`` — reward is certain on the right, and
        * the trial was *not* auto-responded to the right
          (``is_auto_response_right`` is ``None`` or ``False``).

        The left port is the mirror image (``p_reward_left == 1`` and not
        auto-responded to the left, i.e. ``is_auto_response_right`` is ``None``
        or ``True``).

        Parameters
        ----------
        trial : Trial or None
            The per-trial task-logic model, or ``None`` when unavailable.
        is_right : bool
            ``True`` for the right port, ``False`` for the left port.

        Returns
        -------
        bool
            Whether the requested side is baited. A missing ``trial`` is treated
            as not baited (``False``).

        Examples
        --------
        Right port guaranteed reward, no auto-response → baited:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_response_right=None)
        >>> TrialTableBuilder._is_baited(trial, is_right=True)
        True

        Same trial, but auto-responded to the right collects (forfeits) the bait:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_response_right=True)
        >>> TrialTableBuilder._is_baited(trial, is_right=True)
        False

        Left port without guaranteed reward → not baited:

        >>> trial = Trial(p_reward_right=1, p_reward_left=0, is_auto_response_right=None)
        >>> TrialTableBuilder._is_baited(trial, is_right=False)
        False
        """
        if trial is None:
            return False
        auto = trial.is_auto_response_right
        if is_right:
            # Right stays baited unless the animal was auto-responded right.
            return trial.p_reward_right == 1 and auto in (None, False)
        # Left stays baited unless the animal was auto-responded left.
        return trial.p_reward_left == 1 and auto in (None, True)

    @staticmethod
    def _auto_water(trial: t.Optional[Trial], *, is_right: bool) -> t.Optional[int]:
        """Encode autowater for a side from ``is_auto_response_right``.

        ``None`` maps to ``None``; otherwise ``1`` if the auto response was to
        the requested side, else ``0``. ``is_right`` is ``True`` for right.
        """
        if trial is None or trial.is_auto_response_right is None:
            return None
        return int(trial.is_auto_response_right is is_right)

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
        # ``min_block_reward`` is coupled-only; uncoupled generators omit it.
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
        outcome: t.Optional[TrialOutcome],
        start: float,
        stop: float,
        response: t.Any,
        left_valve_times: np.ndarray,
        right_valve_times: np.ndarray,
        go_cue_times: np.ndarray,
        session: t.Dict[str, t.Any],
        lickspout: t.Dict[str, t.Optional[float]],
    ) -> TrialConfig:
        """Assemble a single ``TrialConfig`` from aligned per-trial inputs."""
        trial = outcome.trial if outcome is not None else None
        is_right_choice = outcome.is_right_choice if outcome is not None else None
        is_rewarded = bool(outcome.is_rewarded) if outcome is not None else False

        return TrialConfig(
            start_time=start,
            stop_time=stop,
            delay_start_time=start,
            animal_response=self._animal_response(response),
            rewarded_historyL=self._rewarded_history(is_rewarded, is_right_choice, is_right=False),
            rewarded_historyR=self._rewarded_history(is_rewarded, is_right_choice, is_right=True),
            goCue_start_time=self._closest_time_in_window(go_cue_times, start, stop),
            left_valve_open_time=self._closest_time_in_window(left_valve_times, start, stop),
            right_valve_open_time=self._closest_time_in_window(right_valve_times, start, stop),
            bait_left=self._is_baited(trial, is_right=False),
            bait_right=self._is_baited(trial, is_right=True),
            reward_probabilityL=trial.p_reward_left if trial is not None else None,
            reward_probabilityR=trial.p_reward_right if trial is not None else None,
            response_duration=trial.response_deadline_duration if trial is not None else None,
            reward_consumption_duration=(
                trial.reward_consumption_duration if trial is not None else None
            ),
            ITI_duration=trial.inter_trial_interval_duration if trial is not None else None,
            delay_duration=trial.quiescence_period_duration if trial is not None else None,
            auto_waterL=self._auto_water(trial, is_right=False),
            auto_waterR=self._auto_water(trial, is_right=True),
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

        Hardware streams (valves, go cue) are not per-trial; each trial selects
        the closest hardware event falling inside its ``[start, stop)`` window.

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

        output_set = self._load("Behavior", "HarpBehavior", "OutputSet")
        go_cue = self._load("Behavior", "HarpSoundCard", "PlaySoundOrFrequency")
        manipulator = self._load("Behavior", "SoftwareEvents", "InitialManipulatorPosition")
        task_logic = self._load("Behavior", "InputSchemas", "TaskLogic")

        # Per-trial streams: one payload/timestamp per trial, aligned by index.
        outcome_payloads = self._event_payloads(outcomes)
        start_times = self._event_times(quiescent)
        stop_times = self._event_times(iti)
        response_payloads = self._event_payloads(responses)

        # Guard the positional alignment before we pair streams by index.
        n_trials = len(outcome_payloads)
        warnings = self._check_aligned(
            n_trials,
            {
                "QuiescentPeriod (start_time)": start_times.size,
                "ItiPeriod (stop_time)": stop_times.size,
                "Response": len(response_payloads),
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

        # Hardware streams: not per-trial; each trial picks events in its window.
        left_valve_times = self._valve_open_times(output_set, "SupplyPort0")
        right_valve_times = self._valve_open_times(output_set, "SupplyPort1")
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

            rows.append(
                self._build_row(
                    outcome=outcome,
                    start=start,
                    stop=stop,
                    response=response,
                    left_valve_times=left_valve_times,
                    right_valve_times=right_valve_times,
                    go_cue_times=go_cue_times,
                    session=session,
                    lickspout=lickspout,
                )
            )

        frame = pd.DataFrame(
            [row.model_dump() for row in rows], columns=list(TrialConfig.model_fields)
        )
        return frame.where(pd.notnull(frame), np.nan)
