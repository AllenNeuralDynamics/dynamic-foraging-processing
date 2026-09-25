# Extending it (contribution points)

Back to the [architecture overview](../architecture.md).

The composable layout gives contributors — scientists included — small, self-contained
places to plug in without touching the pipeline plumbing. Each has a defined home, a
clear path into the output, and a test to add.

## Add or adjust a trials-table column
**1. Declare it** — a field (with a `description`) on `TrialConfig`
(`processing/models/trial_config.py`). The `description` is the single source of truth:
`TrialConfig.column_descriptions()` feeds it into the NWB trials column, so there's no
NWB-side change to make.
```python
# processing/models/trial_config.py
class TrialConfig(BaseModel):
    ...
    reaction_time: Optional[float] = Field(
        default=None, description="Time (s) from go cue to the animal's first lick."
    )
```
**2. Populate it** — where each row is assembled, in `TrialTableBuilder._build_row`
(`processing/_trial_table.py`), from the per-trial inputs already in scope (outcome,
start/stop times, response, hardware streams).
```python
# processing/_trial_table.py — inside TrialTableBuilder._build_row(...)
return TrialConfig(
    ...,
    reaction_time=reaction_time,   # computed from the per-trial inputs in scope
)
```
**3. Test it** — under `tests/test_processing/`.
```python
def test_reaction_time_column(dataset):   # dataset: a test Dataset (see existing fixtures)
    trials = TrialTableBuilder(dataset).build()
    assert "reaction_time" in trials.columns
```

## Add a behavior (processed) QC check
**1. Write the check** — a function returning a `QCResult` in `qc/processed/behavior.py`,
alongside `side_bias_result` / `lick_interval_results`.
```python
from dynamic_foraging_processing.qc import QCResult

def finish_rate_result(trials) -> QCResult:
    """Fraction of trials the animal responded to (0/1 = left/right, 2 = no response)."""
    responded = float((trials["animal_response"] != 2).mean())
    return QCResult(
        name="finish rate",
        value=responded,
        passed=responded >= 0.5,
        description="Fraction of trials with a response.",
        tags={"behavior": "finish rate"},   # groups under "behavior" in the QC portal
    )
```
**2. Register it** — add its result to the list `behavior_qc_results` returns
(`qc/processed/results.py`). `ProcessedQC.run` converts that list to metrics, and
`build_quality_control` folds them into the combined `QualityControl` — no pipeline
change.
```python
# qc/processed/results.py — inside behavior_qc_results(...)
results = [
    side_bias_result(side_bias),
    finish_rate_result(trials),                                 # ← new check
    *lick_interval_results(left_lick_times, right_lick_times),
]
```
**3. (Optional) Figure** — add a plotter in `qc/processed/plots.py` and set the
result's `reference` to the filename so the QC portal resolves it.
```python
# qc/processed/plots.py
def plot_finish_rate(trials, results_folder):
    ...
    fig.savefig(Path(results_folder) / "finish_rate.png")
    return "finish_rate.png"
```
**4. Test it** — under `tests/test_qc/`.
```python
import pandas as pd

def test_finish_rate_result():
    trials = pd.DataFrame({"animal_response": [0, 1, 2, 1]})
    result = finish_rate_result(trials)
    assert result.value == 0.75
    assert result.passed
```

## Add a raw / contract QC check
Raw checks come from the upstream `aind-behavior-dynamic-foraging` **data-QC suite**;
`RawQC` runs and surfaces them via `contract_qc_metrics`. New contract checks are added
in that suite, not here — then they appear automatically:
```python
from dynamic_foraging_processing.qc import RawQC

metrics = RawQC().run(loader.dataset)   # includes whatever checks the suite defines
```

## Prototype in a notebook
Everything is callable off the loader — no pipeline or disk writes needed:
```python
from dynamic_foraging_processing.raw_data_loader import RawDataLoader
from dynamic_foraging_processing.processing import TrialTableBuilder
from dynamic_foraging_processing.qc.processed.behavior import finish_rate_result

loader = RawDataLoader(path="/data/<acquisition>")
trials = TrialTableBuilder(loader.dataset).build()
print(finish_rate_result(trials))       # try your new check inline
```

## Guardrails
Contributions must clear the repo's **100% test coverage and 100% docstring** gates,
and inputs must be in the `aind-behavior-dynamic-foraging` data-contract format. For the
full setup, test, and PR workflow — including the exact commands and commit conventions —
see [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
