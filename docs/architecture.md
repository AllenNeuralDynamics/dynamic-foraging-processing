# Architecture

An overview of the `dynamic-foraging-processing` design: how it maps to the
low-level processing **[reference diagram](https://raw.githubusercontent.com/AllenNeuralDynamics/aind-software-docs/325e3284423c24ec8f4f3d5ca960e7410e3bd162/docs/source/diagrams/dynamic_foraging/low_level/dynamic-foraging-low-level-processing.svg)**, how the pieces compose, how to extend it,
and the rationale behind the design choices.

The alignment sections compare the reference diagram (the target design) with what
the repo currently implements, call out where they differ, and document the rationale
— doubling as a checklist for **updating the reference diagram to match the source
code** (each row/section flags what the diagram should be revised to reflect).

## Contents

This document is split across several files:

- **[Alignment](architecture/alignment.md)** — component-by-component comparison
  against the reference diagram, and where/why the repo differs (the rationale behind
  each divergence lives here too).
- **[Composability & usage](architecture/composability.md)** — how the stages
  compose, with worked examples at three tiers.
- **[Extending it](architecture/extending.md)** — contribution points: adding
  trials columns, QC checks, and prototyping in notebooks.
- **[Changelog](changelog.md)** — release history.

## Scope
Delivered across 40+ tracked issues over roughly a four-week period. The work spans
several distinct areas:

- **Data engineering** — the contract loader, stream navigation, trials-table
  assembly, NWB packaging (Zarr, reserved-name collisions, `id` handling, JSON-safe
  serialization).
- **Domain / behavioral knowledge** — licks, rewards, side bias, and auto-reward
  semantics, including the meaning of each QC check.
- **API / architecture design** — the class-vs-function question, composable stages,
  and the `run_qc` reshaping; these were iterative design decisions rather than
  one-pass coding.
- **QC + schema** — `aind-data-schema`, `QualityControl` / `Processing`, and the
  raw-vs-processed split.
- **Release engineering** — semantic-release workflow, rulesets, bypass lists,
  `SERVICE_TOKEN`, branch protection, and the `dev` → `main` flow — a substantial
  area on its own.
- **Deployment / orchestration** — built the NWB, QC, and metadata capsules and the pipeline
  that chains them in Code Ocean, including logging, job-type configuration, and
  end-to-end debugging many issues that came up.
- **Testing rigor** — 100% coverage and docstring gates on every change.
- **Documentation + stakeholder communication** — READMEs, capsule docs, and this
  divergence-rationale doc for review.
- **Organizational / coordination layer** — reconciling a reference
  diagram, scientist requirements, engineering requirements, backward-compatibility constraints, and reviewer
  feedback.
