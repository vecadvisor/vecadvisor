# Benchmark Artifacts

This directory contains deterministic synthetic evidence for the current
VecAdvisor alpha.

- `synthetic-calibration.json`: fitted constants from a small synthetic run.
- `synthetic-sweep.json`: selectivity/correlation sweep with measured and
  predicted strategy outcomes.
- `synthetic-proof.json`: proof summary derived from the sweep.

These files are intentionally small enough to regenerate during local
development. They validate cost-model behavior and safety checks, not
production pgvector latency.
