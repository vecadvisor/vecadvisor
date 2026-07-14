# Contributing

Thanks for helping make VecAdvisor better.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

Run checks before opening a pull request:

```bash
python -m ruff check .
python -m mypy src/vecadvisor
python -m pytest
python -m build
```

PostgreSQL integration tests use:

```text
postgresql://postgres:postgres@localhost:5432/vecadvisor
```

They skip automatically when a reachable pgvector-enabled database is not
available.

## Pull Requests

- Keep changes focused and explain the workload or failure mode they address.
- Add or update tests for behavior changes.
- Avoid benchmark claims without a reproducible command and artifact.
- Do not copy source from AGPL or source-available vector extensions.

## Design Direction

VecAdvisor is a predictive advisor, not a brute-force benchmark harness. New
strategy work should expose statistics, local-selectivity signals, cost
formula inputs, and calibration hooks rather than only timing every option.
