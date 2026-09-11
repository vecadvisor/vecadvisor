# VecAdvisor pgrx Extension

This directory is the MVP2 Part B scaffold for an in-database VecAdvisor
surface. The current extension exposes version/capability metadata plus a
pure post-filter risk estimator. It does not install planner hooks, change
PostgreSQL cost estimates, or run advisory probes.

## Build Locally

Install `cargo-pgrx` matching the crate version:

```bash
cargo install cargo-pgrx --version 0.19.2 --locked
```

Initialize pgrx once. Use the PostgreSQL 17 `pg_config` when available because
the VecAdvisor development database uses pgvector on PostgreSQL 17:

```bash
cargo pgrx init --pg17 /path/to/pg_config
```

Build, generate SQL, or install the extension:

```bash
cd extension/vecadvisor
cargo test --no-default-features
cargo pgrx schema --features pg17
cargo pgrx install --features pg17
```

Then in PostgreSQL:

```sql
CREATE EXTENSION vecadvisor;
SELECT vecadvisor_extension_version();
SELECT vecadvisor_capabilities();
SELECT vecadvisor_postfilter_risk(
  limit_count        => 10,
  ef_search          => 40,
  global_selectivity => 0.05,
  local_selectivity  => 0.00,
  recall_at_ef       => 0.90
);
```

## Scope

`vecadvisor_postfilter_risk()` is deterministic and metadata-free: it only uses
the selectivity inputs supplied by the caller. Pass `NULL` for
`local_selectivity` when no local probe is available; the response falls back
to global selectivity and marks the result as lower confidence.

This scaffold is a safe starting point for future `vector_advise()` and
`explain_vector()` SQL functions. The implementation deliberately avoids
unsafe planner hooks and mutable global state.
