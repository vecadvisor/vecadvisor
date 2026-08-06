# VecAdvisor pgrx Extension

This directory is the MVP2 Part B scaffold for an in-database VecAdvisor
surface. It is intentionally metadata-only for now: the extension exposes
version and capability information, but it does not install planner hooks,
change PostgreSQL cost estimates, or run advisory probes.

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
```

## Scope

This scaffold is a safe starting point for future `vector_advise()` and
`explain_vector()` SQL functions. The first implementation deliberately avoids
unsafe planner hooks and mutable global state.
