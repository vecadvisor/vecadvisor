# Rust/pgrx Extension Design

VecAdvisor MVP2 Part B adds an in-database extension surface without changing
PostgreSQL planner behavior. The first scaffold is intentionally narrow:
metadata SQL functions, a pure post-filter risk estimator, a pgrx crate layout,
and a documented path toward safe catalog/SPI collection.

## Current Surface

The extension crate lives in `extension/vecadvisor` and builds with pgrx
`0.19.2`. PostgreSQL 17 is the default feature because the project benchmark
and CI database use `pgvector/pgvector:pg17`; pgrx features also exist for
PostgreSQL 13 through 19.

Initial SQL functions:

- `vecadvisor_extension_version() RETURNS text`
- `vecadvisor_capabilities() RETURNS jsonb`
- `vecadvisor_postfilter_risk(limit_count int, ef_search int,
  global_selectivity double precision,
  local_selectivity double precision,
  recall_at_ef double precision) RETURNS jsonb`

The capability document reports that SQL metadata functions are enabled while
SPI/catalog probes, planner hooks, and Python CLI parity are not yet enabled.
The post-filter risk function is a pure calculation: it does not read catalogs,
run probes, install hooks, mutate GUCs, or change planner behavior.

Example:

```sql
SELECT vecadvisor_postfilter_risk(
  limit_count        => 10,
  ef_search          => 40,
  global_selectivity => 0.05,
  local_selectivity  => 0.00,
  recall_at_ef       => 0.90
);
```

Use `NULL` for `local_selectivity` or `recall_at_ef` when that input has not
been measured yet.

The returned JSON records expected post-filter survivors, whether the scan is
expected to return `LIMIT` rows, an estimated survivor-bound recall, the
minimum `hnsw.ef_search` needed for expected survivors to reach `LIMIT`, and a
small recommendation list. When `local_selectivity` is omitted, the function
falls back to global selectivity but marks the risk as at least medium because
global selectivity can be wrong for correlated vector/filter workloads.

## SPI And Catalog Access Plan

Future advisor functions should collect metadata through read-only SPI/catalog
queries before they attempt any planner integration:

- resolve table, vector column, metric operator, and pgvector index metadata
  from `pg_class`, `pg_namespace`, `pg_attribute`, `pg_index`, `pg_am`, and
  operator catalogs;
- read selectivity inputs from `pg_statistic` or documented safe wrappers when
  possible;
- run local-selectivity probes through SPI with parameterized values, quoted
  identifiers, `read_only` transactions, and a local `statement_timeout`;
- set `hnsw.ef_search` or `ivfflat.probes` only with `SET LOCAL` inside the
  probe transaction;
- return degraded-confidence diagnostics instead of raising when the ANN index
  is unavailable.

The extension should not cache raw vectors or row IDs. Cache keys may include
query/filter shape, query-vector fingerprints, table stats fingerprints, and
index fingerprints, matching the Python CLI invariants.

## GUC Plan

The first GUCs should be read-only advisor controls, not planner switches:

- `vecadvisor.statement_timeout_ms`
- `vecadvisor.local_probe_rows`
- `vecadvisor.min_confidence`
- `vecadvisor.enable_planner_hooks`, default `off`

Planner hooks must remain disabled until SQL function parity, regression tests,
and opt-in safety controls are in place.

## Python CLI Parity

The pgrx extension should mirror the Python CLI before it becomes a separate
optimizer path:

- identical strategy names and risk labels;
- matching selectivity/correlation diagnostics on shared fixtures;
- matching recommendation choice for calibrated profiles;
- JSON output that can be diffed against `vecadvisor recommend --format json`;
- explicit degraded-confidence behavior when probes fail.

The Python CLI remains the reference implementation until these parity tests
exist. The current `vecadvisor_postfilter_risk()` function is deliberately
smaller than `vecadvisor recommend`; it exists to make the extension surface
testable while preserving the no-planner-regression invariant.

## Local Build

```bash
cargo install cargo-pgrx --version 0.19.2 --locked
cargo pgrx init --pg17 /path/to/pg_config
cd extension/vecadvisor
cargo test --no-default-features
cargo pgrx schema --features pg17
cargo pgrx install --features pg17
```

The `--no-default-features` unit test checks the metadata shape without loading
pgrx, so it can run on machines that have Rust but not a pgrx/PostgreSQL
development environment. Full extension schema generation still requires
`cargo pgrx init`.
