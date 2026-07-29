# Workload Collector

VecAdvisor's workload collector is the MVP3 bridge from one-query advice to
recurring workload intelligence. It scans `pg_stat_statements`, extracts
filtered vector-query shapes, and emits sanitized templates that can be fed
back into `recommend` and benchmark workflows.

The collector is read-only. It never emits raw SQL text, SQL literal values,
or query vectors.

## Command

```bash
vecadvisor workload \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --limit 50 \
  --min-calls 5
```

`--limit` controls how many `pg_stat_statements` rows are inspected, ordered
by total execution time. `--min-calls` filters out one-off statements before
parsing. `--statement-timeout-ms` applies a local timeout to the catalog read.

## Collection Scope

The collector looks for single-statement `SELECT` queries with a pgvector
distance operator in `ORDER BY`:

```sql
ORDER BY embedding <-> $1
ORDER BY embedding <=> $1
ORDER BY embedding <#> $1
```

It summarizes simple filter predicates joined by `AND`:

- equality
- ranges
- `IN`
- `BETWEEN`
- bare boolean columns

Unsupported predicates are reported as shape notes instead of being silently
converted into inaccurate templates. Examples include `OR`, subqueries,
function predicates, and column-to-column comparisons.

## Privacy Rules

`pg_stat_statements.query` is normalized, but VecAdvisor still treats it as
sensitive operational metadata. The emitted JSON follows these rules:

- Raw query text is never included.
- SQL literal values are replaced with `<value>` placeholders.
- Bind parameter numbers such as `$1` and `$2` are not emitted.
- Query vectors are not collected or emitted.
- The fingerprint is a SHA-256 digest of the sanitized shape, not of raw SQL.
- Runtime counters from `pg_stat_statements` are retained: calls, rows, total
  execution time, and mean execution time.

## Fingerprinting And Sampling

Each shape fingerprint is computed from:

- table name, when it can be summarized as one base table
- vector column
- distance operator
- limit shape
- predicate columns, predicate kinds, operators, and value kinds
- unsupported predicate labels

This intentionally groups statements by planner-relevant shape instead of by
literal values. For example, these two statements become the same shape:

```sql
WHERE tenant_id = 42 ORDER BY embedding <-> $1 LIMIT 10
WHERE tenant_id = 84 ORDER BY embedding <-> $1 LIMIT 10
```

The collector samples from the highest-total-time `pg_stat_statements` rows
after `--min-calls`. That keeps the first MVP focused on expensive recurring
query shapes without materializing a full workload trace.

## Output Contract

Each shape includes a `recommend_template` when VecAdvisor can summarize a
single table and supported filters:

```json
{
  "table": "public.documents",
  "vector": "embedding",
  "query": "tenant_id = <value> AND created_at >= <value>",
  "notes": [
    "replace <value> placeholders with representative literal values",
    "supply representative vectors with --q-vectors or --q-vector-sql"
  ]
}
```

To turn a workload shape into a concrete recommendation, replace the
placeholders with representative literal values and provide representative
query vectors:

```bash
vecadvisor recommend \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 42 AND created_at >= '2026-01-01'" \
  --q-vectors examples/query-vectors.json
```

For benchmark reproduction, use the same table/vector/query shape with
`benchmark-db`, `benchmark-sweep-db`, or a purpose-built fixture.

## PostgreSQL Permissions

The collector requires:

- `pg_stat_statements` installed in the database.
- Permission to read visible rows from `pg_stat_statements`.
- Normal connection permission to the target database.

It does not require table write access and does not create, modify, or drop
database objects.

On managed PostgreSQL systems, visibility into `pg_stat_statements` may be
restricted by role, database, or provider policy. Common setups use a
monitoring role with `pg_read_all_stats`, a superuser-equivalent admin role,
or a role that can see only its own statements. VecAdvisor reports only the
rows PostgreSQL exposes to the connected user.

## Current Limits

- This is a shape collector, not a full workload replay engine.
- Literal values and vectors must be supplied separately before running
  `recommend`.
- Join queries are detected but usually do not receive a single-table
  recommend template in MVP3.
- Unsupported predicate labels are intentionally coarse to avoid leaking
  sensitive SQL structure.
