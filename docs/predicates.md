# Predicate Support

VecAdvisor intentionally parses a safe subset of SQL filter predicates. The
goal is to avoid silently turning complex SQL into bad selectivity estimates.
Unsupported filters fail fast with a parser error instead of receiving a
misleading recommendation.

## Supported Shapes

Use simple, unqualified column names joined by `AND`:

```sql
tenant_id = 42
tenant_id = 42 AND region = 'us'
tenant_id IN (1, 2, 3)
created_at >= '2026-01-01'
created_at BETWEEN '2026-01-01' AND '2026-02-01'
is_public
```

Supported predicate forms:

- equality: `column = literal`
- reversed equality: `literal = column`
- ranges: `<`, `<=`, `>`, `>=`
- `BETWEEN` with two literals
- `IN (...)` with one or more literals
- bare boolean columns, equivalent to `column = true`
- conjunctions with `AND`

Supported literal types:

- integers
- floats
- strings
- booleans
- literal casts such as `'2026-01-01'::date`

## Unsupported Shapes

These are intentionally rejected today:

```sql
tenant_id = 42 OR region = 'us'
NOT is_deleted
deleted_at IS NULL
lower(region) = 'us'
tenant_id + 1 = 42
documents.tenant_id = 42
tenant_id = other_column
tenant_id = $1
body ILIKE '%invoice%'
EXISTS (SELECT 1)
```

Unsupported categories:

- `OR` and `NOT`
- `IS NULL` / `IS NOT NULL`
- `LIKE`, `ILIKE`, regex, and full-text predicates
- functions and expressions on columns
- qualified column names
- column-to-column comparisons
- bind parameters inside the predicate string
- subqueries

## Vector Predicate Boundary

Do not include vector distance ordering in `--query`. VecAdvisor receives the
vector column separately through `--vector` and models this query shape:

```sql
WHERE tenant_id = 42
ORDER BY embedding <-> $query_vector
LIMIT 10
```

So this is correct:

```bash
vecadvisor recommend \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 42" \
  --q-vectors examples/query-vectors.json
```

This is not a supported `--query` predicate:

```sql
tenant_id = 42 ORDER BY embedding <-> '[1,2,3]'::vector
```

## Multi-Column Filters

For filters such as:

```sql
tenant_id = 42 AND region = 'us'
```

VecAdvisor compares its selectivity estimate with PostgreSQL's plan rows. If
the estimates diverge, it can suggest extended statistics such as:

```sql
CREATE STATISTICS IF NOT EXISTS ...
ON tenant_id, region
FROM public.documents;
```

That suggestion is not only PostgreSQL hygiene. It also improves the input
signal for the advisor's own cost model.

## Local Selectivity Still Matters

Global selectivity answers "how many rows match in the whole table?" Local
selectivity answers "how many of the nearest vector neighbors match this
filter?" For filtered ANN, the local value is usually the better recall-risk
signal.

Use `--q-vector`, `--q-vectors`, or `--q-vector-sql` whenever possible:

```bash
vecadvisor explain \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --q-vector examples/query-vector.json
```

Without representative query vectors, VecAdvisor clearly marks the result as a
global-selectivity fallback with lower confidence.
