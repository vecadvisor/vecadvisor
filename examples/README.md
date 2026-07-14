# Examples

This directory contains a tiny pgvector demo table and query-vector inputs for
the README commands.

Predicate examples and parser limits are documented in
[`docs/predicates.md`](../docs/predicates.md).

Start PostgreSQL:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Load the demo table:

```bash
psql "postgresql://postgres:postgres@localhost:5432/vecadvisor" -f examples/demo.sql
```

Or through Docker:

```bash
docker compose -f docker/docker-compose.yml exec -T postgres \
  psql -U postgres -d vecadvisor < examples/demo.sql
```

Run a one-vector diagnostic:

```bash
vecadvisor explain \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --q-vector examples/query-vector.json \
  --probe-rows 16 \
  --format text
```

Sample output: [`explain-output.txt`](explain-output.txt)

Run a durable recommendation:

```bash
vecadvisor recommend \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --q-vectors examples/query-vectors.json \
  --probe-rows 16 \
  --max-query-vectors 3 \
  --local-cache-dir .vecadvisor-cache \
  --format text
```

Sample output: [`recommend-output.txt`](recommend-output.txt)
