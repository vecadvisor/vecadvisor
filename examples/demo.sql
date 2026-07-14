CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    tenant_id int NOT NULL,
    region text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    body text NOT NULL,
    embedding vector(3) NOT NULL
);

INSERT INTO documents (tenant_id, region, created_at, body, embedding)
SELECT
    g % 8 AS tenant_id,
    CASE WHEN g % 2 = 0 THEN 'us' ELSE 'eu' END AS region,
    now() - (g || ' hours')::interval AS created_at,
    'demo document ' || g AS body,
    ARRAY[
        (g % 5)::float4,
        (g % 7)::float4,
        (g % 11)::float4
    ]::vector AS embedding
FROM generate_series(1, 256) AS g;

CREATE INDEX documents_embedding_hnsw_idx
ON documents
USING hnsw (embedding vector_l2_ops)
WITH (m = 8, ef_construction = 32);

CREATE INDEX documents_tenant_idx ON documents (tenant_id);

CREATE STATISTICS documents_tenant_region_stats
(dependencies, mcv)
ON tenant_id, region
FROM documents;

ANALYZE documents;
