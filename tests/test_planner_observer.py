from __future__ import annotations

from vecadvisor.models import IndexMeta, PlanNode, PlanSummary, QuerySpec, Strategy, TableStats
from vecadvisor.planner_observer import build_vector_order_sql, classify_observed_plan


def test_build_vector_order_sql_quotes_identifiers_and_uses_params() -> None:
    query = _query()

    sql = build_vector_order_sql(query=query, filter_sql="tenant_id = 1")

    assert sql == (
        'SELECT * FROM "public"."docs" '
        "WHERE tenant_id = 1 "
        'ORDER BY "embedding" <-> %s::vector '
        "LIMIT %s"
    )


def test_classify_observed_plan_detects_nonpartial_vector_index_as_postfilter() -> None:
    plan = PlanSummary(
        root=PlanNode(
            node_type="Limit",
            children=(
                PlanNode(
                    node_type="Index Scan",
                    relation_name="docs",
                    index_name="docs_embedding_hnsw_idx",
                    order_by="(embedding <-> '[1,1,1]'::vector)",
                    filter_text="(tenant_id = 1)",
                ),
            ),
        )
    )

    observed = classify_observed_plan(table=_table(), query=_query(), plan=plan)

    assert observed.strategy is Strategy.POSTFILTER
    assert observed.index_name == "docs_embedding_hnsw_idx"
    assert observed.scan_node_type == "Index Scan"


def test_classify_observed_plan_detects_partial_vector_index() -> None:
    plan = PlanSummary(
        root=PlanNode(
            node_type="Limit",
            children=(
                PlanNode(
                    node_type="Index Scan",
                    relation_name="docs",
                    index_name="docs_embedding_tenant_1_hnsw_idx",
                    order_by="(embedding <-> '[1,1,1]'::vector)",
                ),
            ),
        )
    )

    observed = classify_observed_plan(table=_table(), query=_query(), plan=plan)

    assert observed.strategy is Strategy.PARTIAL
    assert observed.index_name == "docs_embedding_tenant_1_hnsw_idx"


def test_classify_observed_plan_without_vector_index_scan_as_exact() -> None:
    plan = PlanSummary(
        root=PlanNode(
            node_type="Limit",
            children=(
                PlanNode(
                    node_type="Sort",
                    children=(
                        PlanNode(
                            node_type="Bitmap Heap Scan",
                            relation_name="docs",
                            index_name=None,
                            filter_text="(tenant_id = 1)",
                        ),
                    ),
                ),
            ),
        )
    )

    observed = classify_observed_plan(table=_table(), query=_query(), plan=plan)

    assert observed.strategy is Strategy.EXACT
    assert observed.scan_node_type == "Bitmap Heap Scan"
    assert observed.index_name is None


def _query() -> QuerySpec:
    return QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(),
        limit=10,
    )


def _table() -> TableStats:
    return TableStats(
        relname="public.docs",
        n_rows=1_000,
        n_pages=100,
        columns=(),
        vector_dim=3,
        indexes=(
            IndexMeta(
                name="docs_embedding_hnsw_idx",
                method="hnsw",
                columns=("embedding",),
                opclass="vector_l2_ops",
            ),
            IndexMeta(
                name="docs_embedding_tenant_1_hnsw_idx",
                method="hnsw",
                columns=("embedding",),
                opclass="vector_l2_ops",
                is_partial=True,
                predicate="(tenant_id = 1)",
            ),
        ),
    )
