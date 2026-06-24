from __future__ import annotations


SAFE_CLASSES = (
    "'batch_scheduler', "
    "'best_effort_queueable', "
    "'low_priority_flexible', "
    "'medium_flexibility'"
)
RETRY_CLASSES = "'batch_scheduler', 'best_effort_queueable', 'low_priority_flexible'"

VARIANTS = [
    {
        "name": "conservative",
        "order": 0,
        "s1_runtime": 7_200,
        "s1_instances": 3,
        "s1_cpu": 0.20,
        "s1_mem": 0.20,
        "s2_runtime": 7_200,
        "s2_instances": 3,
        "s2_cpu": 0.20,
        "s2_mem": 0.20,
        "s2_incoming": 1,
        "s3_runtime": 14_400,
        "s3_instances": 10,
        "s3_cpu": 0.35,
        "s3_mem": 0.35,
        "r1_runtime": 1_800,
        "r1_instances": 3,
        "r1_cpu": 0.20,
        "r1_mem": 0.20,
    },
    {
        "name": "baseline",
        "order": 1,
        "s1_runtime": 14_400,
        "s1_instances": 5,
        "s1_cpu": 0.25,
        "s1_mem": 0.25,
        "s2_runtime": 14_400,
        "s2_instances": 5,
        "s2_cpu": 0.25,
        "s2_mem": 0.25,
        "s2_incoming": 2,
        "s3_runtime": 28_800,
        "s3_instances": 20,
        "s3_cpu": 0.50,
        "s3_mem": 0.50,
        "r1_runtime": 3_600,
        "r1_instances": 5,
        "r1_cpu": 0.25,
        "r1_mem": 0.25,
    },
    {
        "name": "relaxed",
        "order": 2,
        "s1_runtime": 21_600,
        "s1_instances": 10,
        "s1_cpu": 0.40,
        "s1_mem": 0.40,
        "s2_runtime": 14_400,
        "s2_instances": 10,
        "s2_cpu": 0.40,
        "s2_mem": 0.40,
        "s2_incoming": 4,
        "s3_runtime": 28_800,
        "s3_instances": 40,
        "s3_cpu": 0.75,
        "s3_mem": 0.75,
        "r1_runtime": 3_600,
        "r1_instances": 10,
        "r1_cpu": 0.40,
        "r1_mem": 0.40,
    },
]


def variant_by_name(names: list[str] | None = None) -> list[dict[str, object]]:
    if not names:
        return VARIANTS
    selected = []
    known = {str(v["name"]): v for v in VARIANTS}
    for name in names:
        if name not in known:
            raise ValueError(f"unknown threshold variant: {name}")
        selected.append(known[name])
    return selected


def risk_tier_case(v: dict[str, object], table_alias: str = "e") -> str:
    p = f"{table_alias}."
    return f"""
    CASE
      WHEN {p}safeharvest_seed_candidate = 1
       AND COALESCE({p}instances, 0) = 1
       AND COALESCE({p}bad_terminal_instances, 0) = 0
       AND COALESCE({p}total_cpu_request, 0.0) > 0
       AND COALESCE({p}total_cpu_request, 0.0) <= 0.10
       AND COALESCE({p}total_memory_request, 0.0) <= 0.10
        THEN 'S0_strict_seed'
      WHEN {p}collection_terminal_outcome = 'FINISH'
       AND {p}collection_runtime_sec <= {v["s1_runtime"]}
       AND {p}dependency_graph_role = 'independent_leaf'
       AND COALESCE({p}instances, 0) BETWEEN 1 AND {v["s1_instances"]}
       AND COALESCE({p}bad_terminal_instances, 0) = 0
       AND COALESCE({p}total_cpu_request, 0.0) > 0
       AND COALESCE({p}total_cpu_request, 0.0) <= {v["s1_cpu"]}
       AND COALESCE({p}total_memory_request, 0.0) <= {v["s1_mem"]}
       AND {p}candidate_class IN ({SAFE_CLASSES})
        THEN 'S1_clean_small_independent'
      WHEN {p}collection_terminal_outcome = 'FINISH'
       AND {p}collection_runtime_sec <= {v["s2_runtime"]}
       AND {p}incoming_dependency_edges <= {v["s2_incoming"]}
       AND {p}outgoing_dependent_edges = 0
       AND COALESCE({p}instances, 0) BETWEEN 1 AND {v["s2_instances"]}
       AND COALESCE({p}bad_terminal_instances, 0) = 0
       AND COALESCE({p}total_cpu_request, 0.0) > 0
       AND COALESCE({p}total_cpu_request, 0.0) <= {v["s2_cpu"]}
       AND COALESCE({p}total_memory_request, 0.0) <= {v["s2_mem"]}
       AND {p}candidate_class IN ({SAFE_CLASSES})
        THEN 'S2_clean_small_leaf_dependency'
      WHEN {p}collection_terminal_outcome = 'FINISH'
       AND {p}collection_runtime_sec <= {v["s3_runtime"]}
       AND {p}dependency_graph_role = 'independent_leaf'
       AND COALESCE({p}instances, 0) BETWEEN 1 AND {v["s3_instances"]}
       AND COALESCE({p}bad_terminal_instances, 0) = 0
       AND COALESCE({p}total_cpu_request, 0.0) > 0
       AND COALESCE({p}total_cpu_request, 0.0) <= {v["s3_cpu"]}
       AND COALESCE({p}total_memory_request, 0.0) <= {v["s3_mem"]}
       AND {p}candidate_class IN ({SAFE_CLASSES})
        THEN 'S3_clean_medium_independent'
      WHEN {p}collection_terminal_outcome IN ('EVICT', 'KILL', 'LOST')
       AND {p}collection_runtime_sec <= {v["r1_runtime"]}
       AND {p}dependency_graph_role = 'independent_leaf'
       AND COALESCE({p}instances, 0) BETWEEN 1 AND {v["r1_instances"]}
       AND COALESCE({p}total_cpu_request, 0.0) > 0
       AND COALESCE({p}total_cpu_request, 0.0) <= {v["r1_cpu"]}
       AND COALESCE({p}total_memory_request, 0.0) <= {v["r1_mem"]}
       AND {p}candidate_class IN ({RETRY_CLASSES})
        THEN 'R1_retry_tolerant_signal'
      ELSE 'X_excluded'
    END
    """


def risk_tier_order_case(column: str = "risk_tier") -> str:
    return f"""
    CASE {column}
      WHEN 'S0_strict_seed' THEN 0
      WHEN 'S1_clean_small_independent' THEN 1
      WHEN 'S2_clean_small_leaf_dependency' THEN 2
      WHEN 'S3_clean_medium_independent' THEN 3
      WHEN 'R1_retry_tolerant_signal' THEN 90
      ELSE 99
    END
    """


def risk_tier_family_case(column: str = "risk_tier") -> str:
    return f"""
    CASE
      WHEN {column} IN (
        'S0_strict_seed',
        'S1_clean_small_independent',
        'S2_clean_small_leaf_dependency',
        'S3_clean_medium_independent'
      )
        THEN 'safe_expansion'
      WHEN {column} = 'R1_retry_tolerant_signal'
        THEN 'retry_signal_only'
      ELSE 'excluded'
    END
    """


def variant_union_sql(input_path: str, names: list[str] | None = None) -> str:
    parts = []
    for variant in variant_by_name(names):
        name = variant["name"]
        order = variant["order"]
        case_sql = risk_tier_case(variant)
        parts.append(
            f"""
            SELECT
              '{name}' AS threshold_variant,
              {order} AS threshold_variant_order,
              e.*,
              {case_sql} AS sensitivity_risk_tier
            FROM read_parquet('{input_path}') AS e
            """
        )
    return "\nUNION ALL\n".join(parts)
