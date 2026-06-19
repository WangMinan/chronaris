"""Data contracts for Stage I thesis-facing tables and figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def build_thesis_table_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Return every CSV table used by the thesis materials package."""

    return {
        "evidence_layer_overview.csv": build_evidence_layer_rows(sources),
        "runtime_payload_schema.csv": build_runtime_payload_schema_rows(sources),
        "rigid_body_rotation_audit.csv": build_rigid_body_rotation_rows(sources),
        "weak_label_sweep_ablation.csv": build_weak_label_rows(sources),
        "chronaris_opt_component_ablation.csv": build_private_component_rows(sources),
        "public_transfer_boundary.csv": build_public_transfer_rows(sources),
        "semantic_event_fusion_overview.csv": build_semantic_event_rows(sources),
        "llm_comparison_a0_a4.csv": build_llm_comparison_rows(sources),
    }


def build_evidence_layer_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    live = _payload(sources, "live_sweep") or _payload(sources, "proxy_sweep")
    proxy = _payload(sources, "proxy_sweep")
    private = _payload(sources, "private_component")
    public_calibration = _payload(sources, "public_calibration")
    runtime = _payload(sources, "runtime")
    runtime_service = _payload(sources, "runtime_service")
    support_semantic = _semantic_payload(sources)
    rigid = _payload(sources, "rigid_body")
    rotation = _payload(sources, "rotation_audit")
    llm_pre = _payload(sources, "llm_preprocessing")
    llm_cmp = _payload(sources, "llm_comparison")

    private_rows = _as_list(private.get("rows"))
    public_rows = _as_list(public_calibration.get("rows"))
    semantic_hints = _as_mapping(llm_cmp.get("semantic_hints"))
    human_review = _as_mapping(llm_cmp.get("human_review_packet"))
    runtime_status = _runtime_status_text(runtime_service)
    rigid_body_family = _as_mapping(_as_mapping(rigid.get("families")).get("rigid_body"))
    rigid_diag = _as_mapping(rigid_body_family.get("rigid_body_mapping_diagnostics"))
    enabled_residuals = _as_list(rigid_diag.get("enabled_residuals"))

    rows = [
        _overview_row(
            1,
            "thesis_weak_label",
            "论文 weak-label 主线",
            "Thesis weak-label mainline",
            _run_id(live),
            "sample_count",
            _number(live.get("sample_count")),
            "task_entry_count",
            _number(live.get("task_entry_count")),
            f"best_test_total={_fmt_float(_as_mapping(live.get('best_run')).get('test_total'))}",
            "真实 Stage H 双流窗口；weak-label evidence，不是人工真值。",
            [_source_path(sources, "live_sweep"), _source_path(sources, "proxy_sweep")],
        ),
        _overview_row(
            2,
            "private_proxy",
            "私有代理组件诊断",
            "Private proxy component diagnostics",
            _run_id(private),
            "component_rows",
            len(private_rows),
            "task_count",
            len(_as_mapping(private.get("tasks"))),
            f"variant_count={len(_as_list(private.get('variant_order')))}",
            "T1/T2/T3 用于组件机制对比，不写成论文人工真值任务。",
            [_source_path(sources, "private_component")],
        ),
        _overview_row(
            3,
            "public_adapter_calibration",
            "公开数据适配与校准",
            "Public adapter and calibration",
            _run_id(public_calibration),
            "calibration_rows",
            len(public_rows),
            "baseline_categories",
            len(_as_mapping(public_calibration.get("best_by_category"))),
            "UAB/NASA adapter/calibration evidence",
            "公开数据支撑外部基线和评价接口，不替代私有双流主线。",
            [_source_path(sources, "public_calibration"), _source_path(sources, "public_transfer")],
        ),
        _overview_row(
            4,
            "runtime_schema",
            "运行时 schema 契约",
            "Runtime schema contract",
            _run_id(runtime_service) or _run_id(runtime),
            "replay_window_count",
            _number(runtime.get("sample_count") or runtime_service.get("input_sample_count")),
            "vehicle_feature_gap",
            _number(runtime_service.get("missing_vehicle_feature_count")),
            runtime_status,
            "native replay payload 保持 aligned；canonical service payload 达到 exact。",
            [
                _source_path(sources, "runtime"),
                _source_path(sources, "runtime_service"),
                _source_path(sources, "runtime_schema_contract"),
            ],
        ),
        _overview_row(
            5,
            "semantic_support",
            "语义事件融合支撑",
            "Semantic event fusion support",
            _run_id(_payload(sources, "semantic_event")) or _run_id(_payload(sources, "support")),
            "view_count",
            _number(support_semantic.get("view_count")),
            "query_count",
            _number(support_semantic.get("query_count")),
            f"top_view={support_semantic.get('top_view_id')}",
            "展示 event token 与 query-to-event attribution 的真实 support 统计。",
            [_source_path(sources, "support"), _source_path(sources, "semantic_event")],
        ),
        _overview_row(
            6,
            "rigid_body",
            "刚体约束与 rotation 诊断",
            "Rigid-body and rotation diagnostics",
            _run_id(rotation) or _run_id(rigid),
            "enabled_residual_count",
            len(enabled_residuals) or 2,
            "rotation_status",
            rotation.get("rotation_status", "unknown"),
            "translation+vertical enabled; rotation rate missing",
            "translation + vertical 已启用；rotation 因 pitch/roll/yaw rate 缺失保持诊断状态。",
            [_source_path(sources, "rigid_body"), _source_path(sources, "rotation_audit")],
        ),
        _overview_row(
            7,
            "llm_preprocessing",
            "LLM 预处理与对比",
            "LLM preprocessing and comparison",
            _run_id(llm_cmp) or _run_id(llm_pre),
            "request_count",
            _number(llm_pre.get("request_count")),
            "semantic_query_coverage",
            f"{semantic_hints.get('baseline_query_count')}->{semantic_hints.get('combined_query_count')}",
            f"human_review_packet={human_review.get('item_count')}",
            "LLM 只作为 preprocessing context / whitelisted hints / explanation / review packet。",
            [_source_path(sources, "llm_preprocessing"), _source_path(sources, "llm_comparison")],
        ),
    ]
    return rows


def build_runtime_payload_schema_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    service = _payload(sources, "runtime_service")
    contract = _payload(sources, "runtime_schema_contract")
    native = _as_mapping(contract.get("native_input"))
    canonical = _as_mapping(contract.get("canonical_payload"))
    expected = _as_mapping(contract.get("expected_schema"))
    expected_vehicle = _feature_count(expected, "vehicle")
    native_vehicle = _feature_count(native, "vehicle") or _number(service.get("input_vehicle_feature_count"))
    canonical_vehicle = _feature_count(canonical, "vehicle") or _number(service.get("canonical_vehicle_feature_count"))
    missing_vehicle = _number(service.get("missing_vehicle_feature_count"))
    if missing_vehicle is None and expected_vehicle is not None and native_vehicle is not None:
        missing_vehicle = expected_vehicle - native_vehicle
    missing_groups = _as_mapping(service.get("missing_vehicle_measurement_group_counts"))
    note_native = "native exact 需要补齐 vehicle measurement groups"
    note_canonical = "canonical contract closes service schema"
    rows = [
        {
            "payload_side": "left",
            "payload_name": "native replay payload",
            "payload_name_cn": "原生 replay payload",
            "physiology_feature_count": _feature_count(native, "physiology")
            or _number(service.get("input_physio_feature_count"))
            or _feature_count(expected, "physiology"),
            "vehicle_feature_count": native_vehicle,
            "schema_status": service.get("native_feature_schema_status") or native.get("status"),
            "missing_vehicle_feature_count": missing_vehicle,
            "missing_vehicle_measurement_group_count": len(missing_groups),
            "sample_count": native.get("sample_count") or service.get("input_sample_count"),
            "contract_note": note_native,
            "contract_note_cn": "原生 exact 需要补齐 vehicle measurement groups",
            "schema_source": contract.get("schema_source"),
            "schema_hash": contract.get("schema_hash"),
            "source_path": _join_paths(
                [_source_path(sources, "runtime_service"), _source_path(sources, "runtime_schema_contract")]
            ),
        },
        {
            "payload_side": "right",
            "payload_name": "canonical service payload",
            "payload_name_cn": "契约化 service payload",
            "physiology_feature_count": _feature_count(canonical, "physiology")
            or _feature_count(expected, "physiology"),
            "vehicle_feature_count": canonical_vehicle or expected_vehicle,
            "schema_status": service.get("canonical_feature_schema_status") or canonical.get("status"),
            "missing_vehicle_feature_count": 0,
            "missing_vehicle_measurement_group_count": 0,
            "sample_count": canonical.get("sample_count") or service.get("input_sample_count"),
            "contract_note": note_canonical,
            "contract_note_cn": "canonical 契约闭合 service schema",
            "schema_source": contract.get("schema_source"),
            "schema_hash": contract.get("schema_hash"),
            "source_path": _join_paths(
                [_source_path(sources, "runtime_service"), _source_path(sources, "runtime_schema_contract")]
            ),
        },
    ]
    return rows


def build_rigid_body_rotation_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rigid = _payload(sources, "rigid_body")
    rotation = _payload(sources, "rotation_audit")
    rows: list[dict[str, object]] = []
    for family, payload_value in _as_mapping(rigid.get("families")).items():
        payload = _as_mapping(payload_value)
        for metric_name in ("test_total", "test_alignment", "test_physics_total"):
            rows.append(
                {
                    "row_type": "family_metric",
                    "family": family,
                    "metric_name": metric_name,
                    "metric_value": payload.get(metric_name),
                    "translation_vertical_enabled": True,
                    "rotation_status": rotation.get("rotation_status"),
                    "rotation_disabled_reason": rotation.get("rotation_reading"),
                    "source_path": _source_path(sources, "rigid_body"),
                    "evidence_layer": "rigid_body_rotation_diagnostics",
                }
            )
    feature_candidates = _as_mapping(rotation.get("feature_rotation_candidates"))
    mysql_candidates = _as_mapping(rotation.get("mysql_rotation_candidates"))
    for axis in ("pitch", "roll", "yaw"):
        for field_type, key in (("angle", axis), ("rate", f"{axis}_rate")):
            stage_h_count = len(_as_list(feature_candidates.get(key)))
            mysql_count = len(_as_list(mysql_candidates.get(key)))
            rows.append(
                {
                    "row_type": "rotation_field_matrix",
                    "axis": axis,
                    "field_type": field_type,
                    "stage_h_candidate_count": stage_h_count,
                    "mysql_candidate_count": mysql_count,
                    "available": bool(stage_h_count and mysql_count),
                    "availability_label": "available" if stage_h_count and mysql_count else "missing",
                    "translation_vertical_enabled": True,
                    "rotation_status": rotation.get("rotation_status"),
                    "rotation_disabled_reason": rotation.get("rotation_reading"),
                    "source_path": _source_path(sources, "rotation_audit"),
                    "evidence_layer": rotation.get("evidence_layer", "rotation_diagnostics"),
                }
            )
    return rows


def build_weak_label_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_name in ("proxy_sweep", "live_sweep"):
        if source_name not in sources:
            continue
        payload = _payload(sources, source_name)
        sample_collection = _as_mapping(_as_mapping(payload.get("source_summary")).get("sample_collection"))
        sample_source = sample_collection.get("sample_source") or source_name
        blocked_logs = _as_list(payload.get("blocked_attempt_log_paths"))
        for row in _as_list(payload.get("rows")):
            row_map = _as_mapping(row)
            rows.append(
                {
                    "row_type": "completed_run",
                    "run_status": payload.get("status", "completed"),
                    "sample_source": sample_source,
                    "sample_count": payload.get("sample_count"),
                    "task_entry_count": payload.get("task_entry_count"),
                    "combination_count": payload.get("combination_count"),
                    "target_combination_count": payload.get("combination_count"),
                    "completed_combination_count": payload.get("combination_count"),
                    "blocked_at_run_index": payload.get("blocked_at_run_index"),
                    "blocked_attempt_log_path_count": len(blocked_logs),
                    "child_run_id": row_map.get("child_run_id"),
                    "physics_constraint_family": row_map.get("physics_constraint_family"),
                    "causal_weight": row_map.get("causal_weight"),
                    "task_loss_weight": row_map.get("task_loss_weight"),
                    "causal_lag_window_points": row_map.get("causal_lag_window_points"),
                    "lag_label": "none" if row_map.get("causal_lag_window_points") is None else row_map.get("causal_lag_window_points"),
                    "test_total": row_map.get("test_total"),
                    "test_task_total": row_map.get("test_task_total"),
                    "test_causal_total": row_map.get("test_causal_total"),
                    "best_child_run_id": _as_mapping(payload.get("best_run")).get("child_run_id"),
                    "best_test_total": _as_mapping(payload.get("best_run")).get("test_total"),
                    "derived_from_run_id": payload.get("derived_from_run_id"),
                    "source_path": _source_path(sources, source_name),
                    "evidence_layer": payload.get("evidence_layer", "thesis_weak_label"),
                    "metric_definition": "bounded weak-label sweep metrics; lower test_total is better",
                }
            )
    if "live_partial" in sources:
        partial = _payload(sources, "live_partial")
        target_count = partial.get("target_combination_count") or partial.get("combination_count")
        completed_count = partial.get("combination_count_completed") or len(_as_list(partial.get("completed_child_runs")))
        sample_source = _as_mapping(_as_mapping(partial.get("source_summary")).get("sample_collection")).get("sample_source") or "live_influx"
        rows.append(
            {
                "row_type": "partial_status",
                "run_status": partial.get("status", "partial_blocked"),
                "sample_source": sample_source,
                "sample_count": partial.get("sample_count"),
                "task_entry_count": partial.get("task_entry_count"),
                "combination_count": partial.get("combination_count"),
                "target_combination_count": target_count,
                "completed_combination_count": completed_count,
                "blocked_at_run_index": partial.get("blocked_at_run_index"),
                "blocked_attempt_log_path_count": len(_as_list(partial.get("blocked_attempt_log_paths"))),
                "child_run_id": None,
                "physics_constraint_family": "partial_resume_boundary",
                "causal_weight": None,
                "task_loss_weight": None,
                "causal_lag_window_points": None,
                "lag_label": None,
                "test_total": None,
                "test_task_total": None,
                "test_causal_total": None,
                "best_child_run_id": _as_mapping(partial.get("best_run")).get("child_run_id"),
                "best_test_total": _as_mapping(partial.get("best_run")).get("test_total"),
                "derived_from_run_id": partial.get("derived_from_run_id"),
                "source_path": _source_path(sources, "live_partial"),
                "evidence_layer": partial.get("evidence_layer", "thesis_weak_label"),
                "metric_definition": "partial summary captures completed child runs and blocker logs for resume boundary",
            }
        )
    return rows


def build_private_component_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    payload = _payload(sources, "private_component")
    rows = []
    raw_rows = [_as_mapping(row) for row in _as_list(payload.get("rows"))]
    max_delta_by_task: dict[str, float] = {}
    for row in raw_rows:
        task = str(row.get("task_name"))
        delta = abs(float(row.get("delta_vs_full") or 0.0))
        max_delta_by_task[task] = max(max_delta_by_task.get(task, 0.0), delta)
    for row in raw_rows:
        task_name = str(row.get("task_name"))
        metric_name = str(row.get("primary_metric_name"))
        direction = "lower_is_better" if metric_name.lower() in {"rmse", "mae", "loss"} else "higher_is_better"
        max_delta = max_delta_by_task.get(task_name, 0.0)
        normalized_delta = (float(row.get("delta_vs_full") or 0.0) / max_delta) if max_delta else 0.0
        variant_name = str(row.get("variant_name"))
        rows.append(
            {
                **dict(row),
                "direction": direction,
                "normalized_delta_vs_full": normalized_delta,
                "display_variant": _display_variant(variant_name),
                "variant_role": _variant_role(variant_name),
                "source_path": _source_path(sources, "private_component"),
                "metric_definition": "primary metric split by task; delta_vs_full is relative to chronaris_opt within each task",
            }
        )
    return rows


def build_public_transfer_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    transfer = _payload(sources, "public_transfer")
    calibration = _payload(sources, "public_calibration")
    live = _payload(sources, "live_sweep") or _payload(sources, "proxy_sweep")
    return [
        {
            "segment_order": 1,
            "segment_id": "public_adapter_calibration",
            "segment_title_cn": "公开数据适配与校准",
            "segment_title": "Public adapter and calibration",
            "data_scope_cn": "UAB/NASA；验证数据转换、公开基线、评价接口",
            "evidence_role_cn": "外部公开基线与校准支撑",
            "main_output_cn": f"校准行={len(_as_list(calibration.get('rows')))}",
            "positive_reading_cn": "公开适配支撑外部基线。",
            "source_path": _join_paths([_source_path(sources, "public_calibration"), _source_path(sources, "public_transfer")]),
            "evidence_layer": "public_adapter_calibration",
        },
        {
            "segment_order": 2,
            "segment_id": "private_stage_h_weak_label",
            "segment_title_cn": "私有 Stage H 弱标注主线",
            "segment_title": "Private Stage H weak-label mainline",
            "data_scope_cn": "真实生理流 + 真实航电流；风险/负荷/事件任务闭环",
            "evidence_role_cn": "论文主线弱标注证据",
            "main_output_cn": f"样本={live.get('sample_count')} / 任务条目={live.get('task_entry_count')}",
            "positive_reading_cn": "私有双流支撑论文主线。",
            "source_path": _source_path(sources, "public_transfer"),
            "evidence_layer": "thesis_weak_label",
        },
        {
            "segment_order": 3,
            "segment_id": "private_proxy_component",
            "segment_title_cn": "私有代理消融",
            "segment_title": "Private proxy ablation",
            "data_scope_cn": "T1/T2/T3；chronaris_opt 与组件移除对比",
            "evidence_role_cn": "组件诊断与机制对比",
            "main_output_cn": f"组件行={len(_as_list(_payload(sources, 'private_component').get('rows')))}",
            "positive_reading_cn": "代理基准支撑组件分析。",
            "source_path": _source_path(sources, "private_component"),
            "evidence_layer": "private_proxy",
        },
    ]


def build_semantic_event_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    semantic = _semantic_payload(sources)
    rows: list[dict[str, object]] = []
    for index, row_value in enumerate(_as_list(semantic.get("view_rows")), start=1):
        row = _as_mapping(row_value)
        rows.append(
            {
                "row_type": "view_attribution",
                "view_rank": index,
                "view_id": row.get("view_id"),
                "sortie_id": row.get("sortie_id"),
                "pilot_id": row.get("pilot_id"),
                "sample_count": row.get("sample_count"),
                "dominant_query": row.get("dominant_query_name"),
                "mean_event_token_count": row.get("mean_event_token_count"),
                "mean_top_event_attribution": row.get("mean_top_event_attribution"),
                "top_sample": row.get("top_sample_id"),
                "top_sample_query": row.get("top_sample_query_name"),
                "top_sample_event_attribution": row.get("top_sample_event_attribution"),
                "query_count": semantic.get("query_count"),
                "query_names": ";".join(str(name) for name in _as_list(semantic.get("query_names"))),
                "source_path": _join_paths([_source_path(sources, "support"), _source_path(sources, "semantic_event")]),
                "evidence_layer": "semantic_support",
                "case_definition": "view-level event-token and query-to-event attribution support, not a performance claim",
            }
        )
    return rows


def build_llm_comparison_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    pre = _payload(sources, "llm_preprocessing")
    cmp_summary = _payload(sources, "llm_comparison")
    task = _as_mapping(cmp_summary.get("task_context"))
    hints = _as_mapping(cmp_summary.get("semantic_hints"))
    runtime = _as_mapping(cmp_summary.get("runtime_explanations"))
    review = _as_mapping(cmp_summary.get("human_review_packet"))
    rows = [
        _llm_row(
            "A0_baseline",
            "baseline query bank",
            "内置 query bank",
            "baseline_query_count",
            hints.get("baseline_query_count"),
            "built_in_query_bank",
            "对照条件；不含 LLM 生成内容。",
            _source_path(sources, "support"),
        ),
        _llm_row(
            "A1_llm_context",
            "attach preprocessing context",
            "接入 P20 context",
            "attached_entry_count",
            task.get("attached_entry_count"),
            f"label_unchanged={task.get('label_unchanged')}; label_changed_count={task.get('label_changed_count')}",
            "只 attach context 和规则复核，不改写 weak-label 值。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A2_llm_semantic_hints",
            "whitelisted semantic hints",
            "白名单语义 hints",
            "query_count",
            hints.get("combined_query_count"),
            f"{hints.get('baseline_query_count')}->{hints.get('combined_query_count')}; added={hints.get('added_query_count')}",
            "仅通过 recipe whitelist 扩展 query coverage；未重算 attribution 改善。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A3_llm_runtime_explanation",
            "runtime explanation subset",
            "runtime 解释子集",
            "explained_case_count",
            runtime.get("llm_explained_case_count"),
            f"{runtime.get('llm_explained_case_count')}/{runtime.get('runtime_case_count')}; completeness={runtime.get('with_llm_average_completeness_for_explained_cases')}",
            "解释层补充 runtime case 文本，不改变 native/canonical schema 边界。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A4_human_review_packet",
            "human review packet",
            "人工复核 packet",
            "review_item_count",
            review.get("item_count"),
            f"human_review_completed={review.get('human_review_completed')}",
            "已生成待复核材料；人工复核未完成前不写成验证完成。",
            _source_path(sources, "llm_comparison"),
        ),
    ]
    rows.append(
        _llm_row(
            "P20_preprocessing_run",
            "LLM preprocessing harness",
            "P20 预处理 harness",
            "request_count",
            pre.get("request_count"),
            f"errors={pre.get('error_count')}; hints={pre.get('semantic_query_hint_count')}; runtime_explanations={pre.get('runtime_explanation_count')}",
            "DeepSeek 输出是 preprocessing context，不是人工真值。",
            _source_path(sources, "llm_preprocessing"),
        )
    )
    return rows


def _semantic_payload(sources: Mapping[str, Mapping[str, object]]) -> Mapping[str, object]:
    semantic_event = _payload(sources, "semantic_event")
    if semantic_event:
        return semantic_event
    support = _payload(sources, "support")
    causal = _as_mapping(support.get("causal_support"))
    return _as_mapping(causal.get("semantic_event"))


def _overview_row(
    display_order: int,
    evidence_layer: str,
    title_cn: str,
    title: str,
    source_run_id: object,
    primary_metric_name: str,
    primary_metric_value: object,
    secondary_metric_name: str,
    secondary_metric_value: object,
    key_status: object,
    boundary_cn: str,
    source_paths: Sequence[object],
) -> dict[str, object]:
    return {
        "display_order": display_order,
        "evidence_layer": evidence_layer,
        "layer_title_cn": title_cn,
        "layer_title": title,
        "source_run_id": source_run_id,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "secondary_metric_name": secondary_metric_name,
        "secondary_metric_value": secondary_metric_value,
        "key_status": key_status,
        "boundary_cn": boundary_cn,
        "source_path": _join_paths(source_paths),
    }


def _llm_row(
    condition: str,
    stage: str,
    stage_cn: str,
    metric_name: str,
    metric_value: object,
    metric_note: str,
    boundary_cn: str,
    source_path: object,
) -> dict[str, object]:
    return {
        "condition": condition,
        "stage": stage,
        "stage_cn": stage_cn,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_note": metric_note,
        "boundary_cn": boundary_cn,
        "source_path": source_path,
        "evidence_layer": "llm_preprocessing_comparison",
    }


def _feature_count(section: Mapping[str, object], stream: str) -> int | None:
    stream_payload = _as_mapping(section.get(stream))
    return _number(stream_payload.get("feature_count"))


def _runtime_status_text(service: Mapping[str, object]) -> str:
    native = service.get("native_feature_schema_status")
    canonical = service.get("canonical_feature_schema_status")
    if native or canonical:
        return f"native={native}; canonical={canonical}"
    return "runtime schema contract"


def _display_variant(variant_name: str) -> str:
    mapping = {
        "chronaris_opt": "chronaris_opt",
        "chronaris_opt_no_causal_mask": "no_causal_mask",
        "chronaris_opt_no_time_residual": "no_time_residual",
        "chronaris_opt_no_task_head": "no_task_head",
        "naive_sync": "naive",
        "e_baseline": "E",
        "f_full": "F",
        "g_min": "G(min)",
        "g_no_causal_mask": "G no mask",
    }
    return mapping.get(variant_name, variant_name)


def _variant_role(variant_name: str) -> str:
    if variant_name == "chronaris_opt":
        return "full_candidate"
    if variant_name.startswith("chronaris_opt_no_"):
        return "component_removed"
    return "reference_baseline"


def _payload(sources: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, object]:
    return _as_mapping(_as_mapping(sources.get(name)).get("payload"))


def _source_path(sources: Mapping[str, Mapping[str, object]], name: str) -> str:
    return str(_as_mapping(sources.get(name)).get("path") or "")


def _run_id(payload: Mapping[str, object]) -> object:
    return payload.get("run_id") or ""


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _number(value: object) -> int | float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _fmt_float(value: object) -> str:
    number = _number(value)
    if number is None:
        return "NA"
    return f"{float(number):.4f}"


def _join_paths(paths: Sequence[object]) -> str:
    return ";".join(str(Path(str(path))) for path in paths if path)
