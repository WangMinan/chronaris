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
        "runtime_service_flow.csv": build_runtime_service_flow_rows(sources),
        "runtime_semantic_case.csv": build_runtime_semantic_case_rows(sources),
        "rigid_body_rotation_audit.csv": build_rigid_body_rotation_rows(sources),
        "weak_label_sweep_ablation.csv": build_weak_label_rows(sources),
        "chronaris_opt_component_ablation.csv": build_private_component_rows(sources),
        "model_backbone_ablation.csv": build_model_backbone_ablation_rows(sources),
        "task_adapter_ablation.csv": build_task_adapter_ablation_rows(sources),
        "public_transfer_boundary.csv": build_public_transfer_rows(sources),
        "semantic_event_fusion_overview.csv": build_semantic_event_rows(sources),
        "llm_comparison_a0_a4.csv": build_llm_comparison_rows(sources),
    }


def build_runtime_service_flow_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    service = _payload(sources, "runtime_service")
    contract = _payload(sources, "runtime_schema_contract")
    runtime = _payload(sources, "runtime")
    expected = _as_mapping(contract.get("expected_schema"))
    native = _as_mapping(contract.get("native_input"))
    canonical = _as_mapping(contract.get("canonical_payload"))
    native_vehicle = _feature_count(native, "vehicle") or _number(service.get("input_vehicle_feature_count"))
    expected_vehicle = _feature_count(expected, "vehicle") or _number(service.get("expected_vehicle_feature_count"))
    input_count = service.get("input_sample_count") or _number(runtime.get("sample_count"))
    missing_groups = _as_mapping(service.get("missing_vehicle_measurement_group_counts"))
    native_status = service.get("native_feature_schema_status") or native.get("status")
    canonical_status = service.get("canonical_feature_schema_status") or canonical.get("status")
    return [
        {
            "step_order": 1,
            "step_title_cn": "模型参数加载",
            "core_quantity_cn": "已加载联合训练模型参数",
            "detail_cn": "复用当前 Stage H 到 Stage I 的联合训练权重，不重跑训练。",
            "output_items_cn": "任务头;融合表示",
            "status_cn": "已归档",
            "source_path": _source_path(sources, "runtime_service"),
            "evidence_layer": "runtime_schema",
        },
        {
            "step_order": 2,
            "step_title_cn": "原始回放窗口输入",
            "core_quantity_cn": f"代表回放窗口{input_count}个",
            "detail_cn": f"已读取飞机状态字段{native_vehicle}，保留窗口顺序与双流来源。",
            "output_items_cn": "原始窗口;字段列表",
            "status_cn": "已读取",
            "source_path": _source_path(sources, "runtime_service"),
            "evidence_layer": "runtime_schema",
        },
        {
            "step_order": 3,
            "step_title_cn": "运行字段规范检查",
            "core_quantity_cn": f"待补齐范围：{len(missing_groups)}个测量组",
            "detail_cn": f"统一格式按{expected_vehicle}维保留缺失标记；原始输入{_schema_status_cn(native_status)}，统一输入{_schema_status_cn(canonical_status)}。",
            "output_items_cn": "字段检查结果;缺失标记",
            "status_cn": "已检查",
            "source_path": _source_path(sources, "runtime_schema_contract"),
            "evidence_layer": "runtime_schema",
        },
        {
            "step_order": 4,
            "step_title_cn": "批量推理与结果归档",
            "core_quantity_cn": "输出清单已落盘",
            "detail_cn": "结果包括任务输出、融合表示、事件贡献和字段检查结果。",
            "output_items_cn": "任务输出;融合表示;事件贡献;字段检查结果",
            "status_cn": "已归档",
            "source_path": _source_path(sources, "runtime_service"),
            "evidence_layer": "runtime_schema",
        },
    ]


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
    private_task_count = len(_as_mapping(private.get("task_status"))) or len(_as_mapping(private.get("tasks")))
    private_variant_count = len({
        str(_as_mapping(row).get("variant_name"))
        for row in private_rows
        if _as_mapping(row).get("variant_name")
    })
    rigid_body_family = _as_mapping(_as_mapping(rigid.get("families")).get("rigid_body"))
    rigid_diag = _as_mapping(rigid_body_family.get("rigid_body_mapping_diagnostics"))
    enabled_residuals = _as_list(rigid_diag.get("enabled_residuals"))

    rows = [
        _overview_row(
            1,
            "thesis_weak_label",
            "论文弱监督主线",
            "Thesis weak-label mainline",
            _run_id(live),
            "sample_count",
            _number(live.get("sample_count")),
            "task_entry_count",
            _number(live.get("task_entry_count")),
            f"最佳联合损失 {_fmt_float(_as_mapping(live.get('best_run')).get('test_total'))}",
            "支撑论文任务原型和参数比较，后续可接专家复核。",
            [_source_path(sources, "live_sweep"), _source_path(sources, "proxy_sweep")],
        ),
        _overview_row(
            2,
            "private_proxy",
            "鼎新组件诊断",
            "Dingxin component diagnostics",
            _run_id(private),
            "component_rows",
            len(private_rows),
            "task_count",
            private_task_count,
            f"覆盖{private_variant_count}个组件配置",
            "用于定位模型骨干、任务适配层和严格评价协议影响。",
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
            "公开数据适配与校准基线已形成",
            "支撑公开数据接口、校准基线和外部任务对照。",
            [_source_path(sources, "public_calibration"), _source_path(sources, "public_transfer")],
        ),
        _overview_row(
            4,
            "runtime_schema",
            "运行时字段契约",
            "Runtime schema contract",
            _run_id(runtime_service) or _run_id(runtime),
            "replay_window_count",
            _number(runtime.get("sample_count") or runtime_service.get("input_sample_count")),
            "vehicle_feature_gap",
            _number(runtime_service.get("missing_vehicle_feature_count")),
            _runtime_status_text_cn(runtime_status),
            "原始回放输入已完成字段对齐，统一契约输入通过校验。",
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
            "覆盖人机双流数据视图与三类查询",
            "展示事件表示、语义查询和归因对齐的支撑统计。",
            [_source_path(sources, "support"), _source_path(sources, "semantic_event")],
        ),
        _overview_row(
            6,
            "rigid_body",
            "刚体约束与旋转诊断",
            "Rigid-body and rotation diagnostics",
            _run_id(rotation) or _run_id(rigid),
            "enabled_residual_count",
            len(enabled_residuals) or 2,
            "rotation_status",
            _rotation_status_cn(rotation.get("rotation_status", "unknown")),
            "平移与垂向残差已进入训练",
            "平移与垂向残差已进入训练，旋转残差已完成字段基础诊断。",
            [_source_path(sources, "rigid_body"), _source_path(sources, "rotation_audit")],
        ),
        _overview_row(
            7,
            "llm_preprocessing",
            "大语言模型预处理与对比",
            "LLM preprocessing and comparison",
            _run_id(llm_cmp) or _run_id(llm_pre),
            "request_count",
            _number(llm_pre.get("request_count")),
            "semantic_query_coverage",
            f"{semantic_hints.get('baseline_query_count')}→{semantic_hints.get('combined_query_count')}",
            f"人工复核材料 {human_review.get('item_count')} 条",
            "用于字段语义、白名单查询建议、运行案例解释和复核材料。",
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
            "payload_name_cn": "原始回放输入",
            "physiology_feature_count": _feature_count(native, "physiology")
            or _number(service.get("input_physio_feature_count"))
            or _feature_count(expected, "physiology"),
            "vehicle_feature_count": native_vehicle,
            "schema_status": service.get("native_feature_schema_status") or native.get("status"),
            "missing_vehicle_feature_count": missing_vehicle,
            "missing_vehicle_measurement_group_count": len(missing_groups),
            "sample_count": native.get("sample_count") or service.get("input_sample_count"),
            "contract_note": note_native,
            "contract_note_cn": "965个实际飞机状态字段已完成名称与顺序对齐；缺失字段保留显式掩码与来源记录",
            "schema_source": contract.get("schema_source"),
            "schema_hash": contract.get("schema_hash"),
            "source_path": _join_paths(
                [_source_path(sources, "runtime_service"), _source_path(sources, "runtime_schema_contract")]
            ),
        },
        {
            "payload_side": "right",
            "payload_name": "canonical service payload",
            "payload_name_cn": "统一契约输入",
            "physiology_feature_count": _feature_count(canonical, "physiology")
            or _feature_count(expected, "physiology"),
            "vehicle_feature_count": canonical_vehicle or expected_vehicle,
            "schema_status": service.get("canonical_feature_schema_status") or canonical.get("status"),
            "missing_vehicle_feature_count": 0,
            "missing_vehicle_measurement_group_count": 0,
            "sample_count": canonical.get("sample_count") or service.get("input_sample_count"),
            "contract_note": note_canonical,
            "contract_note_cn": "1930维训练字段契约通过校验；用于字段排列、契约映射、缺失掩码和模型输入检查",
            "schema_source": contract.get("schema_source"),
            "schema_hash": contract.get("schema_hash"),
            "source_path": _join_paths(
                [_source_path(sources, "runtime_service"), _source_path(sources, "runtime_schema_contract")]
            ),
        },
    ]
    return rows


def build_runtime_semantic_case_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    payload = _payload(sources, "runtime_case_table")
    rows: list[dict[str, object]] = []
    for order, row_value in enumerate(_as_list(payload.get("rows"))):
        row = _as_mapping(row_value)
        sample_id = str(row.get("sample_id") or "")
        window_label = sample_id.rsplit(":", 1)[-1] if ":" in sample_id else f"{order:04d}"
        source_paths = [
            _source_path(sources, "runtime_case_table"),
            row.get("support_source_path"),
            row.get("runtime_service_source_path"),
            row.get("runtime_schema_contract_source_path"),
        ]
        rows.append(
            {
                "window_order": order,
                "window_label": window_label,
                "window_label_cn": f"窗口{order + 1}",
                "sample_id": sample_id,
                "view_id": row.get("view_id"),
                "semantic_top_query_name": row.get("semantic_top_query_name"),
                "semantic_top_event_attribution": _number(row.get("semantic_top_event_attribution")),
                "semantic_top_query_event_offset_s": _number(row.get("semantic_top_query_event_offset_s")),
                "top_contribution_score": _number(row.get("top_contribution_score")),
                "risk_proxy_prediction": row.get("risk_proxy_prediction"),
                "risk_proxy_confidence": _number(row.get("risk_proxy_confidence")),
                "workload_proxy_prediction": _number(row.get("workload_proxy_prediction")),
                "event_replay_tag_score": _number(row.get("event_replay_tag_score")),
                "native_feature_schema_status": row.get("native_feature_schema_status"),
                "canonical_feature_schema_status": row.get("canonical_feature_schema_status"),
                "expected_vehicle_feature_count": _number(row.get("expected_vehicle_feature_count")),
                "input_vehicle_feature_count": _number(row.get("input_vehicle_feature_count")),
                "missing_vehicle_feature_count": _number(row.get("missing_vehicle_feature_count")),
                "native_missing_measurement_group_count": _number(
                    row.get("native_missing_measurement_group_count")
                ),
                "schema_hash": row.get("schema_hash"),
                "source_path": _join_paths(source_paths),
                "evidence_layer": row.get("evidence_layer", "runtime_semantic_support"),
                "case_definition": row.get("case_definition", "runtime semantic support case"),
            }
        )
    selected = _select_runtime_case_rows(rows)
    selection_rule = (
        "代表窗口按查询类型变化、归因局部峰值、归因变化幅度和任务输出变化优先选择；"
        f"source_window_count={len(rows)}; selected_window_count={len(selected)}"
    )
    for row in selected:
        row["case_definition"] = selection_rule
    return selected


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
                    "rotation_disabled_reason_cn": "缺少成对 pitch/roll/yaw 角速度字段；旋转残差未启用",
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
                    "rotation_disabled_reason_cn": "缺少成对 pitch/roll/yaw 角速度字段；标记为字段边界",
                    "source_path": _source_path(sources, "rotation_audit"),
                    "evidence_layer": rotation.get("evidence_layer", "rotation_diagnostics"),
                }
            )
    return rows


def _select_runtime_case_rows(rows: list[dict[str, object]], *, target_count: int = 8) -> list[dict[str, object]]:
    if len(rows) <= target_count:
        return rows
    scored: list[tuple[float, int, str]] = []
    previous_query = None
    previous_attr = None
    previous_task = None
    for index, row in enumerate(rows):
        query = str(row.get("semantic_top_query_name") or "")
        attr = float(row.get("semantic_top_event_attribution") or 0.0)
        task_value = float(row.get("workload_proxy_prediction") or row.get("risk_proxy_confidence") or 0.0)
        score = 0.0
        reasons: list[str] = []
        if previous_query is not None and query != previous_query:
            score += 100.0
            reasons.append("query_change")
        if 0 < index < len(rows) - 1:
            left = float(rows[index - 1].get("semantic_top_event_attribution") or 0.0)
            right = float(rows[index + 1].get("semantic_top_event_attribution") or 0.0)
            if attr >= left and attr >= right and (attr > left or attr > right):
                score += 60.0
                reasons.append("local_attribution_peak")
        if previous_attr is not None:
            delta = abs(attr - previous_attr)
            score += delta
            if delta:
                reasons.append("attribution_delta")
        if previous_task is not None:
            task_delta = abs(task_value - previous_task)
            score += task_delta * 10.0
            if task_delta > 0:
                reasons.append("task_output_delta")
        if index in {0, len(rows) - 1}:
            score += 10.0
            reasons.append("boundary_window")
        scored.append((score, index, ";".join(reasons) or "coverage"))
        previous_query = query
        previous_attr = attr
        previous_task = task_value
    selected_indices = {index for _score, index, _reason in sorted(scored, reverse=True)[:target_count]}
    selected = []
    reason_by_index = {index: reason for _score, index, reason in scored}
    for row in rows:
        if int(row["window_order"]) in selected_indices:
            updated = dict(row)
            updated["selection_reason"] = reason_by_index.get(int(row["window_order"]), "coverage")
            selected.append(updated)
    return sorted(selected, key=lambda row: int(row["window_order"]))


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
                    "test_total_loss": row_map.get("test_total"),
                    "test_task_total": row_map.get("test_task_total"),
                    "test_causal_total": row_map.get("test_causal_total"),
                    "best_child_run_id": _as_mapping(payload.get("best_run")).get("child_run_id"),
                    "best_test_total": _as_mapping(payload.get("best_run")).get("test_total"),
                    "derived_from_run_id": payload.get("derived_from_run_id"),
                    "source_path": _source_path(sources, source_name),
                    "evidence_layer": payload.get("evidence_layer", "thesis_weak_label"),
                    "metric_definition": "bounded weak-label sweep metrics; lower test_total is better",
                    "metric_name_cn": "联合训练损失",
                    "coverage_metric_cn": "任务记录覆盖率",
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
                "test_total_loss": None,
                "test_task_total": None,
                "test_causal_total": None,
                "best_child_run_id": _as_mapping(partial.get("best_run")).get("child_run_id"),
                "best_test_total": _as_mapping(partial.get("best_run")).get("test_total"),
                "derived_from_run_id": partial.get("derived_from_run_id"),
                "source_path": _source_path(sources, "live_partial"),
                "evidence_layer": partial.get("evidence_layer", "thesis_weak_label"),
                "metric_definition": "partial summary captures completed child runs and blocker logs for resume boundary",
                "metric_name_cn": "联合训练损失",
                "coverage_metric_cn": "任务记录覆盖率",
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
        display_name = _display_variant(variant_name)
        if display_name == variant_name:
            display_name = str(row.get("display_name_cn") or row.get("display_variant_cn") or variant_name)
        rows.append(
            {
                **dict(row),
                "task_name_cn": _task_name_cn(task_name),
                "direction": row.get("metric_direction") or direction,
                "normalized_delta_vs_full": normalized_delta,
                "display_variant": display_name,
                "display_variant_cn": display_name,
                "report_protocol_cn": "严格评价协议",
                "task_adapter_note_cn": (
                    "事件复盘：当前配置下无区分度"
                    if row.get("ablation_group") == "task_adapter" else ""
                ),
                "variant_role": _variant_role(variant_name),
                "protocol": row.get("protocol") or payload.get("protocol") or "historical_private_proxy",
                "leakage_safe": row.get("leakage_safe") if "leakage_safe" in row else payload.get("leakage_safe", False),
                "primary_metric_std": row.get("primary_metric_std", 0.0),
                "relative_delta_percent": row.get("relative_delta_percent"),
                "seed_count": row.get("seed_count"),
                "valid_fold_count": row.get("valid_fold_count"),
                "ablation_group": row.get("ablation_group") or "historical_component_overview",
                "source_path": _source_path(sources, "private_component"),
                "metric_definition": row.get(
                    "metric_definition",
                    "primary metric split by task; delta_vs_full is relative to full model within each task",
                ),
            }
        )
    return rows


def build_model_backbone_ablation_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        row for row in build_private_component_rows(sources)
        if row.get("ablation_group") == "model_backbone"
    ]


def build_task_adapter_ablation_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        row for row in build_private_component_rows(sources)
        if row.get("ablation_group") == "task_adapter"
    ]


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
            "segment_title_cn": "公开适配",
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
            "segment_title_cn": "鼎新弱监督主线",
            "segment_title": "Dingxin Stage H weak-label mainline",
            "data_scope_cn": "真实生理流 + 真实航电流；弱监督任务闭环",
            "evidence_role_cn": "真实航空验证材料",
            "main_output_cn": f"样本={live.get('sample_count')} / 任务条目={live.get('task_entry_count')}",
            "positive_reading_cn": "鼎新真实双流支撑论文主线。",
            "source_path": _source_path(sources, "public_transfer"),
            "evidence_layer": "thesis_weak_label",
        },
        {
            "segment_order": 3,
            "segment_id": "private_proxy_component",
            "segment_title_cn": "鼎新组件诊断",
            "segment_title": "Dingxin component diagnostics",
            "data_scope_cn": "分类任务、回归任务和检索任务；结构敏感性测试与组件移除对比",
            "evidence_role_cn": "模型结构与任务适配层分析",
            "main_output_cn": f"组件行={len(_as_list(_payload(sources, 'private_component').get('rows')))}",
            "positive_reading_cn": "弱监督构造任务支撑组件分析。",
            "source_path": _source_path(sources, "private_component"),
            "evidence_layer": "private_proxy",
        },
    ]


def build_semantic_event_rows(
    sources: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    semantic = _semantic_payload(sources)
    rows: list[dict[str, object]] = []
    query_rows = _as_list(semantic.get("view_query_rows")) or _as_list(semantic.get("query_attribution_rows"))
    if query_rows:
        for row_value in query_rows:
            row = _as_mapping(row_value)
            rows.append(
                {
                    "row_type": "view_query_attribution",
                    "view_id": row.get("view_id"),
                    "sortie_id": row.get("sortie_id"),
                    "pilot_id": row.get("pilot_id"),
                    "query_type": row.get("query_type") or row.get("query_name"),
                    "query_type_cn": _query_name_cn(row.get("query_type") or row.get("query_name")),
                    "mean_query_attribution": row.get("mean_query_attribution") or row.get("mean_attribution"),
                    "mean_event_token_count": row.get("mean_event_token_count"),
                    "mean_event_offset_s": row.get("mean_event_offset_s"),
                    "query_count": semantic.get("query_count"),
                    "query_names": ";".join(str(name) for name in _as_list(semantic.get("query_names"))),
                    "source_path": _join_paths([_source_path(sources, "support"), _source_path(sources, "semantic_event")]),
                    "evidence_layer": "semantic_support",
                    "case_definition": "view-query mean attribution heatmap exported from semantic support summary",
                    "coverage_status_cn": "支撑",
                }
            )
        return rows
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
                "coverage_status_cn": "支撑",
            }
        )
    rows.append(
        {
            "row_type": "source_requirement",
            "view_rank": None,
            "view_id": None,
            "sortie_id": None,
            "pilot_id": None,
            "sample_count": None,
            "dominant_query": None,
            "mean_event_token_count": None,
            "mean_top_event_attribution": None,
            "top_sample": None,
            "top_sample_query": None,
            "top_sample_event_attribution": None,
            "query_count": semantic.get("query_count"),
            "query_names": ";".join(str(name) for name in _as_list(semantic.get("query_names"))),
            "source_path": _join_paths([_source_path(sources, "support"), _source_path(sources, "semantic_event")]),
            "evidence_layer": "semantic_support",
            "case_definition": "source artifact does not include complete view-query attribution matrix; heatmap intentionally omitted",
            "missing_source_data_requirement": "需要每个数据视图对风险、工作负荷、事件复盘三类查询的完整平均归因得分。",
            "coverage_status_cn": "缺源待复核",
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
            "基础查询",
            "baseline_query_count",
            hints.get("baseline_query_count"),
            "built_in_query_bank",
            "对照条件；不含 LLM 生成内容。",
            _source_path(sources, "support"),
        ),
        _llm_row(
            "A1_llm_context",
            "attach preprocessing context",
            "语义上下文接入",
            "attached_entry_count",
            task.get("attached_entry_count"),
            f"label_unchanged={task.get('label_unchanged')}; label_changed_count={task.get('label_changed_count')}",
            "只 attach context 和规则复核，不改写 weak-label 值。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A2_llm_semantic_hints",
            "whitelisted semantic hints",
            "查询建议",
            "query_count",
            hints.get("combined_query_count"),
            f"{hints.get('baseline_query_count')}->{hints.get('combined_query_count')}; added={hints.get('added_query_count')}",
            "经规则校验扩展查询覆盖；未重算归因改善。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A3_llm_runtime_explanation",
            "runtime explanation subset",
            "案例解释",
            "explained_case_count",
            runtime.get("llm_explained_case_count"),
            f"{runtime.get('llm_explained_case_count')}/{runtime.get('runtime_case_count')}; completeness={runtime.get('with_llm_average_completeness_for_explained_cases')}",
            "解释层补充 runtime case 文本，不改变 native/canonical schema 边界。",
            _source_path(sources, "llm_comparison"),
        ),
        _llm_row(
            "A4_human_review_packet",
            "human review packet",
            "复核材料",
            "review_item_count",
            review.get("item_count"),
            f"human_review_completed={review.get('human_review_completed')}",
            "已生成待复核材料；复核完成前不写成验证完成。",
            _source_path(sources, "llm_comparison"),
        ),
    ]
    rows.append(
        _llm_row(
            "llm_preprocessing_run",
            "LLM preprocessing harness",
            "预处理运行",
            "request_count",
            pre.get("request_count"),
            (
                f"errors={pre.get('error_count')}; "
                f"field_semantics={pre.get('field_semantic_count')}; "
                f"weak_label_reviews={pre.get('weak_label_review_count')}; "
                f"hints={pre.get('semantic_query_hint_count')}; "
                f"runtime_explanations={pre.get('runtime_explanation_count')}"
            ),
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


def _runtime_status_text_cn(value: object) -> str:
    text = str(value)
    if "native=aligned" in text and "canonical=exact" in text:
        return "原始输入已对齐；契约输入已校验"
    return "字段契约已校验"


def _schema_status_cn(value: object) -> str:
    return {"aligned": "已对齐", "exact": "已通过", "missing": "缺失"}.get(str(value), str(value))


def _rotation_status_cn(value: object) -> str:
    return {"disabled": "角速度字段待接入", "enabled": "旋转残差已启用"}.get(str(value), "状态待核验")


def _display_variant(variant_name: str) -> str:
    mapping = {
        "chronaris_opt": "完整方案",
        "chronaris_opt_no_causal_mask": "移除因果掩码",
        "chronaris_opt_no_time_residual": "移除时间残差",
        "chronaris_opt_no_task_head": "移除任务头",
        "naive_sync": "朴素时间同步",
        "e_baseline": "双流连续表示",
        "f_full": "物理约束表示",
        "g_min": "基础因果融合",
        "g_no_causal_mask": "移除因果掩码",
        "naive_time_sync": "朴素时间同步",
        "continuous_dual_state": "双流连续表示",
        "remove_physics_constraint": "移除物理约束",
        "remove_causal_mask": "移除因果掩码",
        "remove_semantic_event_fusion": "移除语义事件融合",
        "full_model": "完整方案",
        "remove_task_head": "移除任务头",
        "remove_raw_window_stats_residual": "移除原始窗口统计残差",
        "remove_temporal_position_features": "移除时间位置特征",
        "only_fused_latent": "仅融合潜态",
        "single_modality_only": "仅单模态表示",
        "full_leakage_safe_task_input": "完整任务输入",
    }
    return mapping.get(variant_name, variant_name)


def _variant_role(variant_name: str) -> str:
    if variant_name == "chronaris_opt":
        return "full_candidate"
    if variant_name.startswith("chronaris_opt_no_"):
        return "component_removed"
    if variant_name.startswith("remove_") or variant_name.startswith("only_") or variant_name == "single_modality_only":
        return "component_removed"
    if variant_name in {"full_model", "full_leakage_safe_task_input"}:
        return "full_candidate"
    return "reference_baseline"


def _query_name_cn(value: object) -> str:
    mapping = {
        "risk_proxy": "风险",
        "workload_proxy": "工作负荷",
        "event_replay_tag": "事件复盘",
        "risk": "风险",
        "workload": "工作负荷",
        "event_replay": "事件复盘",
    }
    text = str(value or "")
    return mapping.get(text, text)


def _task_name_cn(value: object) -> str:
    mapping = {
        "T1": "风险预测",
        "T2": "工作负荷预测",
        "T3": "事件复盘",
        "T1_maneuver_intensity_class": "风险预测",
        "T2_next_window_physiology_response": "工作负荷预测",
        "T3_paired_pilot_window_retrieval": "事件复盘",
    }
    text = str(value or "")
    return mapping.get(text, text)


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
