"""Support-summary builders for task evaluation thesis evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping

import pandas as pd


def _build_alignment_support_summary(
    *,
    e_projection: Mapping[str, object],
    f_projection: Mapping[str, object],
    feature_export_run_manifest: Mapping[str, object],
    case_study_summary: Mapping[str, object],
) -> dict[str, object]:
    e_summary = _projection_summary(e_projection)
    f_summary = _projection_summary(f_projection)
    view_verdict_counts = Counter(
        view_result["view_summary"]["verdict"]
        for view_result in case_study_summary["view_results"]
    )
    partial_data = feature_export_run_manifest.get("partial_data") or {}
    return {
        "alignment_chain": {
            "e_baseline": e_summary,
            "f_full": f_summary,
            "delta_f_minus_e": {
                "sample_count": min(e_summary["sample_count"], f_summary["sample_count"]),
                "mean_projection_cosine": (
                    f_summary["mean_projection_cosine"] - e_summary["mean_projection_cosine"]
                ),
                "mean_projection_l2_gap": (
                    f_summary["mean_projection_l2_gap"] - e_summary["mean_projection_l2_gap"]
                ),
                "threshold_verdict": (
                    "PASS"
                    if f_summary["threshold_verdict"] == "PASS"
                    else f_summary["threshold_verdict"]
                ),
            },
        },
        "feature_export_export": {
            "run_id": feature_export_run_manifest["run_id"],
            "sortie_count": len(feature_export_run_manifest["sortie_ids"]),
            "generated_view_count": int(feature_export_run_manifest["generated_view_count"]),
            "generated_view_ids": list(feature_export_run_manifest["generated_view_ids"]),
            "view_verdict_counts": dict(view_verdict_counts),
            "partial_data_entry_count": int(
                partial_data.get("entry_count", 0)
            ),
            "partial_data_built_entry_count": int(
                partial_data.get("built_entry_count", 0)
            ),
        },
    }


def _build_causal_support_summary(
    *,
    g_causal: Mapping[str, object],
    case_study_summary: Mapping[str, object],
    case_study_ablation: pd.DataFrame,
    private_summary: Mapping[str, object],
    deep_summary: Mapping[str, object],
) -> dict[str, object]:
    ablation_name_column = (
        "ablation_name" if "ablation_name" in case_study_ablation.columns else "name"
    )
    case_study_view_results = case_study_summary["view_results"]
    baseline_view_frame = pd.DataFrame(
        [view_result["view_summary"] for view_result in case_study_view_results]
    )
    pilot_comparison_frame = pd.DataFrame(case_study_summary["pilot_comparisons"])
    ablation_metric_columns = [
        column
        for column in (
            "mean_attention_entropy",
            "mean_top_event_score",
            "mean_top_contribution_score",
            "delta_mean_attention_entropy",
            "delta_mean_top_event_score",
            "delta_mean_top_contribution_score",
            "delta_fused_l2_norm",
            "delta_fused_cosine_to_projection_baseline",
        )
        if column in case_study_ablation.columns
    ]
    ablation_means = (
        case_study_ablation.loc[
            case_study_ablation[ablation_name_column] != "projection_refusion_baseline"
        ]
        .groupby(ablation_name_column, sort=True)[ablation_metric_columns]
        .mean()
        .round(12)
        .to_dict(orient="index")
    )
    for payload in ablation_means.values():
        payload.setdefault("mean_attention_entropy", None)
        payload.setdefault("mean_top_event_score", None)
        payload.setdefault("mean_top_contribution_score", None)
    strongest_ablation_name = min(
        ablation_means,
        key=lambda name: ablation_means[name]["delta_mean_top_contribution_score"],
    )
    private_no_mask = _extract_private_no_mask_summary(private_summary)
    deep_feature_export_case = _extract_deep_feature_export_case_summary(deep_summary)
    semantic_event = g_causal.get("semantic_event") if isinstance(g_causal.get("semantic_event"), Mapping) else None
    if semantic_event is None and "view_rows" in g_causal and "query_names" in g_causal:
        semantic_event = g_causal
    return {
        "g_min": {
            "sample_count": int(g_causal["sample_count"]),
            "mean_attention_entropy": float(g_causal["mean_attention_entropy"]),
            "mean_max_attention": float(g_causal["mean_max_attention"]),
            "mean_top_event_score": float(g_causal["mean_top_event_score"]),
            "mean_top_contribution_score": float(g_causal["mean_top_contribution_score"]),
        },
        "semantic_event": (
            {
                "query_names": list(semantic_event.get("query_names", [])),
                "query_count": int(semantic_event.get("query_count", 0)),
                "mean_event_token_count": float(semantic_event.get("mean_event_token_count", 0.0)),
                "mean_query_entropy": float(semantic_event.get("mean_query_entropy", 0.0)),
                "mean_top_query_score": float(semantic_event.get("mean_top_query_score", 0.0)),
                "mean_top_event_attribution": float(semantic_event.get("mean_top_event_attribution", 0.0)),
                "view_count": int(semantic_event.get("view_count", 0)),
                "top_view_id": semantic_event.get("top_view_id"),
                "view_rows": list(semantic_event.get("view_rows", [])),
                "samples": list(semantic_event.get("samples", [])),
            }
            if semantic_event is not None
            else None
        ),
        "case_study": {
            "view_count": len(case_study_view_results),
            "view_verdict_counts": dict(
                Counter(
                    view_result["view_summary"]["verdict"]
                    for view_result in case_study_view_results
                )
            ),
            "baseline_view_means": {
                "mean_projection_cosine": _frame_mean(
                    baseline_view_frame,
                    "mean_projection_cosine",
                ),
                "mean_projection_l2_gap": _frame_mean(
                    baseline_view_frame,
                    "mean_projection_l2_gap",
                ),
                "mean_attention_entropy": _frame_mean(
                    baseline_view_frame,
                    "mean_attention_entropy",
                ),
                "mean_top_event_score": _frame_mean(
                    baseline_view_frame,
                    "mean_top_event_score",
                ),
                "mean_top_contribution_score": _frame_mean(
                    baseline_view_frame,
                    "mean_top_contribution_score",
                ),
            },
            "pilot_comparisons": list(case_study_summary["pilot_comparisons"]),
            "pilot_delta_means": {
                "delta_mean_projection_cosine": float(
                    pilot_comparison_frame["delta_mean_projection_cosine"].mean()
                )
                if not pilot_comparison_frame.empty
                else 0.0,
                "delta_mean_top_contribution_score": float(
                    pilot_comparison_frame["delta_mean_top_contribution_score"].mean()
                )
                if not pilot_comparison_frame.empty
                else 0.0,
            },
            "ablation_means": ablation_means,
            "strongest_ablation": {
                "name": strongest_ablation_name,
                **ablation_means[strongest_ablation_name],
            },
        },
        "private_no_mask": private_no_mask,
        "deep_auxiliary": {
            "feature_export_case": deep_feature_export_case,
        },
    }


def _build_support_matrix(
    *,
    alignment_support: Mapping[str, object],
    causal_support: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage_name, payload in alignment_support["alignment_chain"].items():
        rows.append(
            {
                "family": "alignment",
                "variant": stage_name,
                "source": "projection_diagnostics_summary",
                "sample_count": payload["sample_count"],
                "mean_projection_cosine": payload["mean_projection_cosine"],
                "mean_projection_l2_gap": payload["mean_projection_l2_gap"],
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": None,
                "delta_mean_top_contribution_score": None,
                "event_mask_interference": None,
                "note": payload["threshold_verdict"],
            }
        )
    rows.append(
        {
            "family": "alignment",
            "variant": "feature_export_export",
            "source": "feature_export_run_manifest",
            "sample_count": alignment_support["feature_export_export"]["generated_view_count"],
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "event_mask_interference": None,
            "note": json.dumps(
                alignment_support["feature_export_export"]["view_verdict_counts"],
                ensure_ascii=False,
            ),
        }
    )
    rows.append(
        {
            "family": "causal",
            "variant": "g_min",
            "source": "causal_fusion_summary",
            "sample_count": causal_support["g_min"]["sample_count"],
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "mean_attention_entropy": causal_support["g_min"]["mean_attention_entropy"],
            "mean_top_event_score": causal_support["g_min"]["mean_top_event_score"],
            "mean_top_contribution_score": causal_support["g_min"]["mean_top_contribution_score"],
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "event_mask_interference": None,
            "note": "g_min_hidden_summary",
        }
    )
    for ablation_name, payload in causal_support["case_study"]["ablation_means"].items():
        rows.append(
            {
                "family": "causal",
                "variant": ablation_name,
                "source": "case_study_ablation_summary",
                "sample_count": causal_support["case_study"]["view_count"],
                "mean_projection_cosine": None,
                "mean_projection_l2_gap": None,
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": payload["delta_mean_top_event_score"],
                "delta_mean_top_contribution_score": payload["delta_mean_top_contribution_score"],
                "event_mask_interference": None,
                "note": "phase2_bundle_only",
            }
        )
    for variant_name, payload in causal_support["private_no_mask"]["matrix_rows"].items():
        rows.append(
            {
                "family": "causal",
                "variant": variant_name,
                "source": "private_benchmark_summary",
                "sample_count": None,
                "mean_projection_cosine": None,
                "mean_projection_l2_gap": None,
                "mean_attention_entropy": None,
                "mean_top_event_score": None,
                "mean_top_contribution_score": None,
                "delta_mean_top_event_score": None,
                "delta_mean_top_contribution_score": None,
                "event_mask_interference": None,
                "note": payload,
            }
        )
    return pd.DataFrame(rows)


def _build_main_ablation_matrix(
    *,
    alignment_support: Mapping[str, object],
    causal_support: Mapping[str, object],
) -> pd.DataFrame:
    case_study = causal_support["case_study"]
    baseline = case_study["baseline_view_means"]
    view_counts = case_study["view_verdict_counts"]
    pilot_delta_means = case_study["pilot_delta_means"]
    no_mask_tasks = causal_support["private_no_mask"]["tasks"]

    rows = [
        {
            "variant": "e_baseline",
            "source": "projection_diagnostics_summary",
            "sample_count": alignment_support["alignment_chain"]["e_baseline"]["sample_count"],
            "mean_projection_cosine": alignment_support["alignment_chain"]["e_baseline"]["mean_projection_cosine"],
            "mean_projection_l2_gap": alignment_support["alignment_chain"]["e_baseline"]["mean_projection_l2_gap"],
            "generated_view_count": 0,
            "pass_view_count": 0,
            "warn_view_count": 0,
            "contract_complete": False,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": None,
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "对齐预览存在",
            "limits": "不含稳定导出与因果解释",
        },
        {
            "variant": "f_full",
            "source": "projection_diagnostics_summary + feature_export_run_manifest",
            "sample_count": alignment_support["alignment_chain"]["f_full"]["sample_count"],
            "mean_projection_cosine": alignment_support["alignment_chain"]["f_full"]["mean_projection_cosine"],
            "mean_projection_l2_gap": alignment_support["alignment_chain"]["f_full"]["mean_projection_l2_gap"],
            "generated_view_count": alignment_support["feature_export_export"]["generated_view_count"],
            "pass_view_count": int(view_counts.get("PASS", 0)),
            "warn_view_count": int(view_counts.get("WARN", 0)),
            "contract_complete": True,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "稳定导出与双 pilot 可读性",
            "limits": "不直接给出因果 ablation 胜负",
        },
        {
            "variant": "g_min",
            "source": "causal_fusion_summary + phase2_case_study",
            "sample_count": causal_support["g_min"]["sample_count"],
            "mean_projection_cosine": baseline["mean_projection_cosine"],
            "mean_projection_l2_gap": baseline["mean_projection_l2_gap"],
            "generated_view_count": alignment_support["feature_export_export"]["generated_view_count"],
            "pass_view_count": int(view_counts.get("PASS", 0)),
            "warn_view_count": int(view_counts.get("WARN", 0)),
            "contract_complete": True,
            "mean_attention_entropy": baseline["mean_attention_entropy"],
            "mean_top_event_score": baseline["mean_top_event_score"],
            "mean_top_contribution_score": baseline["mean_top_contribution_score"],
            "delta_mean_attention_entropy": 0.0,
            "delta_mean_top_event_score": 0.0,
            "delta_mean_top_contribution_score": 0.0,
            "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
            "pilot_delta_mean_top_contribution_score": pilot_delta_means["delta_mean_top_contribution_score"],
            "private_t1_macro_f1": None,
            "private_t2_rmse": None,
            "private_t3_top1_accuracy": None,
            "supports": "非对称注意力与双 pilot 差异可读",
            "limits": "不等价于任务级 superiority",
        },
    ]
    for variant_name in ("no_event_bias", "vehicle_delta_suppressed"):
        payload = case_study["ablation_means"][variant_name]
        rows.append(
            {
                "variant": variant_name,
                "source": "phase2_case_study_bundle_only",
                "sample_count": case_study["view_count"],
                "mean_projection_cosine": baseline["mean_projection_cosine"],
                "mean_projection_l2_gap": baseline["mean_projection_l2_gap"],
                "generated_view_count": alignment_support["feature_export_export"]["generated_view_count"],
                "pass_view_count": int(view_counts.get("PASS", 0)),
                "warn_view_count": int(view_counts.get("WARN", 0)),
                "contract_complete": True,
                "mean_attention_entropy": payload["mean_attention_entropy"],
                "mean_top_event_score": payload["mean_top_event_score"],
                "mean_top_contribution_score": payload["mean_top_contribution_score"],
                "delta_mean_attention_entropy": payload["delta_mean_attention_entropy"],
                "delta_mean_top_event_score": payload["delta_mean_top_event_score"],
                "delta_mean_top_contribution_score": payload["delta_mean_top_contribution_score"],
                "pilot_delta_mean_projection_cosine": pilot_delta_means["delta_mean_projection_cosine"],
                "pilot_delta_mean_top_contribution_score": pilot_delta_means[
                    "delta_mean_top_contribution_score"
                ],
                "private_t1_macro_f1": None,
                "private_t2_rmse": None,
                "private_t3_top1_accuracy": None,
                "supports": "事件/机动敏感性可读",
                "limits": "仅是 frozen feature export view 上的 bundle-only 干预",
            }
        )
    rows.append(
        {
            "variant": "g_no_causal_mask",
            "source": "chronaris_opt_no_causal_mask_private_proxy",
            "sample_count": None,
            "mean_projection_cosine": None,
            "mean_projection_l2_gap": None,
            "generated_view_count": None,
            "pass_view_count": None,
            "warn_view_count": None,
            "contract_complete": None,
            "mean_attention_entropy": None,
            "mean_top_event_score": None,
            "mean_top_contribution_score": None,
            "delta_mean_attention_entropy": None,
            "delta_mean_top_event_score": None,
            "delta_mean_top_contribution_score": None,
            "pilot_delta_mean_projection_cosine": None,
            "pilot_delta_mean_top_contribution_score": None,
            "private_t1_macro_f1": no_mask_tasks["T1_maneuver_intensity_class"]["no_mask_metrics"].get(
                "macro_f1"
            ),
            "private_t2_rmse": no_mask_tasks["T2_next_window_physiology_response"][
                "no_mask_metrics"
            ].get("rmse"),
            "private_t3_top1_accuracy": no_mask_tasks[
                "T3_paired_pilot_window_retrieval"
            ]["no_mask_metrics"].get("top1_accuracy"),
            "supports": "去掉因果掩码后三任务同步退化",
            "limits": "当前来自私有 proxy，不直接等价于公开 benchmark",
        }
    )
    frame = pd.DataFrame(rows)
    variant_order = [
        "e_baseline",
        "f_full",
        "g_min",
        "g_no_causal_mask",
        "vehicle_delta_suppressed",
        "no_event_bias",
    ]
    frame["variant"] = pd.Categorical(
        frame["variant"],
        categories=variant_order,
        ordered=True,
    )
    return frame.sort_values("variant").reset_index(drop=True)


def _projection_summary(payload: Mapping[str, object]) -> dict[str, object]:
    summary = payload["summary"]
    threshold = payload["threshold_evaluation"]
    return {
        "sample_count": int(summary["sample_count"]),
        "mean_projection_cosine": float(summary["mean_projection_cosine"]),
        "mean_projection_l2_gap": float(summary["mean_projection_l2_gap"]),
        "threshold_verdict": str(threshold["verdict"]),
    }


def _frame_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    return float(frame[column].mean())


def _extract_private_no_mask_summary(payload: Mapping[str, object]) -> dict[str, object]:
    conclusion = payload["conclusion"]
    target_variant = conclusion["target_variant_name"]
    no_mask_variant = conclusion["no_mask_variant_name"]
    task_rows: dict[str, object] = {}
    matrix_rows = {
        target_variant: "target_variant",
        no_mask_variant: "no_causal_mask_variant",
    }
    for task_name, task_payload in payload["tasks"].items():
        target_metrics = _extract_variant_metrics(task_payload["variants"][target_variant])
        no_mask_metrics = _extract_variant_metrics(task_payload["variants"][no_mask_variant])
        if task_payload["task_type"] == "classification":
            target_text = (
                f"macro_f1={target_metrics['macro_f1']:.6f}, "
                f"balanced_accuracy={target_metrics['balanced_accuracy']:.6f}"
            )
            no_mask_text = (
                f"macro_f1={no_mask_metrics['macro_f1']:.6f}, "
                f"balanced_accuracy={no_mask_metrics['balanced_accuracy']:.6f}"
            )
            target_beats = float(target_metrics["macro_f1"]) > float(no_mask_metrics["macro_f1"])
        elif task_payload["task_type"] == "regression":
            target_text = (
                f"rmse={target_metrics['rmse']:.6f}, mae={target_metrics['mae']:.6f}"
            )
            no_mask_text = (
                f"rmse={no_mask_metrics['rmse']:.6f}, mae={no_mask_metrics['mae']:.6f}"
            )
            target_beats = float(target_metrics["rmse"]) < float(no_mask_metrics["rmse"])
        else:
            target_text = (
                f"top1_accuracy={target_metrics['top1_accuracy']:.6f}, "
                f"mrr={target_metrics['mrr']:.6f}"
            )
            no_mask_text = (
                f"top1_accuracy={no_mask_metrics['top1_accuracy']:.6f}, "
                f"mrr={no_mask_metrics['mrr']:.6f}"
            )
            target_beats = float(target_metrics["top1_accuracy"]) > float(
                no_mask_metrics["top1_accuracy"]
            )
        task_rows[task_name] = {
            "task_type": task_payload["task_type"],
            "target_metrics": dict(target_metrics),
            "no_mask_metrics": dict(no_mask_metrics),
            "target_metric_text": target_text,
            "no_mask_metric_text": no_mask_text,
            "target_beats_no_mask": bool(target_beats),
        }
    return {
        "target_variant_name": target_variant,
        "no_mask_variant_name": no_mask_variant,
        "criterion_details": dict(conclusion["criterion_details"]),
        "tasks": task_rows,
        "matrix_rows": matrix_rows,
    }


def _extract_deep_feature_export_case_summary(payload: Mapping[str, object]) -> dict[str, object]:
    dataset_payload = payload.get("datasets", {}).get("feature_export_case", {})
    if dataset_payload.get("status") != "completed":
        return {}
    rows: dict[str, object] = {}
    for model_name, model_payload in dataset_payload.get("models", {}).items():
        summary = model_payload["summary"]
        view_metrics = summary["view_metrics"]
        pilot_metrics = summary["pilot_metrics"]
        rows[model_name] = {
            "mean_event_mask_interference": float(
                pd.DataFrame(view_metrics)["event_mask_interference"].mean()
            ),
            "mean_attention_entropy": float(
                pd.DataFrame(view_metrics)["mean_attention_entropy"].mean()
            ),
            "pilot_delta_event_mask_interference": float(
                pilot_metrics[0]["delta_event_mask_interference"]
            )
            if pilot_metrics
            else 0.0,
        }
    return rows


def _extract_variant_metrics(payload: Mapping[str, object]) -> Mapping[str, object]:
    best_metrics = payload.get("best_metrics")
    if isinstance(best_metrics, Mapping):
        return best_metrics
    return payload


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_support_overview_plot(
    matrix: pd.DataFrame,
    *,
    path: Path,
) -> str | None:
    try:
        from matplotlib import pyplot as plt
    except Exception:
        return None

    alignment_rows = matrix.loc[
        matrix["variant"].astype(str).isin(["e_baseline", "f_full"])
    ].copy()
    intervention_rows = matrix.loc[
        matrix["variant"].astype(str).isin(["no_event_bias", "vehicle_delta_suppressed"])
    ].copy()
    no_mask_row = matrix.loc[matrix["variant"].astype(str) == "g_no_causal_mask"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    axes[0].bar(
        ["E baseline", "F(full)"],
        alignment_rows["mean_projection_cosine"].astype(float),
        color=["#5b8ff9", "#61dDAa"],
    )
    axes[0].set_title("Alignment cosine")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        list(intervention_rows["variant"].astype(str)),
        intervention_rows["delta_mean_top_contribution_score"].astype(float),
        color=["#f6bd16", "#e8684a"],
    )
    axes[1].axhline(0.0, color="#999999", linewidth=1.0)
    axes[1].set_title("Intervention delta top contribution")
    axes[1].tick_params(axis="x", rotation=15)

    if not no_mask_row.empty:
        row = no_mask_row.iloc[0]
        axes[2].bar(
            ["T1 macro-F1", "T3 top1"],
            [
                float(row["private_t1_macro_f1"]),
                float(row["private_t3_top1_accuracy"]),
            ],
            color=["#9270ca", "#269a99"],
        )
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_title("No-mask private proxy")
        axes[2].text(
            0.5,
            0.04,
            f"T2 RMSE={float(row['private_t2_rmse']):.2f}",
            ha="center",
            va="bottom",
            transform=axes[2].transAxes,
            fontsize=9,
        )
    else:
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)
