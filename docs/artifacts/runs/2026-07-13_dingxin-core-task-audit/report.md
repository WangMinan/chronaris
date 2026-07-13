# 鼎新核心任务审计

本审计只使用每个外层折的 inner-train 再划分数据，用于锁定机动任务定义和评估任务可学习性；外层测试未读取。

## 任务决定

- 锁定的机动任务：`current_5s`。
- 有效外层折：3。
- 未来任务是否晋升：False。

## 审计范围

- 机动分类诊断记录：72 条。
- 生理响应诊断记录：36 条。
- 比较 5 秒与 30 秒历史、生理单流、航电单流和双流直接观测。
- 这些数值是训练内诊断，不进入论文确认主表。

## 下一步

使用锁定任务定义运行冻结消费者筛选；只有开发门禁通过后才允许训练任务感知 Chronaris。

协议配置：`{"compact_output_root": "docs/artifacts/runs", "context_catalog_path": "docs/artifacts/runs/2026-07-11_dingxin-context-bindings/context_catalog.csv", "e_run_manifest_path": "docs/artifacts/runs/2026-05-02_feature-export-e-allwindow-clean/run_manifest.json", "f_run_manifest_path": "docs/artifacts/runs/2026-05-02_feature-export-f-allwindow-clean/run_manifest.json", "fixed_audit_root": "docs/artifacts/runs/2026-07-10_fixed-data-audit", "inner_split_root": "docs/artifacts/runs/2026-07-11_dingxin-inner-splits", "random_state": 17, "run_id": "2026-07-13_dingxin-core-task-audit", "snapshot_root": "artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot"}`
