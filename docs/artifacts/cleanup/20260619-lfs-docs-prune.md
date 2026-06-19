# Docs LFS prune 2026-06-19

## 目的

本次清理面向 Git LFS 下载配额，移除 `docs/` 下已经由报告、summary、manifest 或 schema contract 覆盖的重型原始/中间载荷。清理后保留当前中期报告入口、工程报告、关键 JSON summary、图表和小型 CSV 表。

GitHub 文档建议用 `git filter-repo` 从历史中移除不再需要的 LFS 文件；同时说明远端 LFS 对象即使从历史移除后仍可能继续计入 storage，彻底清远端对象通常需要删除/重建仓库或联系 GitHub Support。

## 清理原则

- 保留 `docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/artifacts/ARTIFACTS.md`、`docs/midterm/` 中的当前事实入口。
- 保留报告、summary、schema contract、figure/table manifest 和可直接写入中期材料的小型图表。
- 删除可由脚本重新生成、且主要用于历史 replay 或中间 prepared dataset 的重型 raw payload。
- 不把被删 raw payload 继续写成当前可打开产物；需要复跑时按对应历史脚本重新生成。

## 已清理重型文件

| 类别 | 路径/范围 | 近似大小 | 保留证据 |
| --- | --- | ---: | --- |
| Stage H vehicle-only partial raw | `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/vehicle_only_feature_bundle.npz` | 491M | `stage-h-closure-2026-04-27.md` 中的 shape/count 记录 |
| Stage H vehicle-only partial manifest | `docs/artifacts/assets/stage_h/20260427T000000Z-stage-h-closure/partial_data/vehicle_only_window_manifest.jsonl` | 17M | `partial_data_manifest.jsonl` 与 closure 报告 |
| Stage I runtime replay r1 raw samples | `docs/artifacts/assets/stage_i_runtime_inference/20260607T-stage-i-runtime-replay-r1/runtime_samples.jsonl` | 412M | `runtime_inference_summary.json`、predictions CSV、runtime replay 报告 |
| Stage I runtime service r1 raw input | `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r1/input_view_runtime_samples.jsonl` | 138M | r1 summary、runtime inference summary、system figures |
| Runtime schema contract canonical raw payload | `docs/artifacts/assets/stage_i_runtime_service/20260613T-stage-i-runtime-service-smoke-r2-contract/canonical_runtime_samples.jsonl` | 271M | `runtime_schema_contract.json`、canonical runtime summary、r2 contract report |
| Public adapter prepared UAB raw bundle | `docs/artifacts/assets/stage_i_public_opt/20260504T161500Z-stage-i-public-opt-uab-prepared/{task_manifest.jsonl,sequence_bundle.npz}` | 142M | downstream public opt summaries/reports |
| Public adapter prepared NASA raw bundles | `docs/artifacts/assets/stage_i_public_opt/20260506T124000Z-stage-i-public-opt-nasa-prepared/{task_manifest.jsonl,sequence_bundle.npz}` and `20260508T080318Z-stage-i-public-opt-nasa-prepared-v2/{task_manifest.jsonl,sequence_bundle.npz}` | 238M | downstream NASA public opt summaries/reports |
| Deep comparison prepared sequences | `docs/artifacts/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/{uab_sequences,nasa_sequences}/{task_manifest.jsonl,sequence_bundle.npz}` | 230M | probe/full LOSO comparison summaries and reports |
| Stage I phase3 closure heavy tables | `docs/artifacts/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/{uab_window,nasa_attention}/{fold_predictions.csv,task_manifest.jsonl,feature_table.parquet}` | 188M | closure summary, metric JSON, plots, Stage I closure report |

合计从当前 `docs/` 工作树移除约 `2.1G` 重型产物；清理后 `docs/` 约 `335M`。

## 历史处理

本文件记录当前工作树清理范围。若执行 Git 历史清理，应使用同一批路径作为 `git filter-repo --invert-paths` 的 path 清单，然后 `git push --force-with-lease origin main`。历史过滤完成后，当前 HEAD 不应再引用上述 LFS 对象；远端 LFS storage 是否立即下降以 GitHub 计费与支持策略为准。
