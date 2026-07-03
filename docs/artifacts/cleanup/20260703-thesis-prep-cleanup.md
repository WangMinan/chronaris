# 2026-07-03 P42 Thesis Prep Repository Cleanup

## 范围

本次 P42 在 P38 thesis protocol freeze 之后执行，只做仓库收敛、代码入口整理、产物瘦身、外置备份和验证闭环；未启动 P39/P40/P41/P43 新实验，未改写 P30-P38 已确认指标。

起始核对：

- 起始分支：`main`
- 起始 HEAD / `origin/main`：`756555c79627237800458cd8420a064441ab0147`
- 起始工作树：除本次 inventory 外无既有未提交改动
- P38 入口：`docs/artifacts/assets/stage_i_thesis_protocol/20260703T-stage-i-thesis-protocol-r1/`

## 审计输入

详细 inventory：`docs/artifacts/cleanup/20260703-thesis-prep-cleanup-inventory.md`

审计时主要体积：

| scope | before |
| --- | ---: |
| `docs` | 251M |
| `docs/artifacts` | 244M |
| `docs/artifacts/assets` | 242M |
| `.git` | 8.2G |
| `.git/lfs` | 8.0G |
| `.git/objects` | 120M |

审计时 Python 代码规模：

| scope | before |
| --- | ---: |
| `src scripts tests` Python files | 287 |
| `src scripts tests` Python lines | 86,365 |

## 代码收敛

本次只删除低风险兼容 wrapper 和误导性入口，不改变模型训练逻辑或指标计算：

- 删除 `private/third_party_comparison.py`、`scripts/stage_i/private/run_private_third_party_comparison.py`、`tests/test_stage_i_private_third_party_comparison.py`，保留 canonical `thirdparty` 命名。
- 删除 `private/stream_role_private_eval.py` 纯 re-export。
- 将 `scripts/stage_i/run_stream_role_fusion_eval.py` 移入 `scripts/stage_i/evidence/run_stream_role_fusion_eval.py`。
- 将 `public/gpu_runtime.py` 的实现移动到 `common/gpu_runtime.py`，public 训练模块改从 common 导入。
- 将旧 20260504 private benchmark 默认路径改为当前 `20260607T-stage-i-private-opt-package-r2`。

收敛后 Python 代码规模：

| scope | after | delta |
| --- | ---: | ---: |
| `src scripts tests` Python files | 282 | -5 |
| `src scripts tests` Python lines | 86,323 | -42 |

## 产物清理

外置备份目录：

- `/home/wangminan/projects/chronaris-local-artifacts/cleanup-20260703/`
- manifest：`/home/wangminan/projects/chronaris-local-artifacts/cleanup-20260703/backup_manifest.csv`

已备份并从 git 工作树删除：

- P37 `nested_private` / `nested_public` 下可再生成的 `training_curves.csv`、`run.log`、`gpu_perf_batches.csv`、`label_feature_overlap_audit.*`、逐候选 `deep_baseline_summary.json`。
- P35 `private_v3_confirm` / `public_v3_confirm` 下可再生成的 child run 明细、日志、GPU batch 表和重复 task manifest。
- P37 顶层未被 P38 引用的 `t3_retrieval_predictions.csv` 与重复 `training_curves.csv`。
- P34 顶层未被当前索引引用的 `task_manifest.jsonl` 与 `gpu_perf_batches.csv`。

备份结果：

| item | value |
| --- | ---: |
| backed up and deleted files | 67 |
| backed up bytes | 71,677,416 |
| backed up MiB | 68.36 |

当前体积：

| scope | after |
| --- | ---: |
| `docs` | 182M |
| `docs/artifacts` | 176M |
| `docs/artifacts/assets` | 174M |
| `stage_i_optimized_final_polish` | 8.7M |
| `stage_i_stream_role_fusion` | 12M |
| `stage_i_task_heads_optimization` | 5.4M |
| `stage_i_private_thirdparty_comparison` | 10M |

P38 路径完整性检查：

- `experiment_registry.csv` rows：8
- `result_matrix_long.csv` rows：505
- unique checked artifact paths：27
- missing paths：0

## Git/LFS 处理

本次不执行 `git filter-repo` 历史改写。理由：

- `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'` 显示 docs LFS objects 约 132 MB，当前不是明显异常级远端历史膨胀。
- 当前树删除了 67 个 tracked byproduct，并通过 `.gitignore` 阻止 nested byproduct、raw/prepared bundle、dense prediction 和 P34 未索引 batch/manifest 重新入仓。
- 已执行本地 `git lfs prune`，本机 `.git/lfs` 从 8.0G 降到 106M；这是本地缓存清理，不改写远端历史。

当前 Git 存储：

| scope | after |
| --- | ---: |
| `.git` | 295M |
| `.git/lfs` | 106M |
| `.git/objects` | 120M |
| packed git objects | 105.25 MiB |

## 验证

已完成：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m compileall src scripts tests`
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_private_thirdparty_comparison tests.test_stage_i_gpu_runtime tests.test_stream_role_fusion tests.test_stage_i_support`
- `PYTHONDONTWRITEBYTECODE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover tests`：`Ran 222 tests in 148.319s`，`OK (skipped=8)`
- CLI help smoke：
  - `scripts/stage_i/evidence/run_stream_role_fusion_eval.py --help`
  - `scripts/stage_i/private/run_private_thirdparty_comparison.py --help`
  - `scripts/stage_i/evidence/export_anchors.py --help`
  - `scripts/stage_i/public/run_public_fusion_gpuopt.py --help`
  - `scripts/stage_i/training/train_multitask.py --help`
  - `scripts/stage_i/runtime/run_smoke.py --help`
  - `scripts/stage_i/llm/run_preprocessing.py --help`
  - `scripts/stage_i/evidence/build_thesis_protocol.py --help`
  - `scripts/stage_i/public/run_public_fusion_ablation.py --help`
  - `scripts/stage_i/private/run_task_head_optimization.py --help`
- P38 registry/matrix path check：path columns `artifact_path / artifact_root / figure_path / primary_result_path / report_path`，unique paths `42`，missing paths `0`
- `git diff --check`：pass
- `git lfs fsck`：OK
- `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'`：docs LFS objects `132 MB`，`403/404` files

提交前仍需完成非验证性步骤：stage、commit、push 后远端同步核对。

## 保留边界

- P30/P31/P32/P34/P35/P36/P37/P38 confirmed metrics 未改写。
- P38 仍是论文协议冻结入口，不替代原始 source artifact。
- Public 分支仍写作 `public adapter / context-proxy evidence`。
- Private T1/T2/T3 仍写作 proxy/weak-label evidence，不写成人工真值。
- P39 synthetic 只能作为后续附录型 stress-test，不混入真实主结果。
