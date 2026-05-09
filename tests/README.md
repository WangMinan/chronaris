# Tests

运行测试时默认显式使用 `chronaris` 解释器：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`

当前测试组织（域级 suite）：

1. `test_access_cli_overlap.py`
2. `test_access_metadata_live.py`
3. `test_dataset_pipeline.py`
4. `test_e0_input_pipeline.py`
5. `test_sortie_validation.py`
6. `test_alignment_data.py`
7. `test_alignment_model_losses.py`
8. `test_alignment_pipeline.py`
9. `test_alignment_diagnostics.py`
10. `test_stage_h_export.py`
11. `test_stage_i_pipeline.py`
12. `test_stage_i_case_study.py`
13. `test_stage_i_deep_pipeline.py`
14. `test_stage_i_private_optimization.py`
15. `test_stage_i_public_opt.py`
16. `test_stage_i_public_opt_aggregation.py`
17. `test_stage_i_support.py`

测试维护共性规则：

1. 新增测试优先并入现有域级 suite，避免回到碎片化单文件模式。
2. 默认把 `test_*.py` 规模控制在 `8-12` 个区间。
3. 涉及结构性合并后必须跑一次全量 discover，作为 merge 验证门槛。

建议后续至少覆盖：

1. 数据访问层的查询拼装与字段映射
2. 样本构建与时间对齐前处理
3. 特征导出格式
4. Stage H run/sortie/view manifest、feature bundle 读取与 partial-data sidecar
5. 指标计算与关键评测逻辑
6. Stage I task manifest、公开数据适配、session 级 baseline 评测
7. Stage I window contract 兼容读取、UAB window-level workload、NASA CSM attention-state 与 Phase 3 orchestration 落盘
8. Stage I Phase 2 case-study 资产装载、bundle-only 消融、WARN 解释、thesis runtime/demo 与关键工况 anchor 导出
9. Stage I deep sequence contract、真实 sortie smoke comparison 与 mini UAB/NASA comparison orchestration
    - 当前还覆盖 sequence preparation 的 `run.log / progress.json / processing_diagnostics.json`
10. Stage I 私有双流 all-window contract、proxy task 构造与 private benchmark orchestration
    - 基础 orchestration 回归集中在 `tests/test_stage_i_deep_pipeline.py`
    - `chronaris_opt / chronaris_opt_no_causal_mask`、target variant 判据与优化候选 artifacts 回归集中在 `tests/test_stage_i_private_optimization.py`
11. Stage I `chronaris public opt` prepared sequence frame、增强特征组、UAB subjective LOSO、NASA attention-state LOSO、artifact 落盘与有限值兜底
    - 当前最小回归集中在 `tests/test_stage_i_public_opt.py`
    - `session_mean_broadcast / session_median_broadcast` 聚合回归集中在 `tests/test_stage_i_public_opt_aggregation.py`
    - 当前还覆盖 torch UAB `session_mean_broadcast` 的 final prediction broadcast 回归
    - 当前还覆盖 `session_pooled_broadcast` 与 `physiology_only` / `physiology_scalar_only` torch 配置入口
    - 当前还覆盖 `feature_profile` 与 `ensemble_policy` 配置入口
    - 当前还覆盖 `StageIPublicOptTorchUABConfig` 的 `device=auto` CPU fallback 回归
    - 当前还覆盖长任务 `run.log / progress.json`、UAB torch `require_cuda` fail-fast、heat-only `heat_specialist`、UAB `sklearn uab_hybrid` CPU-heavy 显式开关、heat-only `selected_subsets`、fold-safe `target_prior_* / heat_prior_residual_guarded`，以及 NASA `label_leakage_guard` prepared asset 校验
    - 当前还覆盖 unified public mainline 从新 UAB robust-prior summary 与旧 summary 中做 best-of 选择
    - 若本机具备 CUDA，还额外覆盖 `device=cuda` synthetic smoke
12. Stage I `chronaris_public_fusion` 公共深模型入口与 fusion config 透传
    - 当前回归并入 `tests/test_stage_i_deep_pipeline.py`
    - 当前还覆盖 `public_fusion_screen` 的 GPU smoke 筛选入口与 `require_cuda` CPU fail-fast
13. Stage I 论文证据 support / ablation 聚合、固定 6 路径主矩阵与中文报告落盘
    - 当前回归集中在 `tests/test_stage_i_support.py`

当前鼎新私有主线最小回归命令：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_h_export tests.test_stage_i_deep_pipeline tests.test_stage_i_private_optimization`

当前 public opt 最小回归命令：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt`
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_public_opt_aggregation tests.test_stage_i_public_opt tests.test_stage_i_deep_pipeline`
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_support tests.test_stage_i_public_opt tests.test_stage_i_public_opt_aggregation`
- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline tests.test_stage_i_deep_pipeline tests.test_stage_i_public_opt tests.test_stage_i_public_opt_aggregation`

当前 thesis-facing 最小回归命令：

- `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_case_study tests.test_stage_i_support`

模型效果验证属于实验，不完全等价于单元测试。

`test_stage_i_deep_pipeline.py` 默认只跑 synthetic + repo 内 Stage H 真实资产；
如需开启本机 UAB/NASA live sequence 导出回归，显式设置：

- `CHRONARIS_ENABLE_STAGE_I_LIVE_SEQUENCE_TESTS=1`
