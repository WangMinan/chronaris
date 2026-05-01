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
8. Stage I Phase 2 case-study 资产装载、bundle-only 消融与 WARN 解释
9. Stage I deep sequence contract、真实 sortie smoke comparison 与 mini UAB/NASA comparison orchestration

模型效果验证属于实验，不完全等价于单元测试。

`test_stage_i_deep_pipeline.py` 默认只跑 synthetic + repo 内 Stage H 真实资产；
如需开启本机 UAB/NASA live sequence 导出回归，显式设置：

- `CHRONARIS_ENABLE_STAGE_I_LIVE_SEQUENCE_TESTS=1`
