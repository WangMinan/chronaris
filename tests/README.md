# Tests

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

测试维护共性规则：

1. 新增测试优先并入现有域级 suite，避免回到碎片化单文件模式。
2. 默认把 `test_*.py` 规模控制在 `8-12` 个区间。
3. 涉及结构性合并后必须跑一次全量 discover，作为 merge 验证门槛。

建议后续至少覆盖：

1. 数据访问层的查询拼装与字段映射
2. 样本构建与时间对齐前处理
3. 特征导出格式
4. 指标计算与关键评测逻辑

模型效果验证属于实验，不完全等价于单元测试。
