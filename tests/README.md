# 测试入口

当前测试覆盖数据读取与样本分组、融合表示、训练与模型、仿真、应用评价、特征导出、证据材料和运行接口。按实际目录定位：`dataset/`、`representation/`、`modeling/`、`simulation/`、`evaluation/`、`feature_export/`、`evidence/`、`runtime/`、`llm_preprocessing/`；根目录还保留访问层、早期模型与命令入口测试。

完整验证使用项目环境：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q
```

现场数据库或数据测试可能按条件跳过，应记录实际跳过原因；代码错误应修复，不能用聚焦测试替代失败的完整验证。单元测试通过与已训练模型通过研究机制门是两项独立结论。
