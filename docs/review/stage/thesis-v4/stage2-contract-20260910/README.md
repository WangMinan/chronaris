# 阶段 2：共同下游评价可执行合同

本轮将用户确认的共同下游评价落实到数据与表示校验、已有下游拟合入口及真实开发样本验收。引用任务 `01a084c8-627f-7fe2-a726-5a0482e85bfc` 已重新读取；阶段 1 原文件与收据保留。

## 计划与判据

1. 固定任务含义、样本顺序、划分、目标、历史截止点及数据来源。
2. 各方法分别拟合同结构、同参数规则的下游模型；拟合与选择角色可追溯。
3. 声明窗口特征或逐时刻序列能力，拒绝把预测值或复制窗口向量作为序列表示。
4. 区分自监督、汇总标签适配与未来轨迹监督，保留外部预训练来源。
5. 使用现有方法完成鼎新和公开真实开发样本的表示导出、下游拟合、预测、指标与来源记录，验证恢复及合同篡改拒绝。
6. 测试、文档与阶段提交推送收口；旧全部长队列保持停止。

## 执行结果

已完成，详见[阶段报告](../../../../artifacts/runs/2026-09-10_v4-stage2-contract/report.md)和[可执行合同](../../../../requirements/common-downstream-executable-contract-v1.md)。三套真实开发数据完成 12 个评价单元、18 组下游模型，984 条预测保留有效性记录；最终完整显卡验收为 669 项通过、8 项跳过。

实现复用 `v4_grouped_consumers.py`，新增窗口特征容器和方法无关的共同合同；没有增加新的下游算法或依赖。测试覆盖不同表示分别拟合、训练标准化、窗口特征、样本和历史边界、目标变更、实际监督、检查点数据清单及主成分分析拟合角色。初次反例暴露的检查点数据清单缺口已修复。

最终合同加固后的重放复用原检查点与表示，仅重新拟合下游模型。132 份初次验收文件和原实验的 1,106 份文件均未改变。以下命令可重放这一核验；它校验原冻结提取源码、数据规则和旧模型结果散列，不修改旧目录或训练编码器。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/replay_common_downstream_contract.py \
  --source-root /mnt/e/chronaris-v4-results/2026-09-10-stage2-contract/attempt1 \
  --output-root /mnt/e/chronaris-v4-results/2026-09-10-stage2-contract/revalidated \
  --frozen-runner /mnt/e/chronaris-v4-results/2026-09-10-stage2-contract/source_before_hardening/common_downstream_smoke.py
```

从头执行当前源码的真实开发验收时，使用合同文档中的 `common-downstream-contract` 命令并指定新目录。`attempt1` 是初次神经导出来源，不能用后续修改的源码直接续跑该目录。
