# 鼎新未来机动：安全滞后感知融合单折确认

状态：completed（单折单种子真实下游结果）。分支：`research/safe-lag-aware-fusion-20260718`。日期：2026-07-22。

## 目的

在鼎新真实数据上检验安全旁路是否减小未来机动任务的负迁移。这是评价协议 v2 晋级门禁 4/6 的直接证据（“至少一个双流核心任务超过全部融合基线”“机动任务安全退化满足容差”）。

## 设置

- 数据：鼎新冻结 snapshot（2 sortie），真实双流。
- 划分：留一架次（`leave_one_sortie_out__fold01`），训练 30 个机动上下文、留出 60 个。
- 训练：`train_common_pretext_method`，30 epoch、batch 4、seed 17，公共遮挡重构+短期预测+时延判别；三组 `chronaris_safe_lag`、`chronaris_multiscale`（旧）、`vehicle_only`。
- 评价：复用简化下游协议的机动 3 分类目标与相同 Ridge/Logistic 消费者（训练架次拟合、留出架次评价），macro-F1 为导师要求对应指标。
- 与已收口的简化下游确认同口径同消费者，可直接比较相对关系。

## 结果（fold01, seed 17）

| 方法 | macro-F1 | balanced accuracy | Spearman |
| --- | --- | --- | --- |
| Chronaris safe_lag | **0.3387** | 0.4333 | 0.4127 |
| Chronaris multiscale（旧融合） | 0.1667 | 0.3333 | 0.5471 |
| vehicle_only（航电单流） | 0.4821 | 0.5000 | 0.4554 |

## 判断

- **负迁移显著减小**：safe_lag 机动 macro-F1 `0.3387` 是旧 multiscale 融合 `0.1667` 的约 2.0 倍。旧融合在该航电强势任务上仍处灾难性低位（与已收口确认中 Chronaris `0.195` 一致），安全旁路（`vehicle_private` 直达输出、门控近回退）在真实任务上把航电信息保了回来。
- **尚未达到航电单流**：safe_lag 仍低于 vehicle_only `0.4821`。门禁 6（“争取接近或超过航电单流”）未满足——跨模态分支仍轻微干扰，或 30 epoch 未充分、单种子未稳定。
- 同口径同消费者内的相对比较有效；绝对值（vehicle 0.482）低于已收口确认（0.808），因本轮为单折单种子 30 epoch，非 50 epoch 双折三种子锁定确认。

## 边界与下一步

- 单折单种子 30 epoch；非锁定确认。晋级门禁 4（超过全部融合基线）在同口径内已满足（safe_lag > multiscale），但门禁 5/6（超单流、接近航电单流）未满足。
- 下一步杠杆：把 `lag_aware_alignment_loss` 与多统计量汇聚接入训练调度、延长至 50 epoch、三随机种子（17/29/43）、两折；目标让 safe_lag 接近或超过 vehicle_only。
- 重型 checkpoint 位于被忽略目录 `artifacts/application_evaluation/2026-07-22_dingxin-safe-lag-maneuver/`。
