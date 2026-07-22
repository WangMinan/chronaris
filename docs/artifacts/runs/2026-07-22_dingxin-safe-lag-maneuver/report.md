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

| 方法 | macro-F1 | balanced accuracy | 说明 |
| --- | --- | --- | --- |
| **Chronaris safe_lag** | **0.3387** | 0.4333 | 新主线 |
| Chronaris multiscale（旧融合） | 0.1667 | 0.3333 | 旧主干 |
| MulT | 0.1667 | 0.3333 | 融合基线 |
| ContiFormer | 0.1538 | 0.3000 | 融合基线 |
| vehicle_only（航电单流） | 0.4821 | 0.5000 | 单流参考 |

（safe_lag 的 Spearman 0.4127；multiscale 0.5471——连续分数排序与三类 macro-F1 不一致，三类 macro-F1 为导师要求对应主指标。）

## 判断

- **超过全部融合基线（晋级门禁 4 方向性满足）**：safe_lag 机动 macro-F1 `0.3387` 同时高于旧 Chronaris `0.1667`、MulT `0.1667`、ContiFormer `0.1538`。安全旁路（`vehicle_private` 直达输出、门控近回退）使新主线成为该航电强势任务上唯一显著减小负迁移的融合方法。
- **负迁移显著减小但未消除**：safe_lag 是旧融合与本轮全部融合基线的约 2 倍；旧融合仍处灾难性低位（与已收口确认中 Chronaris `0.195` 一致）。
- **尚未达到航电单流**：safe_lag `0.3387` 低于 vehicle_only `0.4821`。门禁 6（“争取接近或超过航电单流”）未满足——`vehicle_private` 仅占 64 维中的 24 维，而 vehicle_only 占全部 64 维；跨模态分支仍轻微干扰，30 epoch 单种子未充分。

## 边界与下一步

- 单折单种子 30 epoch；非锁定确认。晋级门禁 4（同口径超过全部融合基线 MulT/ContiFormer/旧 Chronaris）方向性满足；门禁 5/6（超单流、接近航电单流）未满足，需多种子稳定与进一步缩小与航电单流差距。
- 下一步杠杆：把 `lag_aware_alignment_loss` 与多统计量汇聚接入训练调度、延长至 50 epoch、三随机种子（17/29/43）、两折；目标让 safe_lag 接近或超过 vehicle_only。
- 重型 checkpoint 位于被忽略目录 `artifacts/application_evaluation/2026-07-22_dingxin-safe-lag-maneuver/`。

## 未来生理字段（双流增量检验，gate 5，fold01 seed17）

| 方法 | 标准化 RMSE | 相对持久性技能 | 正技能字段比 | EEG RMSE | SpO2 RMSE |
| --- | --- | --- | --- | --- | --- |
| Chronaris safe_lag | 10.845 | −2080.70 | 0.00 | 12.584 | **2.151** |
| Chronaris multiscale（旧） | 10.766 | −2014.61 | 0.00 | 12.272 | 3.235 |
| MulT | 10.607 | −1649.81 | 0.00 | 12.243 | 2.426 |
| ContiFormer | 8.702 | −2127.80 | 0.00 | 9.990 | 2.263 |
| vehicle_only | 8.673 | −2116.02 | 0.00 | 9.774 | 3.170 |
| physiology_only | 10.753 | −1614.51 | 0.00 | 12.566 | 1.687 |

判断：六方法的正技能字段比均为 `0`（均未超过持久性基线），与已收口确认一致——该任务整体困难，**gate 5（双流增量超最佳单流）未满足**。但 safe_lag 在 SpO2 字段上 RMSE `2.151` 为全部融合方法最低（优于旧融合 `3.235`、航电单流 `3.170`），显示安全旁路在部分生理字段上有边际改善；跨流增量尚未整体形成，需将 `lag_aware_alignment_loss` 接入训练并延长预算。
