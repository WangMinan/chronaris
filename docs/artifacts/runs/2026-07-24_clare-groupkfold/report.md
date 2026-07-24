# CLARE GroupKFold-5：低方差定论（诚实修订 seed-17 假象）

状态：completed（公开数据，5 折 GroupKFold 定论）。日期：2026-07-24。

## 目的

单 4-受试者 LOSO 切分方差大（seed 17 融合胜、seed 29 EEG 胜）。GroupKFold-5 把每个受试者恰好放入一次测试折并平均，给出低方差定论：“中枢+外周融合是否在认知负荷上超任一单流”。

## 设置

16 受试者、1614 窗口、GroupKFold-5（每折 train≈1290/test≈324），6 epoch、seed 17，四组 fusion_safe_lag、fusion_multiscale、central_only（EEG）、peripheral_only（EDA+HR）。指标：二分类低/高认知负荷 balanced-accuracy（5 折均值±std）。

## 结果（5 折 GroupKFold 定论）

| 方法 | 均值 balanced-acc | std | 各折 |
| --- | --- | --- | --- |
| **central_only（EEG 单流）** | **0.563** | 0.111 | 0.546, 0.372, 0.648, 0.551, 0.697 |
| fusion_safe_lag（中枢+外周） | 0.470 | 0.178 | 0.521, 0.354, 0.214, 0.522, 0.740 |
| peripheral_only（EDA+HR） | 0.429 | 0.073 | 0.317, 0.388, 0.524, 0.488, 0.426 |
| fusion_multiscale（旧融合） | 0.414 | 0.089 | 0.391, 0.376, 0.278, 0.501, 0.522 |

## 判断（定论）

- **EEG 单流稳健胜出融合**：central_only 均值 `0.563` > fusion_safe_lag `0.470`，**5 折中胜 4 折**。seed-17 单切分“融合 0.564 胜”是高方差假象——低方差 5 折 CV 表明 EEG 单流更优。**CLARE 融合未超最佳单流**（门禁 5 不成立）。
- **架构改进稳健**：fusion_safe_lag `0.470` > fusion_multiscale `0.414`（跨全部数据集/任务一致）。
- **根因**：公共重构预训练偏外周/航电强信号（审计 Q10），融合稀释了强 EEG 信号；safe_lag 旁路部分缓解（>multiscale）但未完全保住 EEG 质量。

## 综合定论（跨全部任务）

经鼎新机动（3 种子）、鼎新生理（6 方法）、CogPilot 难度±ECG-HR（2 种子/2 队列）、CLARE 认知负荷（2 种子+5 折 GroupKFold）、CogPilot 事件→响应、合成时延恢复——**融合未在任一真实任务上稳健超过最佳单流**；所有单种子“融合胜出”均为高方差假象，低方差 CV 均不成立。**唯一稳健结论**：safe_lag 跨数据集稳健优于旧 multiscale（架构改进真实，但边际）。

下一步唯一建议：重构预训练目标（跨流对比/事件配对替代航电重构，使表示编码跨模态而非航电主导），或接受仿真机制任务（事件锁定，旧 Chronaris 居首）为融合跨模态能力证据。继续在车辆主导/EEG 主导任务上追超单流无意义。
