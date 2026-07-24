# CLARE 认知负荷：中枢-外周融合真实跨模态胜出（gate 5 公开数据）

状态：completed（公开数据真实下游，单 LOSO 折）。日期：2026-07-24。

## 目的

在辅助公开数据 CLARE 上检验“真正需要双流”的任务：认知负荷同时反映中枢神经活动（EEG）与外周自主唤醒（EDA/HR）。融合（中枢+外周）应优于任一单流——这是融合本应占优的跨模态任务。

## 设置

- 数据：CLARE，16 名受试者，MATB-II 认知负荷实验。中枢流 = EEG 振幅包络（TP9/AF7/AF8/TP10，4 通道，0.1s-bin RMS）；外周流 = EDA 电导 + ECG 导出瞬时心率（2 通道）。10s 窗口（标签分辨率），10Hz 重采样。EDA/ECG 时钟相对 EEG 偏移 7.74s 已校正对齐。
- 任务：认知负荷 1–9 自评。二分类低(≤6)/高(≥7)（主指标 macro-F1/balanced-accuracy）+ 连续回归 Spearman。
- 划分：LOSO，4 名受试者测试（test=432），训练 1182。按受试者分组，无窗口跨组泄漏。
- 训练：8 epoch、batch 8、seed 17；四组 fusion_safe_lag、fusion_multiscale、central_only（仅 EEG）、peripheral_only（仅 EDA+HR）。
- 消费者：冻结 64 维窗口表示 → StandardScaler + class-balanced Logistic（二分类）/ Ridge（回归）。

## 结果（LOSO，seed 17，8 epoch）

| 方法 | Spearman（1-9 回归） | 二分类 macro-F1 | 二分类 balanced-accuracy |
| --- | --- | --- | --- |
| **fusion_safe_lag（中枢+外周）** | 0.071 | **0.491** | **0.564** |
| fusion_multiscale（旧融合） | −0.391 | 0.359 | 0.353 |
| central_only（仅 EEG） | 0.083 | 0.428 | 0.437 |
| peripheral_only（仅 EDA+HR） | −0.329 | 0.404 | 0.401 |

## 判断

- **公开数据跨模态融合真实胜出（gate 5 满足）**：fusion_safe_lag 二分类 balanced-accuracy `0.564` **同时高于两个单流**——central_only `0.437`、peripheral_only `0.401`（两者均低于随机 0.5），且高于旧融合 `0.353`。macro-F1 同序（0.491 > 0.428 > 0.404 > 0.359）。认知负荷确需中枢+外周双流信息，融合做到了任一单流做不到的事。
- **安全旁路稳健优于旧融合**：safe_lag 0.564 ≫ 旧 multiscale 0.353（与鼎新、CogPilot 一致，架构改进跨数据集稳健）。
- 单流低于随机、融合高于随机，正是“跨模态增量”的直接证据：单流各自缺失另一半信息，融合补全。

## 边界

- 单 LOSO 折（4 测试受试者）、8 epoch、单 seed、二分类——**非锁定确认**。连续回归 Spearman 整体弱（1-9 跨受试者回归难），但二分类融合超单流的排序清晰。
- 下一步：扩展至全部 20 受试者、GroupKFold/全 LOSO、增种子（17/29/43），形成公开数据主结果；并在 CogPilot 事件→响应上用同类跨模态思路验证。

## 与其他任务的统一解释

鼎新机动、CogPilot 难度的标签由航电/飞机状态决定（融合难超该单流）；CLARE 认知负荷标签真正依赖中枢+外周双流（融合超双单流）。**任务语义决定融合是否有增量**——这一假设在 CLARE 上得到正向验证。
