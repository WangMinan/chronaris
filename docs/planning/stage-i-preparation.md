结论先说：阶段 I 最容易落地的版本，应该改成“公开标注数据集上的认知负荷量化实验 + 现有实飞 Stage H 资产上的小样本案例复盘”。不要把阶段 I 设计成“必须拿到更多鼎新实飞数据和专家细粒度标签”的方案。那条路现在不可控，而且会拖死毕设。

这不违背选题报告。选题报告的核心不是“必须在真实军机数据上训练一个完整下游模型”，而是证明一条从异构时序到标准化融合特征、再到下游任务验证的管线。我们现在可以把“真实实飞数据”用于证明 Chronaris 管线能处理真实人机异构数据，把“公开
数据集”用于完成可复现、有标签、可量化的下游实验。

推荐落地版本
主任务定为：认知负荷评估。
副任务定为：注意状态/失能风险代理分析。这里不要直接宣称“真实 G-LOC 预测”，因为公开数据和我们现有实飞数据都缺少可靠 G-LOC 标签。论文里可以写成“空中失能风险相关的异常注意状态与生理风险代理识别”。

阶段 I 的推荐收口方式
不建议把阶段 I 做成无限扩展的“看到一个公开数据集就继续加一个”的阶段。为了让阶段边界可收口，阶段 I 最终按 `4` 个 phase 推进：

1. `Phase 0`：数据 contract、环境和评测入口固化
2. `Phase 1`：UAB 主数据集双轨 baseline（客观分类 + 主观回归）
3. `Phase 2`：Stage H 真实双流资产 case study
4. `Phase 3`：NASA CSM 第二公开数据集接入、对比/消融补齐并完成阶段 I 收口

其中：

- `Phase 0 + Phase 1` 已经是第一轮可运行闭环
- `Phase 2 + Phase 3` 是阶段 I 最终收口所需的剩余主线
- MATB-II、DS007262、Braindecode、EEGNet、MulT、ContiFormer 等更重扩展不计入阶段 I 主线完成条件，默认放到阶段 I 收口后再评估

数据分成四层：

1. Stage H 现有实飞资产
    使用 docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/run_manifest.json，通过 src/chronaris/features/stage_h_bundle.py:1 读取 3 个真实双流 view。它只做案例复盘、注意力解释、掩码干扰实验，不拿来做大规模监督训练。
2. UAB Flight-Deck Workload 数据集，作为第一主数据集
    这是我最建议优先做的。它包含 EEG/ECG，任务包括 N-back、serious game 和 Airbus320 沉浸式飞行模拟，且提供客观和主观 workload ground truth；UAB 页面也说明 zip 约 966.7 MB，数据可公开获取并带 Creative Commons 使用说明。来源：UAB
    DDD 数据集页面 https://ddd.uab.cat/record/259591
3. NASA Flight Crew Physiological Data，作为第二主数据集或扩展实验
    它是飞行员/副驾驶生理数据，包含 simulator LOFT 全飞行测试，并有 CA、DA、SS 等注意状态标签，采样率 256 Hz。它很贴近“飞行员状态监测”，但页面显示 license not specified，所以论文发布和代码复现前要单独确认使用边界。来源：NASA Op
    en Data Portal https://data.nasa.gov/dataset/flight-crew-physiological-data-for-crew-state-monitoring
4. 补充/预训练数据集
    MATB-II workload 数据：15 名被试、3 个负荷等级、62 通道 EEG，适合做跨 session workload 验证。来源：https://zenodo.org/records/4917218
    PhysioNet EEGMAT：mental arithmetic 前/中 EEG，175.1 MB，适合生理编码器预训练或二分类验证。来源：https://www.physionet.org/content/eegmat/
    EEGDash/OpenNeuro DS007262：8 级 arithmetic cognitive workload，18 subjects、24 channels、250 Hz、CC0。来源：https://eegdash.org/api/dataset/eegdash.dataset.DS007262.html

自己构建数据集的正确含义
不是重新采集数据，也不是伪造飞行标签。我们要构建一个 Chronaris-compatible Stage I Benchmark：

- 对公开数据，统一转换成 StageITaskManifest.jsonl
- 每条样本包含：dataset_id、subject_id、window_start、window_end、physiology_path、context_stream、label、label_source、split_group
- 对 UAB/NASA/MATB，context_stream 来自任务难度、飞行模拟事件、注意诱导事件，而不是飞机总线
- 对我们自己的 Stage H，context_stream 来自真实 BUS/causal attention/event scores，但标签只能是弱标签或案例标签
- 公开数据负责“可量化指标”，真实数据负责“项目场景可信度”

这在论文里很好解释：公开数据提供可复现监督评价，专有实飞数据提供真实航空异构管线验证。

算法选择
最容易落地的算法不要一上来就复现 ContiFormer 或 MulT 全量仓库。按三档做：

1. 第一档：传统强基线
    用 EEG/ECG 手工特征加 LogisticRegression / RandomForest / SVM / XGBoost，再加 sktime 的时序分类器。sktime 是统一时间序列机器学习框架，支持 classification/regression/clustering：https://www.sktime.org/
2. 第二档：生理深度学习基线
    用 MNE 读 EDF/BIDS/EEG 数据，用 Braindecode 跑 EEGNet、Deep4Net、ShallowFBCSPNet 这类 EEG 模型。Braindecode 明确是 EEG/ECG/MEG 深度学习解码工具箱：https://github.com/braindecode/braindecode
3. 第三档：Chronaris 自己的方法
    冻结 Stage H/公开数据适配后的融合特征，训练一个轻量 MLP 或 temporal pooling classifier。重点不是模型多复杂，而是对比：
    - 原始生理特征
    - 简单插值/拼接特征
    - Chronaris 连续对齐特征
    - Chronaris + 非对称因果融合特征

MulT 和 ContiFormer 保留为论文对照方向，不作为第一轮必须落地。MulT 有官方 PyTorch 实现，适合做“非对称掩码 vs 通用跨模态注意力”的对照：https://github.com/yaohungt/Multimodal-Transformer 。ContiFormer 适合做连续时间 Transformer
对照，但实现和依赖更重，建议放在第二轮：https://seqml.github.io/contiformer/

代码实施方案
我建议阶段 I 这样写：

- 运行上述 `scripts/*.py` 时默认显式使用 `chronaris` 解释器：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`

- src/chronaris/stage_i/contracts.py：定义 StageITaskSample、StageIDatasetManifest、label schema
- src/chronaris/stage_i/adapters/uab_workload.py：UAB 数据转统一 manifest
- src/chronaris/stage_i/adapters/nasa_csm.py：NASA 数据转统一 manifest
- src/chronaris/stage_i/adapters/stage_h_case.py：读取 load_stage_h_feature_run()
- src/chronaris/stage_i/features.py：EEG/ECG 统计特征、频带特征、窗口聚合
- src/chronaris/stage_i/baselines.py：传统模型和轻量 MLP
- src/chronaris/stage_i/evaluation.py：balanced accuracy、macro-F1、AUROC、混淆矩阵、subject-wise split
- scripts/prepare_stage_i_dataset.py
- scripts/run_stage_i_baseline.py
- scripts/run_stage_i_case_study.py
- docs/planning/stage-i-data-plan-2026-04-28.md

进入阶段 I 前你真正需要准备的东西
你不需要再等鼎新或专家。你需要准备的是：

1. 在仓库外建数据目录，比如 /home/wangminan/data/chronaris-public/
2. 先下载 UAB workload 数据集
3. 再决定是否下载 NASA CSM；如果用它，需要在报告里标注 license 未明确
4. 接受阶段 I 的主任务改成“认知负荷评估”，失能只做“风险代理/异常注意状态”
5. 不把 20251110... vehicle-only 数据说成双流融合样本，它只能做单流补充诊断

我建议下一步实际开工顺序是：先写 Stage I 数据计划文档和 manifest contract，然后接 UAB 数据集，跑一个传统 baseline。只要 UAB 能跑出 subject-wise macro-F1、混淆矩阵和消融表，阶段 I 就有了可写进论文的主实验骨架。随后再把 Stage H
真实三 view 加进去做案例复盘，整个毕设就既可落地，又没有偏离“航空人机异构时序融合”的初衷。
