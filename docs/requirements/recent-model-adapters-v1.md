# 近期模型开发接入 v1

日期：2026-09-11。本文固定阶段 3 的接口和适配含义，承接[共同下游合同](common-downstream-executable-contract-v1.md)。这批小样本运行用于工程可行性，正式预算与方法比较在后续阶段固定。

## 官方来源与实际表示

下表说明各方法读取什么信息、导出什么表示，支撑共同下游入口的可核验性。输入均复用已有逐字段历史前向填充和训练角色中位数／四分位距归一化；缺失标记保留，窗口末端表示不宣称逐时刻在线能力。

| 方法 | 来源版本 | 真实表示与目标域训练 |
| --- | --- | --- |
| TimeCMA 语言模型增强方法 | [官方实现](https://github.com/ChenxiLiu-HNU/TimeCMA)，提交 `223e4ae9364bec3e3a2d8bb39ab6eed2cf510296`；[GPT-2 语言模型权重](https://huggingface.co/openai-community/gpt2)，修订 `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 官方解码器的通道隐藏状态，预测投影前提取并展开；使用已有汇总任务标签短训，移除任务头后接共同下游 |
| Chronos-2 预训练时序模型 | [官方实现](https://github.com/amazon-science/chronos-forecasting)，提交 `4dbf163c2734c089cdf7da2b86fde48862ff9c6f`；[模型权重](https://huggingface.co/amazon/chronos-2)，修订 `29ec3766d36d6f73f0696f85560a422f50e8498c` | 官方多变量编码器的每通道汇总标记，排除输出预测块；分别验收冻结权重与训练历史内部末 16 点预测适配 |
| SensorLLM 传感器—语言对齐变体 | [官方实现](https://github.com/cruiseresearchgroup/SensorLLM)，提交 `ddd17fecb508fcfa5ae156dbd37902e3ddbf652a`；[DeepSeek Llama 蒸馏语言模型权重](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)，修订 `6a6f4aa4197940add57724a7707d069478df56b1` | 官方多通道传感器标记插入路径，末个非填充语言隐藏状态；先做历史趋势文本对齐，再适配分类任务；作为明确更换语言骨干的变体验证 |

**TimeCMA 的语言输入只描述已观测窗口。** 每个字段的数值、缺失标记、相对秒数和趋势形成提示，使用 GPT-2 最后一个非填充分词的真实隐藏状态。与官方按日历时间和整数数值构造提示不同，本适配保留相对时间和三位有效数字，避免生理小数被整数截断；超出语言模型上下文时拒绝，不静默截断。缓存绑定提示、权重和输出散列，分别记录首次生成与复用耗时。

**Chronos-2 每个窗口单独组成多变量组。** 原生编码器可在窗口内跨通道交互，不跨样本、受试者或数据角色交互。训练历史内部后缀属于已有输入观测，不是未来业务目标，因而属于不使用下游标签的适配；外部时序预训练也单独披露。

**SensorLLM 变体用于核验开放权重的本地可执行性。** 原文的 Llama-3-8B 及较小 Llama-3.2-3B 权重访问均在当前环境受限，用户提出尝试 DeepSeek 或替代方法。DeepSeek 聊天 API（应用程序接口）不提供此链路需要的嵌入插入、隐藏状态和反向训练接口，因而采用官方本地 Llama 架构蒸馏权重。它与原论文骨干不同，结果不可写成原版 SensorLLM 的直接复现。[接口文档](https://api-docs.deepseek.com/api/create-chat-completion/)、[原始论文](https://aclanthology.org/2025.emnlp-main.19/)分别支撑这一接口判断与来源区分。

该变体的时序骨干为官方 `amazon/chronos-t5-large`，修订 `0e46c9c7e2e9f74b53db0617fdfcfe42a413e54a`；复用 SensorLLM 的标准化分箱、时间编码器和两隐藏层对齐网络。历史趋势文本是训练观测的确定性派生材料，不是专家或业务真值。阶段检查先用 CLARE 的六个原生生理通道；通道边界嵌入固定为官方均值初始化，只适配对齐层及随后移除的分类头。

## 执行与恢复

**新增入口沿用现有显卡锁、心跳、训练角色、任务损失和共同下游实现。** 不恢复历史完整长队列，不打开正式确认。短训批次按训练角色的任务有效性覆盖声明任务，不依据标签数值选择；检查点记录实际有效任务权重，零覆盖声明被拒绝。主适配预算各为两次更新，另执行一次从首更新检查点恢复的重放核验；SensorLLM 变体另有两次历史文本对齐更新。预算用于检查梯度、状态恢复和可消费表示，不表示训练充分。

以下命令在现有 `chronaris` 环境执行一个真实开发单元，实际资产清单须包含已下载文件的绝对路径、固定修订、权重散列及依赖版本。入口会核验清单中的全部文件，并将已安装 Chronos 源码与固定官方源码逐文件比较。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/evaluation/application_tasks/run_thesis_v4.py recent-model-smoke \
  --domain clare --recent-method timecma \
  --recent-model-assets /mnt/e/chronaris-v4-results/2026-09-11-stage3-models/assets.json \
  --output-root /mnt/e/chronaris-v4-results/2026-09-11-stage3-models/fresh
```

近期方法参数还支持 `chronos2` 和 `sensorllm_deepseek`；后者需要同时登记两套额外权重的资产清单。已有两个必做方法按鼎新、CogPilot 虚拟飞行数据及 CLARE 生理数据核验，样本来自阶段 2 同一选择函数。

相同目录恢复时重新核验合同、检查点与表示，复用已经完成的更新和下游模型。源码、数据、权重或依赖变更须使用新目录；失败记录带具体异常和时间，保留原现场。模型权重、大缓存和逐样本数组留在结果盘，仓库只保存实现、配置和紧凑证据。

## 复核边界

本轮检查参数量、实际获得梯度的参数量、首次语言缓存成本、适配耗时、统一计算设备架构（CUDA）记录的逻辑分配峰值、表示导出吞吐以及恢复一致性。新入口把显存缓存分配器限制为可见显存的 95%。Windows 子系统可能允许显存溢出到系统内存，因此逻辑分配峰值与物理显存使用需分开记录，不把可完成更新直接视为合理单卡成本。

最终可引用结果由[阶段 3 复核](../review/stage/thesis-v4/stage3-models-20260911/README.md)导航；小样本指标用于发现接口和退化问题，不用于挑选最终排名。
