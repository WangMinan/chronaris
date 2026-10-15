# 第六章写作交接包

**已满足第六章编写条件。** 融合系统在单机、归档与标准观测文件条件下完成最终实测语义配置的真实处理、恢复、性能和浏览器验收，具备需求、架构、功能、测试、案例与标准交付的逐节证据。实际分支为 `feat/chapter6-demo`；验收主体源码提交为 `6b53def8`，推理实现散列和环境见 `environment.json`。

本包服务于“第6章 航空人机时序融合系统的设计与实现”。[逐节映射](chapter-map.md)划分章节职责，[架构说明](architecture.md)给出模块输入输出和主要数据流，[模型来源](model-source.md)区分已有资产、本轮训练和方法适用条件。论文正文及Word由论文侧撰写。

## 阅读顺序

1. 本页最终验收结果与简短交接清单。
2. `chapter-map.md`、`architecture.md`：6.1—6.3的需求和实现依据。
3. `functional-tests.md`、`performance-method.md`及性能原始表：6.4.1—6.4.2的测试条件和结果。
4. `case.md`、同任务界面截图及 `feature-example/`：6.4.3案例输入、处理、查询和交付。
5. `model-source.md`、`model-inventory.json`、训练来源与资产登记：内部追溯及附录材料。

## 复现入口

以下命令使用本工作区既有解释器，应用推理固定为中央处理器单线程。最终资产目录为 `artifacts/chapter6-final/service`，来源登记及恢复说明见下方“资产取得”。

```bash
PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/runtime/run_fusion_workbench.py --data-root artifacts/chapter6-final/service --port 8766
```

启动后在浏览器打开 `http://127.0.0.1:8766`。选择实测观测记录和“连续对齐与事件语义融合”模型，选择窗口，生成特征；随后查看结果分析、特征交付，下载标准包并生成汇总特征。数据接入窗口支持归档时段及标准文件导入追加；模型与字段页展示实际模型配置及已有字段解释。

完整验收入口如下。首次准备使用一个新的产物目录和完成训练的检查点；已有目录可省略 `--checkpoint` 重跑验收，新记录会保留原任务历史。

```bash
PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/runtime/verify_chapter6.py \
  --checkpoint artifacts/chapter6-final/training/chronaris/C/best.pt \
  --root artifacts/chapter6-final/recheck \
  --evidence artifacts/chapter6-final/recheck-evidence
```

[独立重绘包](redraw/README.md)已在仓库外实际运行，含高分辨率PNG和六份可编辑SVG；标准特征示例可直接用 `read_features.py` 回读，性能汇总可由 `summarize_performance.py` 使用标准库重新计算。大型模型和原始快照通过登记位置取得，小型来源、数值证据和截图随Git交付。


## 最终验收

**最终模型实际启用了经过训练的事件语义分支。** 本轮复用原实测种子17权重及训练统计量，在CPU上完成300次补充更新，用时3301.25秒，选中第300次权重。检查点散列为 `7912efbad9a42391f26880d7da067afc1a031a30015cd45576c5e50dbfd5e592`。严格加载、语义参与和历史信息边界分别实测，模型清单中同时保留旧基础模型。该模型属于本章独立应用配置，既有第五章成绩不变。

下表汇总本轮实际结果，完整原始证据保留在对应文件中。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 定向代码测试 | 59项通过，3.53秒 | `tests.log` |
| 真实三窗口与标准回读 | 3×96×64；288个有效参考位置；CSV逐值一致 | `acceptance.json`、`feature-example/` |
| 批量与逐窗口 | 最大绝对差8.34465×10⁻⁷，保持原绝对/相对容差10⁻⁵ | `acceptance.json` |
| 语义参与 | 查询参数非零；末端12个有效事件；关闭语义分支后输出最大差0.895749 | `mechanisms.json` |
| 未来观测扰动 | 15秒及此前表示最大差0 | `mechanisms.json` |
| 真实进程中断恢复 | 15窗口任务重启后复用1个完整片段并完成，原失败任务保留 | `recovery.json` |
| 真实输入错误 | 重复去重；冲突、字段顺序、时间、掩码和非有限值均按预期拒绝 | `http-errors.json` |
| 浏览器功能 | 20项通过，页面脚本异常0；10＋5及1＋2窗口增量通过 | `browser-check.json` |
| 页面响应 | 首次就绪1391.09毫秒，结果切换63.32毫秒；有界单次交互样本 | `browser-check.json` |
| 截图与独立重绘 | 五张实际截图目视通过；仓库外重绘通过，PNG为2倍像素密度 | `ui-evidence.json`、`redraw/` |

**最终配置的运行成本已按1、3、15窗口各测量三次。** 硬件为AMD Ryzen 9 9900X中央处理器、约30.6 GiB系统内存，WSL2环境，Python 3.11.15、PyTorch 2.11.0，推理线程1。下表采用完整管线及提交到汇总交付的平均值；内存列为该规模测量期间服务进程峰值的最大值，计时口径见 `performance-method.md`。

| 窗口数 | 完整管线平均时间（秒） | 接口交付平均时间（秒） | 平均吞吐量（窗口/秒） | 服务进程峰值内存（MiB） |
| --- | --- | --- | --- | --- |
| 1 | 0.7323 | 0.7790 | 1.3716 | 765.92 |
| 3 | 1.1438 | 1.1944 | 2.6255 | 790.59 |
| 15 | 3.6518 | 3.8111 | 4.1108 | 888.18 |

原始九次记录为 `performance.csv` 与 `performance.json`，可重算汇总为 `performance-summary.csv`。原始结果未筛选；首次任务与后续重复、阶段计时及进程内存均保留。

## 资产取得

**小型论文证据随Git交付，运行资产已另行归档并逐文件核对。** 数值记录、五张截图、完整显示数据、标准特征示例、重绘输入和脚本均在本目录；大模型与观测文件位于：

- WSL：`/home/wangminan/chronaris-data/chapter6-delivery-20261015/application-assets.tar.gz`
- Windows：`\\wsl.localhost\Ubuntu-2204\home\wangminan\chronaris-data\chapter6-delivery-20261015\application-assets.tar.gz`
- 散列及包内全部文件清单：[assets.json](assets.json)。归档目录另有 `SHA256SUMS`。

保留现有产物时不要向其覆盖解压；新的检出可在空的 `artifacts/chapter6-final` 中恢复：

```bash
mkdir -p artifacts/chapter6-final
tar -xzf /home/wangminan/chronaris-data/chapter6-delivery-20261015/application-assets.tar.gz \
  -C artifacts/chapter6-final
```

包内包含训练来源、完成权重、登记模型、已接入观测、任务片段和结果、导入追加示例及运行日志。归档时段的新接入继续读取原独立快照，位置也在 `assets.json` 登记；已接入案例与标准文件使用包内数据。原快照及冻结实验模型未改写。

## 论文侧接手清单

- 6.1—6.3：按 `chapter-map.md` 与 `architecture.md` 写需求、架构和四项核心功能。
- 6.4.1：按 `functional-tests.md` 及最终验收表整理功能/接口测试。
- 6.4.2：由性能原始表和复算汇总制表，保留CPU单线程、规模和计时边界。
- 6.4.3：采用 `case.md` 的90—130秒实测案例，引用同任务四视图、实际输出和回读结果。
- 附录：模型配置、训练来源、资产位置和复现命令取自本包；论文正文及Word继续由论文侧编写。

当前实测字段的运动学残差关系集合为空，按第三章规定的字段适用条件处理；本轮验证的是该条件下连续对齐与完整事件语义配置的应用。性能证据覆盖1至15窗口，窗口增量保持独立上下文。最大容量、多用户并发、在线数据库和在线语言模型均未列为本轮已验证能力。
