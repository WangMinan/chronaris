# 固定配置复核：独立绘图包

本目录包含 36 个模型单元、72 条表示路线的完整主线性与 MiniROCKET 时序卷积特征补充结果。所有图由随包表格生成，不需要模型、数据库、GPU 或 `chronaris` Python 包。研究结论见[报告](report.md)。

## 独立重绘与手工编辑

仅需 Python 3.10 以上与 Matplotlib 绘图库。将整个目录复制到任意位置后执行：

```bash
python -m pip install -r requirements.txt
python plot.py
python verify.py
```

`plot.py` 默认相对自身位置读取 `data/` 和 `style.json`，输出到 `figures/`。Linux 可安装文泉驿正黑或 Noto CJK 中文字体；其他系统可在 `style.json` 的 `fonts` 中指定现有中文字体。字体无需依赖仓库路径。

可以直接编辑样式配置中的字体、配色、字号、图幅、分辨率及中文标签。数字修订建议先复制 `data/` 到 `revisions/<日期>/data/`，记录修改原因，然后显式选择版本：

```bash
python plot.py --data-dir revisions/2026-10-01/data --style style.json --output-dir revised_figures
```

绘图脚本只使用已有的指标和汇总值，不重新计算均值、标准差或实验指标。如修订逐设置数据并需修改汇总值，应同时更新 `summary.csv`，使散点与均值/误差线相符。`verify.py` 检查原始导出快照的散列及汇总关系，不用于阻止有记录的展示修订；冻结实验文件不随展示编辑改变。

## 数据与图片对应关系

下表给出完整的重绘依赖；其余数据支持检查汇总、研究结论和成本，后续画新图也可直接使用。

| 图片或数据 | 内容与依赖 |
| --- | --- |
| `figures/linear_comparison.png`、同名 SVG | `data/setting_metrics.csv` 中线性下游行、`data/summary.csv` 对应汇总、`style.json`；渲染入口 `plot.py` |
| `figures/minirocket_comparison.png`、同名 SVG | 相同两个表中的时序补充行及 `style.json`；渲染入口 `plot.py` |
| `data/setting_metrics.csv` | 288 行；域、方法、路线、下游、任务、划分、种子及该设置的分组指标 |
| `data/summary.csv` | 32 行；九个设置的等权均值、总体标准差、最小及最大值；绘图直接使用这些数值 |
| `data/group_metrics.csv` | 528 行；各受试者、任务和字段的指标、样本数、无观测比例及回归尾部误差诊断 |
| `data/cost_units.csv` | 36 行；秒为单位，本轮成本与历史成本分列，并标明完整训练、部分继承或仅下游重验证 |
| `data/training.csv` | 72 行；所选模型位置、已完成更新及本轮新增更新；其中大模型文件本身不用于绘图 |

分类指标为宏平均 F1，越高越好；回归指标为均方根误差，越低越好，两个数据域的原生单位不同，不跨域合并。`fold_index` 从 0 开始；`seed` 是随机种子。`consumer` 的 `linear` 表示共同线性下游，`minirocket` 表示时序卷积特征补充。`route` 的 `self_supervised` 与 `task_guided` 分别为自监督及任务引导。绘图标签在样式文件中映射为中文含义。

误差线是九个设置间的描述性标准差，不是置信区间；重叠受试者和重复种子不被当成九组独立受试者。完整输入包含全部不利结果，没有只选最好种子。

CLARE 回归目标是原始 10 秒区间的负荷评分，误差单位为评分点；CogPilot 回归目标是事件后 8 秒与前 2 秒的皮肤电导中位数之差，误差单位为微西门子（µS）。本批两个回归任务各有一个字段，均未对目标误差再做标准化：先算每位受试者的均方根误差，再对受试者等权平均，最后对九个划分与种子设置等权平均。因此 2.8921 表示评分点误差，0.2650 表示微西门子误差，而非百分比。

## 来源与再导出

`evidence.json` 记录冻结运行来源、原文件散列及随包 CSV 散列，`stage5_report.json` 和 `pipeline_state_snapshot.json` 保留最终机器报告。`verify.py` 仅用标准库检查随包数据，可在无服务器路径环境运行；不声称重新训练模型。

需要从服务器原结果重新导出时，使用仓库中的独立导出入口；这一步与纯绘图分开：

```bash
python scripts/evaluation/application_tasks/export_stage5_closeout.py \
  --run-root /mnt/e/chronaris-v4-results/2026-09-26-stage5-recovery-scaled \
  --output-root docs/artifacts/runs/2026-09-28_v4-stage5-closeout
```
