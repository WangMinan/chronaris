# 论文侧直接使用入口

本包交付第五章初稿所需的种子 17 正式结果，覆盖公开五折、鼎新固定时间块、受控仿真与机制评价。数值、不可用项和不利结果保持原样，正文需按实际结果作判断。种子 29、43 的补充复核不在本包范围内。

下表把章节用途对应到可直接引用的数据与图形。先读 `protocol.json` 与 `data_dictionary.md` 确认单位、划分及误差线含义，再选用对应素材。

| 章节用途 | 直接使用的数据 | 图形文件名（位于 figures/，均有 PNG 和 SVG） |
| --- | --- | --- |
| 5.1—5.2 数据与实验设置 | data/inventory.csv、data/training_contracts.csv.gz、data/protocol/、protocol.json | 依据数据组织论文表格 |
| 5.3 公开数据、鼎新主比较 | data/native_summary.csv、data/native_groups.csv、data/native_predictions/ | native_clare_*、native_cogpilot_*、native_dingxin_* |
| 5.3—5.4 仿真与组件消融 | data/simulation_summary.csv、data/simulation_groups.csv | simulation_* |
| 5.4 参数敏感性 | data/sensitivity.csv | sensitivity_self_supervised、sensitivity_task_guided |
| 5.5 缺失与尾部误差 | data/simulation_summary.csv、data/regression_tails.csv | missing_* |
| 5.5 时间机制与历史信息隔离 | data/timing_metrics.csv、data/timing_predictions/、data/causality_queries.csv | timing_self_supervised、timing_task_guided、causality |
| 5.6 训练与推理成本 | data/costs.csv、data/inference_costs.csv、data/historical_attempts.csv | inference |

**图片可直接插入论文，也可继续编辑。** `figures/*.png` 为高分辨率图片，`figures/*.svg` 为矢量图；所有 35 组图片及其输入、脚本和样式对应关系见 `figures/figure_inputs.json`。Word 中较宽的组合图可使用横向页面或在论文侧拆分面板；不得把坐标含义或误差线定义裁去。

**重绘只需随包文件与绘图库。** 复制整个目录，安装 `requirements.txt` 后执行 `python plot.py`、`python verify.py`。样式在 `style.json`，渲染脚本为 `plot.py`。需要中文字体；本轮使用系统思源黑体，最小环境的依赖版本见 `redraw_environment.json`。

**压缩文件保留全部原数据。** CSV 是逗号分隔文本，`.csv.gz` 是其无损压缩形式，可用支持 gzip 的表格读取工具，或运行 `gzip -dk data/training_contracts.csv.gz` 解压（约 1 GB）。其他预测目录中的 `.csv.gz` 同样可解压。`evidence.json` 校验交付数据，`source_evidence.json` 保留原导出清单，压缩前后校验值见 `packaging.json`；原冻结结果未改写。

公开结果的误差线为折间描述性标准差，仿真结果为生成参数档案间描述性标准差，均不作为置信区间；鼎新仅描述单记录时间块。时间机制结果表示冻结特征的线性可解码性。敏感性曲线来自内部验证，不用于据留出成绩重新选参。
