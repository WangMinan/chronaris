# 融合系统修复验收产物

日期：2026-10-14。应用已贯通归档时段接入、文件导入追加、模型选择、任务恢复、增量处理、结果查询及标准特征交付。完整修复说明见[修复验收](../../../review/stage/fusion-workbench-20261014/repair.md)，入口和数据格式见[使用说明](../../../implementation/fusion-workbench.md)。

本轮中央处理器验证使用既有冻结权重，36 项定向测试和 20 项浏览器操作通过；三窗口批量与逐窗口输出最大绝对差约 5.96×10⁻⁷，保持原 10⁻⁵ 容差。真实归档任务按 10＋5 个窗口完成增量处理，文件导入按 1＋2 个窗口验证追加去重；本轮未启动训练。

以下文件分别保存数值校验、操作结果与显示数据，支撑复核和重绘。

| 文件 | 内容 |
| --- | --- |
| [acceptance.json](acceptance.json) | 真实推理、批量一致性、下载回读和 CSV 对照 |
| [browser-check.json](browser-check.json) | 20 项浏览器检查、任务标识、首屏按钮与窄屏刻度测量 |
| [preview-data.json](preview-data.json) | 截图所用三窗口观测预览、融合特征、标签、模型和任务元信息 |
| [ui-evidence.json](ui-evidence.json) | 各截图的页面、视口、游标、模型、历史任务及字段查询状态 |

下列图片记录实际服务页面，展示修复后的操作层级、查询能力和交付入口。它们是应用截图，数值来自同一已保存处理结果。

- [数据与任务](data-and-tasks.png)：首屏输入、主操作、真实进度与历史记录。
- [结果分析](fusion-replay.png)：统一时间游标、数值色阶及维度查询。
- [模型与字段](models-and-fields.png)：实际模型配置与已有字段解释记录。
- [特征交付](feature-export.png)：特征下载、汇总消费及来源详情。
- [窄屏结果](mobile-replay.png)：手机宽度的字段刻度与特征查询。

**显示输入与绘制源码随 Git 交付。** 曲线及热力图渲染为 `src/chronaris/serving/fusion_workbench/static/charts.js`，交互为同目录 `app.js`，配色、字体、尺寸为 `style.css`。浏览器捕获流程为 `tests/runtime/fusion_workbench_browser.mjs`；截图状态见 `ui-evidence.json`，数值输入见 `preview-data.json`。

纯展示生成器复用前一交付目录的 `render_preview.py`，读取本次输入和当前界面源码，生成可复制到仓库外的展示包；不导入算法模块、不加载模型、不计算新指标、不请求网络。字体、颜色和尺寸可在生成包的 `ui/style.css` 中修改，完整输入保留在包内。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  docs/artifacts/runs/2026-10-13_fusion-workbench/render_preview.py \
  --data docs/artifacts/runs/2026-10-14_fusion-workbench-repair/preview-data.json \
  --output artifacts/fusion-workbench/repair-preview.html \
  --bundle artifacts/fusion-workbench/repair-preview-package
```

复制展示包到仓库外后，以 Python 标准库运行包内 `render_preview.py` 即可再次生成 HTML 页面；浏览器打开后可查询保存的窗口与特征。输入文件中保留原始数值和内部追溯标识，页面通过中文显示映射呈现；调整展示数据时另存版本，不覆盖冻结实验结果。

当前部署的模型采用连续对齐与基础因果融合配置。事件语义配置的应用验收等待相应已训练、兼容的权重；本轮完成模型登记和能力读取接口，未把资产条件写成已完成的模型验证。512 窗口容量上界、多用户部署及在线语言模型调用未在本轮验证。
