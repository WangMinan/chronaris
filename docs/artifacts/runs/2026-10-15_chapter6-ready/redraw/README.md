# 独立界面重绘包

**本包可复制到仓库外，直接从保存的真实显示数据重绘结果页。** `preview-data.json` 包含三窗口的完整热力图矩阵、参考时间、有效性、观测曲线点和标签；`ui/` 保存对应界面、交互、颜色、字体和布局源码。生成器只读这些文件，不导入算法、不加载权重、不计算新指标、不联网。

先运行 `python render_preview.py`，再用浏览器打开 `preview.html`。修改显示数据请另存版本并使用 `--data 新版本.json` 指定，保留原导出数据。字体、色彩和尺寸集中在 `ui/style.css`；曲线与热力图绘制在 `ui/charts.js`，后者保留原界面配色。快照说明系统结果，实际处理入口仍在真实服务中。

下表说明图片与输入、脚本之间的对应关系。全部数据、显示标签、样式和脚本随Git保存。

| 输出 | 输入 | 渲染源 |
| --- | --- | --- |
| `preview.html` | `preview-data.json`及`ui/` | `render_preview.py`，只依赖Python标准库 |
| `redrawn-result.png` | 上述HTML，窗口2、参考索引40、维度13 | `export_images.mjs`，2倍像素密度 |
| `trace-1.svg`至`trace-3.svg` | 窗口2生理观测三条曲线及标签 | 同一浏览器页面中的SVG（可缩放矢量图形）导出 |
| `trace-4.svg`至`trace-6.svg` | 窗口2航电观测三条曲线及标签 | 同上 |

若需自动导出高分辨率PNG和矢量曲线，安装Playwright浏览器自动化工具及Chromium浏览器后运行 `node export_images.mjs`。已有安装可通过 `PLAYWRIGHT_MODULE_PATH` 指定模块位置；绘制仍只使用包内文件。系统需有中文字体；本轮使用工作区原有浏览器及中文字体，在仓库外 `/tmp/chronaris-chapter6-redraw` 实际完成重绘，结果见 `render-check.json`，关键PNG已目视检查。

原始应用截图位于上级目录，来自实际服务；本包的重绘图另行命名，两者共享同一真实案例的显示数据。矢量曲线保留坐标刻度及数值路径，完整字段标签可在显示输入和HTML中对应编辑。
