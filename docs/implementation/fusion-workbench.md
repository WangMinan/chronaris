# Chronaris 融合特征工作台

更新时间：2026-10-14。工作区：`/home/wangminan/projects/chronaris-demo`；分支：`feat/chapter6-demo`。当前已完成工程自验，待用户验收后安排合并。

## 应用关系与交付范围

**独立演示界面已贯通离线观测、冻结模型、融合表示、历史回放和特征下载。** Zorathos 接入整合工程提供项目背景和数据基础，Chronaris 连续对齐与融合工程负责特征生成；本界面提供独立的融合特征工作台，支撑论文系统实现部分。原平台前端由分包方开发，不纳入本次实现。范围依据[背景会话](thread://01a13478-7a66-7cf3-94e9-177f9154d828?hostId=remote-control%3Aenv_e_6acc8bf506d08321a12d3ecdb4d6905c)以及宿主机论文目录中的第六章确认。

**本次改动集中在应用入口，不改变训练与实验算法。** 可复用服务位于 `src/chronaris/serving/fusion_workbench`，命令行文件仅编排；网页使用本地 HTML（超文本标记语言）、CSS（样式表）和 JavaScript（浏览器脚本），无在线资源或运行时前端依赖。既有旧版推理入口保持兼容，新增入口直接复用当前模型的加载、归一化、推理和标准表示导出函数。

下表说明功能与论文目录的对应关系，支撑第六章围绕实际使用过程展开。

| 页面或模块 | 已实现行为 | 目录对应 |
| --- | --- | --- |
| 数据与任务 | 案例、窗口起止、字段数、冻结模型、运行状态 | 6.3.1、6.3.3 |
| 融合结果与回放 | 双流观测、字段观测率、融合表示、窗口播放与参考时刻选择 | 6.3.3、6.5.3 |
| 特征导出与追溯 | 序列特征预览、完整包下载、模型与输入散列、耗时、下游读取示例 | 6.3.4、6.4.2 |
| 测试与复现 | 异常请求、完整性校验、历史恢复、批量与逐窗口一致性、浏览器交互 | 6.5.1、6.5.2 |

## 命名规范

**文件按系统职责命名。** 服务包为 `chronaris.serving.fusion_workbench`；启动、案例准备和验收入口分别为 `run_fusion_workbench.py`、`prepare_fusion_case.py`、`verify_fusion_workbench.py`。文档和产物目录统一使用 `fusion-workbench`，截图按“数据与任务、融合回放、特征导出、移动回放”命名。2026-10-14 已迁移本工作区的案例和历史运行目录，并完成新入口、下载与浏览器回归。已有实验记录中的来源散列保持原值。

## 启动与首次准备

**现有工作区已准备好一个约 6.4 MiB 的独立鼎新案例包。** 它包含三个内部验证窗口、模型副本和字段说明；无需数据库、网络或训练显卡即可运行。命令显式选择既有解释器，入口将当前工作区 `src` 放在模块搜索首位，隐藏显卡并限制数值计算为单线程。

在本工作区根目录执行：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/run_fusion_workbench.py
```

浏览器访问 <http://127.0.0.1:8766>。默认只监听本机回环地址；若端口已占用，可使用 `--port 8767`。界面操作顺序为选择案例与起止窗口、生成融合特征、打开结果、切换窗口或播放、下载完整特征包。启动时可直接打开历史记录。

**新检出需要从已有冻结资产生成案例包，原始观测和模型不进入 Git。** 以下准备命令只读共享输入，写入当前工作区；已有目标目录会被拒绝，避免覆盖既有案例。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/prepare_fusion_case.py \
  --checkpoint /home/wangminan/chronaris-data/chronaris-v4-results/2026-10-02-stage5-performance-recovery/training/255/dingxin/chronaris/self_supervised/chronaris/C/best.pt \
  --snapshot-root /home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot
```

产物均位于 `artifacts/fusion-workbench/`：`cases/` 为独立输入包，`runs/` 为按唯一标识保存的任务与输出，`acceptance/` 为有界验收记录，`screenshots/` 为浏览器截图。页面和服务通过本工作区源码启动，无需安装或切换共享环境中的可编辑包。

## 当前模型与输入合同

**演示案例使用已完成的当前连续融合编码器及其原有归一化参数。** 冻结检查点的实现版本为 `causal_fusion_v4`，种子为 17，字段合同为 12 个生理字段与 965 个航电字段。本包选择该模型内部验证集合的前三个相邻窗口，覆盖相对时间 90—130 秒；每窗口输入长度为 30 秒。仅读取观测，不加载任务目标、不重新拟合归一化、不计算实验排名。

**当前检查点展示连续融合特征生成能力，事件语义模块的状态按检查点如实呈现。** 该检查点未启用事件语义模块，页面因此展示原始双流与融合表示；后续具有同一检查点合同的新冻结编码器可由现有加载器读取自身配置。当前应用入口接受完成的自监督 Chronaris 检查点，任务引导检查点和业务预测头接入另行验收。

**窗口级增量处理具有明确的处理单位。** 服务按窗口逐个执行，保留每个窗口原始异步时间和缺失掩码；窗口之间不持有连续潜态。历史回放读取完成记录，不触发重复计算。每次请求最多八个连续窗口；一个产物目录同时只允许一个服务进程和一个推理任务。退出时未完成的任务在下次启动后标记失败，可重新提交。

**案例包可搬移，并保留输入与模型的完整性校验。** 每个子目录包含 `case.json`、相对路径模型和窗口 NumPy 数组归档（NPZ）。服务只发现启动时存在的案例包，不开放任意文件上传或路径读取。新增包后重启即可列入案例选择。

以下字段定义支撑其他合规观测片段的接入；可参照 `tests/runtime/test_fusion_workbench.py` 中的独立案例构造。

| 文件或字段 | 内容与约束 |
| --- | --- |
| `format`、`id` | `chronaris.demo_case.v1`；案例标识为小写字母、数字及连字符，目录内唯一 |
| `title`、`description`、`source_label`、`model_label` | 面向读者的案例、数据与模型说明 |
| `schema` | 既有 `ObservationSchema` 字段顺序、来源类型和字段角色；与模型严格匹配 |
| `checkpoint`、`checkpoint_sha256` | 包内相对路径与 SHA-256（完整性散列）；检查点必须完成，加载拒绝历史不兼容版本 |
| `windows` | 1—32 个窗口，每项包含包内文件名、文件散列、样本及分组标识、观测来源散列、起点秒数和时长秒数 |
| 窗口数组 | 双流各自的 `values`、`timestamps_s`、`feature_mask`；沿用既有数值、时序和维度校验 |
| `field_labels` | 可选中文显示标签；优先展示至多三个有效字段，无标签时显示字段序号 |

## 接口与下游读取

**下载包沿用仓库既有标准表示合同。** 每个窗口输出 96 个参考时刻、64 维序列表示、有效掩码及按有效时刻平均的汇总表示。写出后立即通过 `load_fusion_stream_batch` 回读校验；来源清单保留样本顺序、模型和输入散列。网页曲线至多保留 240 个实际观测点用于预览，模型输入和下载的融合表示保持完整。

下表列出本机 HTTP（超文本传输协议）接口，便于复核页面行为与后续调用。

| 方法与路径 | 行为 |
| --- | --- |
| `GET /api/cases` | 案例摘要与可选窗口 |
| `POST /api/runs` | 请求体为 `{"case_id":"dingxin-validation","start":0,"stop":3}`，索引从零开始，结束索引不包含在内 |
| `GET /api/runs` | 历史任务列表 |
| `GET /api/runs/{id}` | 状态、完成窗口数、错误或耗时 |
| `GET /api/runs/{id}/result` | 仅完成任务可读的曲线与表示 |
| `GET /api/runs/{id}/download/{name}` | 固定允许的完整包、CSV（逗号分隔值）、NPZ 或 JSON（结构化文本）清单 |

**完整下载包可由现有下游程序直接读取。** 解压后目录内的 `fusion_stream.npz` 与 `representation_manifest.json` 必须放在一起；`features.csv` 每行包含样本标识、窗口内参考时刻（秒）、有效标记及 64 个特征值；`result.json` 另存本次计算、可视化预览、检查点配置与源码散列。

```python
from chronaris.representation import load_fusion_stream_batch
features = load_fusion_stream_batch("解压目录")
sequence = features.sequence_embedding  # 窗口 × 96 × 64
valid = features.valid_mask             # 窗口 × 96
pooled = features.pooled_embedding      # 窗口 × 64
```

## 验收与交接

**工程自验已覆盖真实运行、表示一致性、下载回读和页面交互。** 结果与截图见[验收记录](../artifacts/runs/2026-10-13_fusion-workbench/README.md)。复现有界真实验收前先启动本地服务，再执行：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/verify_fusion_workbench.py
```

常规测试仅使用构造数据，不读取共享模型或执行训练：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest \
  tests/runtime/test_fusion_workbench.py \
  tests/representation/test_contracts.py tests/representation/test_oof_export.py -q
```

**浏览器测试工具与依赖也保留在工作区内。** 本机通过 `tests/runtime/fusion_workbench_browser.mjs` 验证桌面和移动布局；Playwright（浏览器自动化工具）不属于应用运行依赖。浏览器、缺失动态库及中文字体仅下载或复制到忽略目录，未改系统或共享 conda 环境。本机复现命令为：

```bash
LD_LIBRARY_PATH="$PWD/artifacts/fusion-workbench/browser/libs/extracted/usr/lib/x86_64-linux-gnu" \
FONTCONFIG_FILE="$PWD/artifacts/fusion-workbench/browser/fonts.conf" \
PLAYWRIGHT_BROWSERS_PATH="$PWD/artifacts/fusion-workbench/browser/browsers" \
node tests/runtime/fusion_workbench_browser.mjs
```

新机器可在独立目录安装 Playwright 并下载 Chromium 浏览器，再运行同一脚本；应用本身无需这些测试依赖。截图由自动化脚本操作实际界面后生成，不作为新的算法实验图。

**用户验收聚焦实际使用闭环。** 打开默认案例，改选窗口 02—03，运行并观察结果；播放两个窗口、拖动参考时刻、下载完整包并刷新页面打开历史。确认界面、案例与论文口径后再合并 `feat/chapter6-demo`。持续在线接收、跨窗口状态恢复、业务风险判定及大语言模型在线调用属于后续扩展；第六章训练和辅助解释部分仍引用既有工程实现与证据。
