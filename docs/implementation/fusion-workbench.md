# 航空人机时序融合系统

**2026-10-18 整合状态：应用服务、显式 CPU/GPU 设备路径及既有证据已合并到 main。** 当前入口与验证见[合并验收](../review/stage/chapter6-merge-20261018/README.md)；下文早期隔离工作区与 CPU 阶段安排作为历史实现说明保留。

更新时间：2026-10-15。开发位置为当前独立工作区，分支 `feat/chapter6-demo`；共享实验环境与冻结资产保持原状。

**融合系统已贯通数据接入、任务处理、结果分析和特征交付。** 使用者可以从归档记录选择时段，或导入具有相同字段合同的观测文件；选择已登记模型后执行处理，在统一时间游标下查看双流与融合特征，并生成下游可读取的序列及汇总文件。首次复核意见与本轮修复见[独立验收](../review/stage/fusion-workbench-20261014/README.md)和[修复验收](../review/stage/fusion-workbench-20261014/repair.md)。

最终验收采用下列独立资产目录；旧 `artifacts/fusion-workbench` 仍保留原任务与基础模型。新检出先按写作包中的资产登记恢复模型和案例。

## 启动与操作

当前环境使用既有解释器，服务入口显式加载本工作区源码，隐藏训练显卡并限制数值计算线程。无需改变共享环境的可编辑安装。

```bash
PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python \
  scripts/runtime/run_fusion_workbench.py --data-root artifacts/chapter6-final/service
```

打开 <http://127.0.0.1:8766>。可以通过 `--port` 更改端口，通过 `--data-root` 指定本工作区内的独立数据目录。

1. 在“数据与任务”选择观测记录、起止窗口和模型，点击“开始处理”。主要操作在常见笔记本首屏内可见。
2. “接入数据”提供归档时段接入与标准观测文件导入。新增记录立即进入选择列表。
3. 勾选“处理新增窗口”后，服务仅处理当前记录中与当前模型、实现版本对应且尚未完成的窗口；没有新增窗口时明确提示。
4. 已完成任务进入“结果分析”。时间游标联动生理观测、航电观测和融合特征；点击热力图或使用方向键选择特征维度，色阶在同一任务内固定。
5. “特征交付”下载完整特征包；“生成汇总特征”实际调用标准表示读取接口，生成逐窗口汇总表与交付记录。
6. “模型与字段”展示模型的真实训练来源与配置，以及既有语言模型的字段解释、规则复核记录。人工确认或标记待修订后，复核状态独立持久化。

## 数据接入与模型登记

**归档入口复用现有样本构建逻辑。** 当前部署从已准备记录中的来源路径发现归档快照，读取 `docs/artifacts/runs/2026-07-10_fixed-data-audit` 的字段和窗口清单，再通过既有延迟加载器构建样本。可选范围来自对应模型的训练、内部验证观测标识，不读取外层任务目标；当前两份记录各有 15 个可用窗口。时间范围内的完整窗口写入本工作区独立观测包。

首次部署优先恢复写作包登记的资产。以下命令另建一份独立案例，目标目录必须尚不存在；使用该案例时将服务的 `--data-root` 改为 `artifacts/fusion-workbench-extra`。共享源资产只读。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/prepare_fusion_case.py \
  --checkpoint artifacts/chapter6-final/training/chronaris/C/best.pt \
  --snapshot-root /home/wangminan/projects/chronaris/artifacts/application_evaluation/2026-07-10_dingxin-input-snapshot
```

**文件导入允许接入新的兼容观测。** 在接入对话框下载“输入格式”，填入实际观测值并保留字段顺序；模板中的全假掩码是待填写占位，服务会拒绝没有有效观测的窗口。导入前校验字段、角色、数值维度、掩码类型、时间和样本标识；文件上限为 32 MiB，单份记录最多 512 个窗口，较长记录按时段分组。

以下输入合同用于准备数据，而非重新计算实验指标。

| 字段 | 含义 |
| --- | --- |
| `format` | 固定为 `chronaris.observations.v1` |
| `title` | 面向读者的记录名称 |
| `schema` | 从所选模型输入格式取得的完整字段合同，名称、顺序和角色严格匹配 |
| `field_labels` | 字段标识到中文显示标签的映射，可为空 |
| `windows` | 按实际观测填写的窗口列表 |
| `sample_id` | 稳定且唯一的窗口标识；追加时用于判定重复 |
| `start_s`、`duration_s` | 记录内起点与窗口时长，单位为秒 |
| `physiology_values`、`vehicle_values` | 各流的二维观测矩阵，列顺序与合同一致 |
| `physiology_timestamps_s`、`vehicle_timestamps_s` | 各流在窗口内的实际采样时刻，单位为秒 |
| `physiology_feature_mask`、`vehicle_feature_mask` | 与观测矩阵同形的布尔有效标记 |

选择“追加到当前导入记录”可加入新增窗口。已有标识与内容完全相同则去重，同一标识内容改变则拒绝；合法新增内容发布后，任务可继续按增量方式处理。每次提交保存输入快照，后续追加和重新排序不改变已提交任务的选择。

**模型选择来自真实权重登记。** 现有模型经过训练完成状态、实现版本和字段兼容性校验；新增模型由维护者使用以下入口登记，复制到本工作区的模型目录，原权重保持不变。页面点击“刷新模型”即可更新列表。

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/register_fusion_model.py \
  --checkpoint /已有训练产物/best.pt \
  --schema-record dingxin-validation --data-root artifacts/chapter6-final/service
```

当前加载合同支持已完成训练的自监督编码器。最终独立目录登记实测语义应用模型与旧基础模型；语义模型已完成300次补充更新并通过真实应用验收，页面如实展示保存配置。完整来源、性能、案例及运行资产恢复见[第六章写作交接包](../artifacts/runs/2026-10-15_chapter6-ready/README.md)。旧三个实测候选的原配置保留，运动学残差按字段适用条件记录。

## 任务恢复与交付

**任务选择、模型和实现指纹共同确定计算身份。** 每次任务保存原始参数、窗口选择、模型散列和输入清单。逐块结果先持久化，再更新进度；失败或服务退出后，可以按原参数创建关联重试任务。恢复只复用输入、模型和实现指纹一致的已完成片段，原失败记录保留。

当前运行方式为单服务进程、单推理任务。每份记录最多 512 个窗口，模型按小块加载观测并推理；最后按标准表示合同合并、校验和导出。该容量上界是资源保护条件，本轮实际长任务验收覆盖 15 个窗口，不将其表述为最大容量性能验证。

增量去重以已完成任务记录作为持久依据，按窗口内容、模型及实现版本匹配，支持重启和追加数据。它表示窗口任务的增量处理；模型仍按已训练合同分别处理窗口上下文。

下载包保留 `fusion_stream.npz`、`representation_manifest.json`、`features.csv`、`result.json`。前两者必须放在同一目录，按既有接口读取：

```python
from chronaris.representation import load_fusion_stream_batch
features = load_fusion_stream_batch("解压目录")
sequence = features.sequence_embedding
valid = features.valid_mask
pooled = features.pooled_embedding
```

“生成汇总特征”以该接口读取已完成输出，生成 `pooled_features.csv` 与 `delivery.json`。交付回执记录窗口数、有效时刻数和表示维数；它证明标准特征消费接口已执行，具体业务任务预测由对应任务模型完成。

字段复核读取已有 `llm_preprocessing` 结构化结果，通过现有解析器整理字段和规则。人工结论独立保存在 `field-reviews.json`，按字段合同与解释来源绑定，不改变模型输入、标签或既有语言模型记录。本轮接通保存结果的复核流程，没有新增在线模型调用。

## 接口与目录

下表列出页面使用的接口，便于与下游程序衔接。

| 方法与路径 | 功能 |
| --- | --- |
| `GET /api/cases`、`GET /api/models` | 观测记录与兼容模型 |
| `GET /api/sources` | 已授权观测范围内的归档记录 |
| `POST /api/sources/prepare` | 按 `source_id`、`start_s`、`stop_s`、`model_id` 接入时段 |
| `POST /api/cases/import` | 导入标准观测；可带 `record_id` 追加 |
| `POST /api/runs` | `case_id`、`start`、`stop`，可带 `model_id`、`incremental`；结束索引不包含在内 |
| `GET /api/runs`、`GET /api/runs/{id}` | 任务历史与进度 |
| `POST /api/runs/{id}/retry` | 原参数恢复，复用合格片段 |
| `GET /api/runs/{id}/result` | 已完成结果及展示数据 |
| `POST /api/runs/{id}/deliver`、`GET /api/runs/{id}/delivery` | 执行汇总特征消费并读取回执 |
| `GET /api/runs/{id}/download/{name}` | 白名单中的特征包、表示、清单、汇总特征及回执 |
| `GET /api/fields/{case_id}`、`POST /api/fields/{case_id}` | 字段及规则复核；提交 `ids` 和 `decision` |

所有接口只监听本机回环地址，提交请求需要同源结构化数据。用户文件不能提供模型读取路径；数据标识和模型登记路径独立校验。

产物均在 `artifacts/fusion-workbench/`：`cases/` 保存独立观测包，`models/` 保存登记权重，`runs/` 保存任务、片段和输出，`acceptance/` 保存数值验证，截图保存在验收子目录。数据库服务和上游采集系统仍通过已有工程接口配置，本工作台当前接入归档快照与观测文件。

## 复现验证

定向测试覆盖导入、重复与冲突、任务范围、重试、服务重启、增量去重、模型登记、字段复核和标准表示。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest \
  tests/runtime/test_fusion_workbench.py tests/runtime/test_fusion_workbench_workflow.py \
  tests/representation/test_contracts.py tests/representation/test_oof_export.py -q
```

启动服务后运行真实推理一致性与浏览器回归。浏览器检查使用工作区已有的 Playwright 浏览器自动化工具，不属于系统运行依赖。

```bash
PYTHONNOUSERSITE=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python scripts/runtime/verify_fusion_workbench.py --data-root artifacts/chapter6-final/service

WORKBENCH_DATA_ROOT="$PWD/artifacts/chapter6-final/service" \
DEMO_SCREENSHOTS="$PWD/artifacts/chapter6-final/recheck-screenshots" \
LD_LIBRARY_PATH="$PWD/artifacts/fusion-workbench/browser/libs/extracted/usr/lib/x86_64-linux-gnu" \
FONTCONFIG_FILE="$PWD/artifacts/fusion-workbench/browser/fonts.conf" \
PLAYWRIGHT_BROWSERS_PATH="$PWD/artifacts/fusion-workbench/browser/browsers" \
node tests/runtime/fusion_workbench_browser.mjs
```

浏览器回归会用真实观测创建接入记录和推理任务，全部写在本工作区；测试包含归档接入、文件导入追加、增量处理、断连恢复及结果身份切换。此检查不启动训练或读取外层任务目标。
