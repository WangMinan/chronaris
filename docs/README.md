# Chronaris 文档入口

更新时间：2026-07-04

本目录按 AI coding 使用方式重新组织。顶层只承担导航和状态入口；具体计划、需求、产物、review 记录分别下沉到专门目录。

## 术语约定

- 采自酒泉、当前由鼎新链路进入仓库的真实数据，统一写作“鼎新真实数据”或 `dingxin`。
- 分类、回归、检索三类任务在面向人阅读的文档和图表中直接写完整任务名，不再用 `T1/T2/T3` 作为展示标签。
- 基于规则或弱监督构造的验证任务，写作“弱监督任务”或“弱监督任务证据”；公开数据中由任务/场景上下文构造的第二输入流，写作“上下文构造第二输入流”。
- `private`、`proxy`、`Pxx`、`T1/T2/T3` 等只应出现在稳定文件路径、代码标识、历史 run_id 或必须复现的机器字段中。

## 顶层文件

- [STATE.md](STATE.md)：当前整个工作区状态，包括阶段进展、论文阶段、当前最高优先级和边界。
- [README.md](README.md)：本文档，负责解释目录结构与入口。
- [SECRETS.md](SECRETS.md)：本地敏感连接信息，已由 `.gitignore` 纳管，不得外传或复制到其他文件。

## 一级目录

### implementation

执行入口。用于放置当前任务队列和必要进度笔记：

- [implementation/TASKS.md](implementation/TASKS.md)：当前唯一主动执行入口，包含总路线、阶段结构、任务队列与默认工作方式。
- [implementation/notes/README.md](implementation/notes/README.md)：旧 `planning` 内容、阶段 closure、历史计划和进度笔记索引。

### requirements

需求入口。用于回答“毕业论文最终要什么、代码仓需要提供什么能力”：

- [requirements/SPEC.md](requirements/SPEC.md)：论文目标与仓库能力概括。
- [requirements/选题报告与基金申请书/](requirements/选题报告与基金申请书/)：原始 Word 文档。
- [requirements/foundation/](requirements/foundation/)：项目边界、架构和数据契约。
- [requirements/model-contracts/](requirements/model-contracts/)：模型输入、批处理和早期原型协议。

解析 `.docx` 时必须使用 `$docx` skill 或文档插件，不要把 Word 文件当作普通文本猜读。

### artifacts

产物入口。用于组织报告、图、CSV、JSON、checkpoint、manifest 等：

- [artifacts/ARTIFACTS.md](artifacts/ARTIFACTS.md)：当前产物索引和引用规则。
- [artifacts/stage/](artifacts/stage/)：按 [implementation/TASKS.md](implementation/TASKS.md) 的阶段组织产物。
- [artifacts/mid-term/](artifacts/mid-term/)：中期答辩证据包。
- [artifacts/cleanup/](artifacts/cleanup/)：docs 产物清理记录，说明已清理内容、外置备份、LFS/历史处理和保留边界；当前 仓库收敛清理 记录见 [artifacts/cleanup/20260703-thesis-prep-cleanup.md](artifacts/cleanup/20260703-thesis-prep-cleanup.md)。

### midterm

中期报告写作材料入口。用于冻结当前事实、边界风险和可写 claim：

- [midterm/README.md](midterm/README.md)：中期材料使用顺序。
- [midterm/midterm-fact-sheet-2026-06-13.md](midterm/midterm-fact-sheet-2026-06-13.md)：中期事实清单。
- [midterm/boundaries-and-risks-2026-06-13.md](midterm/boundaries-and-risks-2026-06-13.md)：边界与风险说明。
- [midterm/claims-matrix-2026-06-13.md](midterm/claims-matrix-2026-06-13.md)：报告论断矩阵。

### review

Code review 产物入口：

- [review/REVIEW.md](review/REVIEW.md)：review 记录组织规则。
- [review/stage/](review/stage/)：按阶段存放 review 输出。

## 兼容入口

为避免现有脚本和测试立即失效，以下旧路径仍保留为符号链接：

- `docs/planning` -> `docs/implementation/notes`
- `docs/foundation` -> `docs/requirements/foundation`
- `docs/models` -> `docs/requirements/model-contracts`

后续统一使用 `docs/artifacts` 与 `docs/requirements/选题报告与基金申请书`。
