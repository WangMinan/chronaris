# Chronaris 文档入口

更新时间：2026-06-21

本目录按 AI coding 使用方式重新组织。顶层只承担导航和状态入口；具体计划、需求、产物、review 记录分别下沉到专门目录。

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
- [artifacts/cleanup/](artifacts/cleanup/)：docs 产物清理记录，说明已清理内容和保留边界。

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
