# 研究基线合入主分支复核

日期：2026-09-05。

本次按用户“推送所有本地变更，先合并再继续改进”的指示，将已完成的实现、材料和仿真审计集成为主分支 `main` 的研究基线。该决定更新此前暂不合并的分支安排；研究判据仍沿用冻结协议，未来信息隔离与安全旁路失败、公开数据与鼎新外层关闭的结论保持不变。

## 范围与计划

- 合并来源为 `research/thesis-continuous-semantic-fusion-202609`，检查时最新提交为 `a6f8b577`；远端 `main` 为 `7e6a3af9`，是来源分支的祖先，来源分支领先 51 个提交。
- 复核既有实现和实验来源，检查全部未提交变更、完整测试、外层入口阻断和文档一致性，然后提交、推送并通过拉取请求合并；保留原始提交历史。
- 本地新增的中期考核表与 `goal.md` 按原内容归档。后者是用户既有任务原稿，当前执行状态以 `docs/STATE.md` 与 `docs/implementation/TASKS.md` 为准。
- 三份历史指标文件的已有删除一并提交：`2026-07-23_cogpilot-difficulty/difficulty_metrics.json`、同目录 `difficulty_metrics_seed17.json` 和 `2026-07-24_clare-cognitive-load/clare_metrics.json`，均位于 `docs/artifacts/runs/` 下。原内容可从提交 `a6f8b577` 追溯；本轮九月冻结指标不变。

## 已确认事项

- 远端主分支无分叉、无分支保护规则，也没有待合并的其他拉取请求。
- 新增 Word 文档的压缩包与 XML 结构可读，大小约 3.73 MB；新增材料凭据模式检查通过。
- 本次不修改模型、消费者、训练预算或冻结协议，不产生新的研究结果；仅完成材料归档与分支集成。
- 未来依赖已定位但尚未修复，安全旁路仍有三个超阈值单元；这两项作为后续研究任务保留。详情见[冻结仿真复核](frozen-simulation-review-2026-09-04.md)。

## 验证结果

合并前完整测试通过：488 项通过、8 项跳过、319 条警告、零失败、零错误，用时 184.37 秒。命令为 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m pytest -q --junitxml=artifacts/application_evaluation/2026-09-05_main-integration-review/pytest.xml`；结果 XML 的 SHA-256 为 `5f063a4edeb42e5c73752a8dee97b4eb69ad1136d45a4d9a3b08fc64c2682d7f`。

外层入口的四项阻断检查全部通过：公开数据与鼎新入口在默认缺少通过审计时均拒绝执行；在仅供诊断的内存中指向当前 v3.2.3 失败审计后，也均因研究硬门失败而拒绝执行。检查没有读取外层数据或运行外层训练，详细记录位于被忽略的 `artifacts/application_evaluation/2026-09-05_main-integration-review/guard-and-evidence-check.json`。

本次重新核对 49 份复用证据与六份核心审计文件，所有哈希均与冻结记录一致。模型、实验脚本、测试和 v3.2.3 协议与来源提交 `a6f8b577` 一致；新增可读文档的术语检查和 `git diff --check` 均通过。

本次验证支持将该状态作为后续开发的研究基线合入 `main`，不改变四项机制门通过、两项失败的实验判断。后续从主分支建立新改进分支，单独确认模型行为修订范围并重新冻结相关评价。
