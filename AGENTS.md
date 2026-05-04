# Chronaris 协作说明

> **codex 编码八荣八耻**
> - 以瞎猜接口为耻，以认真查询为荣。
> - 以模糊执行为耻，以寻求确认为荣。
> - 以臆想业务为耻，以人类确认为荣。
> - 以创造接口为耻，以复用现有为荣。
> - 以跳过验证为耻，以主动测试为荣。
> - 以破坏架构为耻，以遵循规范为荣。
> - 以假装理解为耻，以诚实无知为荣。
> - 以盲目修改为耻，以谨慎重构为荣。

## 1. 项目定位

`chronaris` 是“航空人机异构时序数据连续对齐与语义融合”仓库。整个项目都在为基于 `docs/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx` 的我的毕业设计服务。

默认承接：

- 下游研究与原型实现
- 数据读取、组织、建模、导出、验证

默认不承接：

- 历史文件接收器重写
- 上游入库链路重建
- 原始大数据文件入仓

上游现状默认视为：

- 生理数据、飞机时序数据已进入 InfluxDB
- 业务元数据已进入 MySQL

## 2. 研究主线

后续工作默认沿这条链路推进：

1. 读取指定架次的人机多源数据及元信息
2. 建立统一 schema、统一时间参考和统一样本组织
3. 实现双流连续潜态建模
4. 实现物理一致性约束时间对齐
5. 实现因果掩码跨模态融合
6. 输出标准化融合特征与中间态接口
7. 面向典型任务开展对比、消融和案例验证

典型任务：

- 空中失能风险分析
- 认知负荷评估
- 飞行事件复盘

## 3. 事实优先级

如果在本地开发环境，默认按下面顺序判断事实：

1. 当前仓库代码、文档、配置
2. `D:\code\zorathos\zorathos-data-model`
3. `D:\code\zorathos\zorathos-data-receiver`
4. 本地真实样例与 `D:\0_大学\2024.9\实验室\数据中台\0_实采数据\AGENTS.md`
5. 选题报告与基金申请书

冲突时：

- 当前仓库内已沉淀结论优先
- 外部源码事实优先于论文式表述
- 上游链路问题优先回看 `zorathos`

## 4. 当前关键事实

- 工作语言：中文
- 理论依据：
  - `docs/选题报告与基金申请书/西北工业大学硕士学位研究生论文选题报告表.docx`
  - `docs/选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx`
- 生理流时间精度：微秒级
- 飞机流时间精度：毫秒级
- 飞机原始时间只有时分秒，完整日期来自 `flight_batch.fly_date`
- 飞机完整时间拼接沿用 `TimeSequenceProcessor` 跨日规则
- 当前主线 sortie：
  - `20251005_四01_ACT-4_云_J20_22#01`
  - `20251002_单01_ACT-8_翼云_J16_12#01`
- 当前应优先引用的事实源：
  - Stage H 收口：`docs/reports/stage-h-closure-2026-04-27.md`
  - Stage I 公开数据收口：`docs/reports/stage-i-closure-2026-04-30.md`
  - Stage I 私有主线计划：`docs/planning/stage-i-private-benchmark-plan-2026-05-02.md`
  - Stage I 主线迁移计划：`docs/planning/stage-i-mainline-transition-2026-05-04.md`
  - 当前 `chronaris_opt` package：`docs/reports/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
- 当前已验证：
  - 目标总线 measurement：`BUS6000019110020`
  - 重点生理 measurement：`eeg`、`spo2`
  - overlap-focused preview 在 `5s` 窗口下可生成 `25` 个联合窗口
  - Stage H `validation` profile 已稳定导出 `3` 个双流 view，`load_stage_h_feature_run()` 可直接读取 run manifest
  - Stage I 公开 benchmark `Phase 0/1/2/3` 已完成并收口
  - `MulT / ContiFormer` 真实 sortie smoke 与公开数据 full LOSO 已完成，作为历史对照保留
  - `chronaris_opt` 已在鼎新私有 proxy benchmark 的 `T1/T2/T3` 三任务上达到当前对照矩阵最优
  - `chronaris_opt` 真实 package 已固化，可作为当前最佳私有工件引用
  - `20251110_单01_ACT-2_涛_J20_26#01` 仍是 vehicle-only partial-data，不是双流 Stage H view

## 5. 目录与边界

新增代码默认遵守：

- `src/chronaris/access`: InfluxDB / MySQL 访问
- `src/chronaris/schema`: 统一 schema
- `src/chronaris/dataset`: 样本组织、窗口切分、时间基准
- `src/chronaris/models/alignment`: 连续对齐
- `src/chronaris/models/fusion`: 因果融合
- `src/chronaris/features`: 特征导出与中间态
- `src/chronaris/pipelines`: 训练 / 导出 / 验证流程
- `src/chronaris/serving`: 服务化或近实时接口
- `src/chronaris/evaluation`: 对比、消融、案例分析

约束：

- 可复用逻辑必须进入 `src/chronaris`
- `scripts` 不承载核心业务实现
- notebook 只用于探索，不能成为唯一事实来源
- `docs` 默认使用中文

## 6. 当前阶段

当前仓库已完成：

- 阶段 B preview 路径
- 阶段 C 真实重叠核验
- 阶段 E0 preview 路径
- 阶段 E/F/G(min)/H 收口
- 阶段 I 公开 benchmark `Phase 0 + Phase 1 + Phase 2 + Phase 3` 收口
- 第三方 deep baseline 对照（真实 sortie smoke + UAB/NASA full LOSO）
- 鼎新私有 proxy benchmark `chronaris_opt` 最优性验证
- `chronaris_opt` package 固化

当前默认判断：

- `阶段 G(min) 已完成`
- `阶段 H 已完成收口（可进入阶段 I）`
- `阶段 I 已完成收口`
- `当前鼎新私有任务验证主线 = chronaris_opt`
- `E/F/G/H 收口工件保留为 chronaris_opt 的历史基线与导出依赖，不删除`

阶段 E/F/G/H 默认参考：

- `docs/planning/coding-roadmap.md`
- `docs/planning/stage-e-closure-2026-04-21.md`
- `docs/planning/stage-f-closure-2026-04-22.md`
- `docs/planning/stage-g-closure-2026-04-22.md`
- `docs/planning/stage-h-closure-2026-04-27.md`
- `docs/models/stage-e-prototype-design.md`
- `docs/models/stage-e-reference-repos.md`

## 7. 环境约定

### 本地开发环境

- 平台：Windows
- 默认环境：`D:\env\anaconda3\envs\chronaris`
- 默认要求：除非明确说明只做静态文本处理，否则本仓库相关的 Python 运行、测试、脚本验证默认都在 `chronaris` conda 环境内执行，不要默认落到 `base`
- 推荐启动方式：
  - Windows：`conda activate chronaris`
  - WSL / Linux 显式解释器：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python`
- 主要用途：
  - 文档整理
  - 协议与纯 Python 模块开发
  - CPU 安全测试

注意：

- 当前本机 `numpy` / `torch` 包虽可安装，但运行时并不稳定
- 本机不作为可靠训练环境，也不作为严格的 `numpy/torch` runtime 验证环境

### 远程训练环境

- 首选：实验室服务器 `10.70.4.57`
- 用户：`wangminan`
- 训练平台：WSL Ubuntu 22.04 + RTX 4090
- 备用：家里工作站 WSL Ubuntu 22.04 + RTX 4070 Ti
- 默认环境：`/home/wangminan/env/anaconda3/envs/chronaris`
- 强制约定：远程环境下的训练、评测、artifact 构建、`unittest`、阶段脚本实跑，默认一律使用 `chronaris` conda 环境；不要因为 shell 停在 `base` 就直接运行
- 推荐执行方式：
  - 先激活：`conda activate chronaris`
  - 或显式调用：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python <script>`

当前已知：

- 从 windows 机器可探测到 SSH 入口，已配置公钥免密
- `chronaris` 环境已可用并能执行阶段 E runtime 测试与真实训练回归
- 如果出现“当前 namespace 正确但缺包”的情况，先检查是否误用了 `base` 或其他解释器，再判断是否真缺依赖
- 当前数据库服务跑在 Docker 中，但 MySQL `3306` 与 InfluxDB `8086` 已映射到本机端口；真实验证默认优先使用 `127.0.0.1` 访问，不要先假设宿主机原生服务。

环境依赖文件位置：

- `configs/environments/chronaris-stage-e-cpu.yml`
- `configs/environments/chronaris-stage-e-gpu.yml`
- `configs/environments/chronaris-stage-i-cpu.yml`
- `configs/environments/chronaris-stage-i-gpu.yml`

## 8. 编码规范

- 以提高最终可读性为标准，大于500行的文件建议拆分，大于800行的文件请一定拆分。

## 9. 安全约束

- 对于远程训练环境，如需 MySQL 数据库密码、sudo 用户名与密码、 InfluxDB token、连接串，请尽管参考并使用 `docs/SECRETS.md`， 该文件已经被 `.gitignore` 纳管，但请不要把对应信息写入其他文件。
- 不要把原始大数据复制进仓库
- 临时验证脚本中可复用部分要及时回收进正式模块
- 实验尽量保留可复现配置、关键指标和架次范围

## 10. 默认工作方式

- 继续研究主线时，默认先看 `docs/planning/coding-roadmap.md`
- 需要规划“单轮会话如何收敛”时，默认同步参考 `docs/planning/iteration-playbook.md`
- 运行任何 Python 脚本、测试、基准或收口命令前，默认先确认解释器属于 `chronaris` 环境；若有歧义，优先使用显式解释器路径 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`
- 阶段 E 已收口，默认冻结阶段 E 基线（仅修复缺陷，不再扩展范围）
- 阶段 F 已收口，默认冻结阶段 F 基线（仅修复缺陷，不再扩展范围）
- 阶段 G 已收口，默认冻结 G(min) 基线（仅修复缺陷，不提前扩展完整因果融合）
- 阶段 H 已收口，默认冻结 Stage H 导出 contract（仅修复缺陷，不再扩展范围）
- 阶段 I 当前默认优先：
  - `chronaris_opt` 作为当前鼎新私有任务验证主线，默认优先维护其文档、测试与后续迁移
  - 保持 `load_stage_h_feature_run()` 与 `Stage H all-window` contract 稳定：它们是 `chronaris_opt` 的输入依赖
  - 保持 `E/F/G(min)/H` 导出路径稳定：它们是 `chronaris_opt` 的历史基线与输入依赖，不直接废弃
  - 继续把 UAB/NASA `Phase 0/1/2/3` 视为公开 benchmark 历史事实，不与私有 proxy 最优性混写
  - 明确 `20251110...` vehicle-only partial bundle 只用于单流预训练/补充诊断，不作为双流融合 view
  - 下一步优先把 `chronaris_opt` 思路迁到公开 sequence contract，先做 `chronaris public opt`
- 切到远程环境前，先同步代码、测试和文档
- 在编写和维护 `docs` 目录下的文档时保持简洁，及时清理冗余文档

## 11. 共性执行模板

跨阶段默认使用同一套执行模板（详见 `docs/planning/iteration-playbook.md`）：

1. 单轮会话节奏：`目标锁定 -> 代码实现 -> 测试闭环 -> 文档回写 -> 冗余清理`
2. 阶段收口 gate：`真实实跑`、`判据可复现`、`测试全通过`、`状态文档一致`
3. 文档治理：阶段状态只在 `coding-roadmap.md` 维护；阶段收口细节只在对应 closure 文档维护
4. 测试治理：测试文件按域合并，默认将 `test_*.py` 规模控制在 `8-12` 个
