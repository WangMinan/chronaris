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

`chronaris` 是“航空人机异构时序数据连续对齐与语义融合”仓库。

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

默认按下面顺序判断事实：

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
- 当前联调目标架次：
  - `20251005_四01_ACT-4_云_J20_22#01`
- 当前真实验证已确认：
  - 目标总线 measurement：`BUS6000019110020`
  - 重点生理 measurement：`eeg`、`spo2`
  - overlap-focused preview 在 `5s` 窗口下可生成 `25` 个联合窗口
  - 阶段 E 对照实跑（`none` / `zscore_train`）已完成，默认阈值模板评估均为 `PASS`
  - 阶段 E 收口报告：`docs/reports/alignment-preview-stage-e-closure-2026-04-21.md`
  - 阶段 E 报告已支持自动产图与样本级诊断（JSON/CSV）与阈值模板判定

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
- 阶段 E 最小训练闭环与 `relative_mse` 真实回归
- 阶段 E 样本级中间态投影诊断与可视化产物导出
- 阶段 E 输入归一化对照（`none` / `zscore_train`）与阈值模板收口

当前默认判断：

- `阶段 E 已完成（可进入阶段 F）`
- `阶段 F / G 未启动`

阶段 E 默认参考：

- `docs/planning/coding-roadmap.md`
- `docs/planning/stage-e-closure-2026-04-21.md`
- `docs/models/stage-e-prototype-design.md`
- `docs/models/stage-e-reference-repos.md`

## 7. 环境约定

### 本地开发环境

- 平台：Windows
- 默认环境：`D:\env\anaconda3\envs\chronaris`
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

当前已知：

- 从 windows 机器可探测到 SSH 入口，已配置公钥免密
- `chronaris` 环境已可用并能执行阶段 E runtime 测试与真实训练回归

环境依赖文件位置：

- `configs/environments/chronaris-stage-e-cpu.yml`
- `configs/environments/chronaris-stage-e-gpu.yml`

## 8. 编码规范

- 以提高最终可读性为标准，大于500行的文件建议拆分，大于800行的文件请一定拆分。

## 9. 安全约束

- 把数据库密码、SSH 密码、token、连接串请参考 `docs/SECRETS.md`， 该文件已经被 `.gitignore` 纳管，因此请尽管使用。但请不要把对应信息写入其他文件。
- 不要把原始大数据复制进仓库
- 临时验证脚本中可复用部分要及时回收进正式模块
- 实验尽量保留可复现配置、关键指标和架次范围

## 10. 默认工作方式

- 继续研究主线时，默认先看 `docs/planning/coding-roadmap.md`
- 需要规划“单轮会话如何收敛”时，默认同步参考 `docs/planning/iteration-playbook.md`
- 阶段 E 已收口，默认冻结阶段 E 基线（仅修复缺陷，不再扩展范围）
- 阶段 F 当前默认优先：
  - 在阶段 E 基线上接入最小物理一致性约束项
  - 保持与阶段 E 同一对照脚手架，补充 `E baseline` vs `E+F(min)` 对比
  - 继续沿用阈值模板与报告模板，确保阶段切换可追溯
- 因果融合继续后置到阶段 G
- 切到远程环境前，先同步代码、测试和文档
- 在编写和维护 `docs` 目录下的文档时保持简洁，及时清理冗余文档

## 11. 共性执行模板

跨阶段默认使用同一套执行模板（详见 `docs/planning/iteration-playbook.md`）：

1. 单轮会话节奏：`目标锁定 -> 代码实现 -> 测试闭环 -> 文档回写 -> 冗余清理`
2. 阶段收口 gate：`真实实跑`、`判据可复现`、`测试全通过`、`状态文档一致`
3. 文档治理：阶段状态只在 `coding-roadmap.md` 维护；阶段收口细节只在对应 closure 文档维护
4. 测试治理：测试文件按域合并，默认将 `test_*.py` 规模控制在 `8-12` 个
