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
- 当前联调目标架次：
  - `20251005_四01_ACT-4_云_J20_22#01`
- 当前真实验证已确认：
  - 目标总线 measurement：`BUS6000019110020`
  - 重点生理 measurement：`eeg`、`spo2`
  - overlap-focused preview 在 `5s` 窗口下可生成 `25` 个联合窗口
  - 阶段 E 对照实跑（`none` / `zscore_train`）已完成，默认阈值模板评估均为 `PASS`
  - 阶段 E 收口报告：`docs/reports/alignment-preview-stage-e-closure-2026-04-21.md`
  - 阶段 E 报告已支持自动产图与样本级诊断（JSON/CSV）与阈值模板判定
  - 阶段 F 完整物理约束族对照实跑（`E baseline` / `E+F(full)`）已完成，默认阈值模板评估均为 `PASS`
  - 阶段 F 收口报告：`docs/reports/alignment-preview-stage-f-closure-2026-04-22.md`
  - 阶段 F 已确认 MySQL RealBus 字段语义映射可加载（当前字段映射数 `96`）
  - 阶段 G 最小因果融合对照实跑（`F baseline` / `F+G(min)`）已完成，默认阈值模板评估均为 `PASS`
  - 阶段 G 收口报告：`docs/reports/alignment-preview-stage-g-min-closure-2026-04-22.md`
  - 阶段 G 已导出因果注意力热力图、事件贡献 JSON/CSV 与 `96` 维最小融合表示
  - 阶段 H v1 双架次标准化导出已实跑：`20251005` 导出 `1` 个 view，`20251002` 导出 `2` 个 view
  - 阶段 H v1 主报告：`docs/reports/stage-h-export-v1-2026-04-26.md`
  - 阶段 H v1 机器资产根目录：`docs/reports/assets/stage_h/20260426T072340Z-stage-h-v1/`
  - 阶段 H v1 已补齐下游读取接口：`src/chronaris/features/stage_h_bundle.py`
  - 阶段 H 默认 `preview` profile 的每 measurement `500` 点上限是查询防护，不是收口标准；`validation/full_clip` 可取消默认点数上限
  - 阶段 H 报告里的 `WARN` 是投影诊断阈值提醒，不代表 view 包导出失败
  - 阶段 H 收口已完成：`docs/planning/stage-h-closure-2026-04-27.md`
  - 阶段 H 收口主报告：`docs/reports/stage-h-closure-2026-04-27.md`
  - 阶段 H 收口机器资产根目录：`docs/reports/assets/stage_h/20260427T000000Z-stage-h-closure/`
  - 阶段 H `validation` profile 已验证当前两条双流 sortie 可导出 `3` 个 view，`load_stage_h_feature_run()` 可直接读取 run manifest
  - 阶段 I `Phase 0 + Phase 1` 已跑通：UAB `87` session manifest、`416` 维 session 特征与双轨 baseline 已导出
  - 阶段 I `Phase 0 + Phase 1` 主报告：`docs/reports/stage-i-uab-baseline-2026-04-29.md`
  - 阶段 I `Phase 0 + Phase 1` 机器资产根目录：`docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase0-1-uab/`
  - 阶段 I 当前主线样本：`n_back=48`、`heat_the_chair=34`；辅助 `flight_simulator=5`
  - 阶段 I 当前已确认 `n_back` 有 `9` 个 session 缺失 ECG，但不阻塞当前 CPU baseline
  - 阶段 I `Phase 2` 已跑通：`3` 个真实双流 view case study、`4` 条 bundle-only 消融与 `WARN` view 主线解释已导出
  - 阶段 I Phase 2 主报告：`docs/reports/stage-i-case-study-phase2-2026-04-29.md`
  - 阶段 I Phase 2 机器资产根目录：`docs/reports/assets/stage_i/20260429T000000Z-stage-i-phase2-case-study/`
  - 阶段 I 当前 `WARN` view：`20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033`
  - 阶段 I `Phase 3` 已完成：UAB `window_v2` workload 主线、NASA CSM attention-state 主线、cross-dataset window count 图、UAB session-vs-window 对比图与 closure summary 已落盘
  - 阶段 I `Phase 3` UAB window 主报告：`docs/reports/stage-i-uab-window-baseline-2026-04-29.md`
  - 阶段 I `Phase 3` NASA 主报告：`docs/reports/stage-i-nasa-attention-baseline-2026-04-29.md`
  - 阶段 I 收口主报告：`docs/reports/stage-i-closure-2026-04-30.md`
  - 阶段 I 收口记录：`docs/planning/stage-i-closure-2026-04-30.md`
  - 阶段 I 收口机器资产根目录：`docs/reports/assets/stage_i/20260430T035013Z-stage-i-phase3-closure/`
  - 阶段 I 收口测试：`/home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest tests.test_stage_i_pipeline` 与 `CHRONARIS_ENABLE_NUMPY_RUNTIME_TESTS=1 CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1 /home/wangminan/env/anaconda3/envs/chronaris/bin/python -m unittest discover -s tests -p 'test_*.py'` 均已通过
  - 阶段 I 增强实验第一批已实跑：`stage_h_case` sequence contract、`MulT / ContiFormer` 真实 sortie smoke comparison 已落盘
  - 阶段 I 增强实验第一批主报告：`docs/reports/stage-i-real-sortie-deep-comparison-2026-05-01.md`
  - 阶段 I 增强实验第一批机器资产根目录：`docs/reports/assets/stage_i/20260501T000000Z-stage-i-deep-real-sortie/`
  - 阶段 I 增强实验第二批 probe 已实跑：`UAB / NASA` sequence 导出与统一 comparison probe 已落盘
  - 阶段 I 增强实验第二批 probe 主报告：`docs/reports/stage-i-deep-comparison-probe-2026-05-01.md`
  - 阶段 I 增强实验第二批 probe 机器资产根目录：`docs/reports/assets/stage_i/20260501T043348Z-stage-i-deep-comparison/`
  - 阶段 I 增强实验第二批 full LOSO 已实跑：`UAB / NASA` 双模型实跑与统一 comparison summary 已落盘
  - 阶段 I 增强实验第二批 full LOSO 主报告：`docs/reports/stage-i-deep-comparison-full-loso-2026-05-01.md`
  - 阶段 I 增强实验第二批 full LOSO 机器资产根目录：`docs/reports/assets/stage_i/20260501T-full-loso-deep-comparison/`
  - partial-data v1 已纳入 repo 标准入口：`configs/partial-data/stage-h-seed-v1.jsonl`
  - partial-data v1 已补齐 `20251110_单01_ACT-2_涛_J20_26#01` 的真实 `bucket / time_range / measurement_family / tag_filters`
  - partial-data v1 已生成 `vehicle_only_window_manifest.jsonl` 与 `vehicle_only_feature_bundle.npz`；该架次仍是 vehicle-only partial-data，不是双流 Stage H view

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
- 阶段 F 完整物理约束族接入与真实库对照收口
- 阶段 G 最小非对称因果融合接入与真实库对照收口
- 阶段 H 标准化特征导出、下游读取接口与 vehicle-only partial-data 收口
- 阶段 I `Phase 0 + Phase 1` UAB 公共数据双轨 baseline
- 阶段 I `Phase 2` Stage H 真实双流资产 case study
- 阶段 I `Phase 3` window-level UAB / NASA CSM 真跑、对比/消融与 closure 收口
- 阶段 I 增强实验第一批真实 sortie deep baseline
- 阶段 I 增强实验第二批公开数据 probe comparison

当前默认判断：

- `阶段 G(min) 已完成`
- `阶段 H 已完成收口（可进入阶段 I）`
- `阶段 I 已完成收口`

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
  - 保持已完成的 `Phase 2` case-study contract 冻结：继续消费 `load_stage_h_feature_run()`，不回到 E/F/G 训练中间对象
  - 继续保持 UAB `Phase 0 + Phase 1` contract 冻结，避免在完成第二数据集前回到大而全特征工程
  - 明确 `20251110...` vehicle-only partial bundle 只用于单流预训练/补充诊断，不作为双流融合 view
  - 在不破坏 frozen E/F/G(min)/H 导出路径的前提下，优先接入 `NASA CSM`
- 切到远程环境前，先同步代码、测试和文档
- 在编写和维护 `docs` 目录下的文档时保持简洁，及时清理冗余文档

## 11. 共性执行模板

跨阶段默认使用同一套执行模板（详见 `docs/planning/iteration-playbook.md`）：

1. 单轮会话节奏：`目标锁定 -> 代码实现 -> 测试闭环 -> 文档回写 -> 冗余清理`
2. 阶段收口 gate：`真实实跑`、`判据可复现`、`测试全通过`、`状态文档一致`
3. 文档治理：阶段状态只在 `coding-roadmap.md` 维护；阶段收口细节只在对应 closure 文档维护
4. 测试治理：测试文件按域合并，默认将 `test_*.py` 规模控制在 `8-12` 个
