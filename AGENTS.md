# Chronaris 协作说明

> [!NOTE]
>
> 本项目中的绝对路径仅适用于本地开发环境

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
  - `docs/选题报告与基金申请书/西北工业大学硕士研究生实践创新能力培育基金项目申请书.docx"`
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

当前默认判断：

- `阶段 E 已启动`

阶段 E 默认参考：

- `docs/planning/coding-roadmap.md`
- `docs/planning/stage-e-execution-plan.md`
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

- 从当前机器可探测到 SSH 入口，已配置公钥免密

环境依赖文件位置：

- `configs/environments/chronaris-stage-e-cpu.yml`
- `configs/environments/chronaris-stage-e-gpu.yml`

## 8. 安全约束

- 把数据库密码、SSH 密码、token、连接串请参考 `docs/SECRETS.md`， 该文件已经被 `.gitignore` 纳管，因此请尽管使用。但请不要把对应信息写入其他文件。
- 不要把原始大数据复制进仓库
- 临时验证脚本中可复用部分要及时回收进正式模块
- 实验尽量保留可复现配置、关键指标和架次范围

## 9. 默认工作方式

- 继续研究主线时，默认先看 `docs/planning/coding-roadmap.md`
- 阶段 E 默认先保住：
  - 最小训练/验证切分
  - 输入适配
  - 最小前向原型
  - 最小损失与训练 loop
- 物理约束与因果融合分别后置到阶段 F / G
- 切到远程环境前，先同步代码、测试和文档
