# Goal Prompt：Stage I 中期前 P10-P15 主动证据推进

下面这段 prompt 用于在新 session 的 goal 模式中启动后续工作。它假定工作区为 `/home/wangminan/projects/chronaris`。

```text
你现在接手 Chronaris 仓库 `/home/wangminan/projects/chronaris`，请以 goal 模式长期推进 Stage I 中期前主动证据闭环。

总目标：在不破坏现有架构和证据边界的前提下，把 Stage I 从当前的 Phase C/D/E/F 已收口状态继续推进到 `P10 evidence runner + P11-P15 五项主动证据`，并形成可复现、可引用、可测试、文档一致的中期前证据包。

开始前必须读取并以这些文档为事实源：

- `AGENTS.md`：仓库协作规则、代码边界、环境约定、执行规则。
- `docs/STATE.md`：当前状态、最新 HEAD、当前主线事实、已入库资产和中期前主动任务。
- `docs/implementation/TASKS.md`：当前唯一主动执行入口，重点阅读 P10-P17。
- `docs/artifacts/ARTIFACTS.md`：当前可引用产物索引。
- `docs/artifacts/stage_i/README.md`：Stage I 报告入口。
- `docs/requirements/SPEC.md`：论文需求与仓库能力边界。
- 如需判断 Word 原始材料，必须使用 `$docx` skill 或文档插件读取 `docs/requirements/选题报告与基金申请书/`，不要猜测 Word 内容。

当前必须遵守的事实边界：

- `chronaris_opt` 与 `T1/T2/T3` 只能写成 `private proxy benchmark / proxy evidence`，不能写成人工真值 thesis task fully closed。
- `risk_proxy / workload_proxy / event_replay_tag` 只能写成 `thesis weak-label evidence`，不等价于人工真值任务。
- NASA/UAB 只能写成 `public adapter / calibration evidence` 或 `context proxy / transfer boundary evidence`，不能写成论文双流本体闭环。
- UAB `target_prior_median`、robust-prior adapter、CPU-heavy sklearn、UAB torch 候选都只能服务于 public adapter baseline / calibration baseline。
- `rigid_body rotation` 当前仍缺成对角速度字段；中期前要核验字段，能启用就补消融，不能启用就固化真实 diagnostics。
- 上游接收器、入库链路、原始大文件入仓中期前不重建。

运行环境：

- 所有 Python 脚本、测试、基准默认使用 `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`。
- 不要因为 shell 在 `base` 环境就直接运行 `python`。
- 如需 MySQL/InfluxDB 密钥，参考被 `.gitignore` 管理的 `docs/SECRETS.md`，不要把密钥写入其他文件。
- 默认 MySQL `127.0.0.1:3306`，InfluxDB `127.0.0.1:8086`。

建议执行顺序：

1. 先复核工作区：
   - `git status --short --branch`
   - 阅读 `docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/artifacts/ARTIFACTS.md`
   - 确认没有用户未提交代码会被误改。

2. 先做 `P10 evidence runner`，因为它是后续五项证据的基础设施：
   - 新增或扩展 `src/chronaris/pipelines/stage_i/evidence/closure_runner.py`
   - 新增 `scripts/stage_i/evidence/run_closure.py`
   - 新增 `tests/test_stage_i_evidence_runner.py`
   - 支持 `--reuse-existing`、`--skip-heavy`、`--only multitask|rigid_body|semantic|runtime|private_proxy|public_adapter|rotation|all`
   - 输出 `evidence_manifest.json`
   - manifest 必须记录输入路径、输出路径、命令参数、git commit、测试摘要、失败状态、`evidence_layer`
   - 每个子任务失败时保留 partial manifest，不覆盖上一轮稳定产物。

3. 做 `P11 thesis weak-label multitask sweep`：
   - 目标是深挖私有 Stage H 双流数据与 Phase C multitask checkpoint。
   - 固定输入：
     - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/run_manifest.json`
     - `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/run_manifest.json`
   - 建议新增 `src/chronaris/pipelines/stage_i/evidence/weak_label_sweep.py` 与 `scripts/stage_i/evidence/run_weak_label_sweep.py`
   - 小网格即可，不要无边界扩搜：
     - `physics_constraint_family=minimal|full|rigid_body`
     - `causal_weight=0|0.05|0.1`
     - `task_loss_weight=0.5|1.0`
     - `causal_lag_window_points=None|3`
   - 输出 `thesis weak-label multitask ablation` 表。
   - 每行必须带 `evidence_layer=thesis_weak_label`、输入 manifest、checkpoint path、git commit、metric definition。

4. 做 `P12 chronaris_opt component ablation`：
   - 目标是把 `chronaris_opt` 从“private proxy 最优候选”推进成“机制贡献可诊断”。
   - 重点文件：
     - `src/chronaris/pipelines/stage_i/private/optimization.py`
     - `src/chronaris/pipelines/stage_i/private/benchmark.py`
     - `scripts/stage_i/private/run_benchmark.py`
     - `tests/test_stage_i_private_optimization.py`
   - 至少拆解：
     - 去掉因果掩码。
     - 去掉时间残差或 lag-aware residual。
     - 去掉 task-aware head。
     - 仅保留 E/F/G/H 既有 baseline。
   - 输出 `chronaris_opt_component_ablation.csv/json` 和中文机制诊断报告。
   - 所有 `T1/T2/T3` 结果必须保持 `evidence_layer=private_proxy`。

5. 做 `P13 public adapter calibration`：
   - 目标是有限预算补做 CPU-heavy `sklearn` / UAB torch 候选，作为公开 adapter baseline / calibration baseline。
   - 不要扩成公开数据上的深度模型竞赛。
   - UAB 只允许 3 到 5 个候选组合，固定 seed、固定 LOSO、固定 `selected_subset`。
   - CPU-heavy `sklearn uab_hybrid` 必须显式 `--allow-cpu-heavy-sklearn`，并在 manifest 写入 `heavy_reason` 与 runtime。
   - NASA 继续以 `NASA enhanced round 1` 为主，只补必要 calibration 对照。
   - 输出 `public_adapter_calibration_summary.json`。
   - 报告必须区分 `public_adapter_baseline`、`calibration_baseline`、`legacy_public_opt`、`torch_uab`。

6. 做 `P14 NASA/UAB transfer boundary`：
   - 目标是形成中文迁移边界报告，说明公开代理数据与私有真实双流数据的差异。
   - 建议新增：
     - `src/chronaris/pipelines/stage_i/evidence/public_transfer_boundary.py`
     - `scripts/stage_i/evidence/build_public_transfer_boundary.py`
     - `tests/test_stage_i_public_transfer_boundary.py`
   - 报告至少包含：
     - 数据边界表：私有 Stage H、UAB、NASA 的模态、标签、任务粒度、时间基准。
     - 任务边界表：thesis weak-label、private proxy、public adapter/calibration 的分层。
     - 性能表：引用 `stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md` 和 P13 calibration summary。
   - 输出 `public_transfer_boundary_summary.json` 和 Markdown 报告。

7. 做 `P15 rigid_body rotation audit`：
   - 目标是核验 `rotation` 是否可以真实启用。
   - 重点查 MySQL label 和 Stage H feature schema 中的 `真航向`、俯仰、横滚、角速度、航向角速度等字段。
   - 若存在可用 rate field，扩充 token，重跑 `minimal / full / rigid_body`，并让报告出现 `vehicle_rigid_body_rotation > 0`。
   - 若只存在角度、没有 rate field，保留 `rotation disabled`，并输出 `missing_requirements.rotation` 诊断。
   - 不能把缺失的 rotation 写成已验证物理约束。

8. 做 `P16 paper table builder` 和 `P17 service boundary`，如果 P10-P15 已经有足够产物：
   - P16：统一导出论文表格，至少包括 thesis weak-label ablation、private proxy component ablation、public adapter calibration、transfer boundary、physical constraint ablation、runtime/semantic case。
   - P17：补离线/准实时推理服务边界、错误样例和 smoke CLI，不重建上游接收器。

每完成一项都必须：

- 真实实跑核心命令，不能只写代码。
- 运行相关测试；涉及 torch runtime 时按需设置 `CHRONARIS_ENABLE_TORCH_RUNTIME_TESTS=1`。
- 更新 `docs/STATE.md`、`docs/implementation/TASKS.md`、`docs/artifacts/ARTIFACTS.md` 和相关报告入口。
- 新增产物必须落在 `docs/artifacts/assets/...`，Markdown 报告落在 `docs/artifacts/stage_i/...` 或既有阶段目录。
- 报告必须写清 `evidence_layer` 与边界，不得混写 public adapter、private proxy、thesis weak-label。
- 运行 `git diff --check`。
- 如果有大文件或 checkpoint，确认是否应走 LFS 或是否已有仓库模式支持。

完成标准：

- P10-P15 至少各有可追溯 JSON summary 或 manifest。
- 至少形成六张论文可引用表：论文本体 weak-label 小网格、`chronaris_opt` 组件诊断、public adapter calibration、公开迁移边界、物理约束消融、runtime/semantic case。
- `docs/STATE.md` 与 `docs/implementation/TASKS.md` 对当前事实、已完成项、下一步和测试结果一致。
- `docs/artifacts/ARTIFACTS.md` 可以直接追溯最新 evidence manifest 和报告。
- 所有新增测试通过，或明确记录未能运行的原因。
```
