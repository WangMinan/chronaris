# Chronaris 当前任务

更新时间：2026-06-07

## 默认工作方式

每轮实现默认按下面顺序收敛：

1. 目标锁定。
2. 代码实现。
3. 测试闭环。
4. 文档回写。
5. 冗余清理。

运行 Python 脚本、测试、基准或阶段命令前，默认使用：

```bash
/home/wangminan/env/anaconda3/envs/chronaris/bin/python
```

## 当前 P0

目标：冻结当前 Stage I thesis mainline Phase C 工作区。

必须处理：

- 复查当前未提交文件，区分 Phase C 文档/代码改动与无关改动。
- 保留并复查新增文件：
  - `src/chronaris/models/alignment/task_heads.py`
  - `src/chronaris/dataset/stage_i_real_task_builders.py`
  - `src/chronaris/pipelines/stage_i/stage_i_multitask_train.py`
  - `tests/test_stage_i_multitask_train.py`
- 补跑：
  - `tests.test_stage_i_multitask_train`
  - `tests.test_stage_i_private_optimization`
  - `tests.test_alignment_model_losses`

## 当前 P1

目标：补一条真实资产上的 Stage I multitask 联合训练证据。

建议输出：

- `multitask_checkpoint.pt`
- `multitask_summary.json`
- `thesis_task_manifest.jsonl`
- 一份明确标注为 `thesis weak-label evidence` 的 Markdown 小报告。

## 当前 P2

目标：补 Stage F 刚体运动物理约束。

建议落点：

- `src/chronaris/models/alignment/physics_state_mapping.py`
- `src/chronaris/models/alignment/physics_residuals.py`
- `src/chronaris/models/alignment/physics.py`
- `src/chronaris/models/alignment/losses.py`
- `tests/test_alignment_model_losses.py`

## 当前 P3

目标：补 Stage G 语义事件融合。

建议落点：

- `src/chronaris/models/fusion/semantic_event.py`
- `src/chronaris/models/fusion/causal.py`
- `src/chronaris/pipelines/stage_i/stage_i_support_builders.py`
- `src/chronaris/pipelines/stage_i/stage_i_support.py`
- `tests/test_stage_i_support.py`

## 当前 P4

目标：补 runtime inference。

建议落点：

- `src/chronaris/dataset/streaming_windows.py`
- `src/chronaris/serving/runtime_inference.py`
- `scripts/run_stage_i_runtime_inference.py`
- `tests/test_runtime_inference.py`
