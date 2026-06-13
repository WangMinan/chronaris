# Stage I Artifacts Index

更新时间：2026-06-07

本目录保存当前 Stage I 报告集合；当前 AI coding 入口请先看 [../ARTIFACTS.md](../ARTIFACTS.md) 与 [../stage/stage-i/README.md](../stage/stage-i/README.md)。

## 当前主入口

- 中期证据整编：`stage-i-midterm-20260607T-stage-i-midterm-r3.md`
- 主动 evidence runner：`stage-i-evidence-closure-20260607T-stage-i-evidence-closure-r2.md`
- bounded weak-label sweep：`stage-i-thesis-weak-label-multitask-sweep-20260607T-stage-i-evidence-closure-r2-multitask.md`
- 历史 Stage I closure：`stage-i-closure-2026-04-30.md`
- Phase C thesis weak-label evidence：`thesis-weak-label-evidence-20260607T-stage-i-multitask-real-closure-r2.md`
- 当前公开主线：`stage-i-public-mainline-20260508T130100Z-stage-i-public-mainline-uab-robust-prior-r1.md`
- UAB robust-prior adapter：`stage-i-public-opt-20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1.md`
- NASA enhanced round 1：`stage-i-public-opt-20260506T161500Z-stage-i-public-opt-nasa-round1.md`
- UAB torch auto-cuda confirm：`stage-i-public-opt-20260506T165558Z-stage-i-public-opt-uab-torch-gpu.md`
- private component ablation：`stage-i-private-component-ablation-20260607T-stage-i-evidence-closure-r2-private-proxy.md`
- public adapter calibration：`stage-i-public-adapter-calibration-20260607T-stage-i-evidence-closure-r2-public-adapter.md`
- public transfer boundary：`stage-i-public-transfer-boundary-20260607T-stage-i-evidence-closure-r2-transfer-boundary.md`
- rigid-body rotation audit：`stage-i-rigid-body-rotation-audit-20260607T-stage-i-rotation-audit-r2.md`
- Phase D/E/F r2：
  - `stage-i-rigid-body-20260607T-stage-i-rigid-body-r2.md`
  - `stage-i-causal-support-20260607T-stage-i-support-semantic-r2.md`
  - `stage-i-semantic-event-support-20260607T-stage-i-semantic-support-r2.md`
  - `stage-i-runtime-inference-20260607T-stage-i-runtime-service-r2.md`
- 历史 support：
  - `stage-i-alignment-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-causal-support-20260506T120000Z-stage-i-support.md`
  - `stage-i-ablation-support-20260506T120000Z-stage-i-support.md`
- Thesis-facing demo：
  - `stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md`
  - `stage-i-anchor-20260506T165435Z-stage-i-anchor.md`

## 引用规则

- `public opt closed` 仍是公开支撑证据，但 UAB `target_prior_median` 只能写成 `public adapter / calibration evidence`。
- UAB/NASA 第二模态统一写成 `context proxy / public adapter evidence`。
- 私有分支需要区分 `T1/T2/T3 = private proxy benchmark / proxy evidence` 与 `risk_proxy / workload_proxy / event_replay_tag = thesis weak-label evidence`。
- 历史 archive 只用于追溯，不作为当前状态入口。
