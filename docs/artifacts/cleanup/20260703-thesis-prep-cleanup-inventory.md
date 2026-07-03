# P42 thesis-prep cleanup inventory

- generated_at_utc: 2026-07-03T13:27:29.872334Z
- scope: pre-cleanup audit only; no tracked artifact deleted before this inventory was written
- backup_root: `/home/wangminan/projects/chronaris-local-artifacts/cleanup-20260703/`
- python: `/home/wangminan/env/anaconda3/envs/chronaris/bin/python`

## 1. Git state
### `git status --short --untracked-files=all`
```text
(empty)
```

### `git log --oneline --decorate -10`
```text
756555c (HEAD -> main, origin/main) feat: add stage i thesis protocol freeze
21c4294 docs: add thesis prep execution plan
6228787 feat: add stage i optimized final polish
e8d367d feat: optimize stage i gpu outputs and docs cleanup
7cf5ed4 feat: add more tests on P30 and P31
587c205 feat: refine train process
edc0731 feat: refresh public evidence and prune artifacts
5e455c6 [DOC]update component pic
5395828 [DOC]update evidence pic
3caca1a feat: refresh stage i thesis figures
```

### `git rev-parse HEAD`
```text
756555c79627237800458cd8420a064441ab0147
```

### `git ls-remote origin refs/heads/main`
```text
756555c79627237800458cd8420a064441ab0147	refs/heads/main
```

## 2. Size footprint
### `du -sh docs`
```text
251M	docs
```

### `du -sh docs/artifacts`
```text
244M	docs/artifacts
```

### `du -sh docs/artifacts/assets`
```text
242M	docs/artifacts/assets
```

### `du -sh src scripts tests .git`
```text
4.7M	src
428K	scripts
740K	tests
8.2G	.git
```

### `git count-objects -vH`
```text
count: 1036
size: 13.57 MiB
in-pack: 8517
packs: 1
size-pack: 105.25 MiB
prune-packable: 0
garbage: 0
size-garbage: 0 bytes
```

### `du -sh .git/lfs .git/objects`
```text
8.0G	.git/lfs
120M	.git/objects
```

### `git lfs status`
```text
On branch main
Objects to be pushed to origin/main:


Objects to be committed:


Objects not staged for commit:
```

### `git lfs ls-files | wc -l`
```text
352
```

### `git lfs migrate info --include-ref=refs/heads/main --include='docs/**'`
```text
migrate: Sorting commits: ..., done.
migrate: Examining commits: 100% (61/61), done.
*.json     	63 MB 	617/617 files(s)	100%
*.log      	63 MB 	120/120 files(s)	100%
*.png      	41 MB 	642/642 files(s)	100%
*.parquet  	11 MB 	    5/5 files(s)	100%
*.docx     	6.0 MB	    2/2 files(s)	100%

LFS Objects	132 MB	403/404 files(s)	100%
```

### `find docs/artifacts/assets -mindepth 1 -maxdepth 2 -type d ... head -80`
```text
51M	docs/artifacts/assets/stage_i_optimized_final_polish
33M	docs/artifacts/assets/stage_h
32M	docs/artifacts/assets/stage_i_stream_role_fusion
31M	docs/artifacts/assets/stage_i_private
13M	docs/artifacts/assets/stage_i_task_heads_optimization
13M	docs/artifacts/assets/stage_i_multitask
10M	docs/artifacts/assets/stage_i_private_thirdparty_comparison
9.6M	docs/artifacts/assets/stage_i_public_opt_torch
9.0M	docs/artifacts/assets/stage_i_public_fusion_ablation
8.0M	docs/artifacts/assets/stage_i_public_opt
7.8M	docs/artifacts/assets/stage_i_private_leakage_safe_ablation
5.9M	docs/artifacts/assets/stage_i
4.6M	docs/artifacts/assets/stage_i_public_fusion_refresh
4.0M	docs/artifacts/assets/stage_i_thesis_figures
2.3M	docs/artifacts/assets/stage_i_private_component_ablation
1.5M	docs/artifacts/assets/stage_i_midterm_runtime
1.2M	docs/artifacts/assets/stage_i_public_fusion_screen
1.2M	docs/artifacts/assets/stage_i_cross_evidence_matrix
932K	docs/artifacts/assets/stage_i_public_model_comparison
860K	docs/artifacts/assets/stage_i_midterm
824K	docs/artifacts/assets/stage_i_runtime_service
768K	docs/artifacts/assets/stage_i_optimized_reevaluation
736K	docs/artifacts/assets/stage_i_multitask_sweep
496K	docs/artifacts/assets/stage_i_thesis_protocol
480K	docs/artifacts/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-g-min
404K	docs/artifacts/assets/alignment-preview-stage-g-min-closure-2026-04-22-stage-f-baseline
404K	docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-stage-f-full
376K	docs/artifacts/assets/stage_i_llm_preprocessing
304K	docs/artifacts/assets/alignment-preview-stage-e-closure-2026-04-21-none
300K	docs/artifacts/assets/alignment-preview-stage-f-closure-2026-04-22-e-baseline
300K	docs/artifacts/assets/alignment-preview-stage-e-closure-2026-04-21-zscore_train
244K	docs/artifacts/assets/stage_i_runtime_inference
236K	docs/artifacts/assets/stage_i_support
216K	docs/artifacts/assets/stage_i_optimized_model_summary
192K	docs/artifacts/assets/stage_i_gpu_parallel_profile
64K	docs/artifacts/assets/stage_i_rotation_audit
60K	docs/artifacts/assets/stage_i_rigid_body
52K	docs/artifacts/assets/stage_i_llm_comparison
48K	docs/artifacts/assets/stage_i_evidence
44K	docs/artifacts/assets/stage_i_semantic_event_support
44K	docs/artifacts/assets/stage_i_public_adapter_calibration
32K	docs/artifacts/assets/stage_i_anchor
24K	docs/artifacts/assets/stage_i_public_transfer_boundary
20K	docs/artifacts/assets/stage_i_public_mainline
12K	docs/artifacts/assets/stage_i_runtime_demo
```

### Largest tracked/current artifact files by filesystem size
| size | path |
| ---: | --- |
| 12.2 MiB | `docs/artifacts/assets/stage_i_multitask/20260607T-stage-i-multitask-real-closure-r2/thesis_task_manifest.jsonl` |
| 7.0 MiB | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/training_curves.csv` |
| 6.4 MiB | `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json` |
| 6.3 MiB | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/training_curves.csv` |
| 6.3 MiB | `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/task_manifest.jsonl` |
| 6.3 MiB | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3/task_manifest.jsonl` |
| 6.3 MiB | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/task_manifest.jsonl` |
| 6.3 MiB | `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_manifest.jsonl` |
| 6.2 MiB | `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_task_manifest.jsonl` |
| 6.2 MiB | `docs/artifacts/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/private_task_manifest.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251005_四01_ACT-4_云_J20_22#01/views/20251005_四01_ACT-4_云_J20_22#01__pilot_10033/raw_window_summary.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/sorties/20251005_四01_ACT-4_云_J20_22#01/views/20251005_四01_ACT-4_云_J20_22#01__pilot_10033/raw_window_summary.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035/raw_window_summary.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-f-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033/raw_window_summary.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10035/raw_window_summary.jsonl` |
| 5.0 MiB | `docs/artifacts/assets/stage_h/20260502T092753Z-stage-h-e-allwindow-clean/sorties/20251002_单01_ACT-8_翼云_J16_12#01/views/20251002_单01_ACT-8_翼云_J16_12#01__pilot_10033/raw_window_summary.jsonl` |
| 4.6 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_context_adapter_only_cap2x_do0p2__seed42/deep_baseline_summary.json` |
| 4.6 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_force_adaptive_context_gate__seed42/deep_baseline_summary.json` |
| 4.6 MiB | `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/t3_similarity_distribution.csv` |
| 4.4 MiB | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3/gpu_perf_batches.csv` |
| 4.2 MiB | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/training_curves.csv` |
| 3.6 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/nasa_csm/p37_public_context_adapter_only_cap2x_do0p2__seed42/deep_baseline_summary.json` |
| 3.5 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/nasa_csm/p37_public_force_adaptive_context_gate__seed42/deep_baseline_summary.json` |
| 3.4 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_context_adapter_only_cap2x_do0p2__seed42/training_curves.csv` |
| 3.4 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_force_adaptive_context_gate__seed42/training_curves.csv` |
| 3.1 MiB | `docs/artifacts/assets/stage_i_public_opt_torch/20260508T090700Z-stage-i-public-opt-uab-heat-specialist-r1/public_opt_torch_feature_frame.parquet` |
| 3.1 MiB | `docs/artifacts/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch/public_opt_torch_feature_frame.parquet` |
| 3.1 MiB | `docs/artifacts/assets/stage_i_public_opt_torch/20260506T165558Z-stage-i-public-opt-uab-torch-gpu/public_opt_torch_feature_frame.parquet` |
| 3.1 MiB | `docs/artifacts/assets/stage_i_public_opt/20260508T125651Z-stage-i-public-opt-uab-robust-prior-r1/public_opt_feature_frame.parquet` |
| 2.9 MiB | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/gpu_perf_batches.csv` |
| 2.9 MiB | `docs/artifacts/assets/stage_i_public_opt/20260506T161500Z-stage-i-public-opt-nasa-round1/public_opt_feature_frame.parquet` |
| 2.6 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/nasa_csm/p37_public_context_adapter_only_cap2x_do0p2__seed42/training_curves.csv` |
| 2.6 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/nasa_csm/p37_public_force_adaptive_context_gate__seed42/training_curves.csv` |
| 2.2 MiB | `docs/artifacts/assets/stage_i_private_component_ablation/20260607T-stage-i-evidence-closure-r2-private-proxy/chronaris_opt_component_ablation.json` |
| 2.1 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_context_adapter_only_cap2x_do0p2__seed42/run.log` |
| 2.0 MiB | `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json` |
| 2.0 MiB | `docs/artifacts/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/private_benchmark_summary.json` |
| 1.9 MiB | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/20260702T-stage-i-optimized-final-polish-r1-public-confirm/confirm/uab_workload_dataset/p37_public_force_adaptive_context_gate__seed42/run.log` |
| 1.6 MiB | `docs/artifacts/assets/stage_i_public_opt/20260506T121000Z-stage-i-public-opt-uab/public_opt_feature_frame.parquet` |
| 1.3 MiB | `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/label_feature_overlap_audit.json` |

## 3. Current artifact reference table
P38 registry is the current thesis protocol freeze entry. `exists=True` for all rows at audit time.

| stage | artifact root | primary result | report | boundary |
| --- | --- | --- | --- | --- |
| P30 | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1` | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv` | `docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md` | Private Stage H proxy-task third-party comparison; not expert truth. |
| P31 | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1` | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/ablation_summary.csv` | `docs/artifacts/stage_i/stage-i-public-fusion-ablation-20260702T-stage-i-public-fusion-ablation-gpuopt-r1.md` | Public adapter context-proxy component ablation; not real aircraft-bus validation. |
| P32 | `docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1` | `docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv` | `docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md` | Cross-evidence routing matrix; do not merge private/public/proxy metrics into one leaderboard. |
| P34 | `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20` | `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/task_head_metrics_long.csv` | `docs/artifacts/stage_i/stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md` | CUDA 20-epoch task-head confirm; T3 remains mixed and must not be overstated. |
| P35 | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20` | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_metrics.csv` | `docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md` | Stream-role v3 confirm separates private real vehicle streams from public context proxy streams. |
| P36 | `docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20` | `docs/artifacts/assets/stage_i_optimized_reevaluation/20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20/optimized_cross_evidence_matrix.csv` | `docs/artifacts/stage_i/stage-i-optimized-chronaris-reevaluation-20260702T-stage-i-optimized-reevaluation-r4-v3-confirm20.md` | Aggregation over fixed references; does not overwrite P30/P31/P32 confirmed results. |
| P36_summary | `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20` | `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/optimized_model_summary.csv` | `docs/artifacts/stage_i/stage-i-optimized-model-summary-20260702T-stage-i-optimized-model-summary-r4-v3-confirm20.md` | Paper-facing summary table and claim boundaries over fixed source artifacts. |
| P37 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1` | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/optimized_final_polish_summary.json` | `docs/artifacts/stage_i/stage-i-optimized-final-polish-20260702T-stage-i-optimized-final-polish-r1.md` | Final polish accepts T1/public route only; T3 rejected and public remains context proxy. |

### P38 result matrix coverage
- rows: 505
- by quadrant: private_component_ablation=204, private_model_comparison=85, public_component_ablation=132, public_model_comparison=84
- by source_stage: P24_via_P32=36, P27_via_P32=80, P30_via_P32=72, P31_via_P32=104, P34=56, P35=112, P36_summary=29, P37=16
- unique referenced paths in artifact_path/source_file/figure_path: 23

Top referenced paths from P38 result matrix:
| refs | path | exists |
| ---: | --- | --- |
| 292 | `docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv` | True |
| 196 | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_metrics.csv` | True |
| 112 | `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/task_head_metrics_long.csv` | True |
| 104 | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/ablation_summary.csv` | True |
| 104 | `docs/artifacts/assets/stage_i_public_fusion_ablation/20260702T-stage-i-public-fusion-ablation-gpuopt-r1/fig_public_ablation_delta_heatmap.png` | True |
| 98 | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/fig_p35_private_task_delta.png` | True |
| 80 | `docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/model_comparison_long.csv` | True |
| 80 | `docs/artifacts/assets/stage_i_public_model_comparison/20260701T-stage-i-public-model-comparison-r1/fig_public_model_delta_heatmap.png` | True |
| 72 | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv` | True |
| 72 | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png` | True |
| 58 | `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/key_metric_summary.csv` | True |
| 56 | `docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/fig_p34_delta_vs_p30_heatmap.png` | True |
| 36 | `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/ablation_summary.json` | True |
| 36 | `docs/artifacts/assets/stage_i_private_leakage_safe_ablation/20260619T-stage-i-leakage-safe-ablation-r2/model_backbone_ablation.png` | True |
| 29 | `docs/artifacts/assets/stage_i_optimized_model_summary/20260702T-stage-i-optimized-model-summary-r4-v3-confirm20/fig_model_summary_metric_delta.png` | True |
| 28 | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_metrics.csv` | True |
| 16 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/t1_calibration_metrics.csv` | True |
| 14 | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/fig_p35_public_ablation_comparison.png` | True |
| 8 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t1_macro_f1_leaderboard.png` | True |
| 8 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/t3_final_polish_metrics.csv` | True |
| 8 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/public_route_calibration_metrics.csv` | True |
| 4 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_t3_delta_vs_p34.png` | True |
| 4 | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/fig_p37_public_route_delta_heatmap.png` | True |

### Claim boundary rows
| claim_id | dataset_role | source_stage | boundary |
| --- | --- | --- | --- |
| private_real_dual_stream_scope | private_real_dual_stream | P30/P34/P35/P37 | Do not describe T1/T2/T3 as expert-label ground truth or claim new Dingxin data was obtained. |
| private_component_ablation_scope | private_real_dual_stream | P32/P34/P35/P36 | Report mixed or failed rows, especially T3; do not write a blanket Chronaris superiority claim. |
| public_context_proxy_scope | public_context_proxy | P31/P35/P37 | Public second stream is context proxy, not private aircraft-bus telemetry. |
| public_component_ablation_scope | public_context_proxy | P31/P35/P36 | Do not let public component wins override private dual-stream limitations. |
| p37_final_polish_scope | mixed_private_public | P37 | P37 T3 is rejected and remains P34 confirmed retrieval; P37 does not rerun or overwrite P30/P31/P34/P35/P36. |
| synthetic_future_scope | synthetic_stress_test | P39-future | Synthetic rows must remain appendix stress tests and cannot replace real data or expert truth. |
| llm_future_scope | llm_semantic_context | P20/P21/P39-future | LLM output is not expert evaluation data and must not directly become truth labels. |

## 4. Code complexity table
- Python files under `src scripts tests`: 287
- Total Python lines: 86371
- Python files under `src/chronaris/pipelines/stage_i scripts/stage_i tests`: 190
- Files with `stage_i` in path/name: 188
- `docs/artifacts/assets/stage_i*` root directories: 38
- `docs/artifacts/stage_i/*.md` reports: 56

Largest Python files:
| lines | path | cleanup reading |
| ---: | --- | --- |
| 2353 | `src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1838 | `src/chronaris/pipelines/stage_i/public/fusion_ablation.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1745 | `src/chronaris/pipelines/stage_i/evidence/thesis_materials_figures.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1625 | `src/chronaris/pipelines/stage_i/public/fusion_refresh.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1542 | `src/chronaris/pipelines/stage_i/public/deep_baseline_runtime.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1489 | `tests/test_stage_i_public_opt.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1476 | `scripts/run_stage_e_relative_preview.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1144 | `src/chronaris/pipelines/stage_i/public/model_comparison.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1128 | `src/chronaris/features/stage_i_sequences.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1052 | `tests/test_stage_h_export.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 1004 | `src/chronaris/pipelines/stage_i/evidence/thesis_materials_data.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 994 | `src/chronaris/serving/runtime_inference.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 969 | `tests/test_stage_i_deep_pipeline.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 963 | `src/chronaris/pipelines/stage_i/evidence/thesis_protocol.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 919 | `src/chronaris/pipelines/stage_i/private/leakage_safe_ablation.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 899 | `src/chronaris/pipelines/alignment_preview.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 823 | `tests/test_alignment_model_losses.py` | must split eventually; P42 targets wrappers/shims first and records large-file backlog |
| 797 | `src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py` | split candidate |
| 758 | `src/chronaris/pipelines/stage_i/llm/preprocessing.py` | split candidate |
| 746 | `src/chronaris/models/alignment/physics.py` | split candidate |
| 738 | `src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py` | split candidate |
| 732 | `src/chronaris/pipelines/stage_i/public/opt_sklearn_heads.py` | split candidate |
| 727 | `tests/test_stage_i_support.py` | split candidate |
| 721 | `src/chronaris/serving/runtime_service_smoke.py` | split candidate |
| 721 | `src/chronaris/pipelines/stage_i/training/multitask_train.py` | split candidate |
| 719 | `src/chronaris/pipelines/stage_i/public/opt_data.py` | split candidate |
| 714 | `src/chronaris/pipelines/causal_fusion.py` | split candidate |
| 712 | `src/chronaris/pipelines/stage_i/common/deep_models.py` | split candidate |
| 703 | `src/chronaris/pipelines/stage_h/export.py` | split candidate |
| 682 | `src/chronaris/pipelines/stage_i/public/fusion_gpuopt.py` | split candidate |

Raw top-80 wc output:
```text
86365 total
   2353 src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py
   1838 src/chronaris/pipelines/stage_i/public/fusion_ablation.py
   1745 src/chronaris/pipelines/stage_i/evidence/thesis_materials_figures.py
   1625 src/chronaris/pipelines/stage_i/public/fusion_refresh.py
   1542 src/chronaris/pipelines/stage_i/public/deep_baseline_runtime.py
   1489 tests/test_stage_i_public_opt.py
   1476 scripts/run_stage_e_relative_preview.py
   1144 src/chronaris/pipelines/stage_i/public/model_comparison.py
   1128 src/chronaris/features/stage_i_sequences.py
   1052 tests/test_stage_h_export.py
   1004 src/chronaris/pipelines/stage_i/evidence/thesis_materials_data.py
    994 src/chronaris/serving/runtime_inference.py
    969 tests/test_stage_i_deep_pipeline.py
    963 src/chronaris/pipelines/stage_i/evidence/thesis_protocol.py
    919 src/chronaris/pipelines/stage_i/private/leakage_safe_ablation.py
    899 src/chronaris/pipelines/alignment_preview.py
    823 tests/test_alignment_model_losses.py
    797 src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py
    758 src/chronaris/pipelines/stage_i/llm/preprocessing.py
    746 src/chronaris/models/alignment/physics.py
    738 src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py
    732 src/chronaris/pipelines/stage_i/public/opt_sklearn_heads.py
    727 tests/test_stage_i_support.py
    721 src/chronaris/serving/runtime_service_smoke.py
    721 src/chronaris/pipelines/stage_i/training/multitask_train.py
    719 src/chronaris/pipelines/stage_i/public/opt_data.py
    714 src/chronaris/pipelines/causal_fusion.py
    712 src/chronaris/pipelines/stage_i/common/deep_models.py
    703 src/chronaris/pipelines/stage_h/export.py
    682 src/chronaris/pipelines/stage_i/public/fusion_gpuopt.py
    645 src/chronaris/pipelines/stage_i/evidence/midterm_pack.py
    641 tests/test_stage_i_case_study.py
    639 src/chronaris/evaluation/stage_i_case_metrics.py
    635 tests/test_alignment_pipeline.py
    635 src/chronaris/pipelines/stage_i/evidence/support_builders.py
    611 src/chronaris/pipelines/stage_i/private/optimization.py
    604 src/chronaris/models/alignment/losses.py
    600 src/chronaris/pipelines/stage_i/private/benchmark_models.py
    587 src/chronaris/pipelines/stage_i/public/opt_torch_candidates.py
    586 src/chronaris/pipelines/stage_i/public/gpu_runtime.py
    583 src/chronaris/pipelines/stage_i/private/benchmark_data.py
    568 tests/test_stage_i_pipeline.py
    568 src/chronaris/pipelines/stage_i/evidence/optimized_model_summary.py
    566 tests/test_access_metadata_live.py
    564 src/chronaris/pipelines/stage_i/public/deep_baseline.py
    533 src/chronaris/pipelines/stage_i/evidence/optimized_reevaluation.py
    519 src/chronaris/pipelines/stage_i/private/task_head_optimization.py
    519 src/chronaris/pipelines/stage_i/evidence/weak_label_sweep_helpers.py
    516 src/chronaris/pipelines/stage_i/private/benchmark.py
    512 src/chronaris/pipelines/partial_data/builder.py
    497 src/chronaris/pipelines/stage_i/public/opt_sklearn.py
    495 src/chronaris/pipelines/stage_i/evidence/anchors.py
    493 src/chronaris/pipelines/stage_i/public/opt_torch.py
    485 src/chronaris/dataset/uab_stage_i.py
    477 src/chronaris/features/stage_i_feature_helpers.py
    473 src/chronaris/pipelines/stage_i/public/mainline_report.py
    472 src/chronaris/pipelines/stage_i/evidence/cross_evidence_matrix.py
    467 src/chronaris/pipelines/stage_i/llm/comparison_builders.py
    467 src/chronaris/pipelines/__init__.py
    455 src/chronaris/access/mysql_metadata.py
    452 src/chronaris/serving/runtime_demo.py
    450 src/chronaris/pipelines/stage_i/evidence/thesis_materials.py
    447 tests/test_stage_i_llm_preprocessing.py
    446 src/chronaris/pipelines/stage_h/export_helpers.py
    436 src/chronaris/pipelines/stage_i/llm/harness.py
    427 src/chronaris/pipelines/stage_i/evidence/closure_runner.py
    422 src/chronaris/dataset/stage_i_sequence_contracts.py
    420 src/chronaris/dataset/stage_i_real_task_builders.py
    410 tests/test_runtime_inference.py
    409 src/chronaris/pipelines/stage_i/legacy/baseline.py
    408 src/chronaris/llm/schemas.py
    408 src/chronaris/features/stage_i_features.py
    404 src/chronaris/pipelines/stage_i/public/fusion_screen.py
    386 src/chronaris/evaluation/alignment_diagnostics.py
    378 src/chronaris/pipelines/stage_i/public/opt_reporting.py
    378 src/chronaris/pipelines/stage_i/evidence/weak_label_sweep.py
    374 src/chronaris/pipelines/stage_i/evidence/private_component_ablation.py
    363 scripts/stage_i/public/run_opt.py
    362 tests/test_stage_i_thesis_materials.py
```

## 5. Import/reference audit for cleanup candidates
### P37 nested public/private child outputs
```text
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:295:            output_root=str(run_root / "nested_private"),
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:342:            artifact_root=str(run_root / "nested_public"),
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:444:    _copy_if_exists(t3_root / "fold_predictions.csv", run_root / "t3_retrieval_predictions.csv")
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py:11:        "t3_screen_root": run_root / "nested_private" / f"{run_id}-t3-screen",
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py:12:        "t1_screen_root": run_root / "nested_private" / f"{run_id}-t1-screen",
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py:13:        "private_confirm_root": run_root / "nested_private" / f"{run_id}-private-confirm",
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py:14:        "public_screen_root": run_root / "nested_public" / f"{run_id}-public-screen",
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish_support.py:15:        "public_confirm_root": run_root / "nested_public" / f"{run_id}-public-confirm",
```

### P35 nested private/public v3 roots
```text
docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md:9:- private_confirm_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/private_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-private-v3`
docs/artifacts/stage_i/stage-i-stream-role-aware-fusion-20260702T-stage-i-stream-role-fusion-r4-v3-confirm20.md:10:- public_confirm_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/public_v3_confirm/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20-public-v3`
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:105:def _run_private_v3_confirm(
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:114:            output_root=str(run_root / "private_v3_confirm"),
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:143:def _run_public_v3_confirm(
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:152:            artifact_root=str(run_root / "public_v3_confirm"),
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:256:        private_result = _run_private_v3_confirm(config, run_root)
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:260:        public_result = _run_public_v3_confirm(config, run_root)
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:492:        ("P35_private_v3_confirm", private_root),
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:493:        ("P35_public_v3_confirm", public_root),
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:515:        ("P35_private_v3_confirm", private_root),
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:516:        ("P35_public_v3_confirm", public_root),
```

### P30 task manifest direct docs/code refs
```text
[exit=1]
```

### P34 task manifest direct docs/code refs
```text
[exit=1]
```

### old private opt packages
```text
tests/test_stage_i_support.py:507:                "run_id": "20260504T120000Z-stage-i-private-opt-package",
docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md:1:# Private Optimization Summary - 20260504T120000Z-stage-i-private-opt-package
docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md:33:- optimized candidate summary: `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_summary.json`
docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md:34:- optimized candidate metrics: `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_metrics.csv`
docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md:35:- optimized candidate package: `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md:36:- optimized package report: `docs/artifacts/private/private-optimized-package-20260504T120000Z-stage-i-private-opt-package.md`
docs/artifacts/private/README.md:9:- `private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md`
docs/artifacts/private/README.md:10:- `private-optimized-package-20260607T-stage-i-private-opt-package-r2.md`
docs/artifacts/private/README.md:14:- [../assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json](../assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json)
docs/artifacts/private/private-optimized-package-20260504T120000Z-stage-i-private-opt-package.md:1:# Private Optimized Package - 20260504T120000Z-stage-i-private-opt-package
docs/artifacts/private/private-optimized-package-20260504T120000Z-stage-i-private-opt-package.md:3:- package path: `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/private/archive/package_support_20260504/private-optimality-summary-20260504T120000Z-stage-i-private-opt-package.md:1:# Private Optimality Summary - 20260504T120000Z-stage-i-private-opt-package
scripts/stage_i/evidence/export_anchors.py:40:            "20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json"
docs/artifacts/private-alignment-support-20260607T-stage-i-private-opt-package-r2.md:1:# Private Alignment Support - 20260607T-stage-i-private-opt-package-r2
docs/artifacts/stage_i/stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md:5:- source_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/stage_i/stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md:9:- package_path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/stage_i/stage-i-runtime-demo-20260506T165435Z-stage-i-runtime-demo.md:11:- source_run_id: `20260504T120000Z-stage-i-private-opt-package`
docs/artifacts/private/archive/package_support_20260504/private-causal-fusion-support-20260504T120000Z-stage-i-private-opt-package.md:1:# Private Causal Fusion Support - 20260504T120000Z-stage-i-private-opt-package
docs/artifacts/private/archive/package_support_20260504/private-alignment-support-20260504T120000Z-stage-i-private-opt-package.md:1:# Private Alignment Support - 20260504T120000Z-stage-i-private-opt-package
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:25:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:26:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt_no_causal_mask | 0.173333 | 0.826667 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:27:| private | T2_next_window_physiology_response | rmse | chronaris_opt | 201.489565 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:28:| private | T2_next_window_physiology_response | rmse | chronaris_opt_no_causal_mask | 313.232477 | 111.742912 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:29:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260509T071500Z-stage-i-midterm-r2.md:30:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt_no_causal_mask | 0.027027 | 0.972973 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:25:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:26:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt_no_causal_mask | 0.173333 | 0.826667 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:27:| private | T2_next_window_physiology_response | rmse | chronaris_opt | 201.489565 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:28:| private | T2_next_window_physiology_response | rmse | chronaris_opt_no_causal_mask | 313.232477 | 111.742912 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:29:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json |  |
docs/artifacts/stage_i/stage-i-midterm-20260607T-stage-i-midterm-r3.md:30:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt_no_causal_mask | 0.027027 | 0.972973 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json | vs target |
docs/implementation/notes/coding-roadmap.md:590:1. `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/private/archive/full_loso_20260502/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md:1:# Private Optimization Summary - 20260502T121815Z-stage-i-private-opt-full
docs/artifacts/private/archive/full_loso_20260502/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md:33:- optimized candidate summary: `docs/artifacts/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/optimized_candidate_summary.json`
docs/artifacts/private/archive/full_loso_20260502/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md:34:- optimized candidate metrics: `docs/artifacts/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/optimized_candidate_metrics.csv`
docs/artifacts/private/archive/full_loso_20260502/private-alignment-support-20260502T121815Z-stage-i-private-opt-full.md:1:# Private Alignment Support - 20260502T121815Z-stage-i-private-opt-full
docs/artifacts/ARTIFACTS.md:83:- Private optimized package summary：[private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md](private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md)
docs/artifacts/ARTIFACTS.md:99:- 当前 `chronaris_opt` package：[assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json](assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json)
docs/implementation/notes/stage-i-mainline-transition-2026-05-04.md:77:- `docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/private/archive/full_loso_20260502/private-causal-fusion-support-20260502T121815Z-stage-i-private-opt-full.md:1:# Private Causal Fusion Support - 20260502T121815Z-stage-i-private-opt-full
docs/artifacts/private/archive/full_loso_20260502/private-optimality-summary-20260502T121815Z-stage-i-private-opt-full.md:1:# Private Optimality Summary - 20260502T121815Z-stage-i-private-opt-full
docs/artifacts/private-causal-fusion-support-20260607T-stage-i-private-opt-package-r2.md:1:# Private Causal Fusion Support - 20260607T-stage-i-private-opt-package-r2
docs/artifacts/private-optimized-package-20260607T-stage-i-private-opt-package-r2.md:1:# Private Optimized Package - 20260607T-stage-i-private-opt-package-r2
docs/artifacts/private-optimized-package-20260607T-stage-i-private-opt-package-r2.md:3:- package path: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json`
docs/implementation/notes/archive/stage_i/stage-i-private-benchmark-plan-2026-05-02.md:136:- `private benchmark full` 已完成：`docs/artifacts/assets/stage_i_private/20260502T121815Z-stage-i-private-opt-full/`
docs/implementation/notes/archive/stage_i/stage-i-private-benchmark-plan-2026-05-02.md:137:- `private benchmark package` 已完成：`docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/`
docs/implementation/notes/archive/stage_i/stage-i-private-benchmark-plan-2026-05-02.md:145:  - 当前可引用 package：`docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/optimized_candidate_package.json`
docs/artifacts/private-optimality-summary-20260607T-stage-i-private-opt-package-r2.md:1:# Private Optimality Summary - 20260607T-stage-i-private-opt-package-r2
docs/STATE.md:185:  - private benchmark 分层资产：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_benchmark_summary.json`
docs/STATE.md:252:   - 输出：`docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/`
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:25:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:26:| private | T1_maneuver_intensity_class | macro_f1 | chronaris_opt_no_causal_mask | 0.173333 | 0.826667 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:27:| private | T2_next_window_physiology_response | rmse | chronaris_opt | 201.489565 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:28:| private | T2_next_window_physiology_response | rmse | chronaris_opt_no_causal_mask | 313.232477 | 111.742912 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:29:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt | 1.000000 | nan | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json |  |
docs/artifacts/stage_i/archive/public_history/stage-i-midterm-20260509T060500Z-stage-i-midterm.md:30:| private | T3_paired_pilot_window_retrieval | top1_accuracy | chronaris_opt_no_causal_mask | 0.027027 | 0.972973 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/private_benchmark_summary.json | vs target |
docs/implementation/TASKS.md:253:- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_manifest.jsonl`。
docs/implementation/TASKS.md:254:- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/private_proxy_task_summary.json`。
docs/implementation/TASKS.md:256:- 已生成 `docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/thesis_task_summary.json`。
docs/artifacts/stage_i/archive/deep_history/thesis-support-assessment-2026-05-01.md:7:> - `docs/artifacts/private/archive/full_loso_20260502/private-optimization-summary-20260502T121815Z-stage-i-private-opt-full.md`
docs/artifacts/stage_i/archive/deep_history/thesis-support-assessment-2026-05-01.md:8:> - `docs/artifacts/private/private-optimization-summary-20260504T120000Z-stage-i-private-opt-package.md`
src/chronaris/pipelines/stage_i/evidence/public_transfer_boundary.py:36:    "docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/"
docs/artifacts/private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md:1:# Private Optimization Summary - 20260607T-stage-i-private-opt-package-r2
docs/artifacts/private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md:36:- optimized candidate summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_summary.json`
docs/artifacts/private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md:37:- optimized candidate metrics: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_metrics.csv`
docs/artifacts/private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md:38:- optimized candidate package: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private/20260607T-stage-i-private-opt-package-r2/optimized_candidate_package.json`
docs/artifacts/private-optimization-summary-20260607T-stage-i-private-opt-package-r2.md:39:- optimized package report: `/home/wangminan/projects/chronaris/docs/artifacts/private-optimized-package-20260607T-stage-i-private-opt-package-r2.md`
src/chronaris/pipelines/stage_i/evidence/midterm_pack.py:35:    "docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
src/chronaris/pipelines/stage_i/evidence/support.py:50:    "docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
src/chronaris/pipelines/stage_i/evidence/anchors.py:21:    "docs/artifacts/assets/stage_i_private/20260504T120000Z-stage-i-private-opt-package/"
```

### legacy third-party wrapper names
```text
tests/test_stage_i_private_third_party_comparison.py:3:from tests.test_stage_i_private_thirdparty_comparison import *  # noqa: F401,F403
scripts/stage_i/private/run_private_thirdparty_comparison.py:22:from chronaris.pipelines.stage_i.private.thirdparty_comparison import (  # noqa: E402
scripts/stage_i/private/run_private_thirdparty_comparison.py:25:    run_stage_i_private_thirdparty_comparison,
scripts/stage_i/private/run_private_thirdparty_comparison.py:47:    parser.add_argument("--output-root", default="docs/artifacts/assets/stage_i_private_thirdparty_comparison")
scripts/stage_i/private/run_private_thirdparty_comparison.py:118:    result = run_stage_i_private_thirdparty_comparison(
tests/test_stage_i_final_polish.py:22:from chronaris.pipelines.stage_i.private.thirdparty_comparison import _class_balanced_weights  # noqa: E402
scripts/stage_i/private/run_private_third_party_comparison.py:12:from run_private_thirdparty_comparison import main  # noqa: E402
docs/artifacts/ARTIFACTS.md:114:- 当前 private third-party comparison assets：[assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/)
docs/artifacts/ARTIFACTS.md:115:- 当前 private third-party comparison CSV/summary：[private_thirdparty_summary.json](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json)、[model_comparison_wide.csv](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_wide.csv)、[improvement_summary.csv](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/improvement_summary.csv)、[gpu_perf_summary.json](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/gpu_perf_summary.json)
docs/artifacts/ARTIFACTS.md:116:- 当前 private third-party comparison figures：[fig_private_thirdparty_t1_macro_f1.png](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t1_macro_f1.png)、[fig_private_thirdparty_t2_rmse.png](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t2_rmse.png)、[fig_private_thirdparty_t3_retrieval.png](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t3_retrieval.png)、[fig_private_thirdparty_delta_heatmap.png](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png)
docs/artifacts/ARTIFACTS.md:117:- 当前 private third-party comparison manifest/log：[evidence_manifest.json](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json)、[run.log](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/run.log)、[progress.json](assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/progress.json)
docs/artifacts/ARTIFACTS.md:162:- P30 `stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/` 是 private real dual-stream Stage H 的 T1/T2/T3 proxy task 第三方对比。它补齐 MulT、ContiFormer、naive time sync 和 classical baseline 对照，但结果为混合对比，不能写成 Chronaris 全面胜出或论文最终人工真值任务。
tests/test_stage_i_private_thirdparty_comparison.py:18:from chronaris.pipelines.stage_i.private.third_party_comparison import (  # noqa: E402
tests/test_stage_i_private_thirdparty_comparison.py:21:    run_stage_i_private_thirdparty_comparison,
tests/test_stage_i_private_thirdparty_comparison.py:51:            result = run_stage_i_private_thirdparty_comparison(
tests/test_stage_i_thesis_protocol.py:52:                        "evidence_quadrant": "private_thirdparty_comparison",
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:19:| private_thirdparty_comparison | 72 |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:34:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | chronaris_full | macro_f1 | 0.1733 |  |  |  | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:35:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | chronaris_full | balanced_accuracy | 0.3333 |  |  |  | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:36:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | chronaris_full | macro_f1 | 0.1733 |  |  |  | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:37:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | chronaris_full | balanced_accuracy | 0.3333 |  |  |  | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:38:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | mult | macro_f1 | 0.2035 | chronaris_full | -0.0301 | -14.8070 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:39:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | mult | balanced_accuracy | 0.3533 | chronaris_full | -0.0199 | -5.6452 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:40:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | mult | macro_f1 | 0.2132 | chronaris_full | -0.0399 | -18.6946 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:41:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | mult | balanced_accuracy | 0.3397 | chronaris_full | -0.0064 | -1.8868 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:42:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | contiformer | macro_f1 | 0.2096 | chronaris_full | -0.0362 | -17.2862 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:43:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | contiformer | balanced_accuracy | 0.3561 | chronaris_full | -0.0228 | -6.4000 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:44:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | contiformer | macro_f1 | 0.2230 | chronaris_full | -0.0497 | -22.2691 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:45:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | contiformer | balanced_accuracy | 0.3376 | chronaris_full | -0.0043 | -1.2658 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:46:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | classical_baseline | macro_f1 | 0.2118 | chronaris_full | -0.0385 | -18.1762 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:47:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | classical_baseline | balanced_accuracy | 0.3504 | chronaris_full | -0.0171 | -4.8780 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:48:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | classical_baseline | macro_f1 | 0.1888 | chronaris_full | -0.0154 | -8.1802 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:49:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T1_maneuver_intensity_class | classical_baseline | balanced_accuracy | 0.3333 | chronaris_full | 0.0000 | 0.0000 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:50:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | rmse | 838.1210 |  |  |  | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:51:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | mae | 762.5579 |  |  |  | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:52:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | nrmse | 8.7810 |  |  |  | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:53:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | rmse | 1111.9789 |  |  |  | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:54:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | mae | 992.5239 |  |  |  | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:55:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | chronaris_full | nrmse | 11.3163 |  |  |  | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:56:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | rmse | 344.2890 | chronaris_full | -493.8321 | -143.4353 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:57:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | mae | 273.3819 | chronaris_full | -489.1760 | -178.9351 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:58:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | nrmse | 2.2379 | chronaris_full | -6.5431 | -292.3790 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:59:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | rmse | 413.4298 | chronaris_full | -698.5492 | -168.9644 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:60:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | mae | 314.7452 | chronaris_full | -677.7786 | -215.3420 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:61:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | mult | nrmse | 2.0426 | chronaris_full | -9.2738 | -454.0296 | leakage_safe_v1/leave_one_sortie_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:62:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | contiformer | rmse | 344.3351 | chronaris_full | -493.7859 | -143.4027 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
docs/artifacts/stage_i/stage-i-cross-evidence-matrix-20260702T-stage-i-cross-evidence-matrix-gpuopt-r1.md:63:| private_thirdparty_comparison | private_real_dual_stream | private_stage_h | T2_next_window_physiology_response | contiformer | mae | 273.5299 | chronaris_full | -489.0280 | -178.7842 | leakage_safe_v1/leave_one_view_out | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png | private model comparison | T1/T2/T3 proxy tasks on private real dual-stream Stage H data |
scripts/stage_i/evidence/build_thesis_protocol.py:34:        default="docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1",
docs/artifacts/stage_i/README.md:57:  - assets root：`../assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/`
src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py:85:DEFAULT_ARTIFACT_ROOT = "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py:240:def run_stage_i_private_thirdparty_comparison(
src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py:248:        stage_name="stage_i_private_thirdparty_comparison",
src/chronaris/pipelines/stage_i/private/thirdparty_comparison.py:2201:        "scripts/stage_i/private/run_private_third_party_comparison.py",
docs/artifacts/stage_i/stage-i-task-aware-heads-20260702T-stage-i-task-heads-optimization-r3-confirm20.md:6:- P30 reference: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1`
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:27:from chronaris.pipelines.stage_i.private.thirdparty_comparison import (
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:29:    run_stage_i_private_thirdparty_comparison,
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:109:    return run_stage_i_private_thirdparty_comparison(
src/chronaris/pipelines/stage_i/private/task_head_optimization.py:28:from chronaris.pipelines.stage_i.private.thirdparty_comparison import (
src/chronaris/pipelines/stage_i/private/task_head_optimization.py:31:    run_stage_i_private_thirdparty_comparison,
src/chronaris/pipelines/stage_i/private/task_head_optimization.py:39:    / "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
src/chronaris/pipelines/stage_i/private/task_head_optimization.py:189:    return run_stage_i_private_thirdparty_comparison(
docs/STATE.md:125:      - 产物：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json`
docs/STATE.md:137:      - 当前结果：`status=completed`，矩阵共 `292` 行，覆盖 `private_thirdparty_comparison=72`、`private_component_ablation=36`、`public_model_comparison=80`、`public_component_ablation=104` 四个象限。
docs/STATE.md:211:  - 最新 private third-party comparison package：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`
src/chronaris/pipelines/stage_i/evidence/cross_evidence_matrix.py:133:                evidence_quadrant="private_thirdparty_comparison",
src/chronaris/pipelines/stage_i/private/third_party_comparison.py:3:from chronaris.pipelines.stage_i.private.thirdparty_comparison import *  # noqa: F401,F403
src/chronaris/pipelines/stage_i/private/third_party_comparison.py:4:from chronaris.pipelines.stage_i.private.thirdparty_comparison import (  # noqa: F401
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:11:- split_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/split_manifest.json`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:15:- audit_json: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/label_feature_overlap_audit.json`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:16:- audit_csv: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/label_feature_overlap_audit.csv`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:144:- `fig_private_third_party_task_leaderboard`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_task_leaderboard.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:145:- `fig_private_third_party_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_delta_heatmap.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:146:- `fig_private_third_party_fold_variance`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_fold_variance.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:147:- `fig_private_third_party_training_curves`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_training_curves.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:148:- `fig_private_third_party_retrieval_topk`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_retrieval_topk.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:149:- `fig_private_third_party_gpu_throughput`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_third_party_gpu_throughput.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:150:- `fig_private_thirdparty_t1_macro_f1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t1_macro_f1.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:151:- `fig_private_thirdparty_t2_rmse`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t2_rmse.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:152:- `fig_private_thirdparty_t3_retrieval`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t3_retrieval.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:153:- `fig_private_thirdparty_delta_heatmap`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:154:- `fig_private_thirdparty_fold_stability`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_fold_stability.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:155:- `fig_private_thirdparty_confusion_t1`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_confusion_t1.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:156:- `fig_private_thirdparty_t2_error_distribution`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t2_error_distribution.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:157:- `fig_private_thirdparty_t3_retrieval_curve`: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_t3_retrieval_curve.png`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:161:- artifact_root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:162:- config: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_config.json`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:163:- evidence_manifest: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:164:- run_log: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/run.log`
docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md:165:- progress: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/progress.json`
docs/midterm/claims-matrix-2026-06-13.md:20:| P30 已完成 private T1/T2/T3 third-party comparison | 中强但限域 | private_proxy / private_thirdparty_comparison | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json`; `docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md` | Chronaris 在所有 private proxy 任务上全面胜出；T1/T2/T3 是论文最终人工真值任务 |
src/chronaris/pipelines/stage_i/evidence/optimized_reevaluation.py:21:    / "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
docs/midterm/midterm-fact-sheet-2026-06-13.md:303:- P30 private third-party comparison：`docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/private_thirdparty_summary.json`
docs/midterm/midterm-fact-sheet-2026-06-13.md:321:P32 cross-evidence matrix 将 private/public/proxy/component 证据统一成 `292` 行矩阵：`private_thirdparty_comparison=72`、`private_component_ablation=36`、`public_model_comparison=80`、`public_component_ablation=104`。中期报告可以用它说明证据层级和边界，但不能把 private Stage H、private proxy、public adapter、public context proxy 结果混成同一个胜负排行榜。
src/chronaris/pipelines/stage_i/evidence/optimized_model_summary.py:23:    / "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
src/chronaris/pipelines/stage_i/evidence/optimized_model_summary.py:196:        _source_row("P30", "completed", "private_thirdparty_comparison", p30_root, "Fixed private T1/T2/T3 reference."),
docs/implementation/notes/thesis-prep-readiness-assessment-2026-07-03.md:97:3. 代码拆分：优先拆 `thirdparty_comparison.py`、`fusion_ablation.py`、`fusion_refresh.py`、`deep_baseline_runtime.py`、`model_comparison.py`、`stage_i_sequences.py`、`runtime_inference.py` 等超长文件；拆分目标是保持 CLI 不变、核心实现按 data/config/train/report/render 分层。
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:40:| P30 | completed | private_model_comparison | private_real_dual_stream | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | Private Stage H proxy-task third-party comparison; not expert truth. |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:72:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | chronaris_full | macro_f1 | 0.173333 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:73:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | chronaris_full | balanced_accuracy | 0.333333 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:74:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | chronaris_full | macro_f1 | 0.173333 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:75:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | chronaris_full | balanced_accuracy | 0.333333 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:76:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | mult | macro_f1 | 0.20346 | chronaris_full | -0.0301264 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:77:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | mult | balanced_accuracy | 0.353276 | chronaris_full | -0.019943 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:78:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | mult | macro_f1 | 0.213188 | chronaris_full | -0.0398546 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:79:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | mult | balanced_accuracy | 0.339744 | chronaris_full | -0.00641026 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:80:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | contiformer | macro_f1 | 0.209558 | chronaris_full | -0.0362246 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:81:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | contiformer | balanced_accuracy | 0.356125 | chronaris_full | -0.022792 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:82:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | contiformer | macro_f1 | 0.222991 | chronaris_full | -0.0496581 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:83:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | contiformer | balanced_accuracy | 0.337607 | chronaris_full | -0.0042735 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:84:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | classical_baseline | macro_f1 | 0.211837 | chronaris_full | -0.0385039 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:85:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_view_out | classical_baseline | balanced_accuracy | 0.350427 | chronaris_full | -0.017094 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:86:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | classical_baseline | macro_f1 | 0.188776 | chronaris_full | -0.0154422 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:87:| private_model_comparison | private_real_dual_stream | T1_maneuver_intensity_class | leave_one_sortie_out | classical_baseline | balanced_accuracy | 0.333333 | chronaris_full | 0 |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:88:| private_model_comparison | private_real_dual_stream | T2_next_window_physiology_response | leave_one_view_out | chronaris_full | rmse | 838.121 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:89:| private_model_comparison | private_real_dual_stream | T2_next_window_physiology_response | leave_one_view_out | chronaris_full | mae | 762.558 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:90:| private_model_comparison | private_real_dual_stream | T2_next_window_physiology_response | leave_one_view_out | chronaris_full | nrmse | 8.78099 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md:91:| private_model_comparison | private_real_dual_stream | T2_next_window_physiology_response | leave_one_sortie_out | chronaris_full | rmse | 1111.98 |  |  |  | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/model_comparison_long.csv | T1/T2/T3 proxy tasks on private real dual-stream Stage H data | P30_via_P32 | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_cross_evidence_matrix/20260702T-stage-i-cross-evidence-matrix-gpuopt-r1/cross_evidence_matrix.csv | private_stage_h | completed | source positive means source reference/full/Chronaris claim is better | /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/fig_private_thirdparty_delta_heatmap.png |
src/chronaris/pipelines/stage_i/evidence/thesis_protocol.py:18:    / "docs/artifacts/assets/stage_i_private_thirdparty_comparison"
src/chronaris/pipelines/stage_i/evidence/thesis_protocol.py:400:    if text == "private_thirdparty_comparison":
src/chronaris/pipelines/stage_i/evidence/thesis_protocol.py:407:    if text == "private_thirdparty_comparison":
docs/implementation/TASKS.md:118:12. P30 已完成 private real dual-stream Stage H third-party comparison；入口为 `docs/artifacts/stage_i/stage-i-private-thirdparty-comparison-20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1.md` 与 `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/evidence_manifest.json`。该结果是 private T1/T2/T3 proxy task 的第三方对比，不能写成论文最终人工真值任务。
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:23:from chronaris.pipelines.stage_i.private.thirdparty_comparison import (
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:25:    run_stage_i_private_thirdparty_comparison,
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:36:DEFAULT_P30_ROOT = REPO_ROOT / "docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1"
src/chronaris/pipelines/stage_i/evidence/optimized_final_polish.py:290:    return run_stage_i_private_thirdparty_comparison(
```

### root stream-role CLI
```text
src/chronaris/pipelines/stage_i/public/stream_role_fusion_eval.py:654:        "scripts/stage_i/run_stream_role_fusion_eval.py",
```

### stage_i old import aliases
```text
docs/implementation/TASKS.md:61:脚本入口统一放到 `scripts/stage_i/<category>/`。根目录不再保留 `run_stage_i_*.py` / `build_stage_i_*.py` 旧脚本文件；需要执行旧命令时，应改用 `scripts/README.md` 里列出的 canonical 路径。Python 模块层保留旧 `chronaris.pipelines.stage_i.stage_i_*` import 的包级兼容映射，以便历史 notebook 或外部调用迁移时不需要立刻重写全部 import。
```

Interpretation:
- `private/stream_role_private_eval.py` has no repository references; it is a pure re-export candidate.
- `private/third_party_comparison.py`, `scripts/stage_i/private/run_private_third_party_comparison.py`, and `tests/test_stage_i_private_third_party_comparison.py` are compatibility wrappers around canonical `thirdparty` names; visible tests/docs can be moved to canonical imports before deletion.
- `scripts/stage_i/run_stream_role_fusion_eval.py` is the only root-level Stage I script; it should move into a category directory and the generated resume command should follow it.
- `stage_i` old import aliases are not used by repo code/tests outside the explanatory `TASKS.md` note; P42 can either remove them or document them as legacy. This pass will keep package-level alias risk low unless tests prove otherwise.
- `public.gpu_runtime` is used as a shared helper through `common.gpu_runtime`; moving the implementation to `common` reduces the misleading public/common wrapper split.

## 6. Deletion candidate table
| action | candidate | path pattern | evidence/reason |
| --- | --- | --- | --- |
| backup then delete if final rg/path checks stay clean | P37 nested public child deep_baseline_summary/training_curves/run.log | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_public/**/{deep_baseline_summary.json,training_curves.csv,run.log,gpu_perf_batches.csv}` | top-level P37 optimized_final_polish_summary / public_route_calibration_metrics / report preserve accepted metrics; current docs do not require per-candidate child rows |
| backup then delete if final rg/path checks stay clean | P37 nested private child logs, duplicate audit CSV/JSON, dense retrieval predictions | `docs/artifacts/assets/stage_i_optimized_final_polish/20260702T-stage-i-optimized-final-polish-r1/nested_private/**/{run.log,training_curves.csv,gpu_perf_*.json,gpu_perf_batches.csv,label_feature_overlap_audit.*,fold_predictions.csv}` | P37 report and P38 freeze only need top-level summary, t1/p37 deltas, accepted/rejected summaries and figures |
| backup then delete if final rg/path checks stay clean | P35 stream-role nested child logs/training curves/gpu batches/task manifest | `docs/artifacts/assets/stage_i_stream_role_fusion/20260702T-stage-i-stream-role-fusion-r4-v3-confirm20/{private_v3_confirm,public_v3_confirm}/**/{task_manifest.jsonl,training_curves.csv,gpu_perf_batches.csv,run.log}` | top-level P35 private_metrics/public_metrics/gate stats/route manifest/report preserve current evidence; report roots may remain with compact summaries |
| backup then delete if final rg/path checks stay clean | P30/P34 repeated private task manifests | `docs/artifacts/assets/stage_i_private_thirdparty_comparison/20260702T-stage-i-private-thirdparty-comparison-gpuopt-r1/task_manifest.jsonl; docs/artifacts/assets/stage_i_task_heads_optimization/20260702T-stage-i-task-heads-optimization-r3-confirm20/task_manifest.jsonl` | no direct docs/code references outside assets; can be regenerated from Stage H manifests |
| direct delete | local Python caches | `./third_party/__pycache__; ./third_party/mult/modules/__pycache__; ./third_party/mult/__pycache__; ./third_party/contiformer/__pycache__; ./third_party/contiformer/physiopro/network/__pycache__; ./third_party/contiformer/physiopro/__pycache__; ./tests/__pycache__; ./.pytest_cache; ./scripts/stage_i/evidence/__pycache__; ./src/chronaris/schema/__pycache__; ./src/chronaris/models/alignment/__pycache__; ./src/chronaris/models/__pycache__; ./src/chronaris/models/fusion/__pycache__; ./src/chronaris/__pycache__; ./src/chronaris/features/__pycache__; ./src/chronaris/pipelines/stage_i/common/__pycache__; ./src/chronaris/pipelines/stage_i/__pycache__; ./src/chronaris/pipelines/stage_i/evidence/__pycache__; ./src/chronaris/pipelines/stage_i/legacy/__pycache__; ./src/chronaris/pipelines/stage_i/public/__pycache__; ./src/chronaris/pipelines/stage_i/private/__pycache__; ./src/chronaris/pipelines/__pycache__; ./src/chronaris/dataset/__pycache__; ./src/chronaris/evaluation/__pycache__` | ignored generated state; direct delete without backup |

## 7. Preserve / defer table
- P38 thesis protocol freeze registry/matrix/summary/boundary/report/manifest under docs/artifacts/assets/stage_i_thesis_protocol/20260703T-stage-i-thesis-protocol-r1/ and docs/artifacts/stage_i/stage-i-thesis-protocol-20260703T-stage-i-thesis-protocol-r1.md
- P37 top-level optimized_final_polish_summary.json, t1_calibration_metrics.csv, public_route_calibration_metrics.csv, p37_delta_vs_p34.csv, p37_delta_vs_p35.csv, accepted/rejected candidate summaries, gpu_perf_summary.json, top-level figures/report/manifest/log/progress/resume
- P30/P31/P32/P34/P35/P36/P36_summary current primary CSV/JSON/report/manifest paths listed in P38 experiment_registry.csv
- P26 r6 thesis figure manifest/quality audit/12 PNG/CSV and runtime_semantic_case.csv historical input used by P20/P21
- Stage H clean run manifests and current Stage H dependency roots; raw window summaries are large but still current training/sample provenance and not pruned in this pass without replacement manifest support
- public adapter baselines explicitly kept by 20260701 cleanup and current public transfer/calibration code paths

## 8. History/LFS decision before cleanup
- Local `.git/lfs` cache is large (`8.0G` before cleanup), while current `.git/objects` is about `120M` and pack history is about `105.25 MiB`.
- `git lfs migrate info --include-ref=refs/heads/main --include=docs/**` still reports docs LFS objects around `132 MB` across `403/404` files, dominated by JSON/log/PNG history.
- Decision is deferred until after tree cleanup, backup manifest validation, and final LFS/history re-check. If bloated paths are backed up and removed from current tree but remain in docs/LFS history, P42 may proceed with `git filter-repo` plus `--force-with-lease` as authorized.
