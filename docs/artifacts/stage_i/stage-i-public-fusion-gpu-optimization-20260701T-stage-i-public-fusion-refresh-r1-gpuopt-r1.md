# Stage I Public Fusion GPU Optimization - 20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1

## 1. Executive Summary
Base 公开融合刷新 run `20260701T-stage-i-public-fusion-refresh-r1` is completed. This GPUOPT run did not continue full LOSO; it profiled representative folds only.
Throughput changed from 559.52 to 5090.93 samples/sec (speedup 9.10x).

## 2. Baseline bottleneck diagnosis
The baseline path repeatedly sliced CPU numpy arrays and rebuilt CUDA tensors inside the batch loop. The optimized path builds train-fold normalization once and reuses device or pinned tensors.

## 3. Optimizations applied
- tensor cache: `auto`
- auto batch size: `True`
- AMP: `bf16`
- torch.compile: `off`

## 4. Throughput before/after
- before_samples_per_sec: `559.5175954001099`
- after_samples_per_sec: `5090.928075174421`
- speedup_ratio: `9.098781016053495`

## 5. Memory and GPU utilization
- before_max_memory_gb: `0.2762179374694824`
- after_max_memory_gb: `3.9131555557250977`
- before_gpu_util_pct: `11.666666666666666`
- after_gpu_util_pct: `15.833333333333334`

## 6. Correctness and protocol checks
Normalization and target transforms are computed from train fold indices only. Split groups are copied from the original LOSO grouping and no label/split/metric definitions are changed.

## 7. Resume status
No 公开融合刷新 full LOSO resume was executed in this run.

## 8. Artifact index
- artifact root: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization`
- gpu_perf_batches: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/gpu_perf_batches.csv`
- gpu_perf_fold_summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/gpu_perf_fold_summary.csv`
- gpu_perf_summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/gpu_perf_summary.json`
- optimization_summary: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/optimization_summary.json`
- fig_gpu_throughput_before_after: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/plots/fig_gpu_throughput_before_after.png`
- fig_gpu_batch_timing_breakdown: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/plots/fig_gpu_batch_timing_breakdown.png`
- fig_gpu_memory_and_batch_size: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/plots/fig_gpu_memory_and_batch_size.png`
- fig_gpu_cache_effect: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/plots/fig_gpu_cache_effect.png`
- fig_gpu_training_progress_heartbeat: `/home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1/gpu_optimization/plots/fig_gpu_training_progress_heartbeat.png`

## 9. Commands run
`python scripts/stage_i/public/run_public_fusion_gpuopt.py --run-id 20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1 --device cuda --require-cuda`

## 10. Next run recommendation
To resume or reproduce this GPUOPT profiling run, use:

```bash
python scripts/stage_i/public/run_public_fusion_gpuopt.py --run-id 20260701T-stage-i-public-fusion-refresh-r1-gpuopt-r1 --base-p28-run-id 20260701T-stage-i-public-fusion-refresh-r1 --base-p28-root /home/wangminan/projects/chronaris/docs/artifacts/assets/stage_i_public_fusion_refresh/20260701T-stage-i-public-fusion-refresh-r1 --profile-batches 20 --profile-epochs 1 --device cuda --require-cuda --tensor-cache auto --max-cache-gb 18 --auto-batch-size --batch-size-candidates 2048 1024 512 256 128 --amp bf16 --torch-compile off --skip-completed
```
