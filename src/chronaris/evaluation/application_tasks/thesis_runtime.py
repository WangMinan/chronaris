"""Frozen CUDA replay benchmark for the selected thesis encoder."""

from __future__ import annotations

import json
import hashlib
import resource
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from chronaris.evaluation.application_tasks.simulation_locked_pretraining_data import (
    load_simulation_locked_pretraining_data,
)
from chronaris.modeling.training import (
    TrainedFusionAdapter,
    load_common_pretraining_checkpoint,
)
from chronaris.representation import select_observation_batch
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


REPO = Path(__file__).resolve().parents[4]
SIMULATION_ROOT = (
    REPO / "artifacts/application_evaluation/2026-07-10_aviation-simulation-formal"
)
CHECKPOINT = (
    REPO
    / "artifacts/application_evaluation/"
    "2026-09-03_thesis-simulation-pretraining-v3p2/"
    "checkpoints/seed_17/chronaris/best.pt"
)


def run_thesis_runtime_benchmark(
    *,
    output_root: str | Path,
    device: str = "cuda",
    warmup_repeats: int = 3,
    measured_repeats: int = 20,
):
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("thesis runtime benchmark requires CUDA")
    if min(warmup_repeats, measured_repeats) <= 0:
        raise ValueError("runtime repeat counts must be positive")
    data = load_simulation_locked_pretraining_data(SIMULATION_ROOT)
    encoder, _heads, normalizer, _payload = load_common_pretraining_checkpoint(
        CHECKPOINT,
        device=device,
    )
    adapter = TrainedFusionAdapter(
        encoder=encoder,
        normalizer=normalizer,
        fold_id=data.fold.fold_id,
        checkpoint_sha256=sha256_file(CHECKPOINT),
    )
    rows = []
    for batch_size in (1, 8, 32):
        sample_ids = data.fold.held_out_sample_ids[:batch_size]
        if len(sample_ids) != batch_size:
            continue
        batch = select_observation_batch(data.batch, sample_ids)
        for _ in range(warmup_repeats):
            adapter(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(measured_repeats):
            started = perf_counter()
            output = adapter(batch)
            torch.cuda.synchronize()
            elapsed.append(perf_counter() - started)
        if output.sequence_embedding.shape != (batch_size, 96, 64):
            raise RuntimeError("runtime representation contract changed")
        if not torch.isfinite(output.sequence_embedding).all():
            raise RuntimeError("runtime replay produced non-finite representation")
        rows.append(
            {
                "batch_size": batch_size,
                **summarize_runtime_timings(elapsed, batch_size=batch_size),
                "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated()
                / 1024**3,
                "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved()
                / 1024**3,
            }
        )
    audit = {
        "format": "chronaris.thesis_runtime_benchmark.v1",
        "protocol_version": "v3.2.2",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO, text=True
        ).strip(),
        "evaluation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO)),
        "checkpoint_sha256": adapter.checkpoint_sha256,
        "warmup_repeats": warmup_repeats,
        "measured_repeats": measured_repeats,
        "cpu_process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "timing_scope": "raw observation batch through normalized [B,96,64] output",
        "rows": rows,
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "runtime_benchmark.json", audit)
    (root / "report.md").write_text(_report(audit), encoding="utf-8")
    return audit


def summarize_runtime_timings(elapsed_s, *, batch_size):
    values = np.asarray(tuple(elapsed_s), dtype=np.float64)
    if not len(values) or np.any(values <= 0) or batch_size <= 0:
        raise ValueError("runtime timing samples are invalid")
    return {
        "latency_p50_ms": float(np.quantile(values, 0.50) * 1_000),
        "latency_p95_ms": float(np.quantile(values, 0.95) * 1_000),
        "throughput_samples_per_s": float(batch_size / np.median(values)),
    }


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _report(audit):
    lines = [
        "# 论文主线离线回放运行评估",
        "",
        "本评估记录原始双流观测窗口到标准化融合表示的图形处理器计算延迟、吞吐、显存和进程内存；不包含磁盘读取与模型参数冷启动。",
        "",
        "| 批量 | 中位延迟（毫秒） | 95 分位延迟（毫秒） | 吞吐（样本/秒） | 峰值分配显存（吉字节，GB） |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {batch_size} | {latency_p50_ms:.3f} | {latency_p95_ms:.3f} | "
        "{throughput_samples_per_s:.3f} | {peak_cuda_allocated_gb:.3f} |".format(
            **row
        )
        for row in audit["rows"]
    )
    lines.extend(
        (
            "",
            f"进程峰值 CPU 内存为 {audit['cpu_process_peak_rss_mb']:.1f} MB。",
            "",
        )
    )
    return "\n".join(lines)
