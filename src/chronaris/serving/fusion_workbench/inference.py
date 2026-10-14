"""Adapt current frozen Chronaris encoders to the existing fusion-stream contract."""
import csv
import hashlib
import inspect
import json
import shutil
from pathlib import Path
import time
import zipfile

import torch

from chronaris.modeling.training import load_common_pretraining_checkpoint, TrainedFusionAdapter
from chronaris.representation import collate_observation_samples, write_fusion_stream_batch, load_fusion_stream_batch
from chronaris.representation.oof_export import _concatenate_fusion_batches
from chronaris.serving.fusion_workbench.cases import MAX_WINDOWS, preview_stream, read_json, write_json
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def load_adapter(case):
    if sha256_file(case.checkpoint) != case.manifest["checkpoint_sha256"]:
        raise ValueError("冻结模型校验失败")
    encoder, _, normalizer, payload = load_common_pretraining_checkpoint(case.checkpoint, device="cpu")
    if payload["method_name"] != "chronaris":
        raise ValueError("请选择兼容的融合模型")
    for stream in ("physiology", "vehicle"):
        if tuple(payload[stream + "_feature_names"]) != getattr(case.schema, stream + "_feature_names"):
            raise ValueError("输入字段及顺序必须与冻结模型一致")
    return TrainedFusionAdapter(encoder=encoder, normalizer=normalizer,
        fold_id=payload["fold"]["fold_id"], checkpoint_sha256=case.manifest["checkpoint_sha256"]), payload


def implementation_digest():
    package = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256(torch.__version__.encode())
    paths = [Path(__file__)]
    for name in ('modeling', 'models', 'representation'):
        paths.extend((package / name).rglob('*.py'))
    for path in sorted(paths):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def window_key(case, i):
    w = case.manifest['windows'][i]
    return hashlib.sha256(json.dumps({'window': w, 'schema': case.schema.schema_sha256,
        'model': case.manifest['checkpoint_sha256']}, sort_keys=True).encode()).hexdigest()


def run_case(case, indices, root, progress=lambda *args: None, *, batch_size=1, resume_root=None):
    """Window-wise inference; independent contexts retain no cross-window latent state."""
    if not indices or len(indices) > MAX_WINDOWS or indices != sorted(set(indices)):
        raise ValueError("请选择按顺序排列的 1—512 个窗口")
    if any(type(i) is not int or not 0 <= i < len(case.manifest["windows"]) for i in indices):
        raise ValueError("窗口索引超出案例范围")
    if batch_size not in (1, 2, 4, 8):
        raise ValueError("不支持的批大小")
    started = time.perf_counter()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    progress("loading", 0)
    adapter, payload = load_adapter(case)
    loaded_s = time.perf_counter() - started
    chunks, windows = [], []
    reused = 0
    implementation = implementation_digest()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        samples = [case.sample(i) for i in selected]
        cache_key = hashlib.sha256((implementation + ''.join(window_key(case, i) for i in selected)).encode()).hexdigest()
        cache = root / 'chunks' / cache_key
        prior = Path(resume_root) / 'chunks' / cache_key if resume_root else cache
        if (prior / 'preview.json').exists():
            chunk = load_fusion_stream_batch(prior)
            saved = read_json(prior / 'preview.json')
            if saved['key'] != cache_key or tuple(s.sample_id for s in samples) != chunk.sample_ids:
                raise ValueError('恢复片段与输入不一致')
            if prior != cache:
                shutil.copytree(prior, cache)
            chunks.append(chunk); windows.extend(saved['windows']); reused += len(selected)
            progress('inference', len(windows))
            continue
        chunk = adapter(collate_observation_samples(samples))
        chunks.append(chunk)
        for row, (i, sample) in enumerate(zip(selected, samples)):
            w = case.manifest["windows"][i]
            windows.append({"label": f"窗口 {i + 1:02}", "start_s": w["start_s"],
                "duration_s": sample.context_duration_s,
                "physiology": preview_stream(sample, "physiology", case.manifest.get("field_labels", {})),
                "vehicle": preview_stream(sample, "vehicle", case.manifest.get("field_labels", {})),
                "times": chunk.timestamps_s[row].tolist(), "valid": chunk.valid_mask[row].tolist(),
                "embedding": chunk.sequence_embedding[row].tolist(), "pooled": chunk.pooled_embedding[row].tolist()})
        write_fusion_stream_batch(chunk, root=cache, export_role="fusion_workbench_chunk")
        write_json(cache / 'preview.json', {'key': cache_key, 'windows': windows[-len(selected):]})
        progress("inference", len(windows))
    output = _concatenate_fusion_batches(chunks)
    progress("exporting", len(windows))
    write_fusion_stream_batch(output, root=root, export_role="fusion_workbench")
    restored = load_fusion_stream_batch(root)
    if (not torch.equal(output.sequence_embedding, restored.sequence_embedding)
            or not torch.equal(output.valid_mask, restored.valid_mask)
            or not torch.equal(output.pooled_embedding, restored.pooled_embedding)
            or output.sample_ids != restored.sample_ids):
        raise ValueError("下游读取复核失败")
    with (root / "features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "query_time_s", "valid", *[f"feature_{i:02}" for i in range(64)]])
        for i, sample_id in enumerate(output.sample_ids):
            for j, t in enumerate(output.timestamps_s[i]):
                writer.writerow([sample_id, float(t), bool(output.valid_mask[i, j]), *output.sequence_embedding[i, j].tolist()])
    config = payload.get("encoder_manifest", {}).get("backbone_config", {})
    result = {"case": case.summary(), "windows": windows, "elapsed_s": time.perf_counter() - started,
        "implementation_digest": implementation, "model_load_s": loaded_s, "reused_windows": reused, "device": "cpu", "batch_size": batch_size,
        "checkpoint_sha256": output.checkpoint_sha256, "source_sample_hashes": list(output.source_sample_hashes),
        "input_archives": [{"file": case.manifest["windows"][i]["file"],
                            "sha256": case.manifest["windows"][i]["sha256"]} for i in indices],
        "model": {k: config.get(k) for k in ("variant", "fusion_kind", "semantic_event_enabled", "attention_kind")},
        "implementation_revision": payload.get("implementation_revision"), "downstream_readback": True,
        "output_dim": 64, "query_points": 96, "selected_indices": indices}
    result["case_manifest_sha256"] = hashlib.sha256(json.dumps(case.manifest, sort_keys=True,
        ensure_ascii=False).encode()).hexdigest()
    result["source_files_sha256"] = {Path(p).name: sha256_file(p) for p in
        (__file__, inspect.getfile(load_common_pretraining_checkpoint), inspect.getfile(write_fusion_stream_batch))}
    write_json(root / "result.json", result)
    with zipfile.ZipFile(root / "features.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("fusion_stream.npz", "representation_manifest.json", "features.csv", "result.json"):
            archive.write(root / name, name)
    return result
