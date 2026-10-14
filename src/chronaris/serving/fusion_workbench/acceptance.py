"""Bounded real-asset acceptance; never trains or reads task targets."""
import csv
import io
import json
from pathlib import Path
import resource
import time
from urllib.request import Request, urlopen
import uuid
import zipfile

import torch

from chronaris.representation import load_fusion_stream_batch
from chronaris.serving.fusion_workbench.cases import Case, write_json
from chronaris.serving.fusion_workbench.inference import run_case
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file


def verify_demo(data_root, *, port=8766):
    root = Path(data_root)
    case = Case(root / "cases/dingxin-validation/case.json")
    output_root = root / "acceptance" / uuid.uuid4().hex
    output_root.mkdir(parents=True)
    base = f"http://127.0.0.1:{port}"
    def request(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        return urlopen(Request(base + path, data=payload,
            headers={} if payload is None else {"Content-Type": "application/json"}), timeout=30)
    job = json.load(request("/api/runs", {"case_id":case.manifest["id"], "start":0, "stop":3}))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = json.load(request("/api/runs/" + job["id"]))
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(.1)
    if status["status"] != "completed":
        raise AssertionError(status)
    with request(f'/api/runs/{job["id"]}/download/features.zip') as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            assert set(archive.namelist()) == {"fusion_stream.npz", "representation_manifest.json", "features.csv", "result.json"}
            archive.extractall(output_root / "download")
    reference = load_fusion_stream_batch(output_root / "download")
    batch = run_case(case, [0, 1, 2], output_root / "batch", batch_size=2)
    other = load_fusion_stream_batch(output_root / "batch")
    torch.testing.assert_close(reference.sequence_embedding, other.sequence_embedding, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reference.pooled_embedding, other.pooled_embedding, atol=1e-5, rtol=1e-5)
    assert reference.sample_ids == other.sample_ids and torch.equal(reference.valid_mask, other.valid_mask)
    with (output_root / "download/features.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 288 and len(rows[0]) == 67
    for i, row in enumerate(rows):
        torch.testing.assert_close(torch.tensor([float(row[f"feature_{d:02}"]) for d in range(64)]),
            reference.sequence_embedding[i // 96, i % 96], atol=0, rtol=0)
    source = case.manifest["source"]["checkpoint"]
    assert sha256_file(source) == case.manifest["checkpoint_sha256"]
    report = {"status":"passed", "http_run_id":job["id"], "device":"cpu", "threads":torch.get_num_threads(),
        "shape":list(reference.sequence_embedding.shape), "http_incremental_s":status["elapsed_s"],
        "direct_batch_s":batch["elapsed_s"],
        "batch_max_absolute_difference":float((reference.sequence_embedding-other.sequence_embedding).abs().max()),
        "tolerance":{"atol":1e-5,"rtol":1e-5}, "csv_values_match_npz":True, "download_readback":True,
        "peak_verifier_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "checkpoint_sha256":case.manifest["checkpoint_sha256"], "shared_checkpoint_unchanged":True}
    write_json(output_root / "acceptance.json", report)
    return output_root / "acceptance.json"
