"""Portable service checks; real frozen-asset acceptance is a separate bounded run."""
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pytest
import torch

from chronaris.representation import FusionStreamBatch, load_fusion_stream_batch
from chronaris.serving.fusion_workbench import inference
from chronaris.serving.fusion_workbench.cases import Case, write_json, preview_stream
from chronaris.serving.fusion_workbench.server import DemoStore, make_server
from chronaris.simulation.aviation_dual_stream.deterministic_npz import sha256_file, write_deterministic_npz


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "cases/test-case"
    root.mkdir(parents=True)
    (root / "encoder.pt").write_bytes(b"unit-test-model")
    arrays = {}
    for stream in ("physiology", "vehicle"):
        arrays.update({stream + "_values": np.array([[1.], [2.]], dtype=np.float32),
                       stream + "_timestamps_s": np.array([0., 1.], dtype=np.float64),
                       stream + "_feature_mask": np.ones((2, 1), dtype=bool)})
    digest = write_deterministic_npz(root / "window.npz", arrays)
    schema = {"schema_id":"test", "source_kind":"unit_test", "physiology_feature_names":["physiology.a"],
              "vehicle_feature_names":["vehicle.a"], "physiology_feature_roles":["observed"], "vehicle_feature_roles":["observed"]}
    write_json(root / "case.json", {"format":"chronaris.demo_case.v1", "id":"test-case", "title":"测试案例",
        "description":"接口测试", "source_label":"构造测试数据", "model_label":"测试模型", "schema":schema,
        "checkpoint":"encoder.pt", "checkpoint_sha256":sha256_file(root / "encoder.pt"),
        "windows":[{"sample_id":f"sample{i}", "group_id":"g", "source_sample_hash":"a"*64,
                    "file":"window.npz", "sha256":digest, "start_s":float(i), "duration_s":30.} for i in range(2)]})
    return Case(root / "case.json")


@pytest.fixture
def fake_adapter(monkeypatch):
    def load(case):
        def encode(batch):
            count = len(batch.sample_ids)
            values = torch.ones(count, 96, 64) * batch.physiology_values[:, :1, :1]
            valid = torch.ones(count, 96, dtype=torch.bool)
            return FusionStreamBatch(batch.sample_ids, batch.query_timestamps_s, values, valid, values.mean(1),
                "chronaris", "test-fold", case.manifest["checkpoint_sha256"], batch.source_sample_hashes)
        return encode, {"implementation_revision":"test", "encoder_manifest":{}}
    monkeypatch.setattr(inference, "load_adapter", load)


def test_portable_case_and_tamper(case):
    assert case.sample(0).physiology_values.shape == (2, 1)
    assert preview_stream(case.sample(0), "physiology", {})["coverage"] == 1.
    (case.root / "window.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="输入窗口校验"):
        case.sample(0)


def test_case_rejects_escape_and_checkpoint_change(case):
    manifest = case.manifest.copy()
    manifest["checkpoint"] = "../outside.pt"
    write_json(case.root / "case.json", manifest)
    with pytest.raises(ValueError, match="案例包内"):
        Case(case.root / "case.json")
    write_json(case.root / "case.json", case.manifest)
    (case.root / "encoder.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="冻结模型"):
        Case(case.root / "case.json")


def test_schema_order_is_checked(case, monkeypatch):
    payload = {"method_name":"chronaris", "physiology_feature_names":["wrong"], "vehicle_feature_names":["vehicle.a"]}
    monkeypatch.setattr(inference, "load_common_pretraining_checkpoint", lambda *a, **k: (None, None, None, payload))
    with pytest.raises(ValueError, match="字段及顺序"):
        inference.load_adapter(case)


def test_export_batch_and_window_readback(case, fake_adapter, tmp_path):
    incremental = inference.run_case(case, [0, 1], tmp_path / "one")
    inference.run_case(case, [0, 1], tmp_path / "batch", batch_size=2)
    a, b = [load_fusion_stream_batch(tmp_path / name) for name in ("one", "batch")]
    torch.testing.assert_close(a.sequence_embedding, b.sequence_embedding, atol=0, rtol=0)
    assert incremental["downstream_readback"]
    assert len((tmp_path / "one/features.csv").read_text().splitlines()) == 193
    with zipfile.ZipFile(tmp_path / "one/features.zip") as archive:
        archive.extractall(tmp_path / "unpacked")
    assert load_fusion_stream_batch(tmp_path / "unpacked").sample_ids == ("sample0", "sample1")


@pytest.mark.parametrize("indices", [[], [1, 0], [0, 0], [99], [True]])
def test_bad_selection(case, indices, tmp_path):
    with pytest.raises(ValueError):
        inference.run_case(case, indices, tmp_path / "bad")


def test_http_history_download_and_security(case, fake_adapter, tmp_path):
    store = DemoStore(case.root.parent, tmp_path / "runs")
    server = make_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    def request(path, payload=None, headers=None):
        data = None if payload is None else json.dumps(payload).encode()
        return urlopen(Request(base + path, data=data, headers=headers or ({"Content-Type":"application/json"} if data else {})), timeout=5)
    try:
        assert request("/").status == 200
        assert json.load(request("/api/cases"))[0]["id"] == "test-case"
        for payload in ({"case_id":"test-case", "start":-1, "stop":1}, {"case_id":"test-case", "start":0, "stop":True}, []):
            with pytest.raises(HTTPError) as exc:
                request("/api/runs", payload)
            assert exc.value.code == 400
        for headers in ({"Host":"evil.example"}, {"Content-Type":"application/json", "Origin":"http://evil.example"}):
            with pytest.raises(HTTPError) as exc:
                request("/api/runs", {"case_id":"test-case","start":0,"stop":1}, headers)
            assert exc.value.code == 403
        job = json.load(request("/api/runs", {"case_id":"test-case","start":0,"stop":2}))
        for _ in range(100):
            saved = json.load(request("/api/runs/" + job["id"]))
            if saved["status"] in ("completed", "failed"):
                break
            time.sleep(.02)
        assert saved["status"] == "completed", saved
        assert len(json.load(request("/api/runs"))) == 1
        assert json.load(request(f'/api/runs/{job["id"]}/result'))["output_dim"] == 64
        assert request(f'/api/runs/{job["id"]}/download/features.zip').headers['Content-Type'] == 'application/zip'
        with pytest.raises(HTTPError):
            request(f'/api/runs/{job["id"]}/download/encoder.pt')
        with pytest.raises(HTTPError):
            request('/api/runs/../../case.json')
        store.close()
        reopened = DemoStore(case.root.parent, tmp_path / "runs")
        assert reopened.history()[0]["status"] == "completed"
        reopened.close()
    finally:
        server.shutdown(); server.server_close(); store.close()


def test_single_job_and_restart_recovery(case, monkeypatch, tmp_path):
    started, release = threading.Event(), threading.Event()
    def pending(*args):
        started.set(); release.wait(5)
        raise ValueError("测试失败必须保留")
    monkeypatch.setattr("chronaris.serving.fusion_workbench.server.run_case", pending)
    store = DemoStore(case.root.parent, tmp_path / "runs")
    try:
        job = store.submit({"case_id":"test-case","start":0,"stop":1})
        assert started.wait(2)
        with pytest.raises(RuntimeError, match="已有演示服务"):
            DemoStore(case.root.parent, tmp_path / "runs")
        with pytest.raises(RuntimeError, match="已有推理任务"):
            store.submit({"case_id":"test-case","start":0,"stop":1})
    finally:
        release.set(); store.close()
    assert store.history()[0]["status"] == "failed"
    assert store.history()[0]["error"] == "测试失败必须保留"
    path = store.job_root(job["id"]) / "job.json"
    job.update(status="inference")
    write_json(path, job)
    reopened = DemoStore(case.root.parent, tmp_path / "runs")
    assert reopened.history()[0]["status"] == "failed"
    reopened.close()
