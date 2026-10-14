"""Loopback HTTP UI with one bounded inference job and persistent local history."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit
import uuid

from chronaris.serving.fusion_workbench.cases import Case, read_json, write_json
from chronaris.serving.fusion_workbench.inference import run_case

STATIC = Path(__file__).with_name("static")
DOWNLOADS = {"features.csv": "text/csv; charset=utf-8", "features.zip": "application/zip",
             "fusion_stream.npz": "application/octet-stream", "representation_manifest.json": "application/json"}


class DemoStore:
    def __init__(self, cases_root, runs_root):
        self.cases = {}
        for path in sorted(Path(cases_root).glob("*/case.json")):
            case = Case(path)
            key = case.manifest["id"]
            if not re.fullmatch(r"[a-z0-9-]{1,64}", key) or key in self.cases:
                raise ValueError("案例标识不合法或重复")
            self.cases[key] = case
        self.root = Path(runs_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_file = (self.root / ".server.lock").open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise RuntimeError("此产物目录已有演示服务运行") from None
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.active = None
        for path in self.root.glob("*/job.json"):
            job = read_json(path)
            if job["status"] not in ("completed", "failed"):
                job.update(status="failed", error="服务上次退出时任务未完成，请重新运行。")
                write_json(path, job)

    def close(self):
        self.executor.shutdown(wait=True)
        self._lock_file.close()

    def job_root(self, job_id):
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise KeyError("运行记录不存在")
        root = self.root / job_id
        if not (root / "job.json").is_file():
            raise KeyError("运行记录不存在")
        return root

    def history(self):
        return sorted((read_json(p) for p in self.root.glob("*/job.json")),
                      key=lambda x: x["created_at"], reverse=True)

    def submit(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"case_id", "start", "stop"}:
            raise ValueError("请选择案例及起止窗口")
        case = self.cases.get(payload["case_id"])
        start, stop = payload["start"], payload["stop"]
        if case is None or type(start) is not int or type(stop) is not int:
            raise ValueError("案例或窗口参数不合法")
        if not 0 <= start < stop <= len(case.manifest["windows"]) or stop - start > 8:
            raise ValueError("每次请选择 1—8 个连续窗口")
        with self.lock:
            if self.active is not None:
                raise RuntimeError("已有推理任务运行，请等待完成")
            job_id = uuid.uuid4().hex
            root = self.root / job_id
            root.mkdir()
            job = {"id": job_id, "case_id": payload["case_id"], "title": case.manifest["title"],
                   "status": "queued", "done": 0, "total": stop - start,
                   "created_at": datetime.now(timezone.utc).isoformat()}
            write_json(root / "job.json", job)
            self.active = job_id
            self.executor.submit(self._run, case, list(range(start, stop)), root, job)
        return job.copy()

    def _run(self, case, indices, root, job):
        def progress(status, done):
            job.update(status=status, done=done)
            write_json(root / "job.json", job)
        try:
            result = run_case(case, indices, root, progress)
            job.update(status="completed", done=len(indices), elapsed_s=result["elapsed_s"])
        except Exception as exc:
            job.update(status="failed", error=str(exc))
        finally:
            write_json(root / "job.json", job)
            with self.lock:
                self.active = None


def make_server(store, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, content_type, status=200, filename=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def json(self, payload, status=200):
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def trusted_host(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            alternate = f"localhost:{self.server.server_port}"
            return self.headers.get("Host") in (expected, alternate)

        def do_GET(self):
            if not self.trusted_host():
                return self.json({"error": "仅允许本机访问"}, 403)
            path = urlsplit(self.path).path
            try:
                if path == "/api/cases":
                    return self.json([c.summary() for c in store.cases.values()])
                if path == "/api/runs":
                    return self.json(store.history())
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                    root = store.job_root(parts[2])
                    job = read_json(root / "job.json")
                    if len(parts) == 3:
                        return self.json(job)
                    if job["status"] != "completed":
                        return self.json({"error": "结果尚未完成"}, 409)
                    if len(parts) == 4 and parts[3] == "result":
                        return self.json(read_json(root / "result.json"))
                    if len(parts) == 5 and parts[3] == "download" and parts[4] in DOWNLOADS:
                        return self.send_bytes((root / parts[4]).read_bytes(), DOWNLOADS[parts[4]], filename=parts[4])
                name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}.get(path)
                if name:
                    return self.send_bytes((STATIC / name).read_bytes(), {"index.html": "text/html; charset=utf-8",
                        "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[name])
                self.json({"error": "资源不存在"}, 404)
            except (KeyError, FileNotFoundError):
                self.json({"error": "资源不存在"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self.trusted_host() or (origin and origin != f"http://{self.headers.get('Host')}")
                or self.headers.get("Content-Type") != "application/json"):
                return self.json({"error": "请求来源或格式不合法"}, 403)
            if self.path != "/api/runs":
                return self.json({"error": "资源不存在"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("请求大小不合法")
                payload = json.loads(self.rfile.read(length))
                self.json(store.submit(payload), 202)
            except (ValueError, TypeError) as exc:
                self.json({"error": str(exc)}, 400)
            except RuntimeError as exc:
                self.json({"error": str(exc)}, 409)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
