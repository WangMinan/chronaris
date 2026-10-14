"""Loopback HTTP transport for the fusion workbench."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

from chronaris.serving.fusion_workbench.cases import read_json
from chronaris.serving.fusion_workbench.presentation import present_result

STATIC = Path(__file__).with_name('static')
DOWNLOADS = {'features.csv':'text/csv; charset=utf-8','features.zip':'application/zip',
    'fusion_stream.npz':'application/octet-stream','representation_manifest.json':'application/json',
    'pooled_features.csv':'text/csv; charset=utf-8','delivery.json':'application/json'}


def make_server(store, port=8766):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body, content_type, status=200, filename=None):
            self.send_response(status)
            for k,v in {'Content-Type':content_type,'Content-Length':str(len(body)), 'Cache-Control':'no-store',
                'X-Content-Type-Options':'nosniff', 'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'"}.items():
                self.send_header(k,v)
            if filename:
                self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def json(self, payload, status=200):
            self.send_bytes(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),'application/json; charset=utf-8',status)

        def trusted_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.trusted_host():
                return self.json({'error':'访问地址不合法'},403)
            path = urlsplit(self.path).path
            try:
                if path=='/api/cases':
                    return self.json(store.records())
                if path=='/api/models':
                    return self.json(store.catalog.model_list())
                if path=='/api/sources':
                    return self.json([{k:v for k,v in s.items() if k not in ('snapshot','rows')} for s in store.catalog.sources()])
                if path=='/api/runs':
                    return self.json(store.history())
                parts = path.strip('/').split('/')
                if len(parts)==3 and parts[:2]==['api','fields']:
                    return self.json(store.field_rows(parts[2]))
                if len(parts)>=3 and parts[:2]==['api','runs']:
                    root = store.job_root(parts[2])
                    job = read_json(root/'job.json')
                    if len(parts)==3:
                        return self.json(job)
                    if job['status']!='completed':
                        return self.json({'error':'结果尚未完成'},409)
                    if len(parts)==4 and parts[3]=='result':
                        return self.json(present_result(read_json(root/'result.json')))
                    if len(parts)==4 and parts[3]=='delivery':
                        return self.json(read_json(root/'delivery.json') if (root/'delivery.json').exists() else {'status':'pending'})
                    if len(parts)==5 and parts[3]=='download' and parts[4] in DOWNLOADS:
                        return self.send_bytes((root/parts[4]).read_bytes(),DOWNLOADS[parts[4]],filename=parts[4])
                names = {'/':'index.html','/app.js':'app.js','/charts.js':'charts.js','/style.css':'style.css'}
                if path in names:
                    name = names[path]
                    mime = 'text/html' if name.endswith('.html') else 'text/css' if name.endswith('.css') else 'text/javascript'
                    return self.send_bytes((STATIC/name).read_bytes(),mime+'; charset=utf-8')
                self.json({'error':'资源不存在'},404)
            except (KeyError,FileNotFoundError):
                self.json({'error':'资源不存在'},404)
            except (ValueError,TypeError,OSError) as exc:
                self.json({'error':str(exc)},400)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if (not self.trusted_host() or origin and origin!=f"http://{self.headers.get('Host')}"
                    or self.headers.get('Content-Type')!='application/json'):
                return self.json({'error':'请求来源或格式不合法'},403)
            try:
                limit = 32*1024*1024 if self.path=='/api/cases/import' else 16384
                length = int(self.headers.get('Content-Length','0'))
                if not 0<length<=limit:
                    raise ValueError('请求大小超出限制')
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                if len(raw)!=length:
                    raise ValueError('请求未完整接收')
                def nonfinite(value):
                    raise ValueError('数值必须有限')
                payload = json.loads(raw,parse_constant=nonfinite)
                if not isinstance(payload,dict):
                    raise ValueError('请求格式不合法')
                if self.path=='/api/runs':
                    return self.json(store.submit(payload),202)
                if self.path in ('/api/cases/import','/api/sources/prepare'):
                    return self.json(store.import_record(payload,archive=self.path=='/api/sources/prepare'),201)
                parts = self.path.strip('/').split('/')
                if len(parts)==4 and parts[:2]==['api','runs']:
                    if parts[3]=='retry':
                        return self.json(store.retry(parts[2]),202)
                    if parts[3]=='deliver':
                        return self.json(store.deliver(parts[2]))
                if len(parts)==3 and parts[:2]==['api','fields']:
                    with store.lock:
                        return self.json(store.reviews.review(store.cases[parts[2]],payload))
                self.json({'error':'资源不存在'},404)
            except (ValueError,TypeError,KeyError,OverflowError) as exc:
                self.json({'error':str(exc)},400)
            except RuntimeError as exc:
                self.json({'error':str(exc)},409)
            except OSError:
                self.json({'error':'数据读写失败，请检查文件或存储空间后重试'},500)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)
