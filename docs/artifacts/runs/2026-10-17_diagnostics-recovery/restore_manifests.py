import hashlib,json,shlex,subprocess
from pathlib import Path
workspace=Path('/home/wangminan/projects/chronaris-runtime-local-validation-20261012')
registry=workspace/'docs/requirements/thesis-v4-simulation-manifest.json';data=json.loads(registry.read_text())
remote='/home/wangminan/projects/chronaris-runtime-chapter5-reload-20261011'
ssh=['/mnt/c/Windows/System32/OpenSSH/ssh.exe','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','wangminan@10.69.120.91']
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[]
for item in data['historical_generation_manifests']:
 target=workspace/item['path'];source=remote+'/'+item['path'];restored=False
 if not target.exists():
  target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix('.download')
  with temporary.open('wb') as output:subprocess.run(ssh+['cat '+shlex.quote(source)],stdout=output,check=True)
  assert sha(temporary)==item['sha256'],'original historical manifest hash differs'
  temporary.replace(target);restored=True
 assert sha(target)==item['sha256']
 rows.append(dict(path=str(target.resolve()),source=source,sha256=item['sha256'],bytes=target.stat().st_size,restored=restored))
for name,digest in data['generator_source_sha256'].items():assert sha(workspace/name)==digest
report=dict(status='completed',registry_sha256=sha(registry),manifests=rows,generator_source_files_verified=len(data['generator_source_sha256']),raw_trajectories_copied=False)
(Path(__file__).parent/'manifest_restoration.json').write_text(json.dumps(report,indent=2)+'\n')
print('restored and hash-verified',len(rows),'historical manifests,',sum(r['bytes'] for r in rows),'bytes')
