"""Pure offline UI rendering: saved JSON + tracked HTML/CSS/JS, no algorithm imports."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从随包数据生成无需服务或模型的离线界面快照")
    parser.add_argument("--ui-root", type=Path, default=here / "ui")
    parser.add_argument("--data", type=Path, default=here / "preview-data.json")
    parser.add_argument("--output", type=Path, default=here / "preview.html")
    parser.add_argument("--bundle", type=Path, help="导出独立可编辑重绘目录")
    args = parser.parse_args()
    ui = args.ui_root
    if not ui.exists():
        ui = here.parents[3] / "src/chronaris/serving/fusion_workbench/static"
    data = json.loads(args.data.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    shim = """
const snapshot = SNAPSHOT;
for (const key of ['title','description','model_label','source_label']) {
  snapshot.case[key] = String(snapshot.case[key] || '').replaceAll('鼎新','实测数据集').replaceAll('Chronaris','融合系统').replaceAll('离线案例','观测记录').replaceAll('内部验证片段','观测片段');
}
const snapshotJob = snapshot.preview_job;
window.fetch = async (path, options={}) => {
  let value, status=200;
  if(options.method==='POST'){value={error:'请在融合系统中提交处理任务。'};status=405;}
  else if(path==='/api/cases')value=[snapshot.case];
  else if(path==='/api/runs')value=[{...snapshotJob,title:snapshot.case.title}];
  else if(path==='/api/models'||path==='/api/sources'||path.startsWith('/api/fields/'))value=[];
  else if(path.endsWith('/delivery'))value={status:'pending'};
  else if(path.endsWith('/result'))value=snapshot;
  else value=snapshotJob;
  return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});
};
""".replace("SNAPSHOT", payload)
    page = (ui / "index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="/style.css">', '<style>' + (ui / 'style.css').read_text() +
        '\n#run,#deliver,#add-data,.download-links,#download-zip,[data-tab=data],[data-tab=models]{display:none}</style>')
    page = page.replace('<script src="/app.js" defer></script>', '')
    page = page.replace('<script src="/charts.js" defer></script>', '')
    script = (ui / 'charts.js').read_text() + '\n' + (ui / "app.js").read_text()
    script += "\nopenResult(snapshotJob.id,false).then(()=>{setProgress(snapshotJob);tab('results');});"
    page = page.replace('</body>', '<script>' + shim + script + '</script></body>')
    args.output.write_text(page, encoding="utf-8")
    if args.bundle:
        args.bundle.mkdir(parents=True, exist_ok=False)
        shutil.copytree(ui, args.bundle / "ui")
        shutil.copyfile(args.data, args.bundle / "preview-data.json")
        shutil.copyfile(__file__, args.bundle / "render_preview.py")
        shutil.copyfile(args.output, args.bundle / "preview.html")
    print(args.output)


if __name__ == "__main__":
    main()
