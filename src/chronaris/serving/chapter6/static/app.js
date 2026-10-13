'use strict';
const $ = id => document.getElementById(id);
const labels = {queued:'等待运行',loading:'加载冻结模型',inference:'生成融合表示',exporting:'校验与导出',completed:'已完成',failed:'运行失败'};
let cases = [], result = null, currentRun = null, timer = null, pollTimer = null, loadingResult = false;
const number = (n, digits=2) => Number(n).toLocaleString('zh-CN', {maximumFractionDigits:digits});
function message(text='') { $('message').textContent = text; $('message').hidden = !text; }
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '请求失败');
  return data;
}
function tab(name) {
  document.querySelectorAll('.view').forEach(el => { el.hidden = el.id !== name; });
  document.querySelectorAll('.nav').forEach(el => { el.classList.toggle('active', el.dataset.tab === name); el.setAttribute('aria-current', el.dataset.tab === name ? 'page' : 'false'); });
  $('page-title').textContent = {data:'数据与任务',results:'融合结果与回放',exports:'特征导出与追溯'}[name];
  window.scrollTo(0, 0);
  if (name !== 'results') stopPlayback();
}
function selectedCase() { return cases.find(c => c.id === $('case').value); }
function configureCase() {
  const c = selectedCase();
  if (!c) return;
  $('case-description').textContent = c.description;
  $('phys-count').textContent = c.field_counts[0]; $('vehicle-count').textContent = c.field_counts[1];
  $('window-count').textContent = c.windows.length; $('model-label').textContent = c.model_label;
  for (const name of ['start','stop']) {
    $(name).replaceChildren(...c.windows.map((w,i) => new Option(`${w.label} · ${number(w.start_s)}—${number(w.start_s+w.duration_s)} 秒`, i)));
  }
  $('stop').value = Math.min(c.windows.length - 1, 2);
  rangeLabel();
}
function rangeLabel() {
  const c = selectedCase(); if (!c) return;
  const start = Number($('start').value), stop = Number($('stop').value);
  $('range-label').textContent = stop < start ? '结束窗口须不早于起始窗口。' : `${stop-start+1} 个观测窗口 · 覆盖 ${number(c.windows[start].start_s)}—${number(c.windows[stop].start_s+c.windows[stop].duration_s)} 秒 · 每次最多 8 个窗口`;
}
function setProgress(job) {
  const busy = !['completed','failed'].includes(job.status);
  const percent = job.status === 'completed' ? 100 : job.status === 'queued' ? 0 : job.status === 'loading' ? 5 : Math.round(10 + job.done/job.total * 80);
  $('progress').value = percent; $('progress-number').textContent = percent;
  $('status-badge').textContent = labels[job.status] || job.status;
  $('progress-text').textContent = job.status === 'failed' ? job.error : `${labels[job.status]} · ${job.done} / ${job.total} 个窗口${job.elapsed_s ? ' · '+number(job.elapsed_s)+' 秒' : ''}`;
  $('run').disabled = busy || !cases.length;
  $('open-results').disabled = job.status !== 'completed';
}
async function poll(id) {
  clearTimeout(pollTimer);
  try {
    const job = await api(`/api/runs/${id}`); setProgress(job);
    if (!['completed','failed'].includes(job.status)) { pollTimer = setTimeout(() => poll(id), 900); return; }
    await history();
    if (job.status === 'completed') { await openResult(id, false); message('融合特征已生成，导出文件已通过下游接口回读校验。'); }
    else message(job.error);
  } catch (e) { message(`状态读取失败：${e.message}。正在重试。`); pollTimer = setTimeout(() => poll(id), 2500); }
}
async function run() {
  try {
    message(); $('run').disabled = true;
    const job = await api('/api/runs', {method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({case_id:$('case').value,start:Number($('start').value),stop:Number($('stop').value)+1})});
    currentRun = job.id; await poll(job.id);
  } catch(e) { message(e.message); $('run').disabled = !cases.length; }
}
function cell(row, text, tag='td') { const el=document.createElement(tag); el.textContent=text; row.append(el); return el; }
async function history() {
  const jobs = await api('/api/runs');
  const container = $('history'); container.replaceChildren();
  if (!jobs.length) { const p=document.createElement('p'); p.className='muted'; p.textContent='还没有运行记录。完成一次推理后，结果会保存在这里。'; container.append(p); return jobs; }
  const table=document.createElement('table'), header=table.insertRow();
  ['案例','运行时间','状态','窗口','耗时','操作'].forEach(x=>cell(header,x,'th'));
  jobs.forEach(job => {
    const row=table.insertRow(); cell(row,job.title); cell(row,new Date(job.created_at).toLocaleString('zh-CN',{hour12:false}));
    cell(row,labels[job.status]); cell(row,`${job.done}/${job.total}`); cell(row,job.elapsed_s ? number(job.elapsed_s)+' 秒' : '—');
    const action=cell(row,''); const b=document.createElement('button');
    b.textContent=job.status==='completed'?'打开回放 →':job.status==='failed'?'查看原因':'查看进度';
    b.onclick=()=>job.status==='completed'?openResult(job.id):job.status==='failed'?message(job.error):poll(job.id);
    action.append(b);
  }); container.append(table); return jobs;
}
async function openResult(id, switchTab=true) {
  if (loadingResult) return;
  loadingResult=true;
  try {
    stopPlayback(); result=await api(`/api/runs/${id}/result`); currentRun=id;
    document.querySelectorAll('.result-empty').forEach(el=>el.hidden=true);
    document.querySelectorAll('.result-content').forEach(el=>el.hidden=false);
    $('result-title').textContent=result.case.title;
    $('window-slider').max=result.windows.length-1; $('window-slider').value=0;
    $('open-results').disabled=false; renderWindow(); renderExport(); if(switchTab) tab('results');
  } catch(e) { message(e.message); } finally { loadingResult=false; }
}
function svgNode(name, attrs={}) { const el=document.createElementNS('http://www.w3.org/2000/svg', name); Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v)); return el; }
function traces(container, stream, duration, color) {
  container.replaceChildren();
  stream.traces.forEach(trace=>{
    const fig=document.createElement('figure'), caption=document.createElement('figcaption');
    caption.textContent=trace.label; fig.append(caption);
    const svg=svgNode('svg',{viewBox:'0 0 460 90',role:'img','aria-label':trace.label+'随时间变化'});
    const vals=trace.points.map(p=>p[1]); let min=Math.min(...vals), max=Math.max(...vals); if(min===max){min-=.5;max+=.5;}
    [15,65].forEach(y=>svg.append(svgNode('line',{x1:72,x2:450,y1:y,y2:y,stroke:'#e5edeb','stroke-dasharray':'3 4'})));
    const d=trace.points.map(([t,v],i)=>`${i?'L':'M'}${72+t/duration*376},${65-(v-min)/(max-min)*50}`).join(' ');
    svg.append(svgNode('path',{d,fill:'none',stroke:color,'stroke-width':1.5}));
    [[2,18,number(max)],[2,68,number(min)],[72,86,'0'],[420,86,number(duration)+' 秒']].forEach(([x,y,text])=>{ const label=svgNode('text',{x,y}); label.textContent=text; svg.append(label); });
    fig.append(svg); container.append(fig);
  });
  if(!stream.traces.length) container.textContent='该窗口暂无有效观测。';
}
function renderWindow() {
  if (!result) return;
  const w=result.windows[Number($('window-slider').value)];
  $('window-label').textContent=`${w.label} · ${number(w.start_s)}—${number(w.start_s+w.duration_s)} 秒`;
  $('phys-coverage').textContent=`字段观测率 ${number(w.physiology.coverage*100,1)}%`;
  $('vehicle-coverage').textContent=`字段观测率 ${number(w.vehicle.coverage*100,1)}%`;
  traces($('phys-traces'),w.physiology,w.duration_s,'#087e83'); traces($('vehicle-traces'),w.vehicle,w.duration_s,'#b18142');
  $('end-time').textContent=number(w.duration_s)+' 秒'; renderHeatmap(w);
}
function renderHeatmap(w) {
  const canvas=$('heatmap'), ctx=canvas.getContext('2d'); const width=canvas.width, height=canvas.height;
  ctx.fillStyle='#f0f3f2';ctx.fillRect(0,0,width,height);
  const scale=Math.max(.001,...w.embedding.flat().map(Math.abs));
  for(let t=0;t<96;t++) for(let f=0;f<64;f++) {
    if(!w.valid[t]) continue;
    const v=w.embedding[t][f]/scale, base=[242,243,235], target=v<0?[33,123,139]:[217,155,80];
    const rgb=base.map((b,i)=>Math.round(b+(target[i]-b)*Math.abs(v)));
    ctx.fillStyle=`rgb(${rgb.join(',')})`; ctx.fillRect(t*width/96,f*height/64,width/96+1,height/64+1);
  }
  const q=Number($('query').value); ctx.strokeStyle='#18343e';ctx.lineWidth=2;ctx.strokeRect(q*width/96,1,width/96,height-2);
  $('query-value').textContent=`参考时刻 ${number(w.times[q],3)} 秒 · ${w.valid[q]?'有效表示':'暂无有效表示'} · 第 1 维 ${number(w.embedding[q][0],5)}`;
}
function renderExport() {
  const base=`/api/runs/${currentRun}/download/`;
  [['zip','features.zip'],['csv','features.csv'],['npz','fusion_stream.npz'],['manifest','representation_manifest.json']].forEach(([id,file])=>$('download-'+id).href=base+file);
  const table=document.createElement('table'), header=table.insertRow(); ['窗口','参考时刻',...Array.from({length:6},(_,i)=>`维 ${i+1}`)].forEach(x=>cell(header,x,'th'));
  result.windows.forEach(w=>{const row=table.insertRow();cell(row,w.label);cell(row,number(w.times[0])+' 秒');w.embedding[0].slice(0,6).forEach(v=>cell(row,number(v,4)));}); $('feature-table').replaceChildren(table);
  const entries=[['数据来源',result.case.source_label],['冻结模型',result.case.model_label],['生成规模',`${result.windows.length} 个窗口 × 96 个时刻 × 64 维`],['运行方式',`CPU · 独立窗口推理 · ${number(result.elapsed_s)} 秒`],['下游回读',result.downstream_readback?'标准接口校验通过':'待校验'],['事件语义模块',result.model.semantic_event_enabled?'此检查点已启用':'此检查点未启用'],['模型文件 SHA-256（完整性散列）',result.checkpoint_sha256],['运行标识',currentRun]];
  $('provenance').replaceChildren();entries.forEach(([label,value])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('provenance').append(dt,dd);});
}
function stopPlayback(){clearInterval(timer);timer=null;$('play').textContent='▶ 播放';}
$('play').onclick=()=>{if(timer){stopPlayback();return;} if(!result)return; $('play').textContent='Ⅱ 暂停'; timer=setInterval(()=>{const next=Number($('window-slider').value)+1;if(next>=result.windows.length){stopPlayback();return;}$('window-slider').value=next;renderWindow();},1600);};
$('window-slider').oninput=()=>{stopPlayback();renderWindow();};
$('query').oninput=()=>{if(result)renderHeatmap(result.windows[Number($('window-slider').value)]);};
$('case').onchange=configureCase; $('start').onchange=rangeLabel; $('stop').onchange=rangeLabel; $('run').onclick=run;
$('refresh').onclick=()=>history().catch(e=>message(e.message)); $('open-results').onclick=()=>currentRun&&openResult(currentRun);
document.querySelectorAll('.nav').forEach(el=>el.onclick=()=>tab(el.dataset.tab));
(async()=>{try{cases=await api('/api/cases');$('case').replaceChildren(...cases.map(c=>new Option(c.title,c.id)));$('run').disabled=!cases.length;
  if(cases.length)configureCase();else message('尚未准备离线案例，请按交付说明生成本地案例包后重启服务。');
  const jobs=await history(); const active=jobs.find(j=>!['completed','failed'].includes(j.status)); if(active)poll(active.id);
}catch(e){message(e.message);$('run').disabled=true;}})();
