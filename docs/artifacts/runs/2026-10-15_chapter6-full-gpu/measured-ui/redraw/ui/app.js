'use strict';
const $ = id => document.getElementById(id);
const labels = {queued:'等待运行',loading:'校验模型',inference:'生成特征',exporting:'封装结果',completed:'已完成',failed:'处理失败'};
const reviewLabels = {pending:'待复核',approved:'已确认',needs_revision:'待修订'};
const number = (n,digits=2) => Number(n).toLocaleString('zh-CN',{maximumFractionDigits:digits});
let runtime={}, cases=[], models=[], sources=[], fields=[], jobs=[], result=null;
let activeJobId=null, viewedRunId=null, activeJob=null, serviceReady=false;
let pollTimer=null, playTimer=null, resultRequest=0, fieldRequest=0, fieldPage=0, scale=1, submitting=false;
function message(text='') { $('message').textContent=text; $('message').hidden=!text; }
function connected(ok) { $('connection').textContent=ok?'服务已连接 · '+(runtime.device?.startsWith('cuda')?'GPU':'CPU'):'连接中断'; $('connection-dot').classList.toggle('error',!ok); }
async function api(path, payload) {
  const response=await fetch(path,payload===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data=await response.json();
  if(!response.ok) { const e=new Error(data.error||'请求失败'); e.status=response.status; throw e; }
  connected(true); return data;
}
function cell(row,text,tag='td') { const el=document.createElement(tag); el.textContent=text; row.append(el); return el; }
function button(text,action) { const b=document.createElement('button'); b.className='text-button'; b.textContent=text; b.onclick=()=>Promise.resolve().then(action).catch(e=>message(e.message)); return b; }
function tab(name) {
  document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==name);
  document.querySelectorAll('.nav').forEach(el=>{el.classList.toggle('active',el.dataset.tab===name);el.setAttribute('aria-current',el.dataset.tab===name?'page':'false');});
  $('page-title').textContent={data:'数据与任务',models:'模型与字段',results:'结果分析',exports:'特征交付'}[name];
  if(name!=='results') stopPlayback();
  if(name==='models') loadFields();
  window.scrollTo(0,0);
  if(name==='results' && result) renderWindow();
}
function selectedCase() { return cases.find(c=>c.id===$('case').value); }
function busy() { return submitting || activeJob && !['completed','failed'].includes(activeJob.status); }
function validRange() {
  const c=selectedCase(), start=Number($('start').value), stop=Number($('stop').value);
  return !!(c && start>=0 && stop>=start && stop<c.windows.length && $('model').value);
}
function syncRunButton() { $('run').disabled=!serviceReady || busy() || !validRange(); }
function rangeLabel() {
  const c=selectedCase(); if(!c) return;
  const start=Number($('start').value), stop=Number($('stop').value), valid=stop>=start;
  $('range-label').textContent=valid?`${stop-start+1} 个窗口 · ${number(c.windows[start].start_s)}—${number(c.windows[stop].start_s+c.windows[stop].duration_s)} 秒`:'结束窗口须不早于起始窗口。';
  $('range-label').classList.toggle('error',!valid);
  $('stop').setAttribute('aria-invalid',String(!valid)); syncRunButton();
}
function configureCase() {
  const c=selectedCase();
  if(!c) { $('start').replaceChildren();$('stop').replaceChildren();$('model').replaceChildren();syncRunButton();return; }
  for(const name of ['start','stop']) $(name).replaceChildren(...c.windows.map((w,i)=>new Option(`${w.label} · ${number(w.start_s)}—${number(w.start_s+w.duration_s)} 秒`,i)));
  $('stop').value=c.windows.length-1;
  const compatible=models.filter(m=>m.compatible_cases.includes(c.id));
  $('model').replaceChildren(...compatible.map(m=>new Option(m.label,m.id)));
  if(compatible.some(m=>m.id===c.checkpoint_sha256)) $('model').value=c.checkpoint_sha256;
  $('append-record').disabled=!c.appendable; $('append-record').checked=false;
  rangeLabel(); if(!$('models').hidden) loadFields();
}
async function refreshCatalog(selectId) {
  const previous=selectId || $('case').value;
  runtime=await api('/api/runtime');
  cases=await api('/api/cases'); models=await api('/api/models');
  $('case').replaceChildren(...cases.map(c=>new Option(c.title,c.id)));
  if(cases.some(c=>c.id===previous)) $('case').value=previous;
  $('record-total').textContent=`${cases.length} 份观测记录`;
  $('import-model').replaceChildren(...models.map(m=>new Option(m.label,m.id)));
  serviceReady=true; configureCase(); renderModels();
  if(!cases.length) message('请接入观测数据。');
}
function setProgress(job) {
  activeJob=job;
  const percent=job.status==='completed'?100:job.status==='queued'?0:job.status==='loading'?5:Math.round(10+job.done/job.total*80);
  $('progress').value=percent; $('progress-number').textContent=percent+'%';
  $('status-badge').textContent=labels[job.status]||job.status;
  $('task-name').textContent=`${job.title}${job.range_s?' · '+job.range_s.map(n=>number(n)).join('—')+' 秒':''}`;
  $('progress-text').textContent=job.status==='failed'?job.error:`${job.device?.startsWith('cuda')?'GPU':'CPU'} · ${job.done} / ${job.total} 个窗口${job.elapsed_s?' · '+number(job.elapsed_s)+' 秒':''}`;
  $('open-results').disabled=job.status!=='completed';
  $('retry-current').hidden=!(job.status==='failed' && job.request);
  const step={queued:0,loading:0,inference:1,exporting:2,completed:3,failed:0}[job.status];
  [...$('pipeline').children].forEach((el,i)=>el.classList.toggle('done',i<step));
  syncRunButton();
}
async function poll(id) {
  clearTimeout(pollTimer);
  try {
    const job=await api(`/api/runs/${id}`);
    if(id!==activeJobId) return;
    setProgress(job);
    if(!['completed','failed'].includes(job.status)) { pollTimer=setTimeout(()=>poll(id),700); return; }
    await history();
    if(job.status==='completed' && !viewedRunId) await openResult(id,false);
    if(job.status==='failed') message(job.error);
  } catch(e) {
    if(id!==activeJobId) return;
    connected(false); message(`任务状态读取失败：${e.message}`);
    if(e.status!==404) pollTimer=setTimeout(()=>poll(id),2000);
  }
}
function trackJob(job) { activeJobId=job.id; localStorage.setItem('fusion-active-job',job.id); setProgress(job); poll(job.id); }
async function run() {
  if(!validRange() || busy()) return;
  submitting=true; syncRunButton(); message();
  try { const job=await api('/api/runs',{case_id:$('case').value,start:Number($('start').value),stop:Number($('stop').value)+1,model_id:$('model').value,incremental:$('incremental').checked});trackJob(job); }
  catch(e) { message(e.message); }
  finally { submitting=false;syncRunButton(); }
}
async function retry(id) {
  const job=await api(`/api/runs/${id}/retry`,{}); message(); trackJob(job); tab('data');
}
async function history() {
  jobs=await api('/api/runs'); const container=$('history');container.replaceChildren();
  if(!jobs.length) { container.textContent='处理记录将在任务创建后显示。'; return; }
  const table=document.createElement('table'), header=table.insertRow();
  ['观测记录','时间范围','创建时间','状态','窗口','操作'].forEach(x=>cell(header,x,'th'));
  jobs.forEach(job=>{
    const row=table.insertRow(); cell(row,job.title);
    cell(row,job.range_s?job.range_s.map(x=>number(x)).join('—')+' 秒':'历史记录');
    cell(row,new Date(job.created_at).toLocaleString('zh-CN',{hour12:false})); cell(row,`${labels[job.status]} · ${job.device?.startsWith('cuda')?'GPU':'CPU'}`);cell(row,`${job.done}/${job.total}`);
    const action=cell(row,'');
    if(job.status==='completed') action.append(button('查看结果',()=>openResult(job.id)));
    else if(job.status==='failed') { action.append(button('失败原因',()=>message(job.error)));if(job.retryable) action.append(button('重试',()=>retry(job.id))); }
    else action.append(button('查看进度',()=>{activeJobId=job.id;setProgress(job);poll(job.id);tab('data');}));
  });container.append(table);
}
async function openResult(id,switchTab=true) {
  const token=++resultRequest;
  try {
    const next=await api(`/api/runs/${id}/result`);
    if(token!==resultRequest) return;
    stopPlayback();result=next;viewedRunId=id;
    scale=.001; result.windows.forEach(w=>w.embedding.forEach(row=>row.forEach(v=>{scale=Math.max(scale,Math.abs(v));})));
    document.querySelectorAll('.result-empty').forEach(el=>el.hidden=true);
    document.querySelectorAll('.result-content').forEach(el=>el.hidden=false);
    $('result-title').textContent=result.case.title;
    const job=jobs.find(j=>j.id===id);
    $('result-identity').textContent=`${job?new Date(job.created_at).toLocaleString('zh-CN',{hour12:false}):'处理记录'} · ${result.windows.length} 个窗口 · ${result.device?.startsWith('cuda')?'GPU':'CPU'}`;
    $('window-slider').max=result.windows.length-1;$('window-slider').value=0;
    $('query').max=result.windows[0].times.length-1;$('query').value=0;
    $('dimension').max=result.output_dim;$('dimension').value=1;
    if(switchTab) tab('results');
    renderWindow();renderExport();
    const receipt=await api(`/api/runs/${id}/delivery`);
    if(viewedRunId===id) renderDelivery(receipt);
  } catch(e) { if(token===resultRequest) message(e.message); }
}
function currentWindow() { return result?.windows[Number($('window-slider').value)]; }
function renderWindow() {
  const w=currentWindow();if(!w)return;
  $('window-label').textContent=`${w.label} · ${number(w.start_s)}—${number(w.start_s+w.duration_s)} 秒`;
  $('phys-coverage').textContent=`观测率 ${number(w.physiology.coverage*100,1)}%`;
  $('vehicle-coverage').textContent=`观测率 ${number(w.vehicle.coverage*100,1)}%`;
  $('end-time').textContent=number(w.duration_s)+' 秒';
  $('last-dimension').textContent=w.embedding[0].length;
  $('scale-min').textContent=number(-scale,3);$('scale-max').textContent=number(scale,3);renderQuery();
}
function renderQuery() {
  const w=currentWindow();if(!w)return;
  const q=Number($('query').value), f=Math.max(0,Math.min(w.embedding[0].length-1,Number($('dimension').value)-1||0));
  $('dimension').value=f+1;
  FusionCharts.heatmap($('heatmap'),w,scale,q,f);
  const t=w.times[q];
  $('query-value').textContent=`记录时间 ${number(w.start_s+t,3)} 秒 · 窗口内 ${number(t,3)} 秒 · 维度 ${f+1} · ${w.valid[q]?number(w.embedding[q][f],5):'无效'}`;
  FusionCharts.traces($('phys-traces'),w.physiology,w.duration_s,'#087e83',t);
  FusionCharts.traces($('vehicle-traces'),w.vehicle,w.duration_s,'#a7722c',t);
  renderFeatureTable();
}
function renderFeatureTable() {
  const w=currentWindow();if(!w)return;
  const q=Number($('query').value),first=Math.max(0,Math.min(w.embedding[0].length-6,Number($('dimension').value)-1));
  $('feature-position').textContent=`${w.label} · 窗口内 ${number(w.times[q],3)} 秒`;
  const table=document.createElement('table'),header=table.insertRow();
  for(let f=first;f<Math.min(first+6,w.embedding[q].length);f++) cell(header,`维度 ${f+1}`,'th');
  const row=table.insertRow();w.embedding[q].slice(first,first+6).forEach(v=>cell(row,w.valid[q]?number(v,4):'无效'));
  $('feature-table').replaceChildren(table);
}
function renderExport() {
  const base=`/api/runs/${viewedRunId}/download/`;
  [['zip','features.zip'],['csv','features.csv'],['npz','fusion_stream.npz'],['manifest','representation_manifest.json'],['pooled','pooled_features.csv']].forEach(([id,file])=>$('download-'+id).href=base+file);
  $('export-summary').textContent=`${result.windows.length} 个窗口 · ${result.query_points} 个参考时刻 · ${result.output_dim} 维特征`;
  const entries=[['运行设备',result.device?.startsWith('cuda')?'GPU（图形处理器）':'CPU（中央处理器）'],['数据来源',result.case.source_label],['模型版本',result.checkpoint_sha256.slice(0,12)],['事件语义聚合',result.model.semantic_event_enabled?'已启用':'基础因果融合配置'],['处理耗时',number(result.elapsed_s)+' 秒'],['标准表示回读',result.downstream_readback?'已通过':'待校验'],['模型完整性散列',result.checkpoint_sha256],['任务标识',viewedRunId]];
  $('provenance').replaceChildren();entries.forEach(([label,value])=>{const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('provenance').append(dt,dd);});
  $('download-pooled').hidden=true;$('delivery-status').textContent='';renderFeatureTable();
}
function renderDelivery(r) {
  $('download-pooled').hidden=r.status!=='completed';
  $('delivery-status').textContent=r.status==='completed'?`${r.windows} 个窗口已由标准接口读取，汇总特征已生成。`:'';
}
async function deliver() {
  const id=viewedRunId;$('deliver').disabled=true;
  try {const r=await api(`/api/runs/${id}/deliver`,{});if(viewedRunId===id)renderDelivery(r);}
  catch(e){message(e.message);}finally{$('deliver').disabled=false;}
}
function stopPlayback() { clearInterval(playTimer);playTimer=null;$('play').textContent='播放'; }
function renderModels() {
  $('model-list').replaceChildren();
  models.forEach(m=>{const card=document.createElement('article');card.className='model-card';const title=document.createElement('b');title.textContent=m.label;card.append(title);
    [`${m.training_status} · ${m.training_kind}`,`训练窗口 ${m.training_windows} · 验证窗口 ${m.validation_windows}`,m.semantic_event_enabled?'事件语义聚合已启用':'连续对齐与因果融合配置',`训练期运动学约束：${m.physics_relation_count || 0} 种`].forEach(t=>{const p=document.createElement('p');p.textContent=t;card.append(p);});$('model-list').append(card);});
}
async function loadFields() {
  const id=$('case').value, token=++fieldRequest;if(!id)return;
  try {const rows=await api(`/api/fields/${id}`);if(token!==fieldRequest)return;fields=rows;fieldPage=0;renderFields();}catch(e){message(e.message);}
}
function renderFields() {
  const query=$('field-search').value.trim(),status=$('field-filter').value;
  const rows=fields.filter(r=>(status==='all'||r.status===status)&&`${r.label} ${r.role} ${r.stream}`.includes(query));
  const pages=Math.max(1,Math.ceil(rows.length/20));fieldPage=Math.min(fieldPage,pages-1);
  $('field-count').textContent=`${rows.length} 项`;const table=document.createElement('table'),header=table.insertRow();
  ['选择','字段或规则','解释','单位','来源','复核状态'].forEach(t=>cell(header,t,'th'));
  rows.slice(fieldPage*20,fieldPage*20+20).forEach(r=>{const row=table.insertRow(),choice=cell(row,'');const input=document.createElement('input');input.type='checkbox';input.value=r.id;input.setAttribute('aria-label','选择'+r.label);choice.append(input);[r.label,r.role,r.unit,r.source,reviewLabels[r.status]].forEach(t=>cell(row,t));});
  $('field-table').replaceChildren(table);$('fields-page').textContent=`${fieldPage+1} / ${pages}`;
  $('fields-prev').disabled=fieldPage===0;$('fields-next').disabled=fieldPage>=pages-1;
}
async function reviewFields(decision) {
  const ids=[...$('field-table').querySelectorAll('input:checked')].map(el=>el.value);
  if(!ids.length){message('请先选择复核条目。');return;}
  try{await api(`/api/fields/${$('case').value}`,{ids,decision});await loadFields();message('复核结论已保存。');}catch(e){message(e.message);}
}
function configureSource() {
  const source=sources.find(s=>s.id===$('source').value);$('prepare-source').disabled=!source;
  $('source-start').value=source?.start_s??'';$('source-stop').value=source?.stop_s??'';
}
async function showImport() {
  $('dialog-message').hidden=true;$('data-dialog').showModal();
  try{sources=await api('/api/sources');$('source').replaceChildren(...sources.map(s=>new Option(s.title,s.id)));if(!sources.length)$('source').add(new Option('暂无可用归档记录',''));configureSource();}
  catch(e){dialogMessage(e.message);}
}
function dialogMessage(text) { $('dialog-message').textContent=text;$('dialog-message').hidden=false; }
async function importData(archive) {
  const target=archive?$('prepare-source'):$('import-record');target.disabled=true;dialogMessage('正在校验并接入数据…');
  try {
    let payload,path;
    if(archive){payload={source_id:$('source').value,start_s:Number($('source-start').value),stop_s:Number($('source-stop').value),model_id:$('import-model').value};path='/api/sources/prepare';}
    else {const file=$('import-file').files[0];if(!file)throw new Error('请选择观测文件');if(file.size>32*1024*1024)throw new Error('文件须小于 32 MiB');payload=JSON.parse(await file.text());payload.model_id=$('import-model').value;if($('append-record').checked)payload.record_id=$('case').value;else delete payload.record_id;path='/api/cases/import';}
    const record=await api(path,payload);await refreshCatalog(record.id);$('data-dialog').close();message('观测记录已接入。');
  }catch(e){dialogMessage(e.message);}finally{target.disabled=false;}
}
function downloadTemplate() {
  const model=models.find(m=>m.id===$('import-model').value);if(!model)return;
  const window={sample_id:'window-001',start_s:0,duration_s:30};
  for(const stream of ['physiology','vehicle']) {const n=model.schema[stream+'_feature_names'].length;window[stream+'_values']=[Array(n).fill(0)];window[stream+'_timestamps_s']=[0];window[stream+'_feature_mask']=[Array(n).fill(false)];}
  const payload={format:'chronaris.observations.v1',title:'观测记录',schema:model.schema,field_labels:{},windows:[window]};
  const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='observation-input.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('case').onchange=configureCase;$('start').onchange=rangeLabel;$('stop').onchange=rangeLabel;$('model').onchange=syncRunButton;
$('run').onclick=run;$('open-results').onclick=()=>activeJobId&&openResult(activeJobId);
$('retry-current').onclick=()=>retry(activeJobId).catch(e=>message(e.message));
$('refresh').onclick=()=>history().catch(e=>message(e.message));
$('to-export').onclick=()=>tab('exports');$('deliver').onclick=deliver;
$('window-slider').oninput=()=>{stopPlayback();renderWindow();};$('query').oninput=renderQuery;$('dimension').oninput=renderQuery;
$('play').onclick=()=>{if(playTimer){stopPlayback();return;}if(!result)return;$('play').textContent='暂停';playTimer=setInterval(()=>{let q=Number($('query').value)+1;if(q>Number($('query').max)){q=0;const next=Number($('window-slider').value)+1;if(next>=result.windows.length){stopPlayback();return;}$('window-slider').value=next;}$('query').value=q;renderWindow();},100);};
$('heatmap').onpointerdown=e=>{const w=currentWindow();if(!w)return;const r=e.currentTarget.getBoundingClientRect();$('query').value=Math.max(0,Math.min(w.times.length-1,Math.floor((e.clientX-r.left)/r.width*w.times.length)));$('dimension').value=Math.max(1,Math.min(w.embedding[0].length,Math.floor((e.clientY-r.top)/r.height*w.embedding[0].length)+1));renderQuery();};
$('heatmap').onkeydown=e=>{if(!currentWindow()||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft'||e.key==='ArrowRight')$('query').value=Number($('query').value)+(e.key==='ArrowLeft'?-1:1);else $('dimension').value=Number($('dimension').value)+(e.key==='ArrowUp'?-1:1);renderQuery();};
$('field-search').oninput=()=>{fieldPage=0;renderFields();};$('field-filter').onchange=()=>{fieldPage=0;renderFields();};
$('fields-prev').onclick=()=>{fieldPage--;renderFields();};$('fields-next').onclick=()=>{fieldPage++;renderFields();};
$('approve-fields').onclick=()=>reviewFields('approved');$('revise-fields').onclick=()=>reviewFields('needs_revision');
$('refresh-models').onclick=()=>refreshCatalog().catch(e=>message(e.message));
$('add-data').onclick=showImport;$('source').onchange=configureSource;$('prepare-source').onclick=()=>importData(true);$('import-record').onclick=()=>importData(false);$('import-template').onclick=downloadTemplate;
new ResizeObserver(()=>{if(result&&!$('results').hidden)renderQuery();}).observe($('phys-traces'));
document.querySelectorAll('.nav').forEach(el=>el.onclick=()=>tab(el.dataset.tab));
(async()=>{try{await refreshCatalog();await history();const job=jobs.find(j=>!['completed','failed'].includes(j.status))||jobs.find(j=>j.id===localStorage.getItem('fusion-active-job'));if(job)trackJob(job);}catch(e){serviceReady=false;syncRunButton();connected(false);message(e.message);}})();
