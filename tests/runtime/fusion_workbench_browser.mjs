// Run against the prepared workbench; all real inference uses the local CPU service.
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import assert from 'node:assert/strict';
const require=createRequire(resolve('artifacts/fusion-workbench/browser/package.json'));
const {chromium}=require('playwright');
const output=resolve(process.env.DEMO_SCREENSHOTS || 'artifacts/fusion-workbench/screenshots');
const base=process.env.WORKBENCH_URL || 'http://127.0.0.1:8766';
const caseId=process.env.WORKBENCH_CASE_ID || 'dingxin-validation';
const dataRoot=resolve(process.env.WORKBENCH_DATA_ROOT || 'artifacts/fusion-workbench');
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
const context=await browser.newContext({viewport:{width:1366,height:768},timezoneId:'Asia/Shanghai'});
const page=await context.newPage();
const errors=[],checks=[],screens=[];
page.on('pageerror',e=>errors.push(e.message));
async function capture(name) {
  await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:resolve(output,name),fullPage:true});
  screens.push(await page.evaluate(name=>({name,viewport:{width:innerWidth,height:innerHeight},
    tab:document.querySelector('.nav.active').dataset.tab,window:Number(document.getElementById('window-slider').value),
    query:Number(document.getElementById('query').value),dimension:Number(document.getElementById('dimension').value),
    cases,models:models.map(({schema,...m})=>m),jobs,activeJob,
    fieldSearch:document.getElementById('field-search').value,fields:fields.filter(r=>r.label.includes(document.getElementById('field-search').value))}),name));
  if(await page.evaluate(()=>!!result)) {
    const snapshot=await page.evaluate(()=>({...result,preview_job:jobs.find(j=>j.id===viewedRunId)}));
    await writeFile(resolve(output,'preview-data.json'),JSON.stringify(snapshot,null,2));
  }
}
async function complete() {
  await page.waitForFunction(()=>document.querySelector('#status-badge').textContent==='已完成',null,{timeout:90000});
  await page.locator('#open-results:not([disabled])').waitFor();
}
async function runAndComplete() {
  const previous=await page.evaluate(()=>activeJobId);
  await page.locator('#run').click();
  await page.waitForFunction(id=>activeJobId && activeJobId!==id,previous,{timeout:30000});
  await complete();
}
try {
  const pageStarted=performance.now();
  await page.goto(base);await page.locator('#run:not([disabled])').waitFor();
  const initialReadyMs=performance.now()-pageStarted;
  await page.locator('#case').selectOption(caseId);
  const button=await page.locator('#run').boundingBox();assert(button.y+button.height<768);
  assert(!/Chronaris|Zorathos|鼎新|离线|演示/.test(await page.locator('body').innerText()));
  checks.push('primary action visible at 1366x768','reader-facing names');
  await page.locator('#start').selectOption('2');await page.locator('#stop').selectOption('0');
  assert(await page.locator('#run').isDisabled());checks.push('invalid range blocks submission');
  await page.locator('#start').selectOption('0');await page.locator('#stop').selectOption('2');
  await runAndComplete();
  const newId=await page.evaluate(()=>activeJobId);
  await page.waitForFunction(id=>jobs[0]?.id===id,newId);
  await page.locator('#history button').nth(1).click();
  await page.waitForFunction(id=>viewedRunId && viewedRunId!==id,newId);
  await page.locator('[data-tab=data]').click();
  const resultStarted=performance.now();await page.locator('#open-results').click();
  await page.waitForFunction(id=>viewedRunId===id,newId);
  const resultSwitchMs=performance.now()-resultStarted;
  checks.push('latest result remains bound after viewing older history');
  const colorScale=await page.locator('#scale-max').textContent();
  await page.locator('#window-slider').fill('1');assert.equal(await page.locator('#scale-max').textContent(),colorScale);
  await page.locator('#query').fill('40');await page.locator('#dimension').fill('12');
  assert.match(await page.locator('#query-value').textContent(),/维度 12/);
  await page.locator('#heatmap').focus();await page.keyboard.press('ArrowDown');
  assert.match(await page.locator('#query-value').textContent(),/维度 13/);
  checks.push('common color scale','keyboard feature selection','shared time cursor');
  await capture('fusion-replay.png');
  await page.locator('#play').click();await page.waitForFunction(()=>Number(document.querySelector('#query').value)>40);await page.locator('#play').click();
  checks.push('time playback');
  await page.locator('[data-tab=exports]').click();
  const download=page.waitForEvent('download');await page.locator('#download-zip').click();
  await (await download).saveAs(resolve(output,'downloaded-features.zip'));
  await page.locator('#deliver').click();await page.locator('#download-pooled:not([hidden])').waitFor();
  assert.match(await page.locator('#delivery-status').textContent(),/3 个窗口/);
  await capture('feature-export.png');checks.push('complete package download','real pooled-feature consumer');
  await page.locator('[data-tab=models]').click();await page.locator('#field-table input').first().waitFor();
  if(caseId==='dingxin-validation') {
    await page.locator('#field-search').fill('脑电');assert.match(await page.locator('#field-table').textContent(),/语言模型解释记录/);
  } else {
    assert.match(await page.locator('#field-table').textContent(),/输入字段定义/);
    assert.match(await page.locator('#model-list').textContent(),/训练期运动学约束：5 种/);
  }
  await capture('models-and-fields.png');checks.push(caseId==='dingxin-validation'?'existing language-model annotations visible':'native simulation fields and five trained physical relations visible');
  await page.reload();await page.locator('#open-results:not([disabled])').waitFor();
  await page.locator('#open-results').click();await page.waitForFunction(id=>viewedRunId===id,newId);
  checks.push('current job restored after reload');
  await page.setViewportSize({width:390,height:844});await capture('mobile-replay.png');
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  const fontHeight=await page.locator('#phys-traces svg text').first().evaluate(e=>e.getBoundingClientRect().height);
  assert(fontHeight>=11);checks.push('mobile axes readable and no horizontal overflow');
  await page.setViewportSize({width:1366,height:768});await page.locator('[data-tab=data]').click();
  await capture('data-and-tasks.png');
  await page.locator('#case').selectOption('dingxin-validation');
  // Archive ingestion uses actual permitted source observations, including more than eight windows.
  await page.locator('#add-data').click();await page.locator('#source option').first().waitFor({state:'attached'});
  await page.locator('#prepare-source:not([disabled])').click();
  await page.waitForFunction(()=>!document.querySelector('#data-dialog').open,null,{timeout:90000});
  const archiveId=await page.locator('#case').inputValue();
  assert(archiveId.startsWith('record-'));assert.equal(await page.locator('#stop').inputValue(),'14');
  await page.locator('#incremental').check();await page.locator('#stop').selectOption('9');
  await runAndComplete();
  const firstJob=await page.evaluate(()=>activeJobId);
  await page.locator('#stop').selectOption('14');await runAndComplete();
  const secondJob=await page.evaluate(()=>activeJobId);
  const saved=await (await page.request.get(`${base}/api/runs/${secondJob}`)).json();
  assert.equal(saved.total,5);assert.deepEqual(saved.selected_indices,[10,11,12,13,14]);
  checks.push('real archive ingestion','ten-window task','incremental task consumes five remaining windows');
  // Export observed input values for a browser upload; no synthetic readings or training.
  const inputFile=resolve(output,'observed-input.json');
  execFileSync('/home/wangminan/env/anaconda3/envs/chronaris/bin/python',['-c',`
import json,sys
from chronaris.serving.fusion_workbench.cases import Case,ARRAYS
case=Case(sys.argv[2]+'/cases/dingxin-validation/case.json')
windows=[]
for i,w in enumerate(case.manifest['windows']):
 s=case.sample(i)
 windows.append({'sample_id':s.sample_id,'start_s':w['start_s'],'duration_s':s.context_duration_s,**{k:getattr(s,k).tolist() for k in ARRAYS}})
with open(sys.argv[1],'w') as f:json.dump({'format':'chronaris.observations.v1','title':'实测数据集 · 观测片段','schema':case.schema.to_dict(),'field_labels':case.manifest.get('field_labels',{}),'windows':windows},f,ensure_ascii=False)
`,inputFile,dataRoot],{env:{...process.env,PYTHONPATH:resolve('src'),PYTHONNOUSERSITE:'1',CUDA_VISIBLE_DEVICES:''}});
  const input=JSON.parse(await readFile(inputFile,'utf8'));
  const firstInput=resolve(output,'first-observation.json');await writeFile(firstInput,JSON.stringify({...input,windows:input.windows.slice(0,1)}));
  await page.locator('#add-data').click();await page.locator('#import-file').setInputFiles(firstInput);await page.locator('#import-record').click();
  await page.waitForFunction(()=>!document.querySelector('#data-dialog').open);
  const importedId=await page.locator('#case').inputValue();
  await page.locator('#incremental').check();await runAndComplete();
  for(const [index,windows] of [input.windows.slice(0,2),input.windows.slice(2)].entries()) {
    const appendFile=resolve(output,`append-observations-${index}.json`);await writeFile(appendFile,JSON.stringify({...input,windows}));
    await page.locator('#add-data').click();await page.locator('#import-file').setInputFiles(appendFile);await page.locator('#append-record').check();await page.locator('#import-record').click();
    await page.waitForFunction(()=>!document.querySelector('#data-dialog').open,null,{timeout:60000});
  }
  assert.equal(await page.locator('#case').inputValue(),importedId);
  assert.equal(await page.locator('#stop').inputValue(),'2');
  await runAndComplete();
  const appendedJob=await page.evaluate(()=>activeJobId);
  const appended=await (await page.request.get(`${base}/api/runs/${appendedJob}`)).json();
  assert.equal(appended.total,2);checks.push('JSON observation import','append deduplicates existing input','incremental processing after append');
  // Drop a status read once; UI retries and reaches the real terminal state.
  let dropped=false;
  await page.route('**/api/runs/*',async route=>{
    if(!dropped && /\/api\/runs\/[a-f0-9]{32}$/.test(route.request().url()) && route.request().method()==='GET') {dropped=true;await route.abort();}
    else await route.continue();
  });
  await page.locator('#incremental').uncheck();await page.locator('#start').selectOption('2');await runAndComplete();
  assert(dropped);await page.unroute('**/api/runs/*');checks.push('transient status disconnect recovers');
  assert.deepEqual(errors,[]);
  await writeFile(resolve(output,'browser-check.json'),JSON.stringify({status:'passed',checks,errors,archiveId,firstJob,secondJob,caseRunId:newId,initialReadyMs,resultSwitchMs,primaryButtonY:button.y,mobileAxisHeight:fontHeight},null,2));
  await writeFile(resolve(output,'ui-evidence.json'),JSON.stringify(screens,null,2));
  console.log('Browser acceptance passed:',output);
} finally {await browser.close();}
