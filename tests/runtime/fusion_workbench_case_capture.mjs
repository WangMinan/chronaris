import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const require=createRequire(resolve('artifacts/fusion-workbench/browser/package.json'));
const {chromium}=require('playwright');
const output=resolve(process.env.DEMO_SCREENSHOTS);
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
try {
 const page=await browser.newPage({viewport:{width:1366,height:768},timezoneId:'Asia/Shanghai'});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.WORKBENCH_URL);
 await page.locator('#run:not([disabled])').waitFor();
 await page.locator('#case').selectOption('dingxin-validation');
 await page.locator('#run').click();
 await page.waitForFunction(()=>activeJobId && activeJob?.status==='completed',null,{timeout:180000});
 const id=await page.evaluate(()=>activeJobId);
 assert.match(await page.locator('#progress-text').textContent(),/GPU/);
 const capture=async name=>{await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:resolve(output,name),fullPage:true});};
 await capture('data-and-tasks.png');
 await page.locator('#open-results').click();
 await page.waitForFunction(id=>viewedRunId===id,id);
 await page.locator('#query').fill('40');await page.locator('#dimension').fill('13');
 await capture('fusion-replay.png');
 await page.locator('[data-tab=models]').click();await page.locator('#field-table input').first().waitFor();
 await page.locator('#field-search').fill('脑电');await capture('models-and-fields.png');
 await page.locator('[data-tab=exports]').click();await page.locator('#deliver').click();
 await page.locator('#download-pooled:not([hidden])').waitFor();await capture('feature-export.png');
 const snapshot=await page.evaluate(()=>({...result,preview_job:jobs.find(j=>j.id===viewedRunId)}));
 assert.equal(snapshot.device,'cuda:0');assert.equal(snapshot.windows.length,3);assert.equal(snapshot.preview_job.id,id);
 assert.deepEqual(errors,[]);
 await writeFile(resolve(output,'preview-data.json'),JSON.stringify(snapshot,null,2));
 await writeFile(resolve(output,'browser-case.json'),JSON.stringify({status:'passed',run_id:id,device:snapshot.device,errors,
  viewport:[1366,768],window:1,query:40,dimension:13,views:['data','results','models','exports']},null,2));
} finally {await browser.close();}
