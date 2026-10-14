// Run against a prepared local demo; Playwright and screenshots stay checkout-local.
import {createRequire} from 'node:module';
import {resolve} from 'node:path';
import {mkdir, writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const require = createRequire(resolve('artifacts/fusion-workbench/browser/package.json'));
const {chromium} = require('playwright');
const output = resolve(process.env.DEMO_SCREENSHOTS || 'artifacts/fusion-workbench/screenshots');
await mkdir(output, {recursive:true});
const browser = await chromium.launch({headless:true, args:['--disable-gpu']});
const page = await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
const errors=[]; page.on('pageerror', e=>errors.push(e.message));
page.on('console', m=>{if(m.type()==='error' && !m.text().includes('favicon'))errors.push(m.text());});
async function screenshot(name) {
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:resolve(output,name),fullPage:true});
}
try {
  await page.goto('http://127.0.0.1:8766');
  await page.locator('#case option').waitFor({state:'attached'});
  await page.locator('#start').selectOption('1');
  await page.locator('#run').click();
  await page.waitForFunction(()=>document.querySelector('#status-badge').textContent==='已完成',null,{timeout:60000});
  await page.locator('#open-results:not([disabled])').waitFor();
  await screenshot('data-and-tasks.png');
  await page.locator('#open-results').click();
  await page.locator('#results .result-content').waitFor({state:'visible'});
  assert.match(await page.locator('#window-label').textContent(), /窗口 02/);
  await page.locator('#play').click();
  await page.waitForFunction(()=>document.querySelector('#window-label').textContent.includes('窗口 03'));
  await page.locator('#query').fill('40');
  await screenshot('fusion-replay.png');
  await page.locator('[data-tab=exports]').click();
  const downloadPromise=page.waitForEvent('download'); await page.locator('#download-zip').click();
  const download=await downloadPromise; assert.equal(download.suggestedFilename(),'features.zip');
  await download.saveAs(resolve(output,'downloaded-features.zip'));
  await screenshot('feature-export.png');
  await page.reload();
  await page.locator('#history button').first().click();
  await page.locator('#results .result-content').waitFor({state:'visible'});
  await page.setViewportSize({width:390,height:844});
  await screenshot('mobile-replay.png');
  const overflow = await page.evaluate(()=>document.documentElement.scrollWidth > innerWidth);
  assert.equal(overflow,false,'mobile viewport overflow');
  assert.deepEqual(errors,[]);
  await writeFile(resolve(output,'browser-check.json'),JSON.stringify({status:'passed',checks:[
    'CPU inference from changed window selection','progress and completion','window playback','query slider',
    'ZIP download','history after reload','mobile no horizontal overflow','no browser errors'],errors},null,2));
  console.log('Browser acceptance passed:', output);
} finally { await browser.close(); }
