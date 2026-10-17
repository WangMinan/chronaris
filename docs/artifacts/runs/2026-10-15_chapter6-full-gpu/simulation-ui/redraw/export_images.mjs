// Pure rendering from the bundled HTML and JSON; no service, weights or network.
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {dirname, join} from 'node:path';
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const {chromium} = require(process.env.PLAYWRIGHT_MODULE_PATH || 'playwright');
const root = dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({headless:true,args:['--disable-gpu']});
try {
  const page = await browser.newPage({viewport:{width:1366,height:900},deviceScaleFactor:2});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(join(root,'preview.html')).href);
  await page.waitForFunction(()=>document.querySelector('#query-value').textContent.includes('维度'));
  await page.locator('#window-slider').fill('1');
  await page.locator('#query').fill('40');
  await page.locator('#dimension').fill('13');
  assert.deepEqual(errors,[]);
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:join(root,'redrawn-result.png'),fullPage:true});
  const vectors=await page.locator('#phys-traces svg, #vehicle-traces svg').evaluateAll(nodes=>nodes.map(svg=>{
    const clone=svg.cloneNode(true);
    clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
    clone.setAttribute('style','font:14px sans-serif;fill:#44636c;background:white');
    return new XMLSerializer().serializeToString(clone);
  }));
  assert.equal(vectors.length,6);
  for(const [i,svg] of vectors.entries())await writeFile(join(root,`trace-${i+1}.svg`),svg);
  await writeFile(join(root,'render-check.json'),JSON.stringify({status:'passed',errors,window:2,query:40,
    dimension:13,pngScale:2,vectorCount:vectors.length},null,2));
} finally {await browser.close();}
