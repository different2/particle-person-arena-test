/**
 * Headless visual verification: screenshots the demo avatar, runs the full
 * ML pipeline on both bundled samples, and captures poses / color modes.
 * Usage: npm run dev  (in another shell)  then  node scripts/shot.mjs [url]
 */
import puppeteer from 'puppeteer';
import fs from 'node:fs';

const URL = process.argv[2] || 'http://127.0.0.1:5173/';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
fs.mkdirSync('shots', { recursive: true });

const browser = await puppeteer.launch({
  headless: true,
  protocolTimeout: 600000,
  args: [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--window-size=1600,900',
  ],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900 });
const errors = [];
page.on('pageerror', (e) => { errors.push(`pageerror: ${e.message}`); });
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(`console: ${m.text().slice(0, 300)}`);
});

console.log('loading', URL);
await page.goto(URL, { waitUntil: 'networkidle', timeout: 120000 });
await sleep(5000);
console.log('stats:', await page.$eval('#stats', (el) => el.textContent));
console.log('badges:', await page.$eval('#detectInfo', (el) => el.textContent));
await page.screenshot({ path: 'shots/01-demo.png' });

async function waitForPipeline(label) {
  for (let i = 0; i < 120; i++) {
    await sleep(2000);
    const cls = await page.$eval('#pipelineStatus', (el) => el.className);
    const txt = await page.$eval('#pipelineStatus', (el) => el.textContent);
    if (!cls.includes('busy')) { console.log(`${label}: ${txt.trim()}`); return true; }
    if (i % 15 === 14) console.log(`${label}… still busy (${txt.trim()})`);
  }
  console.log(`${label}: TIMEOUT`);
  return false;
}

async function dragView(dx) {
  const box = await page.$eval('#viewport', (el) => {
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + dx, box.y + 20, { steps: 25 });
  await page.mouse.up();
  await sleep(1200);
}

// --- sample 1: full pipeline -----------------------------------------------
await page.click('[data-sample="samples/sample-1.jpg"]');
await waitForPipeline('sample-1');
console.log('badges:', await page.$eval('#detectInfo', (el) => el.textContent));
await sleep(2500);
await page.screenshot({ path: 'shots/02-sample1-front.png' });
await dragView(-520);
await page.screenshot({ path: 'shots/03-sample1-side.png' });
// depth color mode from a 3/4 angle
await page.click('#colorSeg button[data-mode="2"]');
await dragView(260);
await page.screenshot({ path: 'shots/04-sample1-depth.png' });
await page.click('#colorSeg button[data-mode="0"]');
// wave preset
await page.click('#presetRow button[data-preset="wave"]');
await sleep(2500);
await page.screenshot({ path: 'shots/05-sample1-wave.png' });

// --- sample 2 ----------------------------------------------------------------
await page.click('#presetRow button[data-preset="idle"]');
await page.click('[data-sample="samples/sample-2.jpg"]');
await waitForPipeline('sample-2');
console.log('badges:', await page.$eval('#detectInfo', (el) => el.textContent));
await sleep(2500);
await page.screenshot({ path: 'shots/06-sample2-front.png' });
await dragView(-520);
await page.screenshot({ path: 'shots/07-sample2-side.png' });
// dance + scatter to prove volumetric coherence
await page.click('#presetRow button[data-preset="dance"]');
await page.evaluate(() => {
  const s = document.getElementById('scatter');
  s.value = '0.15';
  s.dispatchEvent(new Event('input'));
});
await sleep(2000);
await page.screenshot({ path: 'shots/08-sample2-dance-scatter.png' });

console.log(`\n${errors.length} JS errors ${errors.length ? '(see below)' : '✓'}`);
for (const e of errors.slice(0, 12)) console.log(' ', e);
await browser.close();
