/**
 * Full-pipeline simulation without network: hand-estimated 33-joint poses +
 * background-subtraction masks for the bundled samples, exercising the exact
 * ML-path code (landmarksToJoints → mask marching → gated sampling).
 * Run: node scripts/simulate-ml.mjs
 */
import fs from 'node:fs';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';
import {
  buildAvatar,
  createRig,
  createAnimState,
  landmarksToJoints,
  foregroundMask,
  makeMapping,
} from '../src/body.js';
import { renderView } from './splat.mjs';

fs.mkdirSync('shots', { recursive: true });

// Keypoints in normalized image coords (L = person's left = image right).
const POSES = {
  'public/samples/sample-1.jpg': {
    nose: [0.500, 0.163], earL: [0.537, 0.158], earR: [0.463, 0.158],
    shL: [0.605, 0.248], shR: [0.395, 0.248],
    elL: [0.640, 0.378], elR: [0.360, 0.378],
    wrL: [0.668, 0.498], wrR: [0.332, 0.498],
    tipL: [0.672, 0.558], tipR: [0.328, 0.558],
    hipL: [0.552, 0.478], hipR: [0.448, 0.478],
    kneeL: [0.556, 0.658], kneeR: [0.444, 0.658],
    ankL: [0.560, 0.862], ankR: [0.440, 0.862],
    toeL: [0.562, 0.898], toeR: [0.438, 0.898],
  },
  'public/samples/sample-2.jpg': {
    nose: [0.500, 0.228], earL: [0.536, 0.238], earR: [0.464, 0.238],
    shL: [0.618, 0.285], shR: [0.360, 0.275],
    elL: [0.662, 0.420], elR: [0.228, 0.205],
    wrL: [0.660, 0.538], wrR: [0.208, 0.115],
    tipL: [0.658, 0.598], tipR: [0.200, 0.045],
    hipL: [0.552, 0.512], hipR: [0.448, 0.512],
    kneeL: [0.558, 0.682], kneeR: [0.442, 0.682],
    ankL: [0.568, 0.862], ankR: [0.432, 0.862],
    toeL: [0.570, 0.900], toeR: [0.430, 0.900],
  },
};

function makeLandmarks(k) {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }));
  const S = (i, x, y, vis = 0.95) => { lm[i] = { x, y, z: 0, visibility: vis }; };
  S(0, ...k.nose);
  S(7, ...k.earL); S(8, ...k.earR);
  const [nx, ny] = k.nose;
  S(1, nx - 0.009, ny - 0.011); S(2, nx - 0.020, ny - 0.011); S(3, nx - 0.031, ny - 0.011);
  S(4, nx + 0.009, ny - 0.011); S(5, nx + 0.020, ny - 0.011); S(6, nx + 0.031, ny - 0.011);
  S(9, nx - 0.009, ny + 0.020); S(10, nx + 0.009, ny + 0.020);
  S(11, ...k.shL); S(12, ...k.shR);
  S(13, ...k.elL); S(14, ...k.elR);
  S(15, ...k.wrL); S(16, ...k.wrR);
  for (const [side, w, t, ii, pp, tt] of [
    ['L', k.wrL, k.tipL, 19, 17, 21], ['R', k.wrR, k.tipR, 20, 18, 22],
  ]) {
    const dx = t[0] - w[0], dy = t[1] - w[1];
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const s = side === 'L' ? 1 : -1;
    S(ii, w[0] + dx * 0.62 + 0.008 * s, w[1] + dy * 0.62);
    S(pp, w[0] + dx * 0.58 - 0.008 * s, w[1] + dy * 0.58);
    S(tt, w[0] + dx * 0.30 - 0.020 * s, w[1] + dy * 0.30 + 0.004);
  }
  S(23, ...k.hipL); S(24, ...k.hipR);
  S(25, ...k.kneeL); S(26, ...k.kneeR);
  S(27, ...k.ankL); S(28, ...k.ankR);
  for (const [a, t, heel, toe] of [
    [k.ankL, k.toeL, 29, 31], [k.ankR, k.toeR, 30, 32],
  ]) {
    S(toe, ...t);
    S(heel, a[0] + (a[0] - t[0]) * 0.35, a[1] + (a[1] - t[1]) * 0.35);
  }
  return lm;
}

function loadJpg(path) {
  const d = jpeg.decode(fs.readFileSync(path), { useTArray: true });
  return { w: d.width, h: d.height, data: d.data };
}

const views = {
  front: [0.12, 0.06, 1],
  side: [1, 0.06, 0.12],
  threequarter: [0.65, 0.22, 0.72],
  back: [0.1, 0.08, -1],
};

for (const [path, key] of Object.entries(POSES)) {
  const name = path.includes('sample-1') ? 'ml1' : 'ml2';
  const img = loadJpg(path);
  const mapping = makeMapping(img.w, img.h);
  const mask = foregroundMask(img);
  console.log(`${name}: fg-mask ${mask ? `${mask.w}×${mask.h} cov=${(mask.coverage * 100).toFixed(1)}%` : 'FAILED'}`);
  if (mask) {
    const mp = new PNG({ width: mask.w, height: mask.h });
    for (let i = 0; i < mask.w * mask.h; i++) {
      mp.data[i * 4] = mask.data[i]; mp.data[i * 4 + 1] = mask.data[i];
      mp.data[i * 4 + 2] = mask.data[i]; mp.data[i * 4 + 3] = 255;
    }
    fs.writeFileSync(`shots/sim-${name}-mask.png`, PNG.sync.write(mp));
  }
  const joints = landmarksToJoints(makeLandmarks(key), mapping);
  const t0 = performance.now();
  const av = buildAvatar({
    joints, mapping, mask, image: img, palette: 'photo', maxParticles: 40000, seed: 1337,
  });
  console.log(`${name}: built in ${(performance.now() - t0).toFixed(0)} ms, ` +
    `${av.bones.filter((b) => b.enabled).length}/${av.bones.length} bones, r=${av.bounds.radius.toFixed(2)}`);
  for (const bn of ['torso', 'head', 'upperArmL', 'forearmL', 'thighL', 'shankL']) {
    const b = av.bones.find((x) => x.name === bn);
    console.log(`   ${bn.padEnd(10)} r0=${b.r0.toFixed(3)} r1=${b.r1.toFixed(3)} ${b.enabled ? '' : 'DISABLED'}`);
  }
  const rig = createRig(av);
  const anim = createAnimState();
  for (const [vname, dir] of Object.entries(views)) {
    const png = renderView(av, rig, anim, dir);
    fs.writeFileSync(`shots/sim-${name}-${vname}.png`, PNG.sync.write(png));
  }
}
console.log('done → shots/sim-*.png');
