/** Zoomed head crop to diagnose the eye-band artifact. */
import fs from 'node:fs';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';
import {
  buildAvatar, createRig, createAnimState, landmarksToJoints,
  foregroundMask, makeMapping,
} from '../src/body.js';
import { renderView } from './splat.mjs';

const d = jpeg.decode(fs.readFileSync('public/samples/sample-1.jpg'), { useTArray: true });
const img = { w: d.width, h: d.height, data: d.data };
const mapping = makeMapping(img.w, img.h);
const mask = foregroundMask(img);
// same hand pose as simulate-ml.mjs
const k = {
  nose: [0.500, 0.163], earL: [0.537, 0.158], earR: [0.463, 0.158],
  shL: [0.605, 0.248], shR: [0.395, 0.248],
  elL: [0.640, 0.378], elR: [0.360, 0.378],
  wrL: [0.668, 0.498], wrR: [0.332, 0.498],
  tipL: [0.672, 0.558], tipR: [0.328, 0.558],
  hipL: [0.552, 0.478], hipR: [0.448, 0.478],
  kneeL: [0.556, 0.658], kneeR: [0.444, 0.658],
  ankL: [0.560, 0.862], ankR: [0.440, 0.862],
  toeL: [0.562, 0.898], toeR: [0.438, 0.898],
};
const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }));
const S = (i, x, y, vis = 0.95) => { lm[i] = { x, y, z: 0, visibility: vis }; };
S(0, ...k.nose); S(7, ...k.earL); S(8, ...k.earR);
S(11, ...k.shL); S(12, ...k.shR); S(13, ...k.elL); S(14, ...k.elR);
S(15, ...k.wrL); S(16, ...k.wrR);
S(19, k.wrL[0] + 0.004, k.wrL[1] + 0.035); S(17, k.wrL[0] - 0.010, k.wrL[1] + 0.033);
S(21, k.wrL[0] - 0.020, k.wrL[1] + 0.018);
S(20, k.wrR[0] - 0.004, k.wrR[1] + 0.035); S(18, k.wrR[0] + 0.010, k.wrR[1] + 0.033);
S(22, k.wrR[0] + 0.020, k.wrR[1] + 0.018);
S(23, ...k.hipL); S(24, ...k.hipR); S(25, ...k.kneeL); S(26, ...k.kneeR);
S(27, ...k.ankL); S(28, ...k.ankR);
S(31, ...k.toeL); S(32, ...k.toeR);
S(29, k.ankL[0] - 0.002, k.ankL[1] - 0.012); S(30, k.ankR[0] + 0.002, k.ankR[1] - 0.012);
const joints = landmarksToJoints(lm, mapping);
const av = buildAvatar({ joints, mapping, mask, image: img, palette: 'photo', maxParticles: 40000, seed: 1337 });
const rig = createRig(av);
const png = renderView(av, rig, createAnimState(), [0.1, 0.05, 1], 900);
// crop head area
const crop = new PNG({ width: 460, height: 380 });
for (let y = 0; y < 380; y++) {
  for (let x = 0; x < 460; x++) {
    const s = ((y + 10) * 900 + (x + 220)) * 4;
    const o = (y * 460 + x) * 4;
    crop.data[o] = png.data[s]; crop.data[o + 1] = png.data[s + 1];
    crop.data[o + 2] = png.data[s + 2]; crop.data[o + 3] = 255;
  }
}
fs.writeFileSync('shots/zoom-head.png', PNG.sync.write(crop));
// where do head particles project? sample photo colors along face center line
const B = av.base;
const [hx, hy] = mapping.toPixel(B[4 * 3], B[4 * 3 + 1]);
console.log('head joint px:', hx.toFixed(0), hy.toFixed(0));
const px = (x, y) => {
  const o = (Math.round(y) * img.w + Math.round(x)) * 4;
  return [img.data[o], img.data[o + 1], img.data[o + 2]];
};
for (let v = 150; v <= 320; v += 17) console.log('v=' + v, px(hx, v).join(','));
console.log('done');
