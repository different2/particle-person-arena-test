/**
 * Offline geometry preview: heuristic-path avatars (no ML) from the bundled
 * sample photos + the mannequin, rendered as software splats.
 * Run: node scripts/geo-preview.mjs
 */
import fs from 'node:fs';
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';
import {
  buildAvatar,
  createRig,
  createAnimState,
  heuristicJoints,
  mannequinJoints,
  makeMapping,
  JIDX,
} from '../src/body.js';
import { renderView } from './splat.mjs';

fs.mkdirSync('shots', { recursive: true });

function loadJpg(path) {
  const d = jpeg.decode(fs.readFileSync(path), { useTArray: true });
  return { w: d.width, h: d.height, data: d.data };
}

function buildPhotoAvatar(path, seed = 1337) {
  const img = loadJpg(path);
  const mapping = makeMapping(img.w, img.h);
  const joints = heuristicJoints(
    [img.w * 0.24, img.h * 0.03, img.w * 0.76, img.h * 0.985], mapping,
  );
  const t0 = performance.now();
  const av = buildAvatar({
    joints, mapping, mask: null, image: img, palette: 'photo',
    maxParticles: 40000, seed,
  });
  console.log(`${path}: built in ${(performance.now() - t0).toFixed(0)} ms, r=${av.bounds.radius.toFixed(2)}`);
  return av;
}

const views = {
  front: [0.12, 0.06, 1],
  side: [1, 0.06, 0.12],
  threequarter: [0.65, 0.22, 0.72],
  back: [0.1, 0.08, -1],
};

for (const [name, path] of [['sample1', 'public/samples/sample-1.jpg'], ['sample2', 'public/samples/sample-2.jpg']]) {
  const av = buildPhotoAvatar(path);
  const rig = createRig(av);
  const anim = createAnimState();
  for (const [vname, dir] of Object.entries(views)) {
    const png = renderView(av, rig, anim, dir);
    fs.writeFileSync(`shots/geo-${name}-${vname}.png`, PNG.sync.write(png));
  }
  rig.manual[JIDX.shoulderL * 3 + 2] = 2.2;
  rig.manual[JIDX.elbowL * 3] = -0.5;
  rig.manual[JIDX.kneeR * 3] = 1.1;
  rig.manual[JIDX.head * 3 + 1] = 0.5;
  const png = renderView(av, rig, anim, views.front);
  fs.writeFileSync(`shots/geo-${name}-posed.png`, PNG.sync.write(png));
}

{
  const av = buildAvatar({ joints: mannequinJoints(), palette: 'hologram', seed: 7 });
  const rig = createRig(av);
  const anim = createAnimState();
  for (const [vname, dir] of Object.entries(views)) {
    const png = renderView(av, rig, anim, dir);
    fs.writeFileSync(`shots/geo-mannequin-${vname}.png`, PNG.sync.write(png));
  }
}
console.log('done → shots/geo-*.png');
