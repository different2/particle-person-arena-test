/**
 * Node test harness for src/body.js (pure, DOM-free).
 * Builds avatars from synthetic landmark/mask/image data and exercises
 * the rig + every animation preset. Run with: npm test
 */
import assert from 'node:assert';
import {
  MAX_PARTICLES,
  JIDX,
  buildAvatar,
  createRig,
  updateRig,
  computeAnim,
  createAnimState,
  landmarksToJoints,
  heuristicJoints,
  mannequinJoints,
  makeMapping,
  eulerToMat3,
  ANIM_PRESETS,
} from '../src/body.js';

const IMG_W = 320, IMG_H = 400;
const mapping = makeMapping(IMG_W, IMG_H);

// ---- synthetic frontal A-pose landmarks (normalized coords) ----------------
function synthLandmarks() {
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.9 }));
  const S = (i, x, y, z = 0, vis = 0.95) => { lm[i] = { x, y, z, visibility: vis }; };
  S(0, 0.50, 0.100); // nose
  S(1, 0.48, 0.090); S(2, 0.465, 0.090); S(3, 0.45, 0.090);
  S(4, 0.52, 0.090); S(5, 0.535, 0.090); S(6, 0.55, 0.090);
  S(7, 0.44, 0.100); S(8, 0.56, 0.100); // ears
  S(9, 0.49, 0.120); S(10, 0.51, 0.120);
  S(11, 0.38, 0.240); S(12, 0.62, 0.240); // shoulders
  S(13, 0.31, 0.400); S(14, 0.69, 0.400); // elbows
  S(15, 0.27, 0.550); S(16, 0.73, 0.550); // wrists
  S(17, 0.255, 0.600); S(18, 0.745, 0.600); // pinky
  S(19, 0.270, 0.605); S(20, 0.730, 0.605); // index
  S(21, 0.290, 0.585); S(22, 0.710, 0.585); // thumb
  S(23, 0.44, 0.520); S(24, 0.56, 0.520); // hips
  S(25, 0.43, 0.720); S(26, 0.57, 0.720); // knees
  S(27, 0.43, 0.900); S(28, 0.57, 0.900); // ankles
  S(29, 0.425, 0.915); S(30, 0.575, 0.915); // heels
  S(31, 0.445, 0.930); S(32, 0.555, 0.930); // foot index
  return lm;
}

// ---- synthetic person mask (head/ellipsoid torso/capsule limbs) ------------
function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(px - (ax + dx * t), py - (ay + dy * t));
}

function synthMask(w = 160, h = 200) {
  const data = new Uint8Array(w * h);
  const U = (f) => f * w, V = (f) => f * h;
  const limbs = [
    [[0.38, 0.24], [0.31, 0.40], [0.27, 0.55], 0.030],
    [[0.62, 0.24], [0.69, 0.40], [0.73, 0.55], 0.030],
    [[0.44, 0.52], [0.43, 0.72], [0.43, 0.90], 0.048],
    [[0.56, 0.52], [0.57, 0.72], [0.57, 0.90], 0.048],
  ];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const u = x / w, v = y / h;
      let inside = false;
      if (Math.hypot((u - 0.50) * w / h, v - 0.095) < 0.058) inside = true; // head
      const ex = (u - 0.50) / 0.125, ey = (v - 0.375) / 0.175;
      if (ex * ex + ey * ey < 1) inside = true; // torso
      for (const [a, b, c, r] of limbs) {
        const d = Math.min(
          segDist(u, v, a[0], a[1], b[0], b[1]),
          segDist(u, v, b[0], b[1], c[0], c[1]),
          Math.hypot(u - c[0], (v - c[1])) - 0.02,
        );
        if (d < r) { inside = true; break; }
      }
      data[y * w + x] = inside ? 255 : 0;
    }
  }
  let person = 0;
  for (const v of data) if (v > 128) person++;
  return { w, h, data, coverage: person / data.length };
}

// ---- synthetic photo: vertical clothing bands ------------------------------
function synthImage(w = IMG_W, h = IMG_H) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = y / h, u = x / w;
      let r, g, b;
      if (v < 0.16) { r = 232; g = 190; b = 160; } // skin head
      else if (v < 0.54) { r = 20 + 30 * u; g = 140; b = 150; } // teal shirt
      else { r = 30; g = 60; b = 120 + 40 * u; } // jeans
      const i = (y * w + x) * 4;
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
    }
  }
  return { w, h, data };
}

function assertFinite(arr, label) {
  for (let i = 0; i < arr.length; i++) {
    assert.ok(Number.isFinite(arr[i]), `${label}[${i}] = ${arr[i]}`);
  }
}

// ---------------------------------------------------------------- tests ----
console.log('— body.js engine tests —');

{
  // euler sanity: identity + 90° z-rotation
  const M = new Float32Array(9);
  eulerToMat3(0, 0, 0, M);
  for (let i = 0; i < 9; i++) assert.ok(Math.abs(M[i] - (i % 4 === 0 ? 1 : 0)) < 1e-9);
  eulerToMat3(0, 0, Math.PI / 2, M);
  assert.ok(Math.abs(M[0]) < 1e-6 && Math.abs(M[1] + 1) < 1e-6, 'Rz(90°) maps +x→+y');
  console.log('✓ eulerToMat3');
}

const mask = synthMask();
assert.ok(mask.coverage > 0.1 && mask.coverage < 0.6, `mask coverage ${mask.coverage}`);
console.log(`✓ synth mask (coverage ${(mask.coverage * 100).toFixed(1)}%)`);

const image = synthImage();
const joints = landmarksToJoints(synthLandmarks(), mapping);

let av;
{
  const t0 = performance.now();
  av = buildAvatar({ joints, mapping, mask, image, palette: 'photo', seed: 1337 });
  const ms = performance.now() - t0;
  assert.strictEqual(av.particles.count, MAX_PARTICLES);
  const enabled = av.bones.filter((b) => b.enabled).length;
  assert.ok(enabled >= 20, `enabled bones ${enabled}`);
  assert.ok(av.bounds.radius > 0.2 && av.bounds.radius < 3, `radius ${av.bounds.radius}`);
  assertFinite(av.particles.color, 'color');
  for (let i = 0; i < av.particles.count; i++) {
    assert.ok(av.particles.rad[i] > 0, 'radius must be positive');
  }
  console.log(`✓ buildAvatar photo (${enabled}/22 bones, r=${av.bounds.radius.toFixed(2)}, ${ms.toFixed(0)} ms)`);
}

{
  // determinism: same seed ⇒ identical build
  const av2 = buildAvatar({ joints, mapping, mask, image, palette: 'photo', seed: 1337 });
  let diff = 0;
  for (let i = 0; i < 3000; i++) diff += Math.abs(av.particles.color[i] - av2.particles.color[i]);
  assert.strictEqual(diff, 0, 'builds must be bit-identical');
  const av3 = buildAvatar({ joints, mapping, mask, image, palette: 'photo', seed: 42 });
  let diff2 = 0;
  for (let i = 0; i < 3000; i++) diff2 += Math.abs(av.particles.color[i] - av3.particles.color[i]);
  assert.ok(diff2 > 0, 'different seed must differ');
  console.log('✓ determinism (seed-gated)');
}

{
  // rig: hierarchy moves descendants; clamps hold; no NaNs
  const rig = createRig(av);
  const st = createAnimState();
  updateRig(rig, 0.016, st);
  assertFinite(rig.jointPos, 'jointPos rest');
  assertFinite(av.particles.positions, 'positions rest');
  const wristRest = [rig.jointPos[JIDX.wristL * 3], rig.jointPos[JIDX.wristL * 3 + 1]];
  rig.manual[JIDX.shoulderL * 3 + 2] = 1.4; // raise left arm
  rig.manual[JIDX.kneeL * 3] = -1.0; // illegal: must clamp
  for (let i = 0; i < 120; i++) updateRig(rig, 1 / 60, st);
  const wristRaised = [rig.jointPos[JIDX.wristL * 3], rig.jointPos[JIDX.wristL * 3 + 1]];
  const moved = Math.hypot(wristRaised[0] - wristRest[0], wristRaised[1] - wristRest[1]);
  assert.ok(moved > 0.3, `wrist must follow shoulder (moved ${moved})`);
  assert.ok(wristRaised[1] > wristRest[1], 'raised wrist must be higher');
  assert.ok(rig.smooth[JIDX.kneeL * 3] >= -0.121, 'knee clamp');
  assertFinite(av.particles.positions, 'positions posed');
  // depth scale shrinks z-extent
  rig.depthScale = 0.4;
  updateRig(rig, 1 / 60, st);
  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < av.particles.count; i += 7) {
    const z = av.particles.positions[i * 3 + 2];
    if (z < zMin) zMin = z; if (z > zMax) zMax = z;
  }
  rig.depthScale = 1.4;
  updateRig(rig, 1 / 60, st);
  let zMin2 = Infinity, zMax2 = -Infinity;
  for (let i = 0; i < av.particles.count; i += 7) {
    const z = av.particles.positions[i * 3 + 2];
    if (z < zMin2) zMin2 = z; if (z > zMax2) zMax2 = z;
  }
  assert.ok(zMax2 - zMin2 > (zMax - zMin) * 1.5, 'depth scale must widen z range');
  console.log('✓ rig FK + clamps + depth scale');
}

{
  // every animation preset runs cleanly for 3 simulated seconds
  const rig = createRig(av);
  const st = createAnimState();
  for (const preset of ANIM_PRESETS) {
    for (let f = 0; f < 180; f++) {
      computeAnim(st, preset, f / 60, 1, av.restPhi, 1 / 60);
      updateRig(rig, 1 / 60, st);
    }
    assertFinite(av.particles.positions, `positions ${preset}`);
    assertFinite(rig.jointPos, `joints ${preset}`);
  }
  console.log(`✓ presets (${ANIM_PRESETS.join(', ')}) — 180 frames each, all finite`);
}

{
  // mannequin + heuristic + partial-body paths
  const man = buildAvatar({ joints: mannequinJoints(), palette: 'hologram', seed: 7 });
  const rig = createRig(man);
  const st = createAnimState();
  for (let f = 0; f < 60; f++) {
    computeAnim(st, 'dance', f / 60, 1, man.restPhi, 1 / 60);
    updateRig(rig, 1 / 60, st);
  }
  assertFinite(man.particles.positions, 'mannequin');
  const hj = heuristicJoints([40, 10, 280, 390], mapping);
  const heu = buildAvatar({ joints: hj, mapping, mask: null, image, palette: 'photo', seed: 9 });
  assert.ok(heu.bones.every((b) => b.enabled), 'heuristic enables all bones');
  console.log('✓ mannequin + heuristic fallbacks');
}

{
  // partial body: legs invisible ⇒ leg bones skipped, build still succeeds
  const lm = synthLandmarks();
  for (const i of [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) lm[i].visibility = 0;
  const pj = landmarksToJoints(lm, mapping);
  const part = buildAvatar({ joints: pj, mapping, mask, image, palette: 'photo', seed: 5 });
  const thigh = part.bones.find((b) => b.name === 'thighL');
  const torso = part.bones.find((b) => b.name === 'torso');
  assert.ok(!thigh.enabled, 'thigh must be disabled without visible hips');
  assert.ok(torso.enabled, 'torso must survive');
  const rig = createRig(part);
  updateRig(rig, 0.016, createAnimState());
  assertFinite(part.particles.positions, 'partial positions');
  console.log('✓ partial-body degradation (legs dropped, torso kept)');
}

console.log('\nAll body.js tests passed.');
