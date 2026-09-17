/**
 * body.js — deterministic 2D-photo → 3D-particle-character engine.
 *
 * Pure module: no DOM, no three.js, no network. Safe to unit-test in Node.
 *
 * Pipeline:
 *   33 pose landmarks + person mask + photo pixels
 *     → 3D joint skeleton (x/y from image, z = relative pose depth + procedural volume)
 *     → capsule/ellipsoid body-part volumes, radii measured from the mask
 *     → N particles sampled through the *volume* (front AND back), colored from the photo
 *     → forward-kinematics rig so joints & animations move the particles coherently
 *
 * Depth inference (deterministic / procedural, see README + "How it works"):
 *   - Joint depth comes from the pose estimator's relative z (small effect: lean of limbs).
 *   - Body thickness is procedural: every limb/torso segment is a volumetric capsule whose
 *     depth radius equals its (mask-measured) width radius × a per-part flatness factor.
 *   - The hidden back side is hallucinated: particles fill the whole volume and back
 *     particles reuse the mirrored front colors, darkened for a volume cue.
 */

export const MAX_PARTICLES = 40000;

// ---------------------------------------------------------------------------
// Joint table (topologically sorted: parents always come before children)
// ---------------------------------------------------------------------------
export const JOINTS = [
  ['root', -1], // 0  hips center
  ['spine', 0], // 1
  ['chest', 1], // 2  shoulders center
  ['neck', 2], // 3
  ['head', 3], // 4  head center
  ['headTop', 4], // 5
  ['shoulderL', 2], // 6
  ['elbowL', 6], // 7
  ['wristL', 7], // 8
  ['handL', 8], // 9  fingertip center
  ['fIdxL', 8], // 10 index tip
  ['fPkyL', 8], // 11 pinky tip
  ['fThbL', 8], // 12 thumb tip
  ['shoulderR', 2], // 13
  ['elbowR', 13], // 14
  ['wristR', 14], // 15
  ['handR', 15], // 16
  ['fIdxR', 15], // 17
  ['fPkyR', 15], // 18
  ['fThbR', 15], // 19
  ['hipL', 0], // 20
  ['kneeL', 20], // 21
  ['ankleL', 21], // 22
  ['footL', 22], // 23 toe tip
  ['hipR', 0], // 24
  ['kneeR', 24], // 25
  ['ankleR', 25], // 26
  ['footR', 26], // 27
];
export const JIDX = Object.fromEntries(JOINTS.map(([n], i) => [n, i]));
export const JOINT_COUNT = JOINTS.length;

// ---------------------------------------------------------------------------
// Bone table. prior = radius range as a fraction of torso length (chest-root).
// flatZ scales cross-section depth, wideX scales cross-section width.
// ---------------------------------------------------------------------------
const BONES = [
  // name,        ja, jb,              priorMin, priorMax, flatZ, wideX, density, profile, part
  ['pelvis',      0, 1,   0.20, 0.34, 0.78, 1.30, 1.10, 'bulge', 0],
  ['torso',       1, 2,   0.17, 0.30, 0.80, 1.22, 1.15, 'bulge', 0],
  ['neck',        2, 4,   0.05, 0.11, 0.90, 1.00, 1.00, 'capsule', 0], // chest→head: throat (jaw end widens into jaw/hair via mask marching)
  ['head',        3, 5,   0.16, 0.26, 0.92, 1.00, 1.50, 'head', 1],
  ['upperArmL',   6, 7,   0.070, 0.130, 0.95, 1.00, 1.00, 'capsule', 2],
  ['forearmL',    7, 8,   0.055, 0.100, 0.90, 1.00, 1.00, 'taper', 2],
  ['handL',       8, 9,   0.050, 0.085, 0.55, 1.25, 1.40, 'taper', 3],
  ['fIdxL',       8, 10,  0.016, 0.032, 0.70, 1.00, 1.60, 'capsule', 3],
  ['fPkyL',       8, 11,  0.016, 0.032, 0.70, 1.00, 1.60, 'capsule', 3],
  ['fThbL',       8, 12,  0.016, 0.034, 0.70, 1.00, 1.60, 'capsule', 3],
  ['upperArmR',   13, 14, 0.070, 0.130, 0.95, 1.00, 1.00, 'capsule', 4],
  ['forearmR',    14, 15, 0.055, 0.100, 0.90, 1.00, 1.00, 'taper', 4],
  ['handR',       15, 16, 0.050, 0.085, 0.55, 1.25, 1.40, 'taper', 5],
  ['fIdxR',       15, 17, 0.016, 0.032, 0.70, 1.00, 1.60, 'capsule', 5],
  ['fPkyR',       15, 18, 0.016, 0.032, 0.70, 1.00, 1.60, 'capsule', 5],
  ['fThbR',       15, 19, 0.016, 0.034, 0.70, 1.00, 1.60, 'capsule', 5],
  ['thighL',      20, 21, 0.085, 0.160, 0.95, 1.00, 1.00, 'capsule', 6],
  ['shankL',      21, 22, 0.060, 0.115, 0.95, 1.00, 1.00, 'taper', 6],
  ['footL',       22, 23, 0.050, 0.090, 0.60, 1.05, 1.40, 'taper', 7],
  ['thighR',      24, 25, 0.085, 0.160, 0.95, 1.00, 1.00, 'capsule', 8],
  ['shankR',      25, 26, 0.060, 0.115, 0.95, 1.00, 1.00, 'taper', 8],
  ['footR',       26, 27, 0.050, 0.090, 0.60, 1.05, 1.40, 'taper', 9],
];
export const BONE_COUNT = BONES.length;

// Standard BlazePose connections (landmark indices) for the 2D overlay.
export const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/** Euler (XYZ application order: v' = Rx·Ry·Rz·v) → row-major 3×3. */
export function eulerToMat3(x, y, z, out) {
  const cx = Math.cos(x), sx = Math.sin(x);
  const cy = Math.cos(y), sy = Math.sin(y);
  const cz = Math.cos(z), sz = Math.sin(z);
  // M = Rx * Ry * Rz
  out[0] = cy * cz; out[1] = -cy * sz; out[2] = sy;
  out[3] = sx * sy * cz + cx * sz; out[4] = -sx * sy * sz + cx * cz; out[5] = -sx * cy;
  out[6] = -cx * sy * cz + sx * sz; out[7] = cx * sy * sz + sx * cz; out[8] = cx * cy;
  return out;
}

/** World→pixel mapping. World: y-up, image height spans 2 units. */
export function makeMapping(imgW, imgH) {
  const k = 2 / imgH;
  return {
    imgW, imgH, k,
    toPixel(x, y) { return [x / k + imgW / 2, imgH / 2 - y / k]; },
    toWorld(u, v) { return [(u - imgW / 2) * k, (imgH / 2 - v) * k]; },
  };
}

// ---------------------------------------------------------------------------
// Joint extraction
// ---------------------------------------------------------------------------

/** BlazePose 33 landmarks (normalized x/y, relative z) → world-space joints. */
export function landmarksToJoints(lm, mapping) {
  const { imgW, imgH, k } = mapping;
  const aspect = imgW / imgH;
  // Median torso z as the depth anchor (robust to outlier z values).
  const torsoZ = [11, 12, 23, 24].map((i) => lm[i]?.z ?? 0).sort((a, b) => a - b);
  const zMed = torsoZ[1] ?? 0;
  const P = (i) => {
    const p = lm[i] ?? { x: 0.5, y: 0.5, z: 0, visibility: 0 };
    return {
      p: [
        (p.x * imgW - imgW / 2) * k,
        (imgH / 2 - p.y * imgH) * k,
        clamp(-((p.z ?? 0) - zMed) * 2 * aspect, -0.45, 0.45),
      ],
      vis: p.visibility ?? 0.6,
    };
  };
  const mid = (a, b, wa = 0.5) => ({
    p: [lerp(a.p[0], b.p[0], wa), lerp(a.p[1], b.p[1], wa), lerp(a.p[2], b.p[2], wa)],
    vis: Math.min(a.vis, b.vis),
  });
  const ext = (from, through, f) => ({
    p: [
      from.p[0] + (through.p[0] - from.p[0]) * f,
      from.p[1] + (through.p[1] - from.p[1]) * f,
      from.p[2] + (through.p[2] - from.p[2]) * f,
    ],
    vis: Math.min(from.vis, through.vis),
  });

  const L = [];
  for (let i = 0; i < 33; i++) L.push(P(i));

  const j = {};
  j.hipL = L[23]; j.hipR = L[24];
  j.shoulderL = L[11]; j.shoulderR = L[12];
  j.elbowL = L[13]; j.elbowR = L[14];
  j.wristL = L[15]; j.wristR = L[16];
  j.kneeL = L[25]; j.kneeR = L[26];
  j.ankleL = L[27]; j.ankleR = L[28];
  j.root = mid(j.hipL, j.hipR);
  j.chest = mid(j.shoulderL, j.shoulderR);
  j.spine = mid(j.root, j.chest);

  // Head: axis from chest through nose; center/width from ears when visible.
  const nose = L[0];
  const axis = [
    nose.p[0] - j.chest.p[0], nose.p[1] - j.chest.p[1], nose.p[2] - j.chest.p[2],
  ];
  const axisLen = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  axis[0] /= axisLen; axis[1] /= axisLen; axis[2] /= axisLen;
  const shoulderW = Math.max(dist3(j.shoulderL.p, j.shoulderR.p), 1e-4);
  const at = (s) => ({
    p: [j.chest.p[0] + axis[0] * s, j.chest.p[1] + axis[1] * s, j.chest.p[2] + axis[2] * s],
    vis: Math.min(j.chest.vis, nose.vis),
  });
  const earL = L[7], earR = L[8];
  const earsOk = Math.min(earL.vis, earR.vis) > 0.4;
  const earDist = earsOk ? Math.max(dist3(earL.p, earR.p), 1e-4) : shoulderW * 0.45;
  j.headR = clamp(earDist * 0.62, 0.02, 1);
  j.neck = at(shoulderW * 0.30);
  j.head = earsOk && earL.vis > 0.5 && earR.vis > 0.5
    ? mid(earL, earR)
    : at(shoulderW * 0.52);
  const headAxis = [
    j.head.p[0] - j.neck.p[0], j.head.p[1] - j.neck.p[1], j.head.p[2] - j.neck.p[2],
  ];
  const haLen = Math.hypot(headAxis[0], headAxis[1], headAxis[2]) || 1;
  j.headTop = {
    p: [
      j.head.p[0] + (headAxis[0] / haLen) * j.headR * 1.05,
      j.head.p[1] + (headAxis[1] / haLen) * j.headR * 1.05,
      j.head.p[2] + (headAxis[2] / haLen) * j.headR * 1.05,
    ],
    vis: j.head.vis,
  };

  // Hands: extend past the detected hand points toward plausible fingertips.
  const handMidL = mid(L[19], L[17]);
  const handMidR = mid(L[20], L[18]);
  j.handL = ext(j.wristL, handMidL, 1.55);
  j.handR = ext(j.wristR, handMidR, 1.55);
  j.fIdxL = ext(j.wristL, L[19], 1.42); j.fPkyL = ext(j.wristL, L[17], 1.42);
  j.fThbL = ext(j.wristL, L[21], 1.30);
  j.fIdxR = ext(j.wristR, L[20], 1.42); j.fPkyR = ext(j.wristR, L[18], 1.42);
  j.fThbR = ext(j.wristR, L[22], 1.30);

  // Feet: extend past the toe landmark.
  j.footL = ext(j.ankleL, L[31], 1.30);
  j.footR = ext(j.ankleR, L[32], 1.30);

  j.shoulderW = shoulderW;
  return j;
}

/**
 * Heuristic frontal A-pose joints fitted to a bounding box (in pixels).
 * Used when pose detection fails but we still have a person mask (or nothing).
 */
export function heuristicJoints(bbox, mapping) {
  const [x0, y0, x1, y1] = bbox; // pixels, y-down
  const W = x1 - x0, H = y1 - y0;
  const cx = (x0 + x1) / 2;
  const j = {};
  // [xFracOfW from center, yFracOfH from bottom]
  const spec = {
    root: [0, 0.52], spine: [0, 0.64], chest: [0, 0.78],
    neck: [0, 0.852], head: [0, 0.912], headTop: [0, 0.975],
    shoulderL: [0.115, 0.80], elbowL: [0.155, 0.645], wristL: [0.175, 0.50], handL: [0.18, 0.435],
    fIdxL: [0.20, 0.415], fPkyL: [0.165, 0.415], fThbL: [0.155, 0.45],
    shoulderR: [-0.115, 0.80], elbowR: [-0.155, 0.645], wristR: [-0.175, 0.50], handR: [-0.18, 0.435],
    fIdxR: [-0.20, 0.415], fPkyR: [-0.165, 0.415], fThbR: [-0.155, 0.45],
    hipL: [0.052, 0.52], kneeL: [0.058, 0.28], ankleL: [0.062, 0.045], footL: [0.075, 0.015],
    hipR: [-0.052, 0.52], kneeR: [-0.058, 0.28], ankleR: [-0.062, 0.045], footR: [-0.075, 0.015],
  };
  for (const [name, [fx, fy]] of Object.entries(spec)) {
    const u = cx + fx * W;
    const v = y1 - fy * H;
    const [wx, wy] = mapping.toWorld(u, v);
    j[name] = { p: [wx, wy, 0], vis: 0.9 };
  }
  j.headR = H * mapping.k * 0.045;
  j.shoulderW = W * 0.23 * mapping.k;
  return j;
}

/** Fixed hologram-mannequin joints in world units (offline default avatar). */
export function mannequinJoints() {
  const spec = {
    root: [0, 0.95, 0], spine: [0, 1.15, 0], chest: [0, 1.35, 0],
    neck: [0, 1.47, 0], head: [0, 1.585, 0], headTop: [0, 1.72, 0],
    shoulderL: [0.21, 1.35, 0], elbowL: [0.30, 1.10, 0], wristL: [0.36, 0.88, 0],
    handL: [0.375, 0.78, 0.01], fIdxL: [0.39, 0.755, 0.02], fPkyL: [0.365, 0.755, 0.015],
    fThbL: [0.345, 0.79, 0.03],
    shoulderR: [-0.21, 1.35, 0], elbowR: [-0.30, 1.10, 0], wristR: [-0.36, 0.88, 0],
    handR: [-0.375, 0.78, 0.01], fIdxR: [-0.39, 0.755, 0.02], fPkyR: [-0.365, 0.755, 0.015],
    fThbR: [-0.345, 0.79, 0.03],
    hipL: [0.10, 0.95, 0], kneeL: [0.11, 0.50, 0], ankleL: [0.12, 0.08, 0],
    footL: [0.12, 0.03, 0.11],
    hipR: [-0.10, 0.95, 0], kneeR: [-0.11, 0.50, 0], ankleR: [-0.12, 0.08, 0],
    footR: [-0.12, 0.03, 0.11],
  };
  const j = {};
  for (const [name, p] of Object.entries(spec)) j[name] = { p: [...p], vis: 1 };
  j.headR = 0.105;
  j.shoulderW = 0.42;
  return j;
}

// ---------------------------------------------------------------------------
// Radius measurement from the person mask
// ---------------------------------------------------------------------------

function maskScore(mask, u, v) {
  const x = Math.round(u), y = Math.round(v);
  if (x < 0 || y < 0 || x >= mask.w || y >= mask.h) return 0;
  return mask.data[y * mask.w + x];
}

/**
 * Deterministic background-subtraction segmentation for studio-like photos.
 * Estimates a per-row background color from the image edges, then marks
 * pixels that differ from it as foreground. Keeps only the largest connected
 * foreground component (the person). Used as an offline fallback when the ML
 * segmentation model is unreachable.
 * @returns {w,h,data:Uint8Array,covrage} mask (0/255), or null if unusable.
 */
export function foregroundMask(image, maxDim = 240, threshold = 0.16) {
  const scale = Math.min(1, maxDim / Math.max(image.w, image.h));
  const w = Math.max(16, Math.round(image.w * scale));
  const h = Math.max(16, Math.round(image.h * scale));
  const D = image.data, W = image.w, H = image.h;
  const px = (x, y) => {
    const o = (Math.min(H - 1, Math.max(0, Math.round(y))) * W
      + Math.min(W - 1, Math.max(0, Math.round(x)))) * 4;
    return [D[o] / 255, D[o + 1] / 255, D[o + 2] / 255];
  };
  // per-row LEFT/RIGHT edge backgrounds; each pixel is compared against the
  // horizontally interpolated background (robust to side-to-side gradients).
  const bgL = new Float32Array(h * 3), bgR = new Float32Array(h * 3);
  for (let y = 0; y < h; y++) {
    const sy = (y + 0.5) / h * H;
    const L = px(W * 0.015, sy), L2 = px(W * 0.04, sy);
    const R = px(W * 0.985, sy), R2 = px(W * 0.96, sy);
    for (let c = 0; c < 3; c++) {
      bgL[y * 3 + c] = (L[c] + L2[c]) / 2;
      bgR[y * 3 + c] = (R[c] + R2[c]) / 2;
    }
  }
  // light vertical smoothing of the background model
  for (let y = 1; y < h - 1; y++) {
    for (let c = 0; c < 3; c++) {
      bgL[y * 3 + c] = (bgL[(y - 1) * 3 + c] + bgL[y * 3 + c] * 2 + bgL[(y + 1) * 3 + c]) / 4;
      bgR[y * 3 + c] = (bgR[(y - 1) * 3 + c] + bgR[y * 3 + c] * 2 + bgR[(y + 1) * 3 + c]) / 4;
    }
  }
  const fg = new Uint8Array(w * h);
  let count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = px((x + 0.5) / w * W, (y + 0.5) / h * H);
      const f = x / (w - 1);
      const br = bgL[y * 3] + (bgR[y * 3] - bgL[y * 3]) * f;
      const bgc = bgL[y * 3 + 1] + (bgR[y * 3 + 1] - bgL[y * 3 + 1]) * f;
      const bb = bgL[y * 3 + 2] + (bgR[y * 3 + 2] - bgL[y * 3 + 2]) * f;
      const bd = Math.hypot(r - br, g - bgc, b - bb);
      if (bd > threshold) { fg[y * w + x] = 1; count++; }
    }
  }
  const coverage = count / (w * h);
  if (coverage < 0.03 || coverage > 0.85) return null;
  // largest connected component (4-neighborhood flood fill)
  const seen = new Uint8Array(w * h);
  let best = null, bestN = 0;
  const stack = [];
  for (let i = 0; i < w * h; i++) {
    if (!fg[i] || seen[i]) continue;
    stack.length = 0;
    stack.push(i);
    seen[i] = 1;
    const comp = [];
    while (stack.length) {
      const p = stack.pop();
      comp.push(p);
      const cx = p % w, cy = (p / w) | 0;
      if (cx > 0 && fg[p - 1] && !seen[p - 1]) { seen[p - 1] = 1; stack.push(p - 1); }
      if (cx < w - 1 && fg[p + 1] && !seen[p + 1]) { seen[p + 1] = 1; stack.push(p + 1); }
      if (cy > 0 && fg[p - w] && !seen[p - w]) { seen[p - w] = 1; stack.push(p - w); }
      if (cy < h - 1 && fg[p + w] && !seen[p + w]) { seen[p + w] = 1; stack.push(p + w); }
    }
    if (comp.length > bestN) { bestN = comp.length; best = comp; }
  }
  if (!best || bestN / (w * h) < 0.02) return null;
  // slight dilation so edges aren't clipped
  const out = new Uint8Array(w * h);
  for (const p of best) {
    out[p] = 255;
    const cx = p % w, cy = (p / w) | 0;
    if (cx > 0) out[p - 1] = 255;
    if (cx < w - 1) out[p + 1] = 255;
    if (cy > 0) out[p - w] = 255;
    if (cy < h - 1) out[p + w] = 255;
  }
  return { w, h, data: out, coverage: bestN / (w * h) };
}

/**
 * Measure bone half-width at parameter t by marching perpendicular to the bone
 * through the person mask. Returns world-unit radius.
 */
function measureRadius(mask, mapping, A, B, t, maxWorld) {
  const [ax, ay] = mapping.toPixel(A[0], A[1]);
  const [bx, by] = mapping.toPixel(B[0], B[1]);
  const dx = bx - ax, dy = by - ay;
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return maxWorld * 0.5;
  const nx = -dy / len, ny = dx / len; // perpendicular
  const px = ax + dx * t, py = ay + dy * t;
  const sx = mask.w / mapping.imgW, sy = mask.h / mapping.imgH;
  const maxSteps = Math.max(2, Math.ceil((maxWorld / mapping.k) * Math.min(sx, sy)));
  let left = 0, right = 0;
  for (let s = 1; s <= maxSteps; s++) {
    if (maskScore(mask, (px + nx * (s / sx)) * sx, (py + ny * (s / sy)) * sy) > 128) left = s;
    else break;
  }
  for (let s = 1; s <= maxSteps; s++) {
    if (maskScore(mask, (px - nx * (s / sx)) * sx, (py - ny * (s / sy)) * sy) > 128) right = s;
    else break;
  }
  // min() is robust when a neighboring body part touches one side.
  const steps = Math.min(left, right);
  return Math.max(0.5, steps) / Math.min(sx, sy) * mapping.k;
}

// ---------------------------------------------------------------------------
// Avatar construction
// ---------------------------------------------------------------------------

function radiusProfile(kind, t, r0, r1) {
  const tc = clamp(t, 0, 1);
  const base = lerp(r0, r1, tc);
  if (kind === 'head') {
    const grow = smoothstep(0, 0.38, tc); // jaw → skull
    const dome = smoothstep(0.55, 1.0, tc); // rounded crown
    return lerp(r0, r1, grow) * Math.sqrt(Math.max(0.10, 1 - dome * dome * 0.92));
  }
  if (kind === 'taper') return base * (1 - 0.30 * tc) * (0.78 + 0.22 * Math.sin(Math.PI * tc));
  if (kind === 'bulge') return base * (1 + 0.14 * Math.sin(Math.PI * tc)) * (0.80 + 0.20 * Math.sin(Math.PI * tc));
  return base * (0.74 + 0.26 * Math.pow(Math.sin(Math.PI * tc), 0.6));
}

function boneBasis(A, B) {
  let ax = B[0] - A[0], ay = B[1] - A[1], az = B[2] - A[2];
  const len = Math.hypot(ax, ay, az) || 1;
  ax /= len; ay /= len; az /= len;
  // stable perpendicular basis
  let hx = 0, hy = 0, hz = 1;
  if (Math.abs(az) > 0.9) { hx = 0; hy = 1; hz = 0; }
  let ux = ay * hz - az * hy, uy = az * hx - ax * hz, uz = ax * hy - ay * hx;
  const ul = Math.hypot(ux, uy, uz) || 1;
  ux /= ul; uy /= ul; uz /= ul;
  const vx = ay * uz - az * uy, vy = az * ux - ax * uz, vz = ax * uy - ay * ux;
  return { u: [ux, uy, uz], v: [vx, vy, vz], len };
}

function sampleImageBilinear(image, u, v) {
  const x = clamp(u, 0, image.w - 1.001), y = clamp(v, 0, image.h - 1.001);
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = x - x0, fy = y - y0;
  const d = image.data, W = image.w;
  const i00 = (y0 * W + x0) * 4, i10 = i00 + 4, i01 = i00 + W * 4, i11 = i01 + 4;
  const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
  return [
    (d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11) / 255,
    (d[i00 + 1] * w00 + d[i10 + 1] * w10 + d[i01 + 1] * w01 + d[i11 + 1] * w11) / 255,
    (d[i00 + 2] * w00 + d[i10 + 2] * w10 + d[i01 + 2] * w01 + d[i11 + 2] * w11) / 255,
  ];
}

// Hologram gradient stops (linear rgb)
const HOLO_TOP = [0.35, 0.95, 1.0];
const HOLO_MID = [0.55, 0.45, 1.0];
const HOLO_BOT = [1.0, 0.35, 0.75];

export function buildAvatar(opts) {
  const {
    joints, mapping = null, mask = null, image = null,
    palette = 'photo', maxParticles = MAX_PARTICLES, seed = 1337,
  } = opts;
  if (!joints || !joints.chest || !joints.root) throw new Error('buildAvatar: missing joints');

  const rng = mulberry32(seed);
  const J = JOINT_COUNT;

  // ---- joint arrays + presence -------------------------------------------
  const base = new Float32Array(J * 3);
  const vis = new Float32Array(J);
  const present = new Array(J).fill(false);
  for (let i = 0; i < J; i++) {
    const [name, parent] = JOINTS[i];
    const jd = joints[name] || { p: [0, 0, 0], vis: 0 };
    let [x, y, z] = jd.p;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      const pp = parent >= 0 ? [base[parent * 3], base[parent * 3 + 1], base[parent * 3 + 2]] : [0, 0, 0];
      x = pp[0]; y = pp[1]; z = pp[2];
    }
    base[i * 3] = x; base[i * 3 + 1] = y; base[i * 3 + 2] = z;
    vis[i] = jd.vis ?? 0.5;
  }
  // Root fallback: if hips are missing but chest exists (e.g. portrait crop),
  // extrapolate a root below the chest so we still get a bust avatar.
  const chestI = JIDX.chest, rootI = JIDX.root;
  const rootRaw = vis[rootI] >= 0.2;
  const chestRaw = vis[chestI] >= 0.2;
  if (!rootRaw && chestRaw) {
    const sw = joints.shoulderW && Number.isFinite(joints.shoulderW) ? joints.shoulderW : 0.3;
    base[rootI * 3] = base[chestI * 3];
    base[rootI * 3 + 1] = base[chestI * 3 + 1] - sw * 0.85;
    base[rootI * 3 + 2] = base[chestI * 3 + 2];
    vis[rootI] = 0.5;
    // repair the spine link (its visibility was derived from the missing hips)
    const spineI = JIDX.spine;
    base[spineI * 3] = (base[rootI * 3] + base[chestI * 3]) / 2;
    base[spineI * 3 + 1] = (base[rootI * 3 + 1] + base[chestI * 3 + 1]) / 2;
    base[spineI * 3 + 2] = (base[rootI * 3 + 2] + base[chestI * 3 + 2]) / 2;
    vis[spineI] = 0.5;
  }
  for (let i = 0; i < J; i++) {
    const [, parent] = JOINTS[i];
    const need = i === rootI ? 0.2 : 0.3;
    present[i] = vis[i] >= need && (parent < 0 || present[parent]);
  }
  if (!present[chestI]) throw new Error('buildAvatar: torso not detected');

  const jp = (i) => [base[i * 3], base[i * 3 + 1], base[i * 3 + 2]];
  let scaleRef = dist3(jp(chestI), jp(rootI));
  if (!Number.isFinite(scaleRef) || scaleRef < 1e-4) scaleRef = 0.5;

  // descendants per joint (for hierarchical rotations)
  const desc = Array.from({ length: J }, () => []);
  for (let i = 0; i < J; i++) {
    let p = JOINTS[i][1];
    while (p >= 0) { desc[p].push(i); p = JOINTS[p][1]; }
  }

  // ---- bones ---------------------------------------------------------------
  const bones = [];
  const bonesInSubtree = Array.from({ length: J }, () => []);
  for (let b = 0; b < BONES.length; b++) {
    const [name, ja, jb, pMin, pMax, flatZ, wideX, density, profile, part] = BONES[b];
    const A = jp(ja), Bc = jp(jb);
    const len = dist3(A, Bc);
    const minLen = name[0] === 'f' ? 0.008 : 0.02 * scaleRef;
    const enabled = present[ja] && present[jb] && len > minLen;
    const lo = pMin * scaleRef, hi = pMax * scaleRef;
    let r0 = (lo + hi) / 2, r1 = (lo + hi) / 2;
    if (enabled && mask && mapping) {
      const samples = [];
      for (let s = 0; s < 7; s++) {
        const t = 0.12 + (0.76 * s) / 6;
        samples.push(measureRadius(mask, mapping, A, Bc, t, hi));
      }
      // light smoothing along the bone
      const sm = samples.map((v, i) => {
        const a = samples[Math.max(0, i - 1)], c = samples[Math.min(6, i + 1)];
        return (a + v + c) / 3;
      });
      r0 = clamp((sm[0] + sm[1]) / 2, lo, hi);
      r1 = clamp((sm[5] + sm[6]) / 2, lo, hi);
    }
    if (name === 'head' && Number.isFinite(joints.headR)) {
      r1 = clamp(joints.headR, lo, hi);
      r0 = clamp(r1 * 0.45, lo * 0.7, hi);
    }
    const { u, v } = boneBasis(A, Bc);
    bones.push({ name, ja, jb, len, r0, r1, flatZ, wideX, density, profile, part, enabled, u, v });
    // register bone under every ancestor joint of ja (so rotations propagate)
    let a = ja;
    const seen = new Set();
    while (a >= 0 && !seen.has(a)) { seen.add(a); bonesInSubtree[a].push(b); a = JOINTS[a][1]; }
  }
  if (!bones.some((b) => b.enabled)) throw new Error('buildAvatar: no usable body parts');

  // rest arm angles (frontal-plane elevation from straight-down) for adaptive wave
  const restPhi = {};
  for (const [s, sh, el] of [['L', 6, 7], ['R', 13, 14]]) {
    const A = jp(sh), Bc = jp(el);
    restPhi[s] = Math.atan2(Bc[0] - A[0], -(Bc[1] - A[1])); // 0 = down
  }

  // ---- particle allocation ---------------------------------------------------
  const weights = bones.map((bn) => {
    if (!bn.enabled) return 0;
    const rm = 0.5 * (bn.r0 + bn.r1);
    return bn.len * rm * rm * bn.density;
  });
  const wSum = weights.reduce((a, b) => a + b, 0) || 1;
  const counts = weights.map((w, b) => {
    if (!bones[b].enabled) return 0;
    const minC = bones[b].name[0] === 'f' ? 24 : 60;
    return Math.max(minC, Math.round((maxParticles * w) / wSum));
  });
  // fix rounding residual on the largest bone
  let total = counts.reduce((a, b) => a + b, 0);
  let big = 0;
  for (let b = 1; b < counts.length; b++) if (counts[b] > counts[big]) big = b;
  counts[big] += maxParticles - total;
  total = maxParticles;

  const P = {
    count: total,
    bone: new Uint8Array(total),
    t: new Float32Array(total),
    ang: new Float32Array(total),
    rad: new Float32Array(total),
    color: new Float32Array(total * 3),
    size: new Float32Array(total),
    part: new Uint8Array(total),
    scatter: new Float32Array(total * 3),
    positions: new Float32Array(total * 3),
  };

  // y-range for hologram gradient (approx from joints)
  let yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < J; i++) {
    if (!present[i]) continue;
    yMin = Math.min(yMin, base[i * 3 + 1]); yMax = Math.max(yMax, base[i * 3 + 1]);
  }
  if (!(yMax > yMin)) { yMin = -1; yMax = 1; }

  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  let n = 0;
  const photoMode = palette === 'photo' && image && mapping;

  // Per-row LEFT/RIGHT edge background colors (the person is assumed roughly
  // centered). Each particle is compared against the horizontally interpolated
  // background — robust to studio gradients in both directions.
  let bgRowsL = null, bgRowsR = null;
  if (photoMode) {
    const W = image.w, H = image.h, D = image.data;
    const xs = [0.005, 0.02, 0.98, 0.995].map((f) => Math.min(W - 1, Math.floor(f * W)));
    bgRowsL = new Float32Array(H * 3);
    bgRowsR = new Float32Array(H * 3);
    for (let y = 0; y < H; y++) {
      const o0 = (y * W + xs[0]) * 4, o1 = (y * W + xs[1]) * 4;
      const o2 = (y * W + xs[2]) * 4, o3 = (y * W + xs[3]) * 4;
      bgRowsL[y * 3] = (D[o0] + D[o1]) / 2 / 255;
      bgRowsL[y * 3 + 1] = (D[o0 + 1] + D[o1 + 1]) / 2 / 255;
      bgRowsL[y * 3 + 2] = (D[o0 + 2] + D[o1 + 2]) / 2 / 255;
      bgRowsR[y * 3] = (D[o2] + D[o3]) / 2 / 255;
      bgRowsR[y * 3 + 1] = (D[o2 + 1] + D[o3 + 1]) / 2 / 255;
      bgRowsR[y * 3 + 2] = (D[o2 + 2] + D[o3 + 2]) / 2 / 255;
    }
    // vertical smoothing
    const rad = Math.max(1, Math.round(H * 0.008));
    for (const arr of [bgRowsL, bgRowsR]) {
      const sm = new Float32Array(H * 3);
      for (let y = 0; y < H; y++) {
        let r = 0, g = 0, b = 0, n = 0;
        for (let k = -rad; k <= rad; k++) {
          const yy = Math.min(H - 1, Math.max(0, y + k));
          r += arr[yy * 3]; g += arr[yy * 3 + 1]; b += arr[yy * 3 + 2]; n++;
        }
        sm[y * 3] = r / n; sm[y * 3 + 1] = g / n; sm[y * 3 + 2] = b / n;
      }
      arr.set(sm);
    }
  }

  for (let b = 0; b < bones.length; b++) {
    const bn = bones[b];
    if (!bn.enabled || counts[b] <= 0) continue;
    const A = jp(bn.ja), Bc = jp(bn.jb);
    const dx = Bc[0] - A[0], dy = Bc[1] - A[1], dz = Bc[2] - A[2];
    const [ux, uy, uz] = bn.u, [vx, vy, vz] = bn.v;
    const maxR = Math.max(bn.r0, bn.r1, 1e-6);

    for (let k = 0; k < counts[b]; k++, n++) {
      const surface = rng() < 0.42;
      // Candidate sampling: prefer mask-inside, non-background samples
      // so the silhouette stays crisp even with imperfect masks.
      const attempts = mask && mapping ? 7 : 2;
      let best = null;
      for (let a = 0; a < attempts; a++) {
        const t = -0.04 + rng() * 1.08;
        const ang = rng() * Math.PI * 2;
        const rf = surface ? 0.90 + rng() * 0.10 : 0.03 + 0.94 * Math.sqrt(rng());
        const rp = radiusProfile(bn.profile, t, bn.r0, bn.r1) * rf;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const ox = (ux * ca + vx * sa) * rp;
        const oy = (uy * ca + vy * sa) * rp;
        const oz = (uz * ca + vz * sa) * rp;
        const px = A[0] + dx * t + ox * bn.wideX;
        const py = A[1] + dy * t + oy;
        const pz = A[2] + dz * t + oz;
        let ok = true, bg = 0, cr = 1, cg = 1, cb = 1;
        if (photoMode) {
          const [u, v] = mapping.toPixel(px, py);
          const iu = (u / mapping.imgW) * image.w, iv = (v / mapping.imgH) * image.h;
          [cr, cg, cb] = sampleImageBilinear(image, iu, iv);
          if (bgRowsL) {
            const row = clamp(Math.round((v / mapping.imgH) * (image.h - 1)), 0, image.h - 1);
            const f = clamp(u / mapping.imgW, 0, 1);
            const br = lerp(bgRowsL[row * 3], bgRowsR[row * 3], f);
            const bgc = lerp(bgRowsL[row * 3 + 1], bgRowsR[row * 3 + 1], f);
            const bb = lerp(bgRowsL[row * 3 + 2], bgRowsR[row * 3 + 2], f);
            const bd = Math.hypot(cr - br, cg - bgc, cb - bb);
            bg = 1 - clamp((bd - 0.03) / 0.22, 0, 1);
            // Chroma-aware term: neutral-colored samples near a neutral
            // background's brightness are background even across gradients.
            const smx = Math.max(cr, cg, cb), smn = Math.min(cr, cg, cb);
            const rmx = Math.max(br, bgc, bb), rmn = Math.min(br, bgc, bb);
            const neutral = (1 - clamp((smx - smn) / (smx + 1e-3) / 0.18, 0, 1))
              * (1 - clamp((rmx - rmn) / (rmx + 1e-3) / 0.18, 0, 1));
            const lum = 0.299 * cr + 0.587 * cg + 0.114 * cb;
            const rlum = 0.299 * br + 0.587 * bgc + 0.114 * bb;
            const lumMatch = 1 - clamp(Math.abs(lum - rlum) / 0.28, 0, 1);
            bg = Math.max(bg, neutral * lumMatch);
          }
          if (mask) {
            const mu = (u / mapping.imgW) * mask.w, mv = (v / mapping.imgH) * mask.h;
            ok = maskScore(mask, mu, mv) > 110;
          }
        }
        const score = (ok ? 0 : 10) + bg;
        if (!best || score < best.score) {
          best = { t, ang, rp, px, py, pz, oz, ok, bg, cr, cg, cb, score };
        }
        if (ok && bg < 0.45) break;
      }
      const { t, ang, rp, px, py, pz, oz, ok: inside, bg, cr, cg, cb } = best;
      const zfrac = clamp(oz / maxR, -1, 1);

      P.bone[n] = b; P.t[n] = t; P.ang[n] = ang; P.rad[n] = rp;
      P.part[n] = bn.part;
      P.size[n] = (0.62 + rng() * 0.76) * (inside ? 1 : 0.5) * (photoMode ? 1 - 0.55 * bg : 1);

      // scatter direction (random unit vector × magnitude)
      let sx = rng() * 2 - 1, sy = rng() * 2 - 1, sz = rng() * 2 - 1;
      const sl = Math.hypot(sx, sy, sz) || 1;
      const smag = 0.25 + 0.75 * rng();
      P.scatter[n * 3] = (sx / sl) * smag;
      P.scatter[n * 3 + 1] = (sy / sl) * smag;
      P.scatter[n * 3 + 2] = (sz / sl) * smag;

      // color — deterministic hallucination for the back side: reuse the
      // mirrored front color, darkened with depth for a volume cue.
      const depthShade = 0.72 + 0.28 * (0.5 + 0.5 * zfrac);
      let r, g, bl;
      if (photoMode) {
        // background-colored samples fade out (strongly without a mask,
        // gently with one — the mask stays the authority there).
        const bgDim = mask ? 1 - 0.30 * bg : 1 - 0.85 * bg;
        const shade = (inside ? depthShade : 0.30) * bgDim;
        r = cr * shade; g = cg * shade; bl = cb * shade;
      } else {
        const f = clamp((py - yMin) / (yMax - yMin), 0, 1);
        const top = f > 0.5;
        const a = top ? HOLO_MID : HOLO_BOT, c = top ? HOLO_TOP : HOLO_MID;
        const ff = top ? (f - 0.5) * 2 : f * 2;
        r = lerp(a[0], c[0], ff) * depthShade;
        g = lerp(a[1], c[1], ff) * depthShade;
        bl = lerp(a[2], c[2], ff) * depthShade;
        if (bn.part === 1) { r = 1.0 * depthShade; g = 0.60 * depthShade; bl = 0.22 * depthShade; }
      }
      P.color[n * 3] = r; P.color[n * 3 + 1] = g; P.color[n * 3 + 2] = bl;

      if (px < min[0]) min[0] = px; if (px > max[0]) max[0] = px;
      if (py < min[1]) min[1] = py; if (py > max[1]) max[1] = py;
      if (pz < min[2]) min[2] = pz; if (pz > max[2]) max[2] = pz;
    }
  }

  // Shuffle so draw-range subsets (density slider) stay uniformly distributed.
  for (let i = n - 1; i > 0; i--) {
    const j2 = Math.floor(rng() * (i + 1));
    if (j2 === i) continue;
    const tb = P.bone[i]; P.bone[i] = P.bone[j2]; P.bone[j2] = tb;
    const tt = P.t[i]; P.t[i] = P.t[j2]; P.t[j2] = tt;
    const ta = P.ang[i]; P.ang[i] = P.ang[j2]; P.ang[j2] = ta;
    const tr = P.rad[i]; P.rad[i] = P.rad[j2]; P.rad[j2] = tr;
    const tp = P.part[i]; P.part[i] = P.part[j2]; P.part[j2] = tp;
    const ts = P.size[i]; P.size[i] = P.size[j2]; P.size[j2] = ts;
    for (let c = 0; c < 3; c++) {
      const tc = P.color[i * 3 + c]; P.color[i * 3 + c] = P.color[j2 * 3 + c]; P.color[j2 * 3 + c] = tc;
      const sc = P.scatter[i * 3 + c];
      P.scatter[i * 3 + c] = P.scatter[j2 * 3 + c]; P.scatter[j2 * 3 + c] = sc;
    }
  }

  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(
    Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 2, 1e-4,
  );
  return {
    joints: JOINTS.map(([name, parent], i) => ({ name, parent, present: present[i], vis: vis[i] })),
    base, present, desc, bones, bonesInSubtree,
    particles: P, bounds: { min, max, center, radius },
    scaleRef, restPhi, seed, palette,
  };
}

// ---------------------------------------------------------------------------
// Rig: hierarchical joint rotations → particle positions (per frame)
// ---------------------------------------------------------------------------

export function createRig(avatar) {
  const J = JOINT_COUNT;
  return {
    avatar,
    manual: new Float32Array(J * 3), // user slider euler per joint
    smooth: new Float32Array(J * 3), // damped euler actually applied
    rootOff: new Float32Array(3),
    smoothRoot: new Float32Array(3),
    jointPos: new Float32Array(J * 3), // current world joint positions
    depthScale: 1,
    _M: new Float32Array(9),
  };
}

const KNEE = [JIDX.kneeL, JIDX.kneeR];
const ELBOW = [JIDX.elbowL, JIDX.elbowR];

/**
 * Advance the rig one frame.
 * @param anim {euler: Float32Array(J*3) | null, root: [x,y,z] | null}
 * @returns particle positions Float32Array(3N)
 */
export function updateRig(rig, dt, anim) {
  const av = rig.avatar;
  const J = JOINT_COUNT;
  const target = rig._target || (rig._target = new Float32Array(J * 3));
  const ae = anim && anim.euler;

  for (let i = 0; i < J * 3; i++) target[i] = rig.manual[i] + (ae ? ae[i] : 0);
  // anatomical hinge clamps
  for (const k of KNEE) target[k * 3] = clamp(target[k * 3], -0.12, 2.4);
  for (const e of ELBOW) target[e * 3] = clamp(target[e * 3], -2.5, 0.12);

  const s = 1 - Math.exp(-dt * 9);
  for (let i = 0; i < J * 3; i++) rig.smooth[i] += (target[i] - rig.smooth[i]) * s;
  const ar = (anim && anim.root) || [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    rig.rootOff[c] = ar[c];
    rig.smoothRoot[c] += (ar[c] - rig.smoothRoot[c]) * s;
  }

  // current joint positions start at rest, then rotate subtrees root→leaves
  rig.jointPos.set(av.base);
  const M = rig._M;
  const bu = rig._bu || (rig._bu = new Float32Array(BONE_COUNT * 3));
  const bv = rig._bv || (rig._bv = new Float32Array(BONE_COUNT * 3));
  for (let b = 0; b < BONE_COUNT; b++) {
    bu[b * 3] = av.bones[b].u[0]; bu[b * 3 + 1] = av.bones[b].u[1]; bu[b * 3 + 2] = av.bones[b].u[2];
    bv[b * 3] = av.bones[b].v[0]; bv[b * 3 + 1] = av.bones[b].v[1]; bv[b * 3 + 2] = av.bones[b].v[2];
  }
  const rotSub = (arr, idx3, count, isJoint) => {
    const ex = rig.smooth[idx3], ey = rig.smooth[idx3 + 1], ez = rig.smooth[idx3 + 2];
    if (Math.abs(ex) + Math.abs(ey) + Math.abs(ez) < 1e-7) return;
    eulerToMat3(ex, ey, ez, M);
    const j = idx3 / 3;
    const px = rig.jointPos[j * 3], py = rig.jointPos[j * 3 + 1], pz = rig.jointPos[j * 3 + 2];
    if (isJoint) {
      const list = av.desc[j];
      for (let li = 0; li < list.length; li++) {
        const d = list[li] * 3;
        const x = arr[d] - px, y = arr[d + 1] - py, z = arr[d + 2] - pz;
        arr[d] = px + M[0] * x + M[1] * y + M[2] * z;
        arr[d + 1] = py + M[3] * x + M[4] * y + M[5] * z;
        arr[d + 2] = pz + M[6] * x + M[7] * y + M[8] * z;
      }
    } else {
      const list = av.bonesInSubtree[j];
      for (let li = 0; li < list.length; li++) {
        const o = list[li] * 3;
        for (const B of [bu, bv]) {
          const x = B[o], y = B[o + 1], z = B[o + 2];
          B[o] = M[0] * x + M[1] * y + M[2] * z;
          B[o + 1] = M[3] * x + M[4] * y + M[5] * z;
          B[o + 2] = M[6] * x + M[7] * y + M[8] * z;
        }
      }
    }
  };
  for (let j = 0; j < J; j++) {
    if (!av.present[j]) continue;
    rotSub(rig.jointPos, j * 3, 0, true);
    rotSub(null, j * 3, 0, false);
  }

  // bone endpoints
  const ax = rig._ax || (rig._ax = new Float32Array(BONE_COUNT * 3));
  const dd = rig._dd || (rig._dd = new Float32Array(BONE_COUNT * 3));
  for (let b = 0; b < BONE_COUNT; b++) {
    const bn = av.bones[b];
    const A = bn.ja * 3, Bc = bn.jb * 3, o = b * 3;
    ax[o] = rig.jointPos[A]; ax[o + 1] = rig.jointPos[A + 1]; ax[o + 2] = rig.jointPos[A + 2];
    dd[o] = rig.jointPos[Bc] - rig.jointPos[A];
    dd[o + 1] = rig.jointPos[Bc + 1] - rig.jointPos[A + 1];
    dd[o + 2] = rig.jointPos[Bc + 2] - rig.jointPos[A + 2];
  }

  // particles
  const P = av.particles;
  const pos = P.positions;
  const N = P.count;
  const rx = rig.smoothRoot[0], ry = rig.smoothRoot[1], rz = rig.smoothRoot[2];
  const ds = rig.depthScale;
  const zs = rig._zs || (rig._zs = new Float32Array(BONE_COUNT));
  const wx = rig._wx || (rig._wx = new Float32Array(BONE_COUNT));
  for (let b = 0; b < BONE_COUNT; b++) {
    zs[b] = av.bones[b].flatZ * ds;
    wx[b] = av.bones[b].wideX;
  }
  for (let i = 0; i < N; i++) {
    const b = P.bone[i] * 3;
    const t = P.t[i], rr = P.rad[i];
    const ca = Math.cos(P.ang[i]), sa = Math.sin(P.ang[i]);
    const ox = (bu[b] * ca + bv[b] * sa) * rr;
    const oy = (bu[b + 1] * ca + bv[b + 1] * sa) * rr;
    const oz = (bu[b + 2] * ca + bv[b + 2] * sa) * rr;
    const bi = b / 3;
    pos[i * 3] = ax[b] + dd[b] * t + ox * wx[bi] + rx;
    pos[i * 3 + 1] = ax[b + 1] + dd[b + 1] * t + oy + ry;
    pos[i * 3 + 2] = ax[b + 2] + dd[b + 2] * t + oz * zs[bi] + rz;
  }
  return pos;
}

// ---------------------------------------------------------------------------
// Procedural animation presets (additive euler targets + root offset)
// ---------------------------------------------------------------------------

export const ANIM_PRESETS = ['none', 'idle', 'wave', 'walk', 'dance', 'spin'];

export function createAnimState() {
  return { euler: new Float32Array(JOINT_COUNT * 3), root: [0, 0, 0], spinPhase: 0 };
}

/**
 * Fill `st` with the pose for `preset` at `time`. Additive on top of manual.
 * `rest` = avatar.restPhi ({L, R} rest arm elevations) for the adaptive wave.
 */
export function computeAnim(st, preset, time, amp, rest, dt = 0.016) {
  const E = st.euler;
  E.fill(0);
  st.root[0] = 0; st.root[1] = 0; st.root[2] = 0;
  if (preset === 'none' || amp <= 0.001) return st;
  const TAU = Math.PI * 2;
  const S = (j, x = 0, y = 0, z = 0) => {
    E[j * 3] += x * amp; E[j * 3 + 1] += y * amp; E[j * 3 + 2] += z * amp;
  };
  const JX = JIDX;
  if (preset === 'idle') {
    const b = Math.sin(time * 1.7);
    S(JX.spine, 0.035 * b);
    S(JX.shoulderL, 0, 0, 0.07 * b); S(JX.shoulderR, 0, 0, -0.07 * b);
    S(JX.elbowL, -0.05 - 0.03 * b); S(JX.elbowR, -0.05 - 0.03 * b);
    S(JX.head, 0.045 * Math.sin(time * 0.9), 0.07 * Math.sin(time * 0.53));
    S(JX.root, 0, 0.03 * Math.sin(time * 0.4));
    st.root[1] = 0.012 * b * amp;
  } else if (preset === 'wave') {
    // adaptive: raise the person's right arm to ~160° from wherever it rests
    const phiR = rest && Number.isFinite(rest.R) ? rest.R : -0.15;
    const raise = -2.75 - phiR; // z-rotation shifts frontal elevation 1:1
    S(JX.shoulderR, 0.10 * Math.sin(time * 4.4), 0, raise - 0.10 * Math.sin(time * 4.4));
    S(JX.elbowR, 0, 0, 0.55 * Math.sin(time * 4.4));
    S(JX.shoulderL, 0, 0, 0.10);
    S(JX.head, 0, 0.10 * Math.sin(time * 1.1), 0.06);
    S(JX.spine, 0, 0, 0.05);
    st.root[1] = 0.008 * Math.sin(time * 4.4) * amp;
  } else if (preset === 'walk') {
    const f = time * TAU * 1.05;
    S(JX.hipL, 0.55 * Math.sin(f)); S(JX.hipR, 0.55 * Math.sin(f + Math.PI));
    S(JX.kneeL, 0.18 + 0.85 * Math.pow(Math.max(0, Math.sin(f + 2.4)), 1.4));
    S(JX.kneeR, 0.18 + 0.85 * Math.pow(Math.max(0, Math.sin(f + Math.PI + 2.4)), 1.4));
    S(JX.shoulderL, -0.45 * Math.sin(f)); S(JX.shoulderR, -0.45 * Math.sin(f + Math.PI));
    S(JX.elbowL, -0.28 - 0.12 * Math.sin(f)); S(JX.elbowR, -0.28 - 0.12 * Math.sin(f + Math.PI));
    S(JX.spine, 0.05, 0.06 * Math.sin(f));
    S(JX.head, -0.04);
    st.root[1] = 0.030 * Math.sin(2 * f + 0.5) * amp;
  } else if (preset === 'dance') {
    const f = time * TAU * 1.55;
    st.root[1] = Math.abs(Math.sin(f)) * 0.075 * amp;
    st.root[0] = 0.05 * Math.sin(f * 0.5) * amp;
    S(JX.root, 0, 0.16 * Math.sin(f * 0.5), 0.10 * Math.sin(f));
    S(JX.spine, 0.06 * Math.sin(f + 0.6), 0, 0.13 * Math.sin(f + 0.6));
    S(JX.shoulderL, 0.25 * Math.sin(f + 1.2), 0, 1.15 + 0.55 * Math.sin(f));
    S(JX.shoulderR, 0.25 * Math.sin(f + 1.2 + Math.PI), 0, -1.15 - 0.55 * Math.sin(f + Math.PI));
    S(JX.elbowL, -0.75 - 0.35 * Math.sin(f + 1)); S(JX.elbowR, -0.75 - 0.35 * Math.sin(f + 1 + Math.PI));
    S(JX.hipL, 0.40 * Math.sin(f), 0, 0.08 * Math.sin(f));
    S(JX.hipR, 0.40 * Math.sin(f + Math.PI), 0, -0.08 * Math.sin(f));
    S(JX.kneeL, Math.max(0, 0.55 + 0.55 * Math.sin(f - 1.1)));
    S(JX.kneeR, Math.max(0, 0.55 + 0.55 * Math.sin(f + Math.PI - 1.1)));
    S(JX.head, 0, 0, 0.09 * Math.sin(f + 0.4));
  } else if (preset === 'spin') {
    st.spinPhase += dt * 1.5 * amp;
    S(JX.root, 0, st.spinPhase / Math.max(amp, 1e-3));
    S(JX.shoulderL, 0, 0, 0.38); S(JX.shoulderR, 0, 0, -0.38);
    S(JX.elbowL, -0.22); S(JX.elbowR, -0.22);
    S(JX.head, 0, -0.3 * Math.sin(time * 0.8));
  }
  return st;
}

// ---------------------------------------------------------------------------
// Joint slider definitions for the UI
// ---------------------------------------------------------------------------
export const JOINT_SLIDERS = [
  { group: 'Arms', joint: 'shoulderL', axis: 2, label: 'L shoulder · raise', min: -2.9, max: 2.9 },
  { group: 'Arms', joint: 'shoulderL', axis: 0, label: 'L shoulder · swing', min: -2.5, max: 2.5 },
  { group: 'Arms', joint: 'elbowL', axis: 0, label: 'L elbow · bend', min: -2.4, max: 0 },
  { group: 'Arms', joint: 'shoulderR', axis: 2, label: 'R shoulder · raise', min: -2.9, max: 2.9 },
  { group: 'Arms', joint: 'shoulderR', axis: 0, label: 'R shoulder · swing', min: -2.5, max: 2.5 },
  { group: 'Arms', joint: 'elbowR', axis: 0, label: 'R elbow · bend', min: -2.4, max: 0 },
  { group: 'Legs', joint: 'hipL', axis: 0, label: 'L hip · swing', min: -1.8, max: 1.8 },
  { group: 'Legs', joint: 'hipL', axis: 2, label: 'L hip · spread', min: -0.9, max: 0.9 },
  { group: 'Legs', joint: 'kneeL', axis: 0, label: 'L knee · bend', min: 0, max: 2.2 },
  { group: 'Legs', joint: 'hipR', axis: 0, label: 'R hip · swing', min: -1.8, max: 1.8 },
  { group: 'Legs', joint: 'hipR', axis: 2, label: 'R hip · spread', min: -0.9, max: 0.9 },
  { group: 'Legs', joint: 'kneeR', axis: 0, label: 'R knee · bend', min: 0, max: 2.2 },
  { group: 'Head & torso', joint: 'head', axis: 0, label: 'Head · nod', min: -0.7, max: 0.7 },
  { group: 'Head & torso', joint: 'head', axis: 1, label: 'Head · turn', min: -1.1, max: 1.1 },
  { group: 'Head & torso', joint: 'spine', axis: 0, label: 'Spine · lean', min: -0.8, max: 0.8 },
  { group: 'Head & torso', joint: 'spine', axis: 2, label: 'Spine · tilt', min: -0.8, max: 0.8 },
  { group: 'Whole body', joint: 'root', axis: 1, label: 'Turn', min: -Math.PI, max: Math.PI },
  { group: 'Whole body', joint: 'root', axis: 0, label: 'Bow', min: -0.7, max: 0.7 },
];
