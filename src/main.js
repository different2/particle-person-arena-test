/**
 * main.js — app orchestration: intake → perception → avatar build → rig loop → UI.
 */
import './styles.css';
import {
  MAX_PARTICLES,
  JIDX,
  JOINT_SLIDERS,
  buildAvatar,
  createRig,
  updateRig,
  computeAnim,
  createAnimState,
  landmarksToJoints,
  heuristicJoints,
  mannequinJoints,
  foregroundMask,
  makeMapping,
} from './body.js';
import { detectPerson } from './ml.js';
import { createViewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const BASE = import.meta.env.BASE_URL || './';

const state = {
  bitmap: null, // processed (downscaled) ImageBitmap
  image: null, // {w,h,data} pixels for color sampling
  mapping: null,
  landmarks: null,
  mask: null,
  method: 'mannequin',
  avatar: null,
  rig: null,
  anim: createAnimState(),
  preset: 'idle',
  amp: 1,
  playing: true,
  time: 0,
  seed: 1337,
  busy: false,
  photoName: '',
};

// ---------------------------------------------------------------- setup ----
const viewer = createViewer($('viewport'));
viewer.setOnFps((fps) => {
  const drawn = state.avatar ? drawnCount() : 0;
  $('stats').textContent =
    `${fps.toFixed(0)} fps · ${drawn.toLocaleString()} particles · ${state.preset}`;
});
const drawnCount = () => Math.min(state.avatar.particles.count, +$('density').value);

// ---------------------------------------------------------------- toast ----
let toastTimer = 0;
function toast(msg, ms = 4200) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), ms);
}

// --------------------------------------------------------------- status ----
function setStatus(text, mode = '') {
  const el = $('pipelineStatus');
  el.className = `status ${mode}`;
  el.innerHTML = `<span class="pulse"></span> ${text}`;
  if (mode === 'busy') {
    $('progressBar').classList.add('indet');
    $('veil').classList.remove('hidden');
    $('veilText').textContent = text;
  } else {
    $('progressBar').classList.remove('indet');
    $('progressBar').style.width = mode === 'error' ? '0' : '100%';
    $('veil').classList.add('hidden');
  }
}

function setStepper(stage) {
  document.querySelectorAll('#stepper li').forEach((li) => {
    const s = +li.dataset.stage;
    li.classList.toggle('done', s < stage);
    li.classList.toggle('active', s === stage);
  });
}

// ---------------------------------------------------------- joint sliders --
const sliderInputs = [];
function buildJointSliders() {
  const host = $('jointSliders');
  host.innerHTML = '';
  sliderInputs.length = 0;
  let lastGroup = '';
  for (const def of JOINT_SLIDERS) {
    if (def.group !== lastGroup) {
      lastGroup = def.group;
      const h = document.createElement('div');
      h.className = 'joint-group';
      h.innerHTML = `<h3>${def.group}</h3>`;
      h.dataset.group = def.group;
      host.appendChild(h);
    }
    const group = host.querySelector(`[data-group="${def.group}"]`);
    const label = document.createElement('label');
    label.className = 'slider';
    label.innerHTML = `${def.label} <b>0.00</b>
      <input type="range" min="${def.min}" max="${def.max}" step="0.01" value="0" />`;
    const input = label.querySelector('input');
    const val = label.querySelector('b');
    input.addEventListener('input', () => {
      val.textContent = (+input.value).toFixed(2);
      if (state.rig) state.rig.manual[JIDX[def.joint] * 3 + def.axis] = +input.value;
    });
    group.appendChild(label);
    sliderInputs.push({ def, input, label, val });
  }
}

function refreshJointAvailability() {
  for (const { def, input, label } of sliderInputs) {
    const ok = !state.avatar || state.avatar.present[JIDX[def.joint]];
    label.classList.toggle('disabled', !ok);
    input.disabled = !ok;
  }
}

function resetPose() {
  state.rig?.manual.fill(0);
  state.rig?.smooth.fill(0);
  for (const { input, val } of sliderInputs) { input.value = 0; val.textContent = '0.00'; }
}

// --------------------------------------------------------------- intake ----
async function bitmapFromFile(file) {
  const raw = await createImageBitmap(file, { imageOrientation: 'from-image' }).catch(
    () => createImageBitmap(file),
  );
  return downscale(raw);
}

async function bitmapFromUrl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const blob = await res.blob();
  return downscale(await createImageBitmap(blob));
}

async function downscale(bmp) {
  const MAXD = 1024;
  const m = Math.max(bmp.width, bmp.height);
  if (m <= MAXD) return bmp;
  const s = MAXD / m;
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return createImageBitmap(c);
}

function loadPhoto(bitmap, name) {
  if (state.bitmap && state.bitmap !== bitmap) state.bitmap.close?.();
  state.bitmap = bitmap;
  state.photoName = name;
  const c = document.createElement('canvas');
  c.width = bitmap.width; c.height = bitmap.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  state.image = {
    w: c.width, h: c.height,
    data: ctx.getImageData(0, 0, c.width, c.height).data,
  };
  state.mapping = makeMapping(c.width, c.height);
  drawRefCanvas();
}

function drawRefCanvas() {
  const cv = $('refCanvas');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!state.bitmap) return;
  const s = Math.min(cv.width / state.bitmap.width, cv.height / state.bitmap.height);
  const w = state.bitmap.width * s, h = state.bitmap.height * s;
  ctx.drawImage(state.bitmap, (cv.width - w) / 2, (cv.height - h) / 2, w, h);
}

// -------------------------------------------------------------- pipeline ---
function maskBbox(mask, imgW, imgH) {
  let x0 = imgW, y0 = imgH, x1 = 0, y1 = 0;
  for (let y = 0; y < mask.h; y++) {
    for (let x = 0; x < mask.w; x++) {
      if (mask.data[y * mask.w + x] > 128) {
        const ix = (x / mask.w) * imgW, iy = (y / mask.h) * imgH;
        if (ix < x0) x0 = ix; if (iy < y0) y0 = iy;
        if (ix > x1) x1 = ix; if (iy > y1) y1 = iy;
      }
    }
  }
  if (x1 <= x0) return null;
  const pad = 0.06 * Math.max(x1 - x0, y1 - y0);
  return [Math.max(0, x0 - pad), Math.max(0, y0 - pad),
    Math.min(imgW, x1 + pad), Math.min(imgH, y1 + pad)];
}

function defaultBbox(imgW, imgH) {
  return [imgW * 0.22, imgH * 0.03, imgW * 0.78, imgH * 0.97];
}

async function runPipeline() {
  if (state.busy || !state.bitmap) return;
  state.busy = true;
  setStepper(2);
  // Snapshot the photo: if the user picks another image mid-flight, the stale
  // result is discarded and the pipeline restarts for the newest photo.
  const ticket = {
    bitmap: state.bitmap, mapping: state.mapping,
    image: state.image, name: state.photoName,
  };
  const isStale = () => state.bitmap !== ticket.bitmap;
  try {
    setStatus('Analyzing photo (pose + silhouette)…', 'busy');
    const { landmarks, mask, method } = await detectPerson(
      ticket.bitmap, (s) => setStatus(s[0].toUpperCase() + s.slice(1) + '…', 'busy'),
    );
    if (isStale()) { state.busy = false; runPipeline(); return; }
    state.landmarks = landmarks;
    state.mask = mask;
    state.method = method;

    let joints, note;
    if (landmarks) {
      joints = landmarksToJoints(landmarks, ticket.mapping);
      note = 'pose + silhouette';
    } else if (mask) {
      const bb = maskBbox(mask, ticket.image.w, ticket.image.h)
        || defaultBbox(ticket.image.w, ticket.image.h);
      joints = heuristicJoints(bb, ticket.mapping);
      note = 'silhouette-only fallback';
    } else {
      // Offline: deterministic background-subtraction mask + neutral stance.
      let fg = null;
      try { fg = foregroundMask(ticket.image); } catch { fg = null; }
      state.mask = fg;
      const bb = (fg && maskBbox(fg, ticket.image.w, ticket.image.h))
        || defaultBbox(ticket.image.w, ticket.image.h);
      joints = heuristicJoints(bb, ticket.mapping);
      note = fg ? 'offline bg-subtraction + neutral stance' : 'offline procedural fallback';
    }
    setStatus(`Building particle character (${note})…`, 'busy');
    await new Promise((r) => setTimeout(r, 30)); // let the veil paint
    if (isStale()) { state.busy = false; runPipeline(); return; }
    buildCharacter(joints, 'photo', method);
    setStatus(`Ready — ${ticket.name}`, '');
    setStepper(3);
    if (!landmarks) {
      toast(mask
        ? 'No pose detected — built from the silhouette with a neutral stance instead.'
        : 'ML models unreachable (offline?) — built a procedural body from photo colors instead.');
    }
  } catch (err) {
    console.error(err);
    setStatus(`Failed: ${err.message}`, 'error');
    toast(`Could not build character: ${err.message}`);
  } finally {
    state.busy = false;
  }
}

function buildCharacter(joints, palette, method) {
  const t0 = performance.now();
  const avatar = buildAvatar({
    joints,
    mapping: state.mapping,
    mask: state.mask,
    image: state.image,
    palette,
    maxParticles: MAX_PARTICLES,
    seed: state.seed,
  });
  state.avatar = avatar;
  state.rig = createRig(avatar);
  state.rig.depthScale = +$('depth').value;
  state.method = method;
  resetPose();
  refreshJointAvailability();
  viewer.setAvatar(avatar, state.rig);
  viewer.setDensity(drawnCount());
  updateBadges(performance.now() - t0);
  drawCutCanvas();
  $('seedLabel').textContent = `seed ${state.seed}`;
}

function showDemo() {
  state.bitmap = null; state.image = null; state.mapping = null;
  state.landmarks = null; state.mask = null;
  state.photoName = '';
  setStatus('Ready — hologram demo', '');
  setStepper(3);
  const t0 = performance.now();
  state.avatar = buildAvatar({ joints: mannequinJoints(), palette: 'hologram', seed: state.seed });
  state.rig = createRig(state.avatar);
  state.rig.depthScale = +$('depth').value;
  state.method = 'mannequin';
  resetPose();
  refreshJointAvailability();
  viewer.setAvatar(state.avatar, state.rig);
  viewer.setDensity(drawnCount());
  updateBadges(performance.now() - t0);
  const rc = $('refCanvas').getContext('2d');
  rc.clearRect(0, 0, 240, 300);
  const cc = $('cutCanvas').getContext('2d');
  cc.clearRect(0, 0, 240, 300);
  cc.fillStyle = '#8b95a9';
  cc.font = '12px sans-serif';
  cc.textAlign = 'center';
  cc.fillText('No photo —', 120, 140);
  cc.fillText('procedural mannequin', 120, 156);
  $('seedLabel').textContent = `seed ${state.seed}`;
}

// ------------------------------------------------------------ cutout view --
function drawCutCanvas() {
  const cv = $('cutCanvas');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!state.bitmap || !state.mapping) return;
  const s = Math.min(cv.width / state.bitmap.width, cv.height / state.bitmap.height);
  const w = state.bitmap.width * s, h = state.bitmap.height * s;
  const ox = (cv.width - w) / 2, oy = (cv.height - h) / 2;

  ctx.save();
  ctx.beginPath();
  ctx.rect(ox, oy, w, h);
  ctx.clip();
  ctx.drawImage(state.bitmap, ox, oy, w, h);
  if (state.mask) {
    // punch the background out with the person mask
    const mc = document.createElement('canvas');
    mc.width = state.mask.w; mc.height = state.mask.h;
    const mctx = mc.getContext('2d');
    const id = mctx.createImageData(mc.width, mc.height);
    for (let i = 0; i < state.mask.w * state.mask.h; i++) {
      id.data[i * 4] = 255; id.data[i * 4 + 1] = 255; id.data[i * 4 + 2] = 255;
      id.data[i * 4 + 3] = state.mask.data[i];
    }
    mctx.putImageData(id, 0, 0);
    ctx.globalCompositeOperation = 'destination-in';
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(mc, ox, oy, w, h);
    ctx.globalCompositeOperation = 'source-over';
  } else {
    ctx.fillStyle = 'rgba(251,191,36,0.10)';
    ctx.fillRect(ox, oy, w, h);
  }
  ctx.restore();

  // rig skeleton overlay (projected back to 2D)
  if (state.avatar && state.mapping) {
    const m = state.mapping;
    const px = (wx, wy) => {
      const [u, v] = m.toPixel(wx, wy);
      return [ox + (u / m.imgW) * w, oy + (v / m.imgH) * h];
    };
    const J = state.avatar.base;
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = 'rgba(45,212,191,0.95)';
    ctx.fillStyle = '#fef08a';
    for (const bn of state.avatar.bones) {
      if (!bn.enabled) continue;
      const [x1, y1] = px(J[bn.ja * 3], J[bn.ja * 3 + 1]);
      const [x2, y2] = px(J[bn.jb * 3], J[bn.jb * 3 + 1]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    for (let i = 0; i < state.avatar.joints.length; i++) {
      if (!state.avatar.present[i]) continue;
      const [x, y] = px(J[i * 3], J[i * 3 + 1]);
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 7); ctx.fill();
    }
  }
}

function updateBadges(buildMs) {
  const host = $('detectInfo');
  host.innerHTML = '';
  const add = (txt, cls = '') => {
    const s = document.createElement('span');
    s.className = `badge ${cls}`;
    s.textContent = txt;
    host.appendChild(s);
  };
  const methodLabel = {
    pose: 'pose + mask ✓', 'pose-mask': 'mask only', selfie: 'mask only',
    none: state.mask ? 'bg-subtraction fallback' : 'procedural fallback',
    mannequin: 'procedural mannequin',
  }[state.method] || state.method;
  add(methodLabel, state.method === 'pose' ? 'ok' : state.method === 'mannequin' ? '' : 'warn');
  if (state.avatar) {
    const n = state.avatar.present.filter(Boolean).length;
    add(`joints ${n}/28`, n >= 24 ? 'ok' : 'warn');
    const nb = state.avatar.bones.filter((b) => b.enabled).length;
    add(`parts ${nb}/${state.avatar.bones.length}`, nb >= 18 ? 'ok' : 'warn');
  }
  if (state.mask) add(`silhouette ${(state.mask.coverage * 100).toFixed(0)}%`);
  if (buildMs != null) add(`built in ${buildMs.toFixed(0)} ms`);
  add(`${MAX_PARTICLES.toLocaleString()} particles`);
}

// --------------------------------------------------------------- main loop -
let lastFrame = performance.now();
function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min((now - lastFrame) / 1000, 0.05);
  lastFrame = now;
  if (state.playing) state.time += dt;
  if (state.avatar && state.rig) {
    computeAnim(state.anim, state.preset, state.time, state.amp, state.avatar.restPhi, state.playing ? dt : 0);
    updateRig(state.rig, dt, state.anim);
  }
  viewer.sync(state.time);
}

// ------------------------------------------------------------------ events -
function bindUI() {
  // upload
  const dz = $('dropzone'), fi = $('fileInput');
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter') fi.click(); });
  fi.addEventListener('change', async () => {
    if (fi.files?.[0]) {
      try {
        loadPhoto(await bitmapFromFile(fi.files[0]), fi.files[0].name);
        runPipeline();
      } catch (err) { toast(`Could not read image: ${err.message}`); }
      fi.value = '';
    }
  });
  for (const ev of ['dragover', 'dragenter']) {
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('over'); });
  }
  for (const ev of ['dragleave', 'drop']) {
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('over'); });
  }
  dz.addEventListener('drop', async (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    if (!f.type.startsWith('image/')) return toast('Please drop an image file.');
    try {
      loadPhoto(await bitmapFromFile(f), f.name);
      runPipeline();
    } catch (err) { toast(`Could not read image: ${err.message}`); }
  });

  // samples + demo
  document.querySelectorAll('[data-sample]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      try {
        setStatus('Loading sample…', 'busy');
        loadPhoto(await bitmapFromUrl(BASE + btn.dataset.sample), btn.dataset.sample.split('/').pop());
        runPipeline();
      } catch (err) {
        setStatus('Failed to load sample', 'error');
        toast(`Could not load sample: ${err.message}`);
      }
    });
  });
  $('demoBtn').addEventListener('click', showDemo);

  // seed / rebuild
  $('seedBtn').addEventListener('click', () => {
    state.seed = 1 + Math.floor(Math.random() * 99999);
    $('seedLabel').textContent = `seed ${state.seed}`;
    rebuildFromCache();
  });
  $('rebuildBtn').addEventListener('click', rebuildFromCache);

  // view
  $('rotToggle').addEventListener('change', (e) => {
    viewer.setAutoRotate(e.target.checked, +$('rotSpeed').value);
  });
  $('rotSpeed').addEventListener('input', (e) => {
    $('rotSpeedVal').textContent = `${(+e.target.value).toFixed(1)}×`;
    viewer.setAutoRotate($('rotToggle').checked, +e.target.value);
  });
  $('skelToggle').addEventListener('change', (e) => viewer.setSkeletonVisible(e.target.checked));
  $('gridToggle').addEventListener('change', (e) => viewer.setGridVisible(e.target.checked));
  $('frameBtn').addEventListener('click', () => viewer.frameCamera(false));
  $('camBtn').addEventListener('click', () => viewer.frameCamera(true));

  // particles
  $('density').addEventListener('input', (e) => {
    $('densityVal').textContent = (+e.target.value).toLocaleString();
    viewer.setDensity(+e.target.value);
  });
  $('psize').addEventListener('input', (e) => {
    $('sizeVal').textContent = `${(+e.target.value).toFixed(2)}×`;
    viewer.setSize(+e.target.value);
  });
  $('scatter').addEventListener('input', (e) => {
    $('scatterVal').textContent = (+e.target.value).toFixed(2);
    viewer.setScatter(+e.target.value);
  });
  $('depth').addEventListener('input', (e) => {
    $('depthVal').textContent = `${(+e.target.value).toFixed(2)}×`;
    if (state.rig) state.rig.depthScale = +e.target.value;
  });
  $('colorSeg').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    document.querySelectorAll('#colorSeg button').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    viewer.setColorMode(+b.dataset.mode);
  });

  // animation
  $('presetRow').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    document.querySelectorAll('#presetRow button').forEach((x) => x.classList.remove('active'));
    b.classList.add('active');
    state.preset = b.dataset.preset;
    if (state.preset !== 'none' && !state.playing) togglePlay();
  });
  $('amp').addEventListener('input', (e) => {
    state.amp = +e.target.value;
    $('ampVal').textContent = `${state.amp.toFixed(2)}×`;
  });
  $('playBtn').addEventListener('click', togglePlay);
  $('poseResetBtn').addEventListener('click', resetPose);

  // top actions + modal
  $('snapshotBtn').addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = viewer.snapshot();
    a.download = `particle-person-${Date.now()}.png`;
    a.click();
  });
  const modal = $('howModal');
  $('howBtn').addEventListener('click', () => modal.classList.remove('hidden'));
  $('howBtn2').addEventListener('click', () => modal.classList.remove('hidden'));
  $('howClose').addEventListener('click', () => modal.classList.add('hidden'));
  modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });
  window.addEventListener('keydown', (e) => { if (e.key === 'Escape') modal.classList.add('hidden'); });
}

function togglePlay() {
  state.playing = !state.playing;
  $('playBtn').textContent = state.playing ? '⏸ Pause motion' : '▶ Resume motion';
}

function rebuildFromCache() {
  if (!state.bitmap || !state.mapping) {
    showDemo();
    return;
  }
  try {
    let joints;
    if (state.landmarks) joints = landmarksToJoints(state.landmarks, state.mapping);
    else if (state.mask) {
      joints = heuristicJoints(
        maskBbox(state.mask, state.image.w, state.image.h) || defaultBbox(state.image.w, state.image.h),
        state.mapping,
      );
    } else {
      joints = heuristicJoints(defaultBbox(state.image.w, state.image.h), state.mapping);
    }
    buildCharacter(joints, 'photo', state.method);
  } catch (err) {
    toast(`Rebuild failed: ${err.message}`);
  }
}

// ------------------------------------------------------------------ boot ---
buildJointSliders();
bindUI();
showDemo();
if (!window.WebGL2RenderingContext && !window.WebGLRenderingContext) {
  toast('WebGL is unavailable in this browser — the 3D view cannot render.');
}
requestAnimationFrame(loop);
