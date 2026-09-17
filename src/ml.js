/**
 * ml.js — browser-only perception layer (MediaPipe, loaded from CDN on demand).
 *
 * Uses the classic MediaPipe Pose solution with `enableSegmentation`, which gives
 * both the 33-joint skeleton and a person segmentation mask in one pass.
 * Falls back to the Selfie Segmentation model when no pose is found.
 * Everything degrades gracefully to the procedural pipeline when offline.
 */

const POSE_VERSION = '0.5.1675469240';
const SELFIE_VERSION = '0.1.1675465747';
const POSE_SCRIPT = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/pose.js`;
const POSE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${POSE_VERSION}/`;
const SELFIE_SCRIPT = `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@${SELFIE_VERSION}/selfie_segmentation.js`;
const SELFIE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation@${SELFIE_VERSION}/`;

const loadedScripts = new Map();

function loadScript(src, globalName, timeoutMs = 45000) {
  if (loadedScripts.has(src)) return loadedScripts.get(src);
  const p = new Promise((resolve, reject) => {
    if (globalName && window[globalName]) return resolve();
    const el = document.createElement('script');
    el.src = src;
    el.crossOrigin = 'anonymous';
    const timer = setTimeout(() => {
      el.remove();
      reject(new Error(`Timed out loading ${src}`));
    }, timeoutMs);
    el.onload = () => { clearTimeout(timer); resolve(); };
    el.onerror = () => { clearTimeout(timer); reject(new Error(`Failed to load ${src}`)); };
    document.head.appendChild(el);
  });
  loadedScripts.set(src, p);
  return p;
}

let poseInstance = null;
let poseBusy = Promise.resolve();

export async function ensurePose(onStatus) {
  if (poseInstance) return poseInstance;
  onStatus?.('loading pose model…');
  await loadScript(POSE_SCRIPT, 'Pose');
  const pose = new window.Pose({ locateFile: (f) => POSE_BASE + f });
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: false,
    enableSegmentation: true,
    smoothSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  if (typeof pose.initialize === 'function') {
    try { await pose.initialize(); } catch { /* older builds lack initialize() */ }
  }
  poseInstance = pose;
  return pose;
}

let selfieInstance = null;

export async function ensureSelfie(onStatus) {
  if (selfieInstance) return selfieInstance;
  onStatus?.('loading segmentation model…');
  await loadScript(SELFIE_SCRIPT, 'SelfieSegmentation');
  const seg = new window.SelfieSegmentation({ locateFile: (f) => SELFIE_BASE + f });
  seg.setOptions({ modelSelection: 1 });
  if (typeof seg.initialize === 'function') {
    try { await seg.initialize(); } catch { /* ignore */ }
  }
  selfieInstance = seg;
  return seg;
}

/** Serialize sends through one instance (MediaPipe solutions dislike overlap). */
function sendOnce(instance, image, timeoutMs = 90000) {
  const run = async () => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Inference timed out')), timeoutMs);
    instance.onResults((results) => {
      clearTimeout(timer);
      resolve(results);
    });
    instance.send({ image }).catch((err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
  const chained = poseBusy.then(run, run);
  poseBusy = chained.catch(() => {});
  return chained;
}

/**
 * Extract a person-probability mask {w,h,data:Uint8Array} from a MediaPipe
 * segmentationMask CanvasImageSource. Capped at `maxDim` for speed.
 */
export function maskFromSource(source, imgW, imgH, maxDim = 320) {
  if (!source) return null;
  const scale = Math.min(1, maxDim / Math.max(imgW, imgH));
  const w = Math.max(8, Math.round(imgW * scale));
  const h = Math.max(8, Math.round(imgH * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const data = new Uint8Array(w * h);
  let person = 0;
  for (let i = 0; i < w * h; i++) {
    const v = (px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2]) / 3;
    data[i] = v;
    if (v > 128) person++;
  }
  return { w, h, data, coverage: person / (w * h) };
}

/**
 * Full perception pass on an ImageBitmap/HTMLImageElement.
 * @returns { landmarks|null, mask|null, method: 'pose'|'selfie'|'none' }
 */
export async function detectPerson(image, onStatus) {
  const W = image.width ?? image.videoWidth ?? image.naturalWidth;
  const H = image.height ?? image.videoHeight ?? image.naturalHeight;
  // Normalize to a canvas: every MediaPipe JS solution accepts canvas input,
  // while ImageBitmap support varies across solution versions.
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  canvas.getContext('2d').drawImage(image, 0, 0, W, H);
  // 1) Pose + segmentation in one shot.
  try {
    const pose = await ensurePose(onStatus);
    onStatus?.('detecting pose & silhouette…');
    const res = await sendOnce(pose, canvas);
    const landmarks = res?.poseLandmarks?.length === 33 ? res.poseLandmarks : null;
    const mask = maskFromSource(res?.segmentationMask, W, H);
    const maskOk = mask && mask.coverage > 0.02 && mask.coverage < 0.95;
    if (landmarks) return { landmarks, mask: maskOk ? mask : null, method: 'pose' };
    if (maskOk) return { landmarks: null, mask, method: 'pose-mask' };
  } catch (err) {
    console.warn('[ml] pose pass failed:', err);
  }
  // 2) Dedicated segmentation fallback.
  try {
    const seg = await ensureSelfie(onStatus);
    onStatus?.('segmenting person…');
    const res = await sendOnce(seg, image);
    const mask = maskFromSource(res?.segmentationMask, W, H);
    if (mask && mask.coverage > 0.02 && mask.coverage < 0.95) {
      return { landmarks: null, mask, method: 'selfie' };
    }
  } catch (err) {
    console.warn('[ml] selfie pass failed:', err);
  }
  return { landmarks: null, mask: null, method: 'none' };
}
