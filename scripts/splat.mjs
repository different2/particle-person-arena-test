/**
 * Minimal perspective splat renderer for offline geometry previews.
 * Returns a pngjs PNG. Not used by the app — verification tooling only.
 */
import { PNG } from 'pngjs';
import { updateRig, computeAnim } from '../src/body.js';

export function renderView(avatar, rig, animState, dir, size = 420) {
  computeAnim(animState, 'none', 0, 0, avatar.restPhi, 0);
  updateRig(rig, 1 / 60, animState);
  updateRig(rig, 1 / 60, animState);
  const pos = rig.avatar.particles.positions;
  const col = rig.avatar.particles.color;
  const N = rig.avatar.particles.count;
  const [cx, cy, cz] = avatar.bounds.center;
  const dist = avatar.bounds.radius * 2.6 + 0.2;
  const dl = Math.hypot(...dir);
  const d = dir.map((v) => v / dl);
  const cam = [cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist];
  let zx = cam[0] - cx, zy = cam[1] - cy, zz = cam[2] - cz;
  const zl = Math.hypot(zx, zy, zz); zx /= zl; zy /= zl; zz /= zl;
  let xx = zz, xy = 0, xz = -zx;
  const xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  const tan = Math.tan((21 * Math.PI) / 180);
  const proj = [];
  for (let i = 0; i < N; i += 2) {
    const vx = pos[i * 3] - cam[0], vy = pos[i * 3 + 1] - cam[1], vz = pos[i * 3 + 2] - cam[2];
    const px = vx * xx + vy * xy + vz * xz;
    const py = vx * yx + vy * yy + vz * yz;
    const pz = -(vx * zx + vy * zy + vz * zz);
    if (pz < 0.05) continue;
    proj.push([pz, (px / (pz * tan)) * size * 0.5 + size / 2,
      size / 2 - (py / (pz * tan)) * size * 0.5, i]);
  }
  proj.sort((a, b) => b[0] - a[0]);
  const png = new PNG({ width: size, height: size });
  for (let i = 0; i < size * size; i++) {
    png.data[i * 4] = 11; png.data[i * 4 + 1] = 14; png.data[i * 4 + 2] = 20; png.data[i * 4 + 3] = 255;
  }
  for (const [pz, sx, sy, i] of proj) {
    const r = Math.max(1, Math.min(3, (22 / pz) * (size / 420)));
    const cr = Math.max(0, Math.min(255, col[i * 3] * 255));
    const cg = Math.max(0, Math.min(255, col[i * 3 + 1] * 255));
    const cb = Math.max(0, Math.min(255, col[i * 3 + 2] * 255));
    for (let oy = -r; oy <= r; oy++) {
      for (let ox = -r; ox <= r; ox++) {
        if (ox * ox + oy * oy > r * r) continue;
        const X = Math.round(sx + ox), Y = Math.round(sy + oy);
        if (X < 0 || Y < 0 || X >= size || Y >= size) continue;
        const o = (Y * size + X) * 4;
        png.data[o] = cr; png.data[o + 1] = cg; png.data[o + 2] = cb; png.data[o + 3] = 255;
      }
    }
  }
  return png;
}
