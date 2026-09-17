/**
 * viewer.js — Three.js stage: particle points, rig skeleton, floor, camera.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const BG = 0x0b0e14;

const POINTS_VERT = /* glsl */`
attribute vec3 aColor;
attribute float aSize;
attribute float aPart;
attribute vec3 aScatter;
uniform float uSize;
uniform float uPixelRatio;
uniform float uScatter;
uniform float uTime;
varying vec3 vColor;
varying float vPart;
varying float vViewZ;
void main() {
  vColor = aColor;
  vPart = aPart;
  vec3 p = position + aScatter * uScatter;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vViewZ = -mv.z;
  float tw = 1.0 + 0.07 * sin(uTime * 2.2 + position.x * 43.0 + position.y * 29.0 + position.z * 37.0);
  gl_PointSize = uSize * aSize * tw * uPixelRatio * (120.0 / max(vViewZ, 0.001));
  gl_Position = projectionMatrix * mv;
}
`;

const POINTS_FRAG = /* glsl */`
precision mediump float;
varying vec3 vColor;
varying float vPart;
varying float vViewZ;
uniform int uColorMode; // 0 original · 1 vivid · 2 depth · 3 parts
uniform float uZCenter;
uniform float uZRange;
uniform float uOpacity;

vec3 heat(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c1 = vec3(0.16, 0.05, 0.35);
  vec3 c2 = vec3(0.85, 0.15, 0.45);
  vec3 c3 = vec3(1.00, 0.85, 0.25);
  return t < 0.5 ? mix(c1, c2, t * 2.0) : mix(c2, c3, t * 2.0 - 1.0);
}
vec3 partColor(float id) {
  if (id < 0.5) return vec3(0.30, 0.80, 1.00);
  if (id < 1.5) return vec3(1.00, 0.75, 0.30);
  if (id < 2.5) return vec3(0.45, 1.00, 0.55);
  if (id < 3.5) return vec3(0.70, 1.00, 0.45);
  if (id < 4.5) return vec3(0.55, 0.60, 1.00);
  if (id < 5.5) return vec3(0.75, 0.65, 1.00);
  if (id < 6.5) return vec3(1.00, 0.45, 0.75);
  if (id < 7.5) return vec3(1.00, 0.60, 0.60);
  if (id < 8.5) return vec3(0.45, 1.00, 0.85);
  return vec3(0.65, 1.00, 0.70);
}
void main() {
  vec2 q = gl_PointCoord - 0.5;
  float d = length(q);
  float alpha = smoothstep(0.5, 0.30, d);
  if (alpha < 0.03) discard;
  vec3 c = vColor;
  if (uColorMode == 1) {
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(luma), c, 1.9) * 1.05;
  } else if (uColorMode == 2) {
    float t = 0.5 + (uZCenter - vViewZ) / max(uZRange, 1e-4);
    c = heat(t);
  } else if (uColorMode == 3) {
    c = partColor(vPart) * (0.55 + 0.45 * clamp(dot(normalize(vColor + 0.15), vec3(0.577)), 0.0, 1.0));
  }
  c *= 1.0 - 0.30 * (d * 2.0); // soft spherical shading
  gl_FragColor = vec4(c, alpha * uOpacity);
}
`;

function makeShadowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 4, 64, 64, 62);
  g.addColorStop(0, 'rgba(0,0,0,0.55)');
  g.addColorStop(0.6, 'rgba(0,0,0,0.28)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  return tex;
}

export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setClearColor(BG, 1);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(BG);
  scene.fog = new THREE.Fog(BG, 6, 16);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.position.set(0.6, 1.2, 3.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.3;
  controls.maxDistance = 20;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 1.6;

  // floor grid + shadow blob (repositioned per avatar)
  const grid = new THREE.GridHelper(8, 40, 0x2dd4bf, 0x1e293b);
  grid.material.transparent = true;
  grid.material.opacity = 0.35;
  scene.add(grid);
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({ map: makeShadowTexture(), transparent: true, depthWrite: false }),
  );
  shadow.rotation.x = -Math.PI / 2;
  scene.add(shadow);

  let points = null;
  let skeleton = null;
  let jointDots = null;
  let avatar = null;
  let rig = null;
  let bounds = { center: [0, 1, 0], radius: 1 };

  const params = {
    size: 1.0, scatter: 0, colorMode: 0, opacity: 1,
    skeletonVisible: false, gridVisible: true,
  };

  function setAvatar(av, rg) {
    avatar = av; rig = rg;
    bounds = av.bounds;
    if (points) {
      scene.remove(points);
      points.geometry.dispose();
      points.material.dispose();
    }
    if (skeleton) { scene.remove(skeleton); skeleton.geometry.dispose(); skeleton.material.dispose(); }
    if (jointDots) { scene.remove(jointDots); jointDots.geometry.dispose(); jointDots.material.dispose(); }

    const P = av.particles;
    const geo = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(P.positions, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', posAttr);
    geo.setAttribute('aColor', new THREE.BufferAttribute(P.color, 3));
    geo.setAttribute('aSize', new THREE.BufferAttribute(P.size, 1));
    geo.setAttribute('aPart', new THREE.BufferAttribute(new Float32Array(P.part), 1));
    geo.setAttribute('aScatter', new THREE.BufferAttribute(P.scatter, 3));
    const mat = new THREE.ShaderMaterial({
      vertexShader: POINTS_VERT,
      fragmentShader: POINTS_FRAG,
      transparent: true,
      depthWrite: true,
      depthTest: true,
      uniforms: {
        uSize: { value: 0.046 * params.size },
        uPixelRatio: { value: renderer.getPixelRatio() },
        uScatter: { value: params.scatter },
        uTime: { value: 0 },
        uColorMode: { value: params.colorMode },
        uZCenter: { value: 3 },
        uZRange: { value: 2 },
        uOpacity: { value: params.opacity },
      },
    });
    points = new THREE.Points(geo, mat);
    points.frustumCulled = false;
    points.renderOrder = 2;
    scene.add(points);
    geo.setDrawRange(0, P.count);

    // skeleton overlay
    const links = [];
    av.joints.forEach((jn, i) => {
      if (jn.parent >= 0 && jn.present && av.joints[jn.parent].present) links.push([jn.parent, i]);
    });
    const skGeo = new THREE.BufferGeometry();
    skGeo.setDrawRange(0, links.length * 2);
    const skPos = new Float32Array(Math.max(links.length, 1) * 6);
    skGeo.setAttribute('position', new THREE.BufferAttribute(skPos, 3).setUsage(THREE.DynamicDrawUsage));
    skeleton = new THREE.LineSegments(
      skGeo,
      new THREE.LineBasicMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.9 }),
    );
    skeleton.frustumCulled = false;
    skeleton.visible = params.skeletonVisible;
    skeleton.userData.links = links;
    scene.add(skeleton);

    const jdGeo = new THREE.BufferGeometry();
    const jdPos = new Float32Array(av.joints.length * 3);
    jdGeo.setAttribute('position', new THREE.BufferAttribute(jdPos, 3).setUsage(THREE.DynamicDrawUsage));
    jointDots = new THREE.Points(jdGeo, new THREE.PointsMaterial({
      color: 0xfef08a, size: 0.022, sizeAttenuation: true, transparent: true, opacity: 0.95,
      depthTest: false,
    }));
    jointDots.frustumCulled = false;
    jointDots.visible = params.skeletonVisible;
    jointDots.renderOrder = 3;
    scene.add(jointDots);

    // floor + shadow follow the avatar
    const floorY = av.bounds.min[1] - 0.015;
    grid.position.set(av.bounds.center[0] * 0.5, floorY, av.bounds.center[2] * 0.5);
    shadow.position.set(av.bounds.center[0], floorY + 0.002, av.bounds.center[2]);
    const sw = Math.max(av.bounds.max[0] - av.bounds.min[0], 0.4) * 0.85;
    shadow.scale.set(sw, sw * 0.8, 1);

    frameCamera(false);
  }

  function frameCamera(resetDirection = true) {
    const c = new THREE.Vector3(...bounds.center);
    const dist = bounds.radius * 2.9 + 0.25;
    let dir;
    if (resetDirection) {
      dir = new THREE.Vector3(0.38, 0.22, 1).normalize();
    } else {
      dir = camera.position.clone().sub(controls.target);
      if (dir.lengthSq() < 1e-6) dir.set(0.38, 0.22, 1);
      dir.normalize();
    }
    controls.target.copy(c);
    camera.position.copy(c).addScaledVector(dir, dist);
    controls.update();
  }

  function setDensity(count) {
    if (!points || !avatar) return;
    points.geometry.setDrawRange(0, Math.max(100, Math.min(avatar.particles.count, Math.round(count))));
  }
  function setSize(v) {
    params.size = v;
    if (points) points.material.uniforms.uSize.value = 0.046 * v;
  }
  function setScatter(v) {
    params.scatter = v;
    if (points) points.material.uniforms.uScatter.value = v;
  }
  function setColorMode(m) {
    params.colorMode = m;
    if (points) points.material.uniforms.uColorMode.value = m;
  }
  function setSkeletonVisible(v) {
    params.skeletonVisible = v;
    if (skeleton) skeleton.visible = v;
    if (jointDots) jointDots.visible = v;
  }
  function setGridVisible(v) {
    params.gridVisible = v;
    grid.visible = v;
  }
  function setAutoRotate(on, speed = 1.6) {
    controls.autoRotate = on;
    controls.autoRotateSpeed = speed;
  }

  function snapshot() {
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png');
  }

  function resize() {
    const w = container.clientWidth || 2;
    const h = container.clientHeight || 2;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (points) points.material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  }
  new ResizeObserver(resize).observe(container);
  resize();

  let lastT = performance.now();
  let fps = 60;
  let onFps = null;

  /** Per-frame sync: particle positions + skeleton + shader uniforms. */
  function sync(time) {
    if (points && avatar && rig) {
      points.geometry.attributes.position.needsUpdate = true;
      const u = points.material.uniforms;
      u.uTime.value = time;
      const camC = new THREE.Vector3(...bounds.center);
      u.uZCenter.value = camera.position.distanceTo(camC);
      u.uZRange.value = Math.max(bounds.radius * 2.2, 0.2);
      if (skeleton.visible) {
        const arr = skeleton.geometry.attributes.position.array;
        const links = skeleton.userData.links;
        for (let i = 0; i < links.length; i++) {
          const [a, b] = links[i];
          arr[i * 6] = rig.jointPos[a * 3];
          arr[i * 6 + 1] = rig.jointPos[a * 3 + 1];
          arr[i * 6 + 2] = rig.jointPos[a * 3 + 2];
          arr[i * 6 + 3] = rig.jointPos[b * 3];
          arr[i * 6 + 4] = rig.jointPos[b * 3 + 1];
          arr[i * 6 + 5] = rig.jointPos[b * 3 + 2];
        }
        skeleton.geometry.attributes.position.needsUpdate = true;
        const jd = jointDots.geometry.attributes.position.array;
        jd.set(rig.jointPos);
        for (let i = 0; i < avatar.joints.length; i++) {
          const jn = avatar.joints[i];
          if (!jn.present && jn.parent >= 0) {
            jd[i * 3] = jd[jn.parent * 3];
            jd[i * 3 + 1] = jd[jn.parent * 3 + 1];
            jd[i * 3 + 2] = jd[jn.parent * 3 + 2];
          }
        }
        jointDots.geometry.attributes.position.needsUpdate = true;
      }
    }
    controls.update();
    renderer.render(scene, camera);
    const now = performance.now();
    const dt = Math.max(now - lastT, 0.01);
    lastT = now;
    fps += (1000 / dt - fps) * 0.05;
    onFps?.(fps);
  }

  return {
    setAvatar, frameCamera, setDensity, setSize, setScatter, setColorMode,
    setSkeletonVisible, setGridVisible, setAutoRotate, snapshot, resize, sync,
    setOnFps(fn) { onFps = fn; },
    get container() { return container; },
  };
}
