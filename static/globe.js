// 3D globe view, Minecraft style (WebGL, three.js). Layer 1: a planet built from cubes; every content zone is a biome with a
// hard border, trees and a beacon beam. Click a biome -> layer 2: a floating island with that zone's points as blocks above it,
// grouped in sub-zones (one block colour per sub-zone). Click a block -> its text. Search results light up their zones / blocks.
// Needs static/three.module.min.js (one file, see README); no other add-ons.
import * as THREE from '/static/three.module.min.js';

const MAX_ZONES = 12;  // must match content_zones() in app.py
const BIOMES = [
  { top: '#5da83c', leaf: '#2f7d2a', trunk: '#6b4a2b', trees: true },    // grass
  { top: '#e3d08a' },                                                       // sand
  { top: '#f2f6ff', leaf: '#2d5a3d', trunk: '#5a3f24', trees: true },     // snow
  { top: '#8a8a8a' },                                                       // stone
  { top: '#7b5d8e' },                                                       // mycelium
  { top: '#b13a3a' },                                                       // nether
  { top: '#4a86e8' },                                                       // ice / water
  { top: '#f5c542' },                                                       // gold
  { top: '#f29bc4', leaf: '#f6b5d3', trunk: '#5a3a3a', trees: true },     // cherry
  { top: '#1faa59', leaf: '#17893f', trunk: '#6b4a2b', trees: true },     // jungle
  { top: '#dcd89a' },                                                       // end stone
  { top: '#4a4360' },                                                       // deepslate
];
const biomeOf = i => BIOMES[((i % BIOMES.length) + BIOMES.length) % BIOMES.length];
const colorOf = i => new THREE.Color(biomeOf(i).top);
const RV = 22;  // planet radius in blocks

const CSS = `
.g-host { position: relative; height: min(78vh, 760px); min-height: 480px; margin-top: 14px; overflow: hidden; background: #0b1026; color: #fff;
  font-family: "VT323", var(--mono), var(--body); font-size: 1.1rem; user-select: none; border: 4px solid #000; box-shadow: inset 0 0 0 3px #555; }
.g-host canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none; image-rendering: pixelated; }
.g-panel { position: absolute; background: rgba(28,28,28,.86); padding: 8px 12px; border: 4px solid #000;
  box-shadow: inset 3px 3px 0 #5a5a5a, inset -3px -3px 0 #151515, 0 0 0 2px #000; text-shadow: 2px 0 #000, -2px 0 #000, 0 2px #000, 0 -2px #000, 2px 2px #000, -2px -2px #000, 2px -2px #000, -2px 2px #000; }
.g-crumb { left: 14px; top: 14px; display: flex; align-items: center; gap: 10px; max-width: 62%; }
.g-crumb button, .g-btn { background: #8b8b8b; color: #fff; border: 3px solid #000; padding: 2px 12px; cursor: pointer; font: inherit; text-shadow: 2px 2px 0 #3f3f3f;
  box-shadow: inset 3px 3px 0 #c6c6c6, inset -3px -3px 0 #555; }
.g-crumb button:hover, .g-btn:hover { background: #7a8bd0; box-shadow: inset 3px 3px 0 #a9b6f0, inset -3px -3px 0 #44508a; }
.g-crumb .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #ffff55; }
.g-stats { right: 14px; top: 14px; color: #aaa; text-align: right; }
.g-search { left: 14px; bottom: 14px; width: min(430px, calc(100% - 28px)); display: flex; flex-direction: column; gap: 8px; max-height: 52%; }
.g-search form { display: flex; gap: 8px; }
.g-search input { flex: 1; background: #000; border: 3px solid #a0a0a0; color: #fff; padding: 3px 9px; font: inherit; }
.g-search input:focus { outline: 3px solid #ffff55; }
.g-hits { overflow: auto; display: flex; flex-direction: column; gap: 3px; }
.g-hit { text-align: left; background: #3b3b3b; border: 2px solid #000; padding: 2px 8px; cursor: pointer; color: #ddd; font: inherit; font-size: 1rem; }
.g-hit:hover { background: #55a4ff55; border-color: #fff; }
.g-hit b { color: #ffaa00; margin-right: 6px; }
.g-legend { right: 14px; top: 62px; max-width: 300px; max-height: 46%; overflow: auto; display: flex; flex-direction: column; gap: 3px; }
.g-chip { display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 1px 6px; border: 2px solid transparent; font-size: 1rem; }
.g-chip:hover { background: #ffffff22; }
.g-chip.on { border-color: #ffff55; }
.g-chip i { width: 14px; height: 14px; flex: none; border: 2px solid #000; box-shadow: inset 2px 2px 0 rgba(255,255,255,.4), inset -2px -2px 0 rgba(0,0,0,.35); }
.g-chip span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.g-chip small { color: #aaa; }
.g-card { right: 14px; bottom: 14px; width: min(390px, calc(100% - 28px)); max-height: 46%; overflow: auto; font-size: 1rem; line-height: 1.25; }
.g-card h4 { margin: 0 0 4px; font: inherit; font-size: 1.2rem; color: #ffff55; }
.g-card small { color: #aaa; }
.g-card p { margin: 8px 0 0; color: #eee; white-space: pre-wrap; text-shadow: none; }
.g-label { position: absolute; transform: translate(-50%, -50%); color: #1b1206; text-shadow: none; padding: 0 9px; background: #b88a4a; border: 4px solid #000;
  box-shadow: inset 2px 2px 0 #d9b57a, inset -2px -2px 0 #8a6330, 0 0 0 2px #3b2a14; cursor: pointer; white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
.g-label:hover, .g-label.hot { background: #ffe27a; border-color: #000; }
.g-tip { position: absolute; pointer-events: none; padding: 3px 11px; background: #100010; border: 4px solid #000; color: #fff; max-width: 320px; z-index: 5;
  box-shadow: inset 0 0 0 3px #3b0a8a, 0 0 0 3px #000, 6px 6px 0 3px rgba(0,0,0,.45); text-shadow: 2px 0 #000, -2px 0 #000, 0 2px #000, 0 -2px #000, 2px 2px #000, -2px -2px #000, 2px -2px #000, -2px 2px #000; }
.g-hint { left: 50%; bottom: 14px; transform: translateX(-50%); color: #8a93b8; background: none; border: 0; box-shadow: none; pointer-events: none; padding: 0; white-space: nowrap; }
.g-msg { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; padding: 24px; color: #bbb; pointer-events: none; }
.hidden { display: none !important; }
@media (prefers-reduced-motion: reduce) { .g-host * { transition: none !important; } }
`;

// ---------------------------------------------------------------------------------------------- helpers
function hash3(x, y, z) {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(z | 0, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function vnoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), sm = t => t * t * (3 - 2 * t), l = (a, b, t) => a + (b - a) * t;
  const u = sm(x - xi), v = sm(y - yi), w = sm(z - zi);
  return l(l(l(hash3(xi, yi, zi), hash3(xi + 1, yi, zi), u), l(hash3(xi, yi + 1, zi), hash3(xi + 1, yi + 1, zi), u), v),
           l(l(hash3(xi, yi, zi + 1), hash3(xi + 1, yi, zi + 1), u), l(hash3(xi, yi + 1, zi + 1), hash3(xi + 1, yi + 1, zi + 1), u), v), w);
}
const terrainHeight = d => Math.floor(vnoise(d.x * 3.2 + 7, d.y * 3.2, d.z * 3.2) * 2.6 + vnoise(d.x * 7.5, d.y * 7.5 + 3, d.z * 7.5) * 1.2);
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const shade = () => [0.8, 0.9, 1.0, 1.1][(Math.random() * 4) | 0];  // 4 fixed brightness steps look like pixel art

function blockTexture() {  // 16 x 16 speckled face with a light top-left and dark bottom-right edge, like a game block
  const c = document.createElement('canvas'); c.width = c.height = 16;
  const g = c.getContext('2d');
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    let v = 205 + Math.floor(Math.random() * 50);
    if (x === 0 || y === 0) v = Math.min(255, v + 25); else if (x === 15 || y === 15) v -= 70;
    g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x, y, 1, 1);
  }
  const t = new THREE.CanvasTexture(c);
  t.magFilter = t.minFilter = THREE.NearestFilter; t.generateMipmaps = false; t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function start(host, { api, el }) {
  const style = document.createElement('style'); style.textContent = CSS; document.head.append(style);
  host.classList.add('g-host');
  const msg = el('div', 'g-msg', 'กำลังสร้างโลก…'); host.append(msg);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x0b1026, 1);
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0, 3.45);
  // hard, low sun: deep shadows between the terrain steps, trees and clouds; the far side of the planet falls into the dark
  scene.add(new THREE.AmbientLight(0x9fb0ff, 0.75));
  scene.add(new THREE.HemisphereLight(0x8ec5ff, 0x1a0f2e, 0.55));
  const sun = new THREE.DirectionalLight(0xfff1d6, 3.4); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0006; sun.shadow.normalBias = 0.02;
  sun.shadow.camera.near = 0.5; sun.shadow.camera.far = 16; scene.add(sun);
  const fitShadow = e => { const c = sun.shadow.camera; c.left = c.bottom = -e; c.right = c.top = e; c.updateProjectionMatrix(); };
  fitShadow(1.9);

  const cube = new THREE.BoxGeometry(1, 1, 1);
  const blockMat = new THREE.MeshLambertMaterial({ map: blockTexture() });
  const fadeMats = [];
  const solid = (extra = {}) => { const m = new THREE.MeshLambertMaterial({ map: blockMat.map, ...extra }); fadeMats.push(m); return m; };
  const flat = (color, opacity, additive) => {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
    m.userData.base = opacity; fadeMats.push(m); return m;
  };

  // ---------- HUD
  const crumb = el('div', 'g-panel g-crumb'); const back = el('button', '', '← โลก'); const crumbName = el('span', 'name', 'โลกความหมาย');
  back.classList.add('hidden'); crumb.append(back, crumbName);
  const stats = el('div', 'g-panel g-stats', '');
  const legend = el('div', 'g-panel g-legend hidden');
  const card = el('div', 'g-panel g-card hidden');
  const hint = el('div', 'g-panel g-hint', 'ลากเพื่อหมุน · ล้อเมาส์ซูม · คลิกไบโอมเพื่อเข้าไปดู');
  const tip = el('div', 'g-tip hidden');
  const search = el('div', 'g-panel g-search');
  const form = el('form'); const input = el('input'); input.placeholder = 'ค้นแล้วดูว่าอยู่ไบโอมไหน'; input.setAttribute('aria-label', 'ค้นหา');
  const go = el('button', 'g-btn', 'ค้น'); go.type = 'submit'; form.append(input, go);
  const hitsBox = el('div', 'g-hits'); search.append(form, hitsBox);
  const labelLayer = el('div'); labelLayer.style.cssText = 'position:absolute;inset:0;pointer-events:none';
  host.append(labelLayer, crumb, stats, legend, card, search, hint, tip);

  // ---------- state
  const S = {
    mode: 'globe', zones: null, version: -1, zoneOf: [], hits: new Map(), hitZones: new Set(), detail: null, anim: null,
    hover: -1, sel: -1, iso: -1, subHover: -1, dragging: false, moved: 0, vel: { x: 0, y: 0 }, last: { x: 0, y: 0 }, loading: false, yaw: 0.5, pitch: 0.45,
  };

  // ---------- backdrop: square stars, drifting cloud blocks
  const stars = (() => {
    const n = 1100, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(18 + Math.random() * 14); pos.set([v.x, v.y, v.z], i * 3);
      const b = 0.5 + Math.random() * 0.5; col.set([b, b, b + 0.1], i * 3);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.16, vertexColors: true, sizeAttenuation: true })); scene.add(p); return p;
  })();

  const cloudGroups = [0, 1, 2, 3].map(k => {
    const parts = []; const cx = (Math.random() - 0.5) * 0.4;
    for (let c = 0; c < 3; c++) {  // three little clouds per orbit, each a flat cluster of cubes
      const ox = cx + (c - 1) * 0.9 + Math.random() * 0.3, oz = (Math.random() - 0.5) * 0.4;
      for (let i = 0; i < 9; i++) parts.push([ox + (Math.floor(Math.random() * 4) - 1.5) * 0.075, (Math.random() - 0.5) * 0.03, oz + (Math.floor(Math.random() * 3) - 1) * 0.075]);
    }
    const mat = solid({ transparent: true, opacity: 0.92 }); mat.userData.base = 0.92;
    const mesh = new THREE.InstancedMesh(cube, mat, parts.length), m = new THREE.Matrix4();
    parts.forEach((p, i) => { m.compose(new THREE.Vector3(...p), new THREE.Quaternion(), new THREE.Vector3(0.075, 0.04, 0.075)); mesh.setMatrixAt(i, m); mesh.setColorAt(i, new THREE.Color().setScalar(0.9 + Math.random() * 0.1)); });
    mesh.castShadow = true; const g = new THREE.Group(); mesh.position.set(0, 0, 1.55 + k * 0.12); g.add(mesh);
    g.rotation.set(Math.random() * 3, Math.random() * 6, Math.random() * 3); g.userData.speed = (0.0006 + Math.random() * 0.0008) * (k % 2 ? 1 : -1);
    scene.add(g); return g;
  });

  // ---------- globe layer (a planet of cubes)
  const globe = new THREE.Group(); scene.add(globe);
  const pick = new THREE.Mesh(new THREE.SphereGeometry(1.06, 24, 24), new THREE.MeshBasicMaterial({ visible: false })); globe.add(pick);
  let planet = null, planetZone = null, planetBase = null, beacons = [], treeMeshes = [];
  const labels = [];

  const zoneAt = p => {
    let best = -1, bestScore = -9;
    S.zones.forEach((z, j) => { const s = p.x * z.dir[0] + p.y * z.dir[1] + p.z * z.dir[2] + z.bias; if (s > bestScore) { bestScore = s; best = j; } });
    return best;
  };

  function clearPlanet() {
    [planet, ...treeMeshes, ...beacons.map(b => b.group)].forEach(o => { if (o) { globe.remove(o); o.geometry && o.geometry.dispose && o.geometry.dispose(); } });
    planet = null; treeMeshes = []; beacons = []; labels.splice(0).forEach(l => l.node.remove());
  }

  function buildPlanet() {
    clearPlanet();
    const zs = S.zones, mean = zs.reduce((a, z) => a + z.count, 0) / zs.length;
    zs.forEach(z => { z.bias = 0.2 * Math.log(Math.max(z.count, 1) / mean); });
    const cells = [], d = new THREE.Vector3(), w = new THREE.Vector3(), lim = RV + 4;
    for (let x = -lim; x <= lim; x++) for (let y = -lim; y <= lim; y++) for (let z = -lim; z <= lim; z++) {
      const r = Math.sqrt(x * x + y * y + z * z); if (r < RV - 2 || r > RV + 4) continue;
      d.set(x / r, y / r, z / r);
      const h = RV + terrainHeight(d); if (r > h || r <= h - 1.7) continue;
      w.set(d.x + (vnoise(x * 0.16, y * 0.16, z * 0.16) - 0.5) * 0.4, d.y + (vnoise(x * 0.16 + 9, y * 0.16, z * 0.16) - 0.5) * 0.4, d.z + (vnoise(x * 0.16, y * 0.16 + 4, z * 0.16 + 9) - 0.5) * 0.4).normalize();
      cells.push({ x, y, z, j: zoneAt(w), top: r > h - 1 });
    }
    const mat = solid(); const mesh = new THREE.InstancedMesh(cube, mat, cells.length), m = new THREE.Matrix4(), s = new THREE.Vector3().setScalar(1.003 / RV);
    planetZone = new Int16Array(cells.length); planetBase = new Float32Array(cells.length * 3);
    const c = new THREE.Color();
    cells.forEach((cell, i) => {
      m.compose(new THREE.Vector3(cell.x / RV, cell.y / RV, cell.z / RV), new THREE.Quaternion(), s); mesh.setMatrixAt(i, m);
      c.copy(colorOf(cell.j)).multiplyScalar(cell.top ? shade() : 0.5 * shade()); planetBase.set([c.r, c.g, c.b], i * 3); planetZone[i] = cell.j;
    });
    mesh.castShadow = mesh.receiveShadow = true; planet = mesh; globe.add(planet);

    // trees: one trunk + a small crown, standing along the dominant axis of the cell (the voxel world has no tilted blocks)
    const trunks = [], leaves = [], axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    cells.forEach(cell => {
      const b = biomeOf(cell.j); if (!cell.top || !b.trees || Math.random() > 0.045) return;
      const comp = [cell.x, cell.y, cell.z], k = comp.map(Math.abs).indexOf(Math.max(...comp.map(Math.abs)));
      const sign = Math.sign(comp[k]) || 1, up = axes[k].map(v => v * sign), p1 = axes[(k + 1) % 3], p2 = axes[(k + 2) % 3];
      const at = n => [cell.x + up[0] * n, cell.y + up[1] * n, cell.z + up[2] * n];
      const off = (n, a, bb) => at(n).map((v, i) => v + p1[i] * a + p2[i] * bb);
      trunks.push({ p: at(1), j: cell.j }, { p: at(2), j: cell.j });
      [[0, 1], [0, -1], [1, 0], [-1, 0]].forEach(([a, bb]) => leaves.push({ p: off(2, a, bb), j: cell.j }));
      [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]].forEach(([a, bb]) => leaves.push({ p: off(3, a, bb), j: cell.j }));
      leaves.push({ p: at(4), j: cell.j });
    });
    [[trunks, 'trunk'], [leaves, 'leaf']].forEach(([list, key]) => {
      if (!list.length) return;
      const tm = new THREE.InstancedMesh(cube, solid(), list.length);
      list.forEach((o, i) => {
        m.compose(new THREE.Vector3(o.p[0] / RV, o.p[1] / RV, o.p[2] / RV), new THREE.Quaternion(), s); tm.setMatrixAt(i, m);
        tm.setColorAt(i, new THREE.Color(biomeOf(o.j)[key]).multiplyScalar(shade()));
      });
      tm.castShadow = tm.receiveShadow = true; tm.userData.zones = list.map(o => o.j); tm.userData.base = Array.from(tm.instanceColor.array); globe.add(tm); treeMeshes.push(tm);
    });

    // beacon beam + sign for every zone
    zs.forEach((z, j) => {
      const dir = new THREE.Vector3(...z.dir), group = new THREE.Group();
      const outer = new THREE.Mesh(cube, flat(colorOf(j), 0.45, true)); outer.scale.set(0.045, 0.34, 0.045);
      const inner = new THREE.Mesh(cube, flat(0xffffff, 0.8, true)); inner.scale.set(0.016, 0.34, 0.016);
      group.add(outer, inner); group.position.copy(dir).multiplyScalar(1.12 + 0.17); group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      globe.add(group); beacons.push({ group, outer, inner });
      const node = el('div', 'g-label', z.name); node.onclick = () => openZone(j);
      node.onpointerenter = () => { S.hover = j; syncBoost(); }; node.onpointerleave = () => { S.hover = -1; syncBoost(); };
      labelLayer.append(node); node.style.pointerEvents = 'auto'; labels.push({ node, dir });
    });
    syncBoost();
  }

  function syncBoost() {
    if (!planet) return;
    const f = j => (j === S.hover ? 1.55 : S.hitZones.has(j) ? 1.3 : 1);
    const c = new THREE.Color();
    for (let i = 0; i < planetZone.length; i++) { const k = f(planetZone[i]); c.setRGB(planetBase[i * 3] * k, planetBase[i * 3 + 1] * k, planetBase[i * 3 + 2] * k); planet.setColorAt(i, c); }
    planet.instanceColor.needsUpdate = true;
    treeMeshes.forEach(tm => {
      const zones = tm.userData.zones, base = tm.userData.base, arr = tm.instanceColor.array;
      for (let i = 0; i < zones.length; i++) { const k = f(zones[i]); arr[i * 3] = base[i * 3] * k; arr[i * 3 + 1] = base[i * 3 + 1] * k; arr[i * 3 + 2] = base[i * 3 + 2] * k; }
      tm.instanceColor.needsUpdate = true;
    });
    beacons.forEach((b, j) => { const on = j === S.hover || S.hitZones.has(j); b.outer.scale.x = b.outer.scale.z = on ? 0.09 : 0.045; b.inner.scale.x = b.inner.scale.z = on ? 0.03 : 0.016; });
    labels.forEach((l, j) => l.node.classList.toggle('hot', j === S.hover || S.hitZones.has(j)));
  }

  // ---------- zone layer: floating island + the zone's points as blocks
  const zoneGroup = new THREE.Group(); zoneGroup.visible = false; scene.add(zoneGroup);
  const CELL = 0.06;
  let zoneMesh = null, zonePos = null, zoneBase = null, cursor = null, posOf = new Map();
  const zoneMat = solid();

  const island = (() => {
    const g = new THREE.Group(), cells = [], m = new THREE.Matrix4(), s = 0.1, tops = [];
    [12, 9, 6, 3, 1].forEach((half, layer) => {
      for (let gx = -half; gx <= half; gx++) for (let gz = -half; gz <= half; gz++) {
        const col = layer === 0 ? new THREE.Color('#5da83c') : new THREE.Color(layer < 3 ? '#7a5a3a' : '#7d7d7d');
        cells.push({ p: [gx * s, -1.5 - layer * s, gz * s], c: col.multiplyScalar(shade()) });
        if (layer === 0 && Math.abs(gx) < 11 && Math.abs(gz) < 11) tops.push([gx, gz]);
      }
    });
    const mesh = new THREE.InstancedMesh(cube, solid(), cells.length);
    cells.forEach((c, i) => { m.compose(new THREE.Vector3(...c.p), new THREE.Quaternion(), new THREE.Vector3(s * 1.003, s * 1.003, s * 1.003)); mesh.setMatrixAt(i, m); mesh.setColorAt(i, c.c); });
    mesh.castShadow = mesh.receiveShadow = true; g.add(mesh);
    const bits = [], pickTop = () => tops[(Math.random() * tops.length) | 0];
    for (let t = 0; t < 7; t++) {  // little trees on the island
      const [gx, gz] = pickTop();
      for (let n = 1; n <= 3; n++) bits.push({ p: [gx * s, -1.5 + n * s, gz * s], c: '#6b4a2b' });
      for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) bits.push({ p: [(gx + a) * s, -1.5 + 4 * s, (gz + b) * s], c: '#2f7d2a' });
      [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]].forEach(([a, b]) => bits.push({ p: [(gx + a) * s, -1.5 + 5 * s, (gz + b) * s], c: '#3a8f34' }));
    }
    const trees = new THREE.InstancedMesh(cube, solid(), bits.length);
    bits.forEach((c, i) => { m.compose(new THREE.Vector3(...c.p), new THREE.Quaternion(), new THREE.Vector3(s * 1.003, s * 1.003, s * 1.003)); trees.setMatrixAt(i, m); trees.setColorAt(i, new THREE.Color(c.c).multiplyScalar(shade())); });
    trees.castShadow = trees.receiveShadow = true; g.add(trees); return g;
  })();
  zoneGroup.add(island);

  function buildZone(d) {
    if (zoneMesh) { zoneGroup.remove(zoneMesh, cursor); zoneMesh.dispose(); }
    const n = d.idx.length, taken = new Set(), pos = new Float32Array(n * 3), base = new Float32Array(n * 3), m = new THREE.Matrix4(), c = new THREE.Color();
    const mesh = new THREE.InstancedMesh(cube, zoneMat, n); mesh.castShadow = mesh.receiveShadow = true;
    posOf = new Map();
    for (let k = 0; k < n; k++) {
      let gx = Math.round(d.xyz[k][0] * 1.0 / CELL * 0.95), gy = Math.round(d.xyz[k][1] * 1.0 / CELL * 0.95), gz = Math.round(d.xyz[k][2] * 1.0 / CELL * 0.95);
      while (taken.has(gx + ',' + gy + ',' + gz)) gy++;  // two points never share a block
      taken.add(gx + ',' + gy + ',' + gz);
      pos.set([gx * CELL, gy * CELL + 0.1, gz * CELL], k * 3);
      c.copy(colorOf(d.sub[k] + S.detail.id * 5)).multiplyScalar(shade()); base.set([c.r, c.g, c.b], k * 3); posOf.set(d.idx[k], k);
    }
    zoneMesh = mesh; zonePos = pos; zoneBase = base;
    cursor = new THREE.LineSegments(new THREE.EdgesGeometry(cube), new THREE.LineBasicMaterial({ color: 0xffff55 })); cursor.scale.setScalar(CELL * 1.9); cursor.visible = false;
    zoneGroup.add(mesh, cursor);
    syncZoneBoost();
    legend.replaceChildren(...d.subs.map((s, j) => {
      const chip = el('div', 'g-chip'); const sw = el('i'); sw.style.background = '#' + colorOf(j + S.detail.id * 5).getHexString();
      chip.append(sw, el('span', '', s.name), el('small', '', String(s.count)));
      chip.onpointerenter = () => { S.subHover = j; syncZoneBoost(); }; chip.onpointerleave = () => { S.subHover = -1; syncZoneBoost(); };
      chip.onclick = () => { S.iso = S.iso === j ? -1 : j; legend.querySelectorAll('.g-chip').forEach((q, i) => q.classList.toggle('on', i === S.iso)); syncZoneBoost(); };
      return chip;
    }));
  }

  function syncZoneBoost() {
    if (!S.detail || !zoneMesh) return;
    const d = S.detail, m = new THREE.Matrix4(), c = new THREE.Color(), q = new THREE.Quaternion();
    for (let k = 0; k < d.idx.length; k++) {
      const hit = S.hits.has(d.idx[k]), sub = d.sub[k];
      let scale = 1, tint = 1;
      if (hit) { scale = 1.9; tint = 1.8; }
      else if (S.iso >= 0 && sub !== S.iso) { scale = 0.35; tint = 0.35; }
      else if (S.subHover >= 0) { if (sub === S.subHover) { scale = 1.25; tint = 1.35; } else { scale = 0.5; tint = 0.4; } }
      m.compose(new THREE.Vector3(zonePos[k * 3], zonePos[k * 3 + 1], zonePos[k * 3 + 2]), q, new THREE.Vector3().setScalar(CELL * 0.98 * scale)); zoneMesh.setMatrixAt(k, m);
      if (hit) c.setRGB(1.0 * tint, 0.82 * tint, 0.2 * tint); else c.setRGB(zoneBase[k * 3] * tint, zoneBase[k * 3 + 1] * tint, zoneBase[k * 3 + 2] * tint);
      zoneMesh.setColorAt(k, c);
    }
    zoneMesh.instanceMatrix.needsUpdate = true; zoneMesh.instanceColor.needsUpdate = true;
  }

  // ---------- fading (dive / surface transitions)
  function setFade(v, mats = fadeMats) {
    mats.forEach(m => { if (m.userData.base != null) { m.opacity = m.userData.base * v; } else { m.transparent = v < 0.999; m.opacity = v; } });
    labelLayer.style.opacity = v;
  }

  // ---------- data
  async function loadZones(force) {
    if (S.loading) return;
    S.loading = true;
    try {
      const st = await api('/api/status');
      if (!force && S.zones && st.version === S.version) return;
      const d = await api('/api/zones');
      S.version = d.version; S.zones = d.zones; S.zoneOf = d.zone_of;
      stats.textContent = `${d.n.toLocaleString()} จุด · ${d.zones.length} ไบโอม`;
      if (!d.zones.length) { msg.textContent = 'ต้องมีอย่างน้อย 60 จุดบนแผนที่ก่อน (อัปโหลดเอกสารในแท็บอัปโหลด)'; msg.classList.remove('hidden'); return; }
      msg.classList.add('hidden');
      buildPlanet();
    } catch (e) { msg.textContent = e.message; msg.classList.remove('hidden'); } finally { S.loading = false; }
  }

  function tween(ms, fn, done) { S.anim = { t0: performance.now(), ms, fn, done }; }

  async function openZone(id, focusIndex) {
    if (S.mode !== 'globe' || !S.zones) return;
    S.mode = 'diving'; tip.classList.add('hidden');
    const fetching = api('/api/zone?id=' + id);
    const from = globe.quaternion.clone(), dir = new THREE.Vector3(...S.zones[id].dir);
    const to = new THREE.Quaternion().setFromUnitVectors(dir, new THREE.Vector3(0, 0, 1));
    const z0 = camera.position.z;
    tween(800, t => {
      globe.quaternion.slerpQuaternions(from, to, ease(t)); camera.position.z = z0 + (1.7 - z0) * ease(t);
      setFade(1 - Math.max(0, (t - 0.5) / 0.5));
    }, async () => {
      try { S.detail = await fetching; } catch (e) { msg.textContent = e.message; msg.classList.remove('hidden'); S.mode = 'globe'; setFade(1); camera.position.z = z0; return; }
      showZone(focusIndex);
    });
  }

  function showZone(focusIndex) {
    S.iso = -1; S.subHover = -1; S.sel = -1; buildZone(S.detail);
    globe.visible = false; stars.visible = true; cloudGroups.forEach(g => { g.visible = false; }); zoneGroup.visible = true; S.yaw = 0.5; S.pitch = 0.45;
    back.classList.remove('hidden'); legend.classList.remove('hidden');
    crumbName.textContent = '› ' + S.detail.name + (S.detail.total > S.detail.shown ? `  (สุ่มแสดง ${S.detail.shown.toLocaleString()} จาก ${S.detail.total.toLocaleString()})` : `  (${S.detail.total.toLocaleString()} บล็อก)`);
    hint.textContent = 'ลากเพื่อหมุน · ล้อเมาส์ซูม · ชี้บล็อกดูชื่อ · คลิกดูข้อความ · คลิกสีด้านขวาเพื่อเลือกกลุ่มย่อย';
    S.mode = 'zone'; fitShadow(2.7); camera.position.set(0, 0.6, 4.6); labelLayer.classList.add('hidden');
    tween(700, t => { setFade(ease(t)); zoneGroup.scale.setScalar(0.6 + 0.4 * ease(t)); });
    if (focusIndex != null && posOf.has(focusIndex)) selectPoint(posOf.get(focusIndex)); else card.classList.add('hidden');
  }

  function backToGlobe() {
    if (S.mode !== 'zone') return;
    S.mode = 'surfacing'; card.classList.add('hidden'); legend.classList.add('hidden'); back.classList.add('hidden'); tip.classList.add('hidden');
    tween(400, t => { setFade(1 - ease(t)); }, () => {
      zoneGroup.visible = false; globe.visible = true; cloudGroups.forEach(g => { g.visible = true; }); labelLayer.classList.remove('hidden');
      setFade(1); fitShadow(1.9); camera.position.set(0, 0, 3.45); crumbName.textContent = 'โลกความหมาย'; S.detail = null; S.mode = 'globe';
      hint.textContent = 'ลากเพื่อหมุน · ล้อเมาส์ซูม · คลิกไบโอมเพื่อเข้าไปดู';
    });
  }
  back.onclick = backToGlobe;

  async function selectPoint(k) {
    S.sel = k; const d = S.detail, i = d.idx[k];
    cursor.visible = true; cursor.position.set(zonePos[k * 3], zonePos[k * 3 + 1], zonePos[k * 3 + 2]);
    card.classList.remove('hidden'); card.replaceChildren(el('h4', '', d.names[d.nm[k]]), el('small', '', d.lab[k] || ''), el('p', '', 'กำลังโหลด…'));
    try { const doc = await api('/api/doc?i=' + i); if (S.sel === k) card.querySelector('p').textContent = doc.text; } catch (e) { card.querySelector('p').textContent = e.message; }
  }

  // ---------- search
  form.onsubmit = async e => {
    e.preventDefault();
    const q = input.value.trim(); if (!q) return;
    hitsBox.replaceChildren(el('div', 'g-hit', 'กำลังค้น…'));
    try {
      if (!S.zones) await loadZones(true);
      const r = await api('/api/search?q=' + encodeURIComponent(q));
      S.hits = new Map(r.results.map((h, rank) => [h.i, rank + 1])); S.hitZones = new Set(r.results.map(h => S.zoneOf[h.i]).filter(z => z != null));
      hitsBox.replaceChildren(...r.results.map((h, rank) => {
        const z = S.zoneOf[h.i], b = el('button', 'g-hit'); b.type = 'button';
        b.append(el('b', '', String(rank + 1)), document.createTextNode(`${h.law}${h.section ? ' · ' + h.section : ''}${z != null && S.zones[z] ? '  → ' + S.zones[z].name : ''}`));
        b.onclick = () => {
          if (z == null) return;
          if (S.mode === 'zone' && S.detail.id === z) { if (posOf.has(h.i)) selectPoint(posOf.get(h.i)); } else if (S.mode === 'globe') openZone(z, h.i);
        };
        return b;
      }));
      syncBoost(); syncZoneBoost();
    } catch (err) { hitsBox.replaceChildren(el('div', 'g-hit', err.message)); }
  };

  // ---------- interaction
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  const dom = renderer.domElement;
  const rectOf = () => dom.getBoundingClientRect();

  dom.addEventListener('pointerdown', e => { tip.classList.add('hidden'); S.dragging = true; S.moved = 0; S.last = { x: e.clientX, y: e.clientY }; dom.setPointerCapture(e.pointerId); });
  dom.addEventListener('pointerup', e => {
    S.dragging = false;
    if (S.moved < 5) {
      if (S.mode === 'globe' && S.hover >= 0) openZone(S.hover);
      else if (S.mode === 'zone') { const k = nearestPoint(e); if (k >= 0) selectPoint(k); }
    }
  });
  dom.addEventListener('pointermove', e => {
    if (S.dragging) {
      const dx = e.clientX - S.last.x, dy = e.clientY - S.last.y; S.moved += Math.abs(dx) + Math.abs(dy); S.last = { x: e.clientX, y: e.clientY };
      tip.classList.add('hidden'); S.vel = { x: dx * 0.005, y: dy * 0.005 }; rotateBy(S.vel.x, S.vel.y); return;
    }
    const r = rectOf(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    if (S.mode === 'globe' && S.zones) {
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(pick)[0];
      const j = hit ? zoneAt(globe.worldToLocal(hit.point.clone()).normalize()) : -1;
      if (j !== S.hover) { S.hover = j; syncBoost(); }
      dom.style.cursor = j >= 0 ? 'pointer' : 'grab';
      if (j >= 0) { tip.textContent = `${S.zones[j].name} · ${S.zones[j].count.toLocaleString()} จุด`; tip.style.left = e.clientX - r.left + 14 + 'px'; tip.style.top = e.clientY - r.top + 12 + 'px'; tip.classList.remove('hidden'); }
      else tip.classList.add('hidden');
    } else if (S.mode === 'zone') {
      const k = nearestPoint(e);
      if (k >= 0) { const d = S.detail; tip.textContent = d.names[d.nm[k]] + (d.lab[k] ? ' · ' + d.lab[k] : ''); tip.style.left = e.clientX - r.left + 14 + 'px'; tip.style.top = e.clientY - r.top + 12 + 'px'; tip.classList.remove('hidden'); dom.style.cursor = 'pointer'; }
      else { tip.classList.add('hidden'); dom.style.cursor = 'grab'; }
    }
  });
  dom.addEventListener('pointerleave', () => { tip.classList.add('hidden'); if (S.hover >= 0) { S.hover = -1; syncBoost(); } });
  dom.addEventListener('wheel', e => {
    e.preventDefault();
    camera.position.z = THREE.MathUtils.clamp(camera.position.z * (1 + Math.sign(e.deltaY) * 0.08), S.mode === 'zone' ? 1.8 : 2.4, 8);
  }, { passive: false });

  function rotateBy(dx, dy) {
    if (S.mode === 'zone') { S.yaw += dx * 1.2; S.pitch = THREE.MathUtils.clamp(S.pitch + dy * 1.2, -0.35, 1.2); return; }
    const qy = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dx), qx = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), dy);
    globe.quaternion.premultiply(qy).premultiply(qx);
  }

  const tmp = new THREE.Vector3();
  function nearestPoint(e) {
    if (!zonePos) return -1;
    const r = rectOf(), mx = e.clientX - r.left, my = e.clientY - r.top;
    zoneGroup.updateMatrixWorld(); camera.updateMatrixWorld();
    let best = -1, bestZ = Infinity;
    for (let k = 0; k < zonePos.length / 3; k++) {
      if (S.iso >= 0 && S.detail.sub[k] !== S.iso) continue;
      tmp.set(zonePos[k * 3], zonePos[k * 3 + 1], zonePos[k * 3 + 2]).applyMatrix4(zoneGroup.matrixWorld).project(camera);
      const sx = (tmp.x * 0.5 + 0.5) * r.width, sy = (-tmp.y * 0.5 + 0.5) * r.height;
      if (Math.abs(sx - mx) < 11 && Math.abs(sy - my) < 11 && tmp.z < bestZ) { bestZ = tmp.z; best = k; }
    }
    return best;
  }

  // ---------- loop
  const size = () => {
    const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5)); renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  new ResizeObserver(size).observe(host); size();

  const worldDir = new THREE.Vector3(), camDir = new THREE.Vector3(), projected = new THREE.Vector3(), look = new THREE.Vector3();
  function frame(now) {
    requestAnimationFrame(frame);
    if (!host.offsetParent) return;  // tab hidden
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (S.anim) {
      const t = Math.min(1, (now - S.anim.t0) / S.anim.ms); S.anim.fn(t);
      if (t >= 1) { const done = S.anim.done; S.anim = null; if (done) done(); }
    }
    if (!S.dragging && !S.anim) {
      if (Math.abs(S.vel.x) > 1e-4 || Math.abs(S.vel.y) > 1e-4) { rotateBy(S.vel.x, S.vel.y); S.vel.x *= 0.94; S.vel.y *= 0.94; }
      else if (!reduce && (S.mode === 'zone' || (S.mode === 'globe' && S.hover < 0))) rotateBy(S.mode === 'zone' ? 0.0012 : 0.0016, 0);
    }
    if (S.mode === 'zone') zoneGroup.rotation.set(S.pitch * 0.35, S.yaw, 0);
    if (!reduce) { stars.rotation.y += 0.00006; cloudGroups.forEach(g => { g.rotation.y += g.userData.speed; }); }
    const pulse = 0.7 + 0.3 * Math.sin(now / 500);
    beacons.forEach(b => { b.outer.material.opacity = 0.45 * pulse * (S.mode === 'globe' ? 1 : labelLayer.style.opacity || 1); });
    sun.position.copy(camera.position).add(look.set(-3.2, 2.4, 1.8)); look.set(0, S.mode === 'zone' ? -0.2 : 0, 0); camera.lookAt(look);
    if (S.mode !== 'zone' && labels.length) {
      const r = rectOf(); camDir.copy(camera.position).normalize();
      labels.forEach(l => {
        worldDir.copy(l.dir).applyQuaternion(globe.quaternion);
        const facing = worldDir.dot(camDir); projected.copy(worldDir).multiplyScalar(1.28).project(camera);
        l.node.style.left = (projected.x * 0.5 + 0.5) * r.width + 'px'; l.node.style.top = (-projected.y * 0.5 + 0.5) * r.height + 'px';
        l.node.style.opacity = facing > 0.12 ? String(Math.min(1, (facing - 0.12) * 3)) : '0'; l.node.style.pointerEvents = facing > 0.3 ? 'auto' : 'none';
      });
    }
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  loadZones(true);

  return { refresh: () => loadZones(false), resize: size };
}
