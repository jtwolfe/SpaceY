import * as THREE from "three";
import { STAGE_LENGTH_M, STAGE_DIAMETER_M } from "./constants";
import type { Sim } from "./sim";
import type { CamMode } from "./store";

const STEEL = 0xb7bec8;
const WHITE = 0xf4f6f8;
const CHAR = 0x16181d;
const BELL = 0x4a3c32;
const PAD = 0x5a5e66;
const OCEAN = 0x0d1c26;
const FLAME = 0xffc48a;

function enuToThree(x: number, y: number, z: number, out: THREE.Vector3) {
  return out.set(x, z, -y);
}

export class SpaceyScene {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  rocket: THREE.Group;
  exhaust: THREE.Mesh;
  exhaustCore: THREE.Mesh;
  exhaustMat: THREE.MeshBasicMaterial;
  exhaustCoreMat: THREE.MeshBasicMaterial;
  flameLight: THREE.PointLight;
  fins: THREE.Group[] = [];
  pad: THREE.Group;
  trail: THREE.Line;
  trailPos: Float32Array;
  trailIdx = 0;
  yaw = 0.62;
  pitch = 0.22;
  zoom = 1;
  dragging = false;
  lastPx = 0;
  lastPy = 0;
  swarm: THREE.Line[] = [];
  swarmPos: Float32Array[] = [];
  swarmMesh: THREE.InstancedMesh;
  swarmDummy = new THREE.Object3D();
  swarmColor = new THREE.Color();
  wide = false;
  unbindNav: (() => void) | null = null;
  tmp = new THREE.Vector3();
  tmp2 = new THREE.Vector3();
  up = new THREE.Vector3(0, 1, 0);
  camTarget = new THREE.Vector3();
  camPos = new THREE.Vector3();
  lookAt = new THREE.Vector3();
  bodyDir = new THREE.Vector3();
  disposed = false;
  resizeObs: ResizeObserver | null = null;
  firstCam = true;
  handoff = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0c141c, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.18;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x0c141c, 0.000008);

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.4, 180_000);
    this.camera.position.set(24, 90, 36);

    this.scene.add(new THREE.HemisphereLight(0xc5d2e0, 0x2a322c, 1.05));
    const sun = new THREE.DirectionalLight(0xfff6ea, 2.15);
    sun.position.set(-500, 900, 420);
    this.scene.add(sun);
    const rim = new THREE.DirectionalLight(0x7e9bb8, 0.55);
    rim.position.set(300, 80, -500);
    this.scene.add(rim);

    this.scene.add(makeSky());
    this.scene.add(makeStars());
    this.scene.add(makeOcean());
    this.pad = makePad();
    this.scene.add(this.pad);

    this.rocket = makeRocket();
    this.scene.add(this.rocket);
    this.exhaust = this.rocket.getObjectByName("exhaust") as THREE.Mesh;
    this.exhaustCore = this.rocket.getObjectByName("exhaustCore") as THREE.Mesh;
    this.exhaustMat = this.exhaust.material as THREE.MeshBasicMaterial;
    this.exhaustCoreMat = this.exhaustCore.material as THREE.MeshBasicMaterial;
    this.flameLight = this.rocket.getObjectByName("flameLight") as THREE.PointLight;
    this.rocket.traverse((o) => {
      if (o.name.startsWith("finAct")) this.fins.push(o as THREE.Group);
    });

    this.trailPos = new Float32Array(600 * 3);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    this.trail = new THREE.Line(
      tg,
      new THREE.LineBasicMaterial({ color: 0xdbe4ee, transparent: true, opacity: 0.8 }),
    );
    this.scene.add(this.trail);

    for (let i = 0; i < 12; i++) {
      const buf = new Float32Array(280 * 3);
      this.swarmPos.push(buf);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(buf, 3));
      geo.setDrawRange(0, 0);
      const line = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({
          color: 0x8aa0b8,
          transparent: true,
          opacity: 0.35,
        }),
      );
      line.visible = false;
      this.swarm.push(line);
      this.scene.add(line);
    }

    const needle = new THREE.CylinderGeometry(1.2, 1.8, 32, 7);
    const needleMat = new THREE.MeshBasicMaterial({ color: 0xd5dee8, transparent: true, opacity: 0.55 });
    this.swarmMesh = new THREE.InstancedMesh(needle, needleMat, 12);
    this.swarmMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.swarmMesh.frustumCulled = false;
    this.swarmMesh.visible = false;
    for (let i = 0; i < 12; i++) this.swarmMesh.setColorAt(i, this.swarmColor.setHex(STEEL));
    if (this.swarmMesh.instanceColor) this.swarmMesh.instanceColor.needsUpdate = true;
    this.scene.add(this.swarmMesh);

    this.fit(canvas);
    this.resizeObs = new ResizeObserver(() => this.fit(canvas));
    this.resizeObs.observe(canvas.parentElement ?? canvas);
    this.unbindNav = this.bindNav(canvas.parentElement ?? canvas);
  }

  fit(canvas: HTMLCanvasElement) {
    const parent = canvas.parentElement ?? canvas;
    const w = Math.max(16, parent.clientWidth);
    const h = Math.max(16, parent.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  sync(sim: Sim, cam: CamMode, dt: number) {
    const p = sim.p;
    const bx = sim.bodyX();
    enuToThree(p.x, p.y, p.z, this.tmp);
    this.rocket.position.copy(this.tmp);
    enuToThree(bx.x, bx.y, bx.z, this.bodyDir).normalize();
    this.rocket.quaternion.setFromUnitVectors(this.up, this.bodyDir);

    const on = sim.lastN > 0 && sim.lastThrottle > 0.05 && sim.term === "none";
    this.exhaust.visible = on;
    this.exhaustCore.visible = on;
    this.flameLight.intensity = on ? 40 + sim.lastThrottle * 70 : 0;
    if (on) {
      const k = (0.7 + sim.lastThrottle * 0.9) * (sim.lastN >= 3 ? 1.7 : 1);
      this.exhaust.scale.set(1.15, k * (2.4 + Math.random() * 0.35), 1.15);
      this.exhaustCore.scale.set(0.7, k * (1.8 + Math.random() * 0.25), 0.7);
      this.exhaustMat.color.setHSL(0.08, 0.9, 0.58 + Math.random() * 0.08);
      this.exhaustCoreMat.color.setHSL(0.12, 0.55, 0.86);
    }

    const [fp, fy] = sim.lastFins;
    for (let i = 0; i < this.fins.length; i++) {
      const th = Math.PI / 4 + i * (Math.PI / 2);
      this.fins[i].rotation.x = fp * Math.cos(th) + fy * Math.sin(th);
    }

    const i = (this.trailIdx % 600) * 3;
    this.trailPos[i] = this.tmp.x;
    this.trailPos[i + 1] = this.tmp.y;
    this.trailPos[i + 2] = this.tmp.z;
    this.trailIdx += 1;
    (this.trail.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    this.trail.geometry.setDrawRange(0, Math.min(600, this.trailIdx));

    this.updateCam(cam, dt);
  }

  setSwarm(
    items: {
      trail: Float32Array;
      len: number;
      landed: boolean;
      dead: boolean;
      hero: boolean;
      px: number;
      py: number;
      pz: number;
      bx: number;
      by: number;
      bz: number;
    }[],
    on: boolean,
  ) {
    this.wide = false;
    this.swarmMesh.visible = on;
    for (let i = 0; i < this.swarm.length; i++) {
      const line = this.swarm[i];
      const item = items[i];
      if (!on || !item || item.len < 2) {
        line.visible = false;
      } else {
        line.visible = true;
        const src = item.trail;
        const dst = this.swarmPos[i];
        const n = Math.min(280, item.len);
        const start = item.len > 280 ? (item.len - 280) % 280 : 0;
        for (let k = 0; k < n; k++) {
          const s = ((start + k) % 280) * 3;
          const d = k * 3;
          dst[d] = src[s];
          dst[d + 1] = src[s + 2];
          dst[d + 2] = -src[s + 1];
        }
        (line.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
        line.geometry.setDrawRange(0, n);
        const mat = line.material as THREE.LineBasicMaterial;
        if (item.landed) {
          mat.color.setHex(0x9fd4b0);
          mat.opacity = 0.45;
        } else if (item.dead) {
          mat.color.setHex(0xe09086);
          mat.opacity = 0.28;
        } else {
          mat.color.setHex(0x8aa0b8);
          mat.opacity = 0.35;
        }
      }

      if (!on || !item) continue;
      enuToThree(item.px, item.py, item.pz, this.tmp);
      enuToThree(item.bx, item.by, item.bz, this.bodyDir).normalize();
      this.swarmDummy.position.copy(this.tmp);
      this.swarmDummy.quaternion.setFromUnitVectors(this.up, this.bodyDir);
      this.swarmDummy.scale.setScalar(0.55);
      this.swarmDummy.updateMatrix();
      this.swarmMesh.setMatrixAt(i, this.swarmDummy.matrix);
      if (item.landed) this.swarmColor.setHex(0x9fd4b0);
      else if (item.dead) this.swarmColor.setHex(0xe09086);
      else this.swarmColor.setHex(0xc5d2e0);
      this.swarmMesh.setColorAt(i, this.swarmColor);
    }
    this.swarmMesh.instanceMatrix.needsUpdate = true;
    if (this.swarmMesh.instanceColor) this.swarmMesh.instanceColor.needsUpdate = true;
  }

  look(dyaw: number, dpitch: number, zoomMul = 1) {
    this.yaw += dyaw;
    this.pitch = Math.min(1.35, Math.max(-0.28, this.pitch + dpitch));
    this.zoom = Math.min(10, Math.max(0.18, this.zoom * zoomMul));
    this.firstCam = true;
  }

  getLook() {
    return { yaw: this.yaw, pitch: this.pitch, zoom: this.zoom, dragging: this.dragging };
  }

  resetLook() {
    this.yaw = 0.72;
    this.pitch = 0.28;
    this.zoom = 1;
    this.firstCam = true;
    this.handoff = 0;
  }

  beginHandoff() {
    this.handoff = 1.25;
    this.firstCam = false;
  }

  private bindNav(root: HTMLElement) {
    root.style.touchAction = "none";
    const pts = new Map<number, { x: number; y: number }>();
    let pinch0 = 0;
    const isUi = (t: EventTarget | null) =>
      t instanceof Element && !!t.closest("button, a, input, textarea, [data-ui]");

    const onDown = (e: PointerEvent) => {
      if (isUi(e.target)) return;
      if (e.pointerType === "mouse" && e.button !== 0) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) {
        this.dragging = true;
        this.lastPx = e.clientX;
        this.lastPy = e.clientY;
      } else if (pts.size === 2) {
        const arr = [...pts.values()];
        pinch0 = Math.hypot(arr[0].x - arr[1].x, arr[0].y - arr[1].y) || 1;
        this.dragging = false;
      }
    };
    const onMove = (e: PointerEvent) => {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size >= 2) {
        const arr = [...pts.values()];
        const d = Math.hypot(arr[0].x - arr[1].x, arr[0].y - arr[1].y) || 1;
        if (pinch0 > 0) this.zoom = Math.min(10, Math.max(0.18, this.zoom * (pinch0 / d)));
        pinch0 = d;
        return;
      }
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPx;
      const dy = e.clientY - this.lastPy;
      this.lastPx = e.clientX;
      this.lastPy = e.clientY;
      this.yaw -= dx * 0.0075;
      this.pitch = Math.min(1.35, Math.max(-0.28, this.pitch + dy * 0.0075));
    };
    const onUp = (e: PointerEvent) => {
      pts.delete(e.pointerId);
      if (pts.size === 0) this.dragging = false;
      if (pts.size === 1) {
        const left = [...pts.values()][0];
        this.dragging = true;
        this.lastPx = left.x;
        this.lastPy = left.y;
      }
    };
    const onWheel = (e: WheelEvent) => {
      if (isUi(e.target)) return;
      e.preventDefault();
      this.zoom = Math.min(10, Math.max(0.18, this.zoom * Math.exp(e.deltaY * 0.00135)));
    };
    const onDbl = (e: MouseEvent) => {
      if (isUi(e.target)) return;
      this.resetLook();
    };
    root.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    root.addEventListener("wheel", onWheel, { passive: false });
    root.addEventListener("dblclick", onDbl);
    return () => {
      root.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      root.removeEventListener("wheel", onWheel);
      root.removeEventListener("dblclick", onDbl);
    };
  }

  private updateCam(cam: CamMode, dt: number) {
    const pos = this.rocket.position;
    if (cam === "orbit" && !this.dragging) this.yaw += dt * 0.08;
    if (cam === "orbit") {
      this.camTarget.copy(pos);
    } else if (cam === "pad") {
      this.camTarget.set(pos.x * 0.22, Math.max(8, pos.y * 0.32), pos.z * 0.22);
    } else if (pos.y < 140) {
      this.camTarget.set(pos.x * 0.12, pos.y * 0.38, pos.z * 0.12);
    } else {
      this.camTarget.set(pos.x, pos.y - 1, pos.z);
    }
    this.handoff = Math.max(0, this.handoff - dt);
    const follow = this.handoff > 0 ? 2.2 : 7;
    const k = this.firstCam || this.dragging ? 1 : 1 - Math.exp(-dt * follow);
    this.firstCam = false;
    this.lookAt.lerp(this.camTarget, k);

    const alt = this.lookAt.y;
    let dist: number;
    let lift: number;
    if (cam === "orbit") {
      dist = (160 + Math.hypot(this.lookAt.x, this.lookAt.z) * 0.28 + alt * 0.28) * this.zoom;
      lift = dist * 0.1;
    } else if (cam === "pad") {
      dist = 140 * this.zoom;
      lift = 24;
    } else if (alt < 140) {
      dist = 92 * this.zoom;
      lift = 16;
    } else {
      dist = 48 * this.zoom;
      lift = 10;
    }
    dist = Math.max(16, dist);
    const cp = this.pitch;
    const cy = this.yaw;
    const ch = Math.cos(cp);
    this.camPos.set(
      this.lookAt.x + Math.sin(cy) * ch * dist,
      this.lookAt.y + Math.sin(cp) * dist + lift,
      this.lookAt.z + Math.cos(cy) * ch * dist,
    );
    this.camera.position.lerp(this.camPos, k);
    this.camera.lookAt(this.lookAt);
  }

  resetTrail() {
    this.trailIdx = 0;
    this.trail.geometry.setDrawRange(0, 0);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    this.unbindNav?.();
    this.resizeObs?.disconnect();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else mat?.dispose();
    });
    this.renderer.dispose();
  }
}

function makeRocket() {
  const g = new THREE.Group();
  const r = STAGE_DIAMETER_M * 0.5;
  const L = STAGE_LENGTH_M;
  const white = new THREE.MeshStandardMaterial({
    color: WHITE,
    metalness: 0.06,
    roughness: 0.42,
    emissive: 0x6a7380,
    emissiveIntensity: 0.55,
  });
  const steel = new THREE.MeshStandardMaterial({
    color: STEEL,
    metalness: 0.5,
    roughness: 0.34,
    emissive: 0x3a424c,
    emissiveIntensity: 0.4,
  });
  const dark = new THREE.MeshStandardMaterial({
    color: CHAR,
    metalness: 0.3,
    roughness: 0.5,
    emissive: 0x1a1e24,
    emissiveIntensity: 0.28,
  });
  const bellM = new THREE.MeshStandardMaterial({
    color: BELL,
    metalness: 0.55,
    roughness: 0.4,
    emissive: 0x2a1810,
    emissiveIntensity: 0.4,
  });

  const body = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L * 0.72, 28), white);
  body.position.y = L * 0.08;
  g.add(body);
  const race = new THREE.Mesh(new THREE.BoxGeometry(0.22, L * 0.62, r * 0.35), dark);
  race.position.set(r * 0.92, L * 0.06, 0);
  g.add(race);
  const stripe = new THREE.Mesh(new THREE.CylinderGeometry(r * 1.012, r * 1.012, 1.35, 28), dark);
  stripe.position.y = 6;
  g.add(stripe);
  const inter = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.92, r, L * 0.12, 24), dark);
  inter.position.y = L * 0.44;
  g.add(inter);
  const aft = new THREE.Mesh(new THREE.CylinderGeometry(r, r * 0.9, L * 0.16, 24), steel);
  aft.position.y = -L * 0.36;
  g.add(aft);

  for (let i = 0; i < 4; i++) {
    const root = new THREE.Group();
    const act = new THREE.Group();
    act.name = `finAct${i}`;
    const frame = new THREE.Mesh(new THREE.BoxGeometry(0.14, 1.85, 1.65), steel);
    frame.position.set(0, 0.15, 0.95);
    act.add(frame);
    for (let k = 0; k < 5; k++) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.05, 1.7, 0.05), steel);
      bar.position.set(-0.5 + k * 0.25, 0.15, 1.15);
      act.add(bar);
    }
    root.add(act);
    const th = Math.PI / 4 + i * (Math.PI / 2);
    root.position.set(Math.cos(th) * r, L * 0.28, Math.sin(th) * r);
    root.rotation.y = -th + Math.PI / 2;
    g.add(root);
  }

  const oct = [
    [0, 0],
    [1.3, 0],
    [-1.3, 0],
    [0.65, 1.12],
    [-0.65, 1.12],
    [0.65, -1.12],
    [-0.65, -1.12],
    [1.05, 0.65],
    [-1.05, 0.65],
  ];
  for (const [x, z] of oct) {
    const bell = new THREE.Mesh(new THREE.ConeGeometry(0.5, 1.6, 12), bellM);
    bell.position.set(x, -L * 0.5 - 0.25, z);
    bell.rotation.x = Math.PI;
    g.add(bell);
  }

  for (let i = 0; i < 4; i++) {
    const th = Math.PI / 4 + i * (Math.PI / 2);
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.18, 8.4, 8), steel);
    leg.position.set(Math.cos(th) * (r + 1.7), -L * 0.42, Math.sin(th) * (r + 1.7));
    leg.rotation.z = Math.cos(th) * 0.35;
    leg.rotation.x = -Math.sin(th) * 0.35;
    g.add(leg);
    const foot = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.16, 1.05), dark);
    foot.position.set(Math.cos(th) * (r + 3.5), -L * 0.5 - 3.3, Math.sin(th) * (r + 3.5));
    g.add(foot);
  }

  const exhaustGeo = new THREE.ConeGeometry(1.35, 12, 14, 1, true);
  exhaustGeo.rotateX(Math.PI);
  const exhaustMat = new THREE.MeshBasicMaterial({
    color: FLAME,
    transparent: true,
    opacity: 0.78,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
  const exhaust = new THREE.Mesh(exhaustGeo, exhaustMat);
  exhaust.name = "exhaust";
  exhaust.position.y = -L * 0.5 - 6.5;
  exhaust.visible = false;
  g.add(exhaust);

  const coreGeo = new THREE.ConeGeometry(0.55, 8, 10, 1, true);
  coreGeo.rotateX(Math.PI);
  const coreMat = new THREE.MeshBasicMaterial({
    color: 0xfff4d2,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
  const core = new THREE.Mesh(coreGeo, coreMat);
  core.name = "exhaustCore";
  core.position.y = -L * 0.5 - 5.2;
  core.visible = false;
  g.add(core);

  const flameLight = new THREE.PointLight(0xffb070, 0, 80, 2);
  flameLight.name = "flameLight";
  flameLight.position.y = -L * 0.5 - 4;
  g.add(flameLight);
  const fill = new THREE.PointLight(0xf2f5f8, 90, 260, 1.15);
  fill.position.set(10, 6, 16);
  g.add(fill);

  g.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) {
      m.castShadow = false;
      m.receiveShadow = false;
    }
  });
  return g;
}

function makePad() {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: PAD,
    metalness: 0.28,
    roughness: 0.58,
    emissive: 0x151820,
    emissiveIntensity: 0.2,
  });
  const deck = new THREE.Mesh(new THREE.CylinderGeometry(32, 32, 1.4, 8), mat);
  deck.position.y = 0.7;
  g.add(deck);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(20, 30, 8),
    new THREE.MeshBasicMaterial({ color: 0xd0d6de, side: THREE.DoubleSide }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 1.42;
  g.add(ring);
  const mark = new THREE.Mesh(
    new THREE.CircleGeometry(5.2, 24),
    new THREE.MeshBasicMaterial({ color: 0x07080c }),
  );
  mark.rotation.x = -Math.PI / 2;
  mark.position.y = 1.43;
  g.add(mark);
  const xMat = new THREE.MeshBasicMaterial({ color: 0xe8eaed });
  for (const rot of [0.6, -0.6]) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(11, 0.1, 0.85), xMat);
    bar.position.y = 1.48;
    bar.rotation.y = rot;
    g.add(bar);
  }
  for (let i = 0; i < 4; i++) {
    const p = new THREE.Mesh(new THREE.BoxGeometry(2.4, 7, 1.5), mat);
    const a = i * (Math.PI / 2) + Math.PI / 4;
    p.position.set(Math.cos(a) * 40, 3.5, Math.sin(a) * 40);
    g.add(p);
    const lamp = new THREE.Mesh(
      new THREE.BoxGeometry(0.6, 0.6, 0.6),
      new THREE.MeshBasicMaterial({ color: 0xe8dcc0 }),
    );
    lamp.position.set(Math.cos(a) * 40, 7.2, Math.sin(a) * 40);
    g.add(lamp);
    const light = new THREE.PointLight(0xffe6c0, 18, 90, 2);
    light.position.set(Math.cos(a) * 38, 8, Math.sin(a) * 38);
    g.add(light);
  }
  return g;
}

function makeOcean() {
  const g = new THREE.Group();
  const water = new THREE.Mesh(
    new THREE.CircleGeometry(60_000, 64),
    new THREE.MeshStandardMaterial({ color: OCEAN, metalness: 0.32, roughness: 0.58 }),
  );
  water.rotation.x = -Math.PI / 2;
  water.position.y = -0.4;
  g.add(water);
  const land = new THREE.Mesh(
    new THREE.CircleGeometry(520, 36),
    new THREE.MeshStandardMaterial({ color: 0x323c30, roughness: 0.92, metalness: 0.02 }),
  );
  land.rotation.x = -Math.PI / 2;
  land.position.y = -0.18;
  g.add(land);
  return g;
}

function makeSky() {
  const geo = new THREE.SphereGeometry(80_000, 32, 20);
  const col = geo.attributes.position;
  const colors = new Float32Array(col.count * 3);
  const cTop = new THREE.Color(0x070b10);
  const cHor = new THREE.Color(0x243242);
  const cBot = new THREE.Color(0x101820);
  for (let i = 0; i < col.count; i++) {
    const y = col.getY(i) / 80_000;
    const t = THREE.MathUtils.clamp(y * 0.5 + 0.5, 0, 1);
    const c = t < 0.48 ? cBot.clone().lerp(cHor, t / 0.48) : cHor.clone().lerp(cTop, (t - 0.48) / 0.52);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, depthWrite: false, fog: false });
  return new THREE.Mesh(geo, mat);
}

function makeStars() {
  const n = 1400;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 70_000;
    const a = Math.random() * Math.PI * 2;
    const b = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(b) * Math.cos(a);
    pos[i * 3 + 1] = Math.abs(r * Math.cos(b));
    pos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(
    geo,
    new THREE.PointsMaterial({ color: 0xe8eef4, size: 90, sizeAttenuation: true, depthWrite: false }),
  );
}
