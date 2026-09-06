import * as THREE from "three";
import { STAGE_LENGTH_M, STAGE_DIAMETER_M } from "./constants";
import { POP } from "./trainer";
import { makeWorld } from "./world";
import type { Sim } from "./sim";
import type { CamMode } from "./store";

const STEEL = 0xb7bec8;
const WHITE = 0xf4f6f8;
const CHAR = 0x16181d;
const BELL = 0x4a3c32;
const FLAME = 0xffc48a;
const ZOOM_MIN = 0.15;
const ZOOM_MAX = 64;

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
  nadir: THREE.Mesh;
  swarmNadir: THREE.InstancedMesh;
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
      logarithmicDepthBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x02040a, 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(48, 1, 0.8, 2.6e7);
    this.camera.position.set(24, 90, 36);

    this.scene.add(new THREE.HemisphereLight(0x8aa7c4, 0x0c1410, 0.42));
    const world = makeWorld();
    this.scene.add(world.group);
    this.pad = world.pad;

    const sun = new THREE.DirectionalLight(0xfff1dc, 2.35);
    sun.position.copy(world.sunDir).multiplyScalar(4.5e6);
    this.scene.add(sun);
    const bounce = new THREE.DirectionalLight(0x4a6a88, 0.28);
    bounce.position.set(800, 120, 200);
    this.scene.add(bounce);

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

    for (let i = 0; i < POP; i++) {
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
    this.swarmMesh = new THREE.InstancedMesh(needle, needleMat, POP);
    this.swarmMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.swarmMesh.frustumCulled = false;
    this.swarmMesh.visible = false;
    for (let i = 0; i < POP; i++) this.swarmMesh.setColorAt(i, this.swarmColor.setHex(STEEL));
    if (this.swarmMesh.instanceColor) this.swarmMesh.instanceColor.needsUpdate = true;
    this.scene.add(this.swarmMesh);

    this.nadir = new THREE.Mesh(
      new THREE.RingGeometry(2.4, 4.1, 28),
      new THREE.MeshBasicMaterial({
        color: 0xe8dcc0,
        transparent: true,
        opacity: 0.92,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.nadir.rotation.x = -Math.PI / 2;
    this.nadir.position.y = 1.55;
    this.scene.add(this.nadir);

    const nadirDisk = new THREE.CircleGeometry(2.2, 14);
    nadirDisk.rotateX(-Math.PI / 2);
    this.swarmNadir = new THREE.InstancedMesh(
      nadirDisk,
      new THREE.MeshBasicMaterial({
        color: 0x8aa0b8,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      }),
      POP,
    );
    this.swarmNadir.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.swarmNadir.frustumCulled = false;
    this.swarmNadir.visible = false;
    this.scene.add(this.swarmNadir);

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

    this.nadir.position.set(this.tmp.x, 1.55, this.tmp.z);
    this.nadir.visible = true;

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
    this.swarmNadir.visible = on;
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

      this.swarmDummy.position.set(item.px, 1.5, -item.py);
      this.swarmDummy.quaternion.identity();
      this.swarmDummy.scale.setScalar(item.landed || item.dead ? 0.7 : 1);
      this.swarmDummy.updateMatrix();
      this.swarmNadir.setMatrixAt(i, this.swarmDummy.matrix);
    }
    const drawn = on ? Math.min(items.length, POP) : 0;
    this.swarmMesh.count = drawn;
    this.swarmNadir.count = drawn;
    this.swarmMesh.instanceMatrix.needsUpdate = true;
    this.swarmNadir.instanceMatrix.needsUpdate = true;
    if (this.swarmMesh.instanceColor) this.swarmMesh.instanceColor.needsUpdate = true;
  }

  look(dyaw: number, dpitch: number, zoomMul = 1) {
    this.yaw += dyaw;
    this.pitch = Math.min(1.35, Math.max(-0.28, this.pitch + dpitch));
    this.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, this.zoom * zoomMul));
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
        if (pinch0 > 0) this.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, this.zoom * (pinch0 / d)));
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
      this.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, this.zoom * Math.exp(e.deltaY * 0.00135)));
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
      if (pos.y > 12_000) {
        this.camTarget.y = pos.y * 0.42;
        this.camTarget.x = pos.x * 0.55;
        this.camTarget.z = pos.z * 0.55;
      }
    } else if (cam === "pad") {
      if (pos.y > 400) {
        this.camTarget.set(0, 6, 0);
      } else {
        this.camTarget.set(pos.x * 0.22, Math.max(8, pos.y * 0.32), pos.z * 0.22);
      }
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
      dist = (160 + Math.hypot(this.lookAt.x, this.lookAt.z) * 0.28 + Math.max(alt, pos.y) * 0.42) * this.zoom;
      lift = dist * 0.16;
    } else if (cam === "pad") {
      if (pos.y > 400) {
        dist = Math.min(320 + pos.y * 0.04, 2_400) * this.zoom;
        lift = 110;
      } else {
        dist = 140 * this.zoom;
        lift = 24;
      }
    } else if (alt < 140) {
      dist = 92 * this.zoom;
      lift = 16;
    } else {
      dist = (48 + Math.min(pos.y, 80_000) * 0.012) * this.zoom;
      lift = 10 + Math.min(pos.y, 40_000) * 0.004;
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
