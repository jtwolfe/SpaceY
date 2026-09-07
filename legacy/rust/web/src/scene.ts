import * as THREE from "three";
import type { Snapshot, PackedTrails, GenerationViz } from "./types";

export type CamMode = "chase" | "pad" | "orbit";

const EARTH_R = 6_371_000;
const TRAIL_MAX = 800;
const BOOM_N = 48;
const BOOM_LIFE = 0.7;
const PATH_SPARKS = 10;
const PATH_POOL = 64 * PATH_SPARKS;
const PATH_LIFE = 0.52;
const WORLD_X = new THREE.Vector3(1, 0, 0);
const WORLD_Y = new THREE.Vector3(0, 1, 0);

function makeEarthTexture(): THREE.CanvasTexture {
  const w = 1024;
  const h = 512;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const g = c.getContext("2d")!;
  const img = g.createImageData(w, h);
  for (let y = 0; y < h; y++) {
    const lat = (y / h) * Math.PI;
    for (let x = 0; x < w; x++) {
      const lon = (x / w) * Math.PI * 2;
      const n =
        Math.sin(lon * 3.0 + 0.4) * Math.sin(lat * 2.2) +
        0.45 * Math.sin(lon * 7.0 + lat * 3.0) +
        0.25 * Math.sin(lon * 13.0 - lat * 8.0) +
        0.35 * Math.cos(lat * 6.0);
      const ice = lat < 0.22 || lat > Math.PI - 0.22;
      const land = n > 0.18;
      let r: number, gg: number, b: number;
      if (ice) {
        r = 210; gg = 220; b = 230;
      } else if (land) {
        r = 46 + 30 * n;
        gg = 78 + 40 * n;
        b = 42;
      } else {
        r = 8;
        gg = 28 + 20 * Math.sin(lat);
        b = 78 + 30 * Math.sin(lon);
      }
      const i = (y * w + x) * 4;
      img.data[i] = r;
      img.data[i + 1] = gg;
      img.data[i + 2] = b;
      img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function makeStars(): THREE.Points {
  const n = 4000;
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 4.2e7;
    const a = Math.random() * Math.PI * 2;
    const b = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(b) * Math.cos(a);
    pos[i * 3 + 1] = r * Math.cos(b);
    pos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(
    geo,
    new THREE.PointsMaterial({ color: 0xdef4ff, size: 18_000, sizeAttenuation: true }),
  );
}

function readEcef(v: ArrayLike<number> | undefined, out: THREE.Vector3) {
  if (!v || v.length < 3) return out.set(0, 0, 0);
  return out.set(Number(v[0]), Number(v[2]), -Number(v[1]));
}

function asNums(v: ArrayLike<number> | undefined | null): number[] {
  if (!v) return [];
  if (Array.isArray(v)) return v.map(Number);
  return Array.from(v as ArrayLike<number>, Number);
}

function makeGridFin(): { root: THREE.Group; act: THREE.Group } {
  const root = new THREE.Group();
  const act = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x9aa3ad,
    metalness: 0.72,
    roughness: 0.28,
  });
  const w = 1.5;
  const h = 1.7;
  const n = 5;
  const frameTop = new THREE.Mesh(new THREE.BoxGeometry(w, 0.07, 0.08), mat);
  frameTop.position.y = h / 2;
  const frameBot = new THREE.Mesh(new THREE.BoxGeometry(w, 0.07, 0.08), mat);
  frameBot.position.y = -h / 2;
  const frameR = new THREE.Mesh(new THREE.BoxGeometry(0.07, h, 0.08), mat);
  frameR.position.x = w / 2;
  const frameL = new THREE.Mesh(new THREE.BoxGeometry(0.07, h, 0.08), mat);
  frameL.position.x = -w / 2;
  act.add(frameTop, frameBot, frameR, frameL);
  for (let i = 1; i < n; i++) {
    const x = -w / 2 + (i / n) * w;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.045, h - 0.08, 0.045), mat);
    bar.position.set(x, 0, 0);
    act.add(bar);
  }
  for (let j = 1; j < n; j++) {
    const y = -h / 2 + (j / n) * h;
    const bar = new THREE.Mesh(new THREE.BoxGeometry(w - 0.08, 0.045, 0.045), mat);
    bar.position.set(0, y, 0);
    act.add(bar);
  }
  root.add(act);
  return { root, act };
}

function fitRgb(f: number, lo: number, hi: number, success: boolean, best: boolean): [number, number, number] {
  if (success) return best ? [0.55, 1.0, 0.72] : [0.28, 0.95, 0.62];
  const span = Math.max(1, hi - lo);
  const t = Math.max(0, Math.min(1, (f - lo) / span));
  if (t < 0.5) {
    const u = t * 2;
    return [0.95, 0.28 + 0.45 * u, 0.32 + 0.08 * u];
  }
  const u = (t - 0.5) * 2;
  return [0.95 - 0.55 * u, 0.73 + 0.15 * u, 0.4 + 0.6 * u];
}

type MapPath = { lat: number[]; lon: number[]; f: number; ok: boolean; best: boolean };

function glowSprite(inner: string, mid: string): THREE.Sprite {
  const mark = document.createElement("canvas");
  mark.width = mark.height = 64;
  const mg = mark.getContext("2d")!;
  const grd = mg.createRadialGradient(32, 32, 2, 32, 32, 30);
  grd.addColorStop(0, inner);
  grd.addColorStop(0.3, mid);
  grd.addColorStop(1, "rgba(126,224,255,0)");
  mg.fillStyle = grd;
  mg.fillRect(0, 0, 64, 64);
  return new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(mark), transparent: true, depthWrite: false }),
  );
}

function makeSparkTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 30);
  grd.addColorStop(0, "rgba(255,255,240,1)");
  grd.addColorStop(0.18, "rgba(255,210,90,0.95)");
  grd.addColorStop(0.45, "rgba(255,110,30,0.45)");
  grd.addColorStop(1, "rgba(20,0,0,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeLabel(text: string, color: string): THREE.Sprite {
  const c = document.createElement("canvas");
  c.width = 256;
  c.height = 64;
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, 256, 64);
  g.font = "700 34px ui-monospace, monospace";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.lineWidth = 8;
  g.strokeStyle = "rgba(4, 8, 16, 0.78)";
  g.strokeText(text, 128, 34);
  g.fillStyle = color;
  g.fillText(text, 128, 34);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }),
  );
}

export class SceneApp {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(55, 1, 20, 1.2e8);
  camMode: CamMode = "pad";

  private earth: THREE.Mesh;
  private atmo: THREE.Mesh;
  private rocket = new THREE.Group();
  private finActuators: THREE.Group[] = [];
  private rcsJets: THREE.Mesh[] = [];
  private plume: THREE.Mesh;
  private plasma: THREE.Mesh;
  private boomSparks: THREE.Sprite[] = [];
  private boomFlash: THREE.Sprite;
  private boomPos = new Float32Array(BOOM_N * 3);
  private boomVel = new Float32Array(BOOM_N * 3);
  private boomSize = new Float32Array(BOOM_N);
  private boomLife = 0;
  private boomSpent = false;
  private trail: THREE.Line;
  private trailPos = new Float32Array(TRAIL_MAX * 3);
  private trailLen = 0;
  private trailLat: number[] = [];
  private trailLon: number[] = [];
  private pad: THREE.Group;
  private sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
  private camReady = false;
  private marker: THREE.Sprite;
  private padMark: THREE.Sprite;
  private startMark: THREE.Sprite;
  private padLabel: THREE.Sprite;
  private startLabel: THREE.Sprite;
  private velArrow: THREE.ArrowHelper;
  private padLine: THREE.Line;
  private startLine: THREE.Line;
  private slamLine: THREE.Line;
  private slamPos = new Float32Array(20 * 3);
  private yaw = 0.35;
  private pitch = 0.36;
  private zoom = 1;
  private dragging = false;
  private lastPx = 0;
  private lastPy = 0;
  private startLatched = false;
  private lastT = -1;
  private swarmLive: THREE.LineSegments;
  private swarmPrev: THREE.LineSegments;
  private pathSparks: THREE.Sprite[] = [];
  private pathSparkLife = new Float32Array(PATH_POOL);
  private pathSparkVel = new Float32Array(PATH_POOL * 3);
  private pathSparkSize = new Float32Array(PATH_POOL);
  private pathSparkCursor = 0;
  private liveEndCount = 0;
  private mapLive: MapPath[] = [];
  private mapPrev: MapPath[] = [];
  private vizStamp = -1;
  private readonly _pos = new THREE.Vector3();
  private readonly _pad = new THREE.Vector3();
  private readonly _localUp = new THREE.Vector3();
  private readonly _padUp = new THREE.Vector3();
  private readonly _vel = new THREE.Vector3();
  private readonly _start = new THREE.Vector3();
  private readonly _back = new THREE.Vector3();
  private readonly _right = new THREE.Vector3();
  private readonly _cam = new THREE.Vector3();
  private readonly _fwd = new THREE.Vector3();
  private readonly _up = new THREE.Vector3();
  private readonly _tmp = new THREE.Vector3();
  private readonly _center = new THREE.Vector3();
  private readonly _along = new THREE.Vector3();
  private readonly _qBody = new THREE.Quaternion();
  private readonly _qMap = new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0, -1),
      new THREE.Vector3(0, 1, 0),
    ),
  );
  private readonly _qBodyToThree = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1),
    -Math.PI / 2,
  );

  constructor(canvas: HTMLCanvasElement) {
    const opts = { canvas, antialias: true, failIfMajorPerformanceCaveat: false as const };
    try {
      this.renderer = new THREE.WebGLRenderer({ ...opts, logarithmicDepthBuffer: true });
    } catch {
      this.renderer = new THREE.WebGLRenderer(opts);
    }
    this.renderer.setPixelRatio(Math.min(typeof devicePixelRatio === "number" ? devicePixelRatio : 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene.background = new THREE.Color(0x02040a);

    const earthMat = new THREE.MeshStandardMaterial({
      map: makeEarthTexture(),
      roughness: 0.82,
      metalness: 0.05,
    });
    this.earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 96, 64), earthMat);
    this.scene.add(this.earth);

    const atmoMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false,
      uniforms: {},
      vertexShader: `
        #include <common>
        #include <logdepthbuf_pars_vertex>
        varying vec3 vN;
        void main() {
          vN = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          #include <logdepthbuf_vertex>
        }`,
      fragmentShader: `
        #include <common>
        #include <logdepthbuf_pars_fragment>
        varying vec3 vN;
        void main() {
          float f = pow(1.0 - abs(vN.z), 2.2);
          gl_FragColor = vec4(0.25, 0.55, 1.0, 0.22 * f);
          #include <logdepthbuf_fragment>
        }`,
    });
    this.atmo = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R * 1.035, 64, 48), atmoMat);
    this.scene.add(this.atmo);

    this.scene.add(makeStars());
    this.sun.position.set(2e7, 8e6, 1.2e7);
    this.scene.add(this.sun);
    this.scene.add(new THREE.AmbientLight(0x8aa4c0, 0.55));
    this.scene.add(new THREE.HemisphereLight(0xa8d4ff, 0x1a2418, 0.4));

    this.pad = this.buildPad();
    this.scene.add(this.pad);

    this.buildRocket();
    this.scene.add(this.rocket);

    this.plume = new THREE.Mesh(
      new THREE.ConeGeometry(2.4, 28, 12, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x7ecbff, transparent: true, opacity: 0.75, side: THREE.DoubleSide }),
    );
    this.plume.rotation.x = Math.PI;
    this.plume.position.y = -38;
    this.rocket.add(this.plume);

    this.plasma = new THREE.Mesh(
      new THREE.SphereGeometry(8, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xff7a3a, transparent: true, opacity: 0.0 }),
    );
    this.plasma.position.y = -18;
    this.rocket.add(this.plasma);

    const tgeo = new THREE.BufferGeometry();
    tgeo.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    tgeo.setDrawRange(0, 0);
    this.trail = new THREE.Line(
      tgeo,
      new THREE.LineBasicMaterial({ color: 0x7ee0ff, transparent: true, opacity: 0.65 }),
    );
    this.scene.add(this.trail);

    this.marker = glowSprite("rgba(255,240,200,1)", "rgba(126,224,255,0.85)");
    this.scene.add(this.marker);
    this.padMark = glowSprite("rgba(255,210,80,1)", "rgba(255,180,40,0.8)");
    this.scene.add(this.padMark);
    this.startMark = glowSprite("rgba(80,230,255,1)", "rgba(40,180,255,0.85)");
    this.scene.add(this.startMark);
    this.padLabel = makeLabel("LZ-1", "#ffd266");
    this.startLabel = makeLabel("START", "#7ee0ff");
    this.scene.add(this.padLabel);
    this.scene.add(this.startLabel);

    this.velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 80, 0x7ee0ff, 18, 10);
    this.scene.add(this.velArrow);

    const pgeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.padLine = new THREE.Line(
      pgeo,
      new THREE.LineBasicMaterial({ color: 0xffcc66, transparent: true, opacity: 0.62 }),
    );
    this.scene.add(this.padLine);
    this.startLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x5ec8ff, transparent: true, opacity: 0.32 }),
    );
    this.scene.add(this.startLine);

    const slamGeo = new THREE.BufferGeometry();
    slamGeo.setAttribute("position", new THREE.BufferAttribute(this.slamPos, 3));
    slamGeo.setDrawRange(0, 0);
    this.slamLine = new THREE.Line(
      slamGeo,
      new THREE.LineDashedMaterial({
        color: 0x5dffb2,
        transparent: true,
        opacity: 0.7,
        dashSize: 28,
        gapSize: 18,
      }),
    );
    this.scene.add(this.slamLine);

    this.swarmPrev = this.makeSwarmLines(0.22);
    this.swarmLive = this.makeSwarmLines(0.9);
    this.scene.add(this.swarmPrev);
    this.scene.add(this.swarmLive);

    const sparkTex = makeSparkTexture();
    this.pathSparks = [];
    for (let i = 0; i < PATH_POOL; i++) {
      const spark = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: sparkTex,
          color: 0xffffff,
          transparent: true,
          opacity: 0,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      );
      spark.visible = false;
      this.pathSparks.push(spark);
      this.scene.add(spark);
    }
    this.boomSparks = [];
    for (let i = 0; i < BOOM_N; i++) {
      const spark = new THREE.Sprite(
        new THREE.SpriteMaterial({
          map: sparkTex,
          color: 0xffeeaa,
          transparent: true,
          opacity: 0,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
        }),
      );
      spark.visible = false;
      this.boomSparks.push(spark);
      this.scene.add(spark);
    }
    this.boomFlash = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: sparkTex,
        color: 0xffc070,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.boomFlash.visible = false;
    this.scene.add(this.boomFlash);

    this.bindNav(canvas);
    this.resize();
    window.addEventListener("resize", () => this.resize());
    window.addEventListener("keydown", (e) => this.onKey(e));
  }

  private bindNav(el: HTMLCanvasElement) {
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      this.lastPx = e.clientX;
      this.lastPy = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPx;
      const dy = e.clientY - this.lastPy;
      this.lastPx = e.clientX;
      this.lastPy = e.clientY;
      const sens = 0.005;
      this.yaw -= dx * sens;
      this.pitch = THREE.MathUtils.clamp(this.pitch + dy * sens, -0.2, 1.32);
    });
    const endDrag = () => {
      this.dragging = false;
    };
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const f = Math.exp(e.deltaY * 0.00115);
        this.zoom = THREE.MathUtils.clamp(this.zoom * f, 0.18, 14);
      },
      { passive: false },
    );
    el.addEventListener("dblclick", () => this.resetLook());
  }

  private onKey(e: KeyboardEvent) {
    const tag = (e.target as HTMLElement | null)?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    const step = 0.07;
    if (e.key === "ArrowLeft") {
      this.yaw += step;
    } else if (e.key === "ArrowRight") {
      this.yaw -= step;
    } else if (e.key === "ArrowUp") {
      this.pitch = THREE.MathUtils.clamp(this.pitch - step, -0.2, 1.32);
    } else if (e.key === "ArrowDown") {
      this.pitch = THREE.MathUtils.clamp(this.pitch + step, -0.2, 1.32);
    } else if (e.key === "=" || e.key === "+") {
      this.zoom = THREE.MathUtils.clamp(this.zoom * 0.88, 0.18, 14);
    } else if (e.key === "-" || e.key === "_") {
      this.zoom = THREE.MathUtils.clamp(this.zoom * 1.14, 0.18, 14);
    } else if (e.key === "Home" || e.key === "0") {
      this.resetLook();
    }
  }

  setCamMode(mode: CamMode) {
    if (this.camMode === mode) return;
    this.camMode = mode;
    this.camReady = false;
    this.resetLook();
  }

  private resetLook() {
    this.yaw = this.camMode === "pad" ? 0.35 : this.camMode === "chase" ? 0.12 : 0.2;
    this.pitch = this.camMode === "chase" ? 0.28 : this.camMode === "pad" ? 0.36 : 0.4;
    this.zoom = 1;
  }

  private buildPad(): THREE.Group {
    const g = new THREE.Group();
    const deck = new THREE.Mesh(
      new THREE.CylinderGeometry(45, 48, 3, 32),
      new THREE.MeshStandardMaterial({ color: 0x8a8f96, roughness: 0.7 }),
    );
    g.add(deck);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(18, 22, 32),
      new THREE.MeshBasicMaterial({ color: 0xffcc33, side: THREE.DoubleSide }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 1.7;
    g.add(ring);
    const x = new THREE.Mesh(
      new THREE.PlaneGeometry(70, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }),
    );
    x.rotation.x = -Math.PI / 2;
    x.position.y = 1.65;
    g.add(x);
    const apron = new THREE.Mesh(
      new THREE.RingGeometry(52, 140, 48),
      new THREE.MeshBasicMaterial({
        color: 0xffcc66,
        transparent: true,
        opacity: 0.2,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    apron.rotation.x = -Math.PI / 2;
    apron.position.y = 1.35;
    g.add(apron);
    return g;
  }

  private buildRocket() {
    const white = new THREE.MeshStandardMaterial({
      color: 0xf2f4f7,
      metalness: 0.28,
      roughness: 0.42,
      emissive: 0x3a4150,
      emissiveIntensity: 0.35,
    });
    const black = new THREE.MeshStandardMaterial({ color: 0x15181c, metalness: 0.5, roughness: 0.45 });
    const soot = new THREE.MeshStandardMaterial({ color: 0x2a241c, metalness: 0.2, roughness: 0.7 });
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1.83, 1.83, 42, 24), white);
    body.position.y = 2;
    this.rocket.add(body);
    const inter = new THREE.Mesh(new THREE.CylinderGeometry(1.83, 1.7, 5, 24), black);
    inter.position.y = 25.5;
    this.rocket.add(inter);
    const lower = new THREE.Mesh(new THREE.CylinderGeometry(1.83, 1.83, 10, 24), soot);
    lower.position.y = -14;
    this.rocket.add(lower);
    this.finActuators = [];
    for (let i = 0; i < 4; i++) {
      const { root, act } = makeGridFin();
      const a = (i * Math.PI) / 2 + Math.PI / 4;
      root.position.set(Math.cos(a) * 2.15, 21, Math.sin(a) * 2.15);
      root.lookAt(0, 21, 0);
      this.rocket.add(root);
      this.finActuators.push(act);
    }
    this.rcsJets = [];
    const rcsMat = new THREE.MeshBasicMaterial({
      color: 0xb8e4ff,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    for (let i = 0; i < 4; i++) {
      const jet = new THREE.Mesh(new THREE.ConeGeometry(0.18, 1.4, 6, 1, true), rcsMat.clone());
      const a = (i * Math.PI) / 2;
      jet.position.set(Math.cos(a) * 2.0, 22.4, Math.sin(a) * 2.0);
      jet.lookAt(Math.cos(a) * 6, 22.4, Math.sin(a) * 6);
      this.rocket.add(jet);
      this.rcsJets.push(jet);
    }
    for (let i = 0; i < 4; i++) {
      const leg = new THREE.Mesh(
        new THREE.CylinderGeometry(0.08, 0.08, 9, 6),
        new THREE.MeshStandardMaterial({ color: 0xc9cdd2, metalness: 0.6 }),
      );
      const a = (i * Math.PI) / 2;
      leg.position.set(Math.cos(a) * 2.4, -20, Math.sin(a) * 2.4);
      leg.rotation.z = Math.cos(a) * 0.55;
      leg.rotation.x = -Math.sin(a) * 0.55;
      this.rocket.add(leg);
    }
    for (let i = 0; i < 9; i++) {
      const bell = new THREE.Mesh(
        new THREE.ConeGeometry(0.45, 1.6, 10),
        new THREE.MeshStandardMaterial({ color: 0x6b4a2a, metalness: 0.7, roughness: 0.25 }),
      );
      const ang = (i / 8) * Math.PI * 2;
      const rad = i === 0 ? 0 : 1.15;
      bell.position.set(Math.cos(ang) * rad, -22.2, Math.sin(ang) * rad);
      bell.rotation.x = Math.PI;
      this.rocket.add(bell);
    }
  }

  resize() {
    const w = innerWidth;
    const h = innerHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  apply(s: Snapshot) {
    readEcef(s.r_ecef, this._pos);
    const [pw, px, py, pz] = s.quat_ecef;
    this._qBody.set(px, py, pz, pw);
    this.rocket.position.copy(this._pos);
    this.rocket.quaternion.copy(this._qMap).multiply(this._qBody).multiply(this._qBodyToThree);
    this.marker.position.copy(this._pos);

    if (!this.startLatched || s.t + 0.25 < this.lastT) {
      this._start.copy(this._pos);
      this.startLatched = true;
      this.trailLen = 0;
      this.trailLat = [];
      this.trailLon = [];
      this.trail.geometry.setDrawRange(0, 0);
    }
    this.lastT = s.t;
    this.startMark.position.copy(this._start);

    readEcef(s.pad_ecef, this._pad);
    this.pad.position.copy(this._pad);
    this._padUp.copy(this._pad).normalize();
    this.pad.quaternion.setFromUnitVectors(this._tmp.set(0, 1, 0), this._padUp);
    this.padMark.position.copy(this._pad);

    this.earth.rotation.y = 0;

    this.plume.visible = s.throttle > 0.05 && s.intact;
    (this.plume.material as THREE.MeshBasicMaterial).opacity = 0.35 + 0.55 * s.throttle;
    this.plume.scale.setScalar((0.7 + s.throttle * 1.4) * Math.max(1, Math.sqrt(s.n_engines || 1)));

    const deltas = s.fin_delta && s.fin_delta.length >= 4 ? s.fin_delta : [0, 0, 0, 0];
    for (let i = 0; i < this.finActuators.length; i++) {
      this.finActuators[i].rotation.y = Number(deltas[i]) || 0;
    }
    const rcs = Math.min(1, Math.abs(Number(s.rcs) || 0));
    for (const jet of this.rcsJets) {
      const mat = jet.material as THREE.MeshBasicMaterial;
      mat.opacity = s.intact ? rcs * 0.7 : 0;
      jet.scale.setScalar(0.6 + rcs * 1.8);
    }

    const heat = Math.min(1, s.heat / 8e6);
    (this.plasma.material as THREE.MeshBasicMaterial).opacity = s.intact ? heat * 0.55 : 0;
    this.plasma.scale.setScalar(1 + heat * 3);

    this.rocket.visible = s.intact;
    if (s.intact) {
      this.boomSpent = false;
      if (this.boomFlash.visible || this.boomLife > 0) this.clearBoom();
    }

    if (s.intact) {
      if (this.trailLen < TRAIL_MAX) {
        const i = this.trailLen * 3;
        this.trailPos[i] = this._pos.x;
        this.trailPos[i + 1] = this._pos.y;
        this.trailPos[i + 2] = this._pos.z;
        this.trailLen++;
        this.trailLat.push(s.lat);
        this.trailLon.push(s.lon);
      } else {
        this.trailPos.copyWithin(0, 3);
        const i = (TRAIL_MAX - 1) * 3;
        this.trailPos[i] = this._pos.x;
        this.trailPos[i + 1] = this._pos.y;
        this.trailPos[i + 2] = this._pos.z;
        this.trailLat.shift();
        this.trailLon.shift();
        this.trailLat.push(s.lat);
        this.trailLon.push(s.lon);
      }
      const attr = this.trail.geometry.getAttribute("position") as THREE.BufferAttribute;
      attr.needsUpdate = true;
      this.trail.geometry.setDrawRange(0, this.trailLen);
    }

    readEcef(s.v_ground, this._vel);
    if (!s.intact && !this.boomSpent) {
      this.spawnBoom(this._pos, this._vel);
      this.boomSpent = true;
    }
    const vlen = this._vel.length();
    if (vlen > 2) {
      this.velArrow.position.copy(this._pos);
      this.velArrow.setDirection(this._tmp.copy(this._vel).multiplyScalar(1 / vlen));
    }
    this.velArrow.visible = false;

    const linePos = this.padLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    linePos.setXYZ(0, this._pos.x, this._pos.y, this._pos.z);
    linePos.setXYZ(1, this._pad.x, this._pad.y, this._pad.z);
    linePos.needsUpdate = true;
    this.padLine.visible = true;

    const startPos = this.startLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    startPos.setXYZ(0, this._start.x, this._start.y, this._start.z);
    startPos.setXYZ(1, this._pad.x, this._pad.y, this._pad.z);
    startPos.needsUpdate = true;

    const slam = s.slam_xyz;
    const slamAttr = this.slamLine.geometry.getAttribute("position") as THREE.BufferAttribute;
    if (slam && slam.length >= 6) {
      const n = Math.min(this.slamPos.length, slam.length);
      for (let i = 0; i < n; i++) this.slamPos[i] = Number(slam[i]);
      slamAttr.needsUpdate = true;
      this.slamLine.geometry.setDrawRange(0, Math.floor(n / 3));
      this.slamLine.computeLineDistances();
      this.slamLine.visible = true;
    } else {
      this.slamLine.visible = false;
    }

    this.updateCamera(s);
  }

  resetTrail() {
    this.trailLen = 0;
    this.trailLat = [];
    this.trailLon = [];
    this.trail.geometry.setDrawRange(0, 0);
    this.clearBoom();
    this.boomSpent = false;
    this.camReady = false;
    this.startLatched = false;
    this.lastT = -1;
    this.clearSwarm();
  }

  clearSwarm() {
    this.vizStamp = -1;
    this.mapLive = [];
    this.mapPrev = [];
    this.swarmLive.geometry.setDrawRange(0, 0);
    this.swarmPrev.geometry.setDrawRange(0, 0);
    this.liveEndCount = 0;
  }

  private makeSwarmLines(opacity: number): THREE.LineSegments {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(6), 3));
    geo.setDrawRange(0, 0);
    return new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity,
        depthWrite: false,
      }),
    );
  }

  applySwarm(viz: GenerationViz | null | undefined) {
    if (!viz) return;
    const stamp = Number(viz.stamp);
    if (stamp === this.vizStamp) return;
    this.vizStamp = stamp;
    const liveN = Number(viz.live?.n) || 0;
    if (liveN < this.liveEndCount) this.liveEndCount = 0;
    this.mapPrev = this.packToLines(viz.prev, this.swarmPrev, false);
    this.mapLive = this.packToLines(viz.live, this.swarmLive, true);
  }

  private packToLines(pack: PackedTrails | undefined, lines: THREE.LineSegments, ends: boolean): MapPath[] {
    const paths: MapPath[] = [];
    if (!pack || !pack.n) {
      lines.geometry.setDrawRange(0, 0);
      return paths;
    }
    const xyz = asNums(pack.xyz);
    const lat = asNums(pack.lat);
    const lon = asNums(pack.lon);
    const counts = asNums(pack.counts);
    const fits = asNums(pack.fitnesses);
    const n = Math.min(Number(pack.n) || counts.length, counts.length, fits.length);
    const bestIdx = Number(pack.best_idx);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const f = fits[i];
      if (!Number.isFinite(f)) continue;
      if (f < lo) lo = f;
      if (f > hi) hi = f;
    }
    if (!Number.isFinite(lo)) {
      lo = 0;
      hi = 1;
    }
    const terms = pack.terms || [];
    const maxSegs = 16000;
    const pos = new Float32Array(maxSegs * 6);
    const col = new Float32Array(maxSegs * 6);
    const endPos = new Float32Array(n * 3);
    const endCol = new Float32Array(n * 3);
    let segs = 0;
    let cursor = 0;
    let endN = 0;
    for (let i = 0; i < n; i++) {
      const c = Math.max(0, Math.floor(counts[i] || 0));
      const ok = terms[i] === "landed";
      const best = i === bestIdx;
      const [r, g, b] = fitRgb(fits[i], lo, hi, ok, best);
      const plat: number[] = [];
      const plon: number[] = [];
      for (let k = 0; k < c; k++) {
        const li = cursor + k;
        if (li < lat.length) plat.push(lat[li]);
        if (li < lon.length) plon.push(lon[li]);
      }
      paths.push({ lat: plat, lon: plon, f: fits[i], ok, best });
      for (let k = 0; k < c - 1 && segs < maxSegs; k++) {
        const a = (cursor + k) * 3;
        const bidx = (cursor + k + 1) * 3;
        if (bidx + 2 >= xyz.length) break;
        const o = segs * 6;
        pos[o] = xyz[a];
        pos[o + 1] = xyz[a + 2];
        pos[o + 2] = -xyz[a + 1];
        pos[o + 3] = xyz[bidx];
        pos[o + 4] = xyz[bidx + 2];
        pos[o + 5] = -xyz[bidx + 1];
        const boost = best ? 1.15 : 1;
        for (let t = 0; t < 6; t += 3) {
          col[o + t] = Math.min(1, r * boost);
          col[o + t + 1] = Math.min(1, g * boost);
          col[o + t + 2] = Math.min(1, b * boost);
        }
        segs++;
      }
      if (c > 0 && ends) {
        const last = (cursor + c - 1) * 3;
        if (last + 2 < xyz.length) {
          const e = endN * 3;
          endPos[e] = xyz[last];
          endPos[e + 1] = xyz[last + 2];
          endPos[e + 2] = -xyz[last + 1];
          endCol[e] = r;
          endCol[e + 1] = g;
          endCol[e + 2] = b;
          endN++;
        }
      }
      cursor += c;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setDrawRange(0, segs * 2);
    lines.geometry.dispose();
    lines.geometry = geo;
    if (ends) {
      for (let i = this.liveEndCount; i < endN; i++) {
        const e = i * 3;
        this.spawnPathBurst(endPos[e], endPos[e + 1], endPos[e + 2], endCol[e], endCol[e + 1], endCol[e + 2]);
      }
      this.liveEndCount = endN;
    }
    return paths;
  }

  private spawnPathBurst(x: number, y: number, z: number, r: number, g: number, b: number) {
    for (let s = 0; s < PATH_SPARKS; s++) {
      const i = this.pathSparkCursor;
      this.pathSparkCursor = (this.pathSparkCursor + 1) % PATH_POOL;
      const a = Math.random() * Math.PI * 2;
      const polar = Math.acos(2 * Math.random() - 1);
      const speed = 10 + Math.random() * 38;
      const i3 = i * 3;
      this.pathSparkVel[i3] = Math.sin(polar) * Math.cos(a) * speed;
      this.pathSparkVel[i3 + 1] = Math.cos(polar) * speed;
      this.pathSparkVel[i3 + 2] = Math.sin(polar) * Math.sin(a) * speed;
      this.pathSparkLife[i] = PATH_LIFE * (0.75 + Math.random() * 0.35);
      this.pathSparkSize[i] = s === 0 ? 7 + Math.random() * 5 : 1.6 + Math.random() * 3.2;
      const spark = this.pathSparks[i];
      const hot = 0.65 + Math.random() * 0.35;
      (spark.material as THREE.SpriteMaterial).color.setRGB(r * hot, g * hot, b * hot);
      spark.position.set(x + (Math.random() - 0.5) * 3, y + (Math.random() - 0.5) * 4, z + (Math.random() - 0.5) * 3);
      spark.scale.setScalar(this.pathSparkSize[i]);
      spark.visible = true;
    }
  }

  private tickPathSparks(dt: number) {
    const drag = Math.exp(-3.4 * dt);
    const grav = 18 * dt;
    for (let i = 0; i < PATH_POOL; i++) {
      if (this.pathSparkLife[i] <= 0) continue;
      this.pathSparkLife[i] -= dt;
      const spark = this.pathSparks[i];
      if (this.pathSparkLife[i] <= 0) {
        spark.visible = false;
        (spark.material as THREE.SpriteMaterial).opacity = 0;
        continue;
      }
      this._tmp.copy(spark.position).normalize();
      const i3 = i * 3;
      this.pathSparkVel[i3] = this.pathSparkVel[i3] * drag - this._tmp.x * grav;
      this.pathSparkVel[i3 + 1] = this.pathSparkVel[i3 + 1] * drag - this._tmp.y * grav;
      this.pathSparkVel[i3 + 2] = this.pathSparkVel[i3 + 2] * drag - this._tmp.z * grav;
      spark.position.x += this.pathSparkVel[i3] * dt;
      spark.position.y += this.pathSparkVel[i3 + 1] * dt;
      spark.position.z += this.pathSparkVel[i3 + 2] * dt;
      const u = this.pathSparkLife[i] / PATH_LIFE;
      (spark.material as THREE.SpriteMaterial).opacity = u * u;
      spark.scale.setScalar(this.pathSparkSize[i] * (0.7 + 0.55 * (1 - u)));
    }
  }

  private spawnBoom(pos: THREE.Vector3, vel: THREE.Vector3) {
    for (let i = 0; i < BOOM_N; i++) {
      const i3 = i * 3;
      const a = Math.random() * Math.PI * 2;
      const b = Math.acos(2 * Math.random() - 1);
      const speed = 14 + Math.random() * 55;
      this.boomPos[i3] = pos.x + (Math.random() - 0.5) * 5;
      this.boomPos[i3 + 1] = pos.y + (Math.random() - 0.5) * 7;
      this.boomPos[i3 + 2] = pos.z + (Math.random() - 0.5) * 5;
      this.boomVel[i3] = vel.x * 0.1 + Math.sin(b) * Math.cos(a) * speed;
      this.boomVel[i3 + 1] = vel.y * 0.1 + Math.cos(b) * speed;
      this.boomVel[i3 + 2] = vel.z * 0.1 + Math.sin(b) * Math.sin(a) * speed;
      this.boomSize[i] = 2.2 + Math.random() * 4.5;
      const spark = this.boomSparks[i];
      const hot = Math.random();
      (spark.material as THREE.SpriteMaterial).color.setRGB(1, 0.5 + 0.5 * hot, 0.16 + 0.4 * hot);
      spark.position.set(this.boomPos[i3], this.boomPos[i3 + 1], this.boomPos[i3 + 2]);
      spark.scale.setScalar(this.boomSize[i]);
      spark.visible = true;
    }
    this.boomLife = BOOM_LIFE;
    this.boomFlash.position.copy(pos);
    this.boomFlash.visible = true;
    this.boomFlash.scale.setScalar(28);
    (this.boomFlash.material as THREE.SpriteMaterial).opacity = 0.9;
  }

  private clearBoom() {
    this.boomLife = 0;
    this.boomFlash.visible = false;
    (this.boomFlash.material as THREE.SpriteMaterial).opacity = 0;
    for (const spark of this.boomSparks) {
      spark.visible = false;
      (spark.material as THREE.SpriteMaterial).opacity = 0;
    }
  }

  tickDebris(dt: number) {
    this.tickPathSparks(dt);
    if (this.boomLife <= 0) {
      if (this.boomFlash.visible) this.clearBoom();
      return;
    }
    this.boomLife -= dt;
    const u = Math.max(0, this.boomLife / BOOM_LIFE);
    this._tmp.copy(this.rocket.position).normalize();
    const drag = Math.exp(-3.2 * dt);
    const g = 22 * dt;
    for (let i = 0; i < BOOM_N; i++) {
      const i3 = i * 3;
      this.boomVel[i3] = this.boomVel[i3] * drag - this._tmp.x * g;
      this.boomVel[i3 + 1] = this.boomVel[i3 + 1] * drag - this._tmp.y * g;
      this.boomVel[i3 + 2] = this.boomVel[i3 + 2] * drag - this._tmp.z * g;
      this.boomPos[i3] += this.boomVel[i3] * dt;
      this.boomPos[i3 + 1] += this.boomVel[i3 + 1] * dt;
      this.boomPos[i3 + 2] += this.boomVel[i3 + 2] * dt;
      const spark = this.boomSparks[i];
      spark.position.set(this.boomPos[i3], this.boomPos[i3 + 1], this.boomPos[i3 + 2]);
      const fade = u * u;
      (spark.material as THREE.SpriteMaterial).opacity = fade;
      spark.scale.setScalar(this.boomSize[i] * (0.65 + 0.7 * (1 - u)));
    }
    const flash = this.boomFlash.material as THREE.SpriteMaterial;
    const flashU = Math.max(0, (this.boomLife - (BOOM_LIFE - 0.2)) / 0.2);
    flash.opacity = flashU * 0.9;
    this.boomFlash.scale.setScalar(18 + 40 * (1 - flashU));
    this.boomFlash.position.copy(this.rocket.position);
    if (this.boomLife <= 0) this.clearBoom();
  }

  private lookAtSafe(target: THREE.Vector3, up: THREE.Vector3) {
    this._fwd.copy(target).sub(this.camera.position);
    if (this._fwd.lengthSq() < 1e-6) return;
    this._fwd.normalize();
    this._up.copy(up).normalize();
    if (Math.abs(this._fwd.dot(this._up)) > 0.97) {
      this._up.crossVectors(this._fwd, WORLD_Y);
      if (this._up.lengthSq() < 1e-8) this._up.crossVectors(this._fwd, WORLD_X);
      this._up.normalize();
    }
    this.camera.up.copy(this._up);
    this.camera.lookAt(target);
  }

  private placeCam(desired: THREE.Vector3, snap = false, k = 0.22) {
    if (snap || !this.camReady || this.camera.position.distanceToSquared(desired) > 2.5e13) {
      this.camera.position.copy(desired);
    } else {
      this.camera.position.lerp(desired, k);
    }
  }

  private flatten(v: THREE.Vector3, up: THREE.Vector3, out: THREE.Vector3) {
    out.copy(v).addScaledVector(up, -v.dot(up));
    return out;
  }

  private eastOf(up: THREE.Vector3, out: THREE.Vector3) {
    out.crossVectors(WORLD_Y, up);
    if (out.lengthSq() < 1e-10) out.crossVectors(WORLD_X, up);
    out.normalize();
    return out;
  }

  private liftAboveGround(p: THREE.Vector3, minH: number) {
    const r = p.length();
    const need = EARTH_R + minH;
    if (r > 1 && r < need) p.multiplyScalar(need / r);
  }

  private frameDistance(target: THREE.Vector3, pts: THREE.Vector3[]): number {
    const vHalf = Math.tan(THREE.MathUtils.degToRad(this.camera.fov * 0.5));
    const hHalf = vHalf * Math.max(this.camera.aspect, 0.5);
    const half = Math.min(vHalf, hHalf) * 0.72;
    let m = 50;
    for (const p of pts) {
      const d = p.distanceTo(target);
      if (d > m) m = d;
    }
    return (m / Math.max(half, 0.14)) * 1.7;
  }

  private orbitCam(target: THREE.Vector3, up: THREE.Vector3, along: THREE.Vector3, dist: number) {
    this._along.copy(along);
    this.flatten(this._along, up, this._along);
    if (this._along.lengthSq() < 1e-10) this.eastOf(up, this._along);
    else this._along.normalize();
    this._right.crossVectors(up, this._along);
    if (this._right.lengthSq() < 1e-10) this.eastOf(up, this._right);
    else this._right.normalize();
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    this._back.copy(this._along).multiplyScalar(cy).addScaledVector(this._right, sy);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    this._cam.copy(target).addScaledVector(this._back, dist * cp).addScaledVector(up, dist * sp);
    this.liftAboveGround(this._cam, 40);
    if (this._cam.distanceToSquared(target) < 400) {
      this._cam.copy(target).addScaledVector(up, Math.max(dist, 80));
      this.liftAboveGround(this._cam, 40);
    }
  }

  private updateCamera(s: Snapshot) {
    const pos = this._pos;
    const pad = this._pad;
    this._localUp.copy(pos).normalize();
    const padUp = this._padUp;

    this.pitch = THREE.MathUtils.clamp(this.pitch, this.camMode === "chase" ? -0.18 : 0.05, 1.28);
    this.zoom = THREE.MathUtils.clamp(
      this.zoom,
      this.camMode === "chase" ? 0.4 : this.camMode === "orbit" ? 0.18 : 0.26,
      this.camMode === "chase" ? 14 : 10,
    );

    this.flatten(this._tmp.copy(pos).sub(pad), padUp, this._fwd);
    const approachLen = this._fwd.length();
    if (approachLen < 8) this.eastOf(padUp, this._fwd);
    else this._fwd.normalize();

    let lerp = 0.22;
    let up = padUp;
    let dist = 400;

    if (this.camMode === "chase") {
      lerp = 0.42;
      up = this._localUp;
      this._along.copy(this._fwd);
      this._center.copy(pos);
      if (approachLen > 30) {
        this._center.addScaledVector(this._fwd, -24);
      }
      const alt = Math.max(0, Number(s.engine_alt) || s.alt || 0);
      dist = THREE.MathUtils.clamp(200 + Math.min(alt, 2_000) * 0.04, 160, 280) * this.zoom;
    } else if (this.camMode === "pad") {
      this._center.copy(pad).lerp(pos, 0.38);
      if (this.startLatched) this._center.lerp(this._start, 0.16);
      const h = Math.abs(this._tmp.copy(pos).sub(pad).dot(padUp));
      this._center.addScaledVector(padUp, THREE.MathUtils.clamp(h * 0.1 + 28, 18, 140_000));
      this._along.crossVectors(padUp, this._fwd);
      if (this._along.lengthSq() < 1e-10) this.eastOf(padUp, this._along);
      else this._along.normalize();
      const pts = [pad, pos];
      if (this.startLatched) pts.push(this._start);
      dist = THREE.MathUtils.clamp(this.frameDistance(this._center, pts) * this.zoom, 140, 3.2e7);
    } else {
      this._center.copy(pad).add(pos);
      let n = 2;
      if (this.startLatched) {
        this._center.add(this._start);
        n++;
      }
      this._center.multiplyScalar(1 / n);
      const span = Math.max(
        pad.distanceTo(this._center),
        pos.distanceTo(this._center),
        this.startLatched ? this._start.distanceTo(this._center) : 0,
      );
      const global = span > EARTH_R * 0.7 || this._center.length() < EARTH_R * 0.45;
      if (global) {
        this._center.set(0, 0, 0);
        this._along.crossVectors(pad, pos);
        if (this._along.lengthSq() < 1e-6) this.eastOf(padUp, this._along);
        else this._along.normalize();
        dist = THREE.MathUtils.clamp(EARTH_R * 2.55 * this.zoom, EARTH_R * 1.22, EARTH_R * 14);
      } else {
        if (this._center.length() < EARTH_R + 120) {
          this._center.normalize().multiplyScalar(EARTH_R + 120);
        }
        this._along.crossVectors(padUp, this._fwd);
        if (this._along.lengthSq() < 1e-10) this.eastOf(padUp, this._along);
        else this._along.normalize();
        const pts = [pad, pos];
        if (this.startLatched) pts.push(this._start);
        const fitted = this.frameDistance(this._center, pts);
        dist = THREE.MathUtils.clamp(Math.max(fitted * 2.2, 22_000) * this.zoom, 8_000, 2.4e7);
      }
    }

    this.orbitCam(this._center, up, this._along, dist);
    this.placeCam(this._cam, !this.camReady || this.camMode === "chase", lerp);
    this.lookAtSafe(this._center, up);

    const distVeh = this.camera.position.distanceTo(pos);
    const distPad = this.camera.position.distanceTo(pad);
    const distStart = this.camera.position.distanceTo(this._start);
    const distSurf = Math.max(40, this.camera.position.length() - EARTH_R * 0.94);
    const near = THREE.MathUtils.clamp(Math.min(distVeh, distPad, distSurf) * 0.012, 1.2, 120_000);
    const far = Math.max(1.2e8, this.camera.position.length() * 10);
    if (Math.abs(this.camera.near - near) / near > 0.08 || Math.abs(this.camera.far - far) > 1e6) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }

    const vlen = this._vel.length();
    if (vlen > 2 && distVeh > 500) {
      const arrowLen = THREE.MathUtils.clamp(distVeh * 0.022, 30, 60_000);
      this.velArrow.setLength(arrowLen, arrowLen * 0.18, arrowLen * 0.1);
      this.velArrow.visible = true;
    } else {
      this.velArrow.visible = false;
    }

    this.marker.scale.setScalar(THREE.MathUtils.clamp(distVeh * 0.016, 8, 180_000));
    this.marker.visible = this.camMode !== "chase" && distVeh > 1_400;
    this.padMark.scale.setScalar(
      THREE.MathUtils.clamp(distPad * (this.camMode === "chase" ? 0.04 : 0.014), 10, 140_000),
    );
    this.padMark.visible = distPad > (this.camMode === "chase" ? 2_000 : 900);
    this.padLabel.visible = distPad > (this.camMode === "chase" ? 8_000 : 1_600);
    this.padLabel.position.copy(pad).addScaledVector(padUp, Math.max(30, distPad * 0.01));
    this.padLabel.scale.set(Math.max(40, distPad * 0.042), Math.max(10, distPad * 0.0105), 1);

    const startAway = this.startLatched && this._start.distanceTo(pos) > 25;
    const startPadSep = this._start.distanceTo(pad);
    this.startLine.visible = startAway && startPadSep / Math.max(distPad, 1) > 0.004;
    this.startMark.visible =
      this.camMode !== "chase" && startAway && distStart > 900 && startPadSep / Math.max(distStart, 1) > 0.004;
    this.startMark.scale.setScalar(THREE.MathUtils.clamp(distStart * 0.014, 10, 140_000));
    this._tmp.copy(this._start).normalize();
    this.startLabel.position.copy(this._start).addScaledVector(this._tmp, Math.max(30, distStart * 0.01));
    this.startLabel.scale.set(Math.max(40, distStart * 0.042), Math.max(10, distStart * 0.0105), 1);
    this.startLabel.visible =
      this.camMode !== "chase" && startAway && distStart > 1_600 && startPadSep / Math.max(distStart, 1) > 0.015;

    this.camReady = true;
  }

  paintMap(canvas: HTMLCanvasElement, s: Snapshot) {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "rgba(4, 10, 22, 0.65)";
    ctx.fillRect(0, 0, w, h);
    const cx = w * 0.5;
    const cy = h * 0.5;
    const rr = Math.min(w, h) * 0.42;
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
    ctx.fillStyle = "#0a1a33";
    ctx.fill();
    ctx.strokeStyle = "rgba(126,224,255,0.35)";
    ctx.stroke();

    const project = (lat: number, lon: number) => {
      const x = cx + (lon / Math.PI) * rr;
      const y = cy - (lat / (Math.PI / 2)) * rr * 0.92;
      return { x, y };
    };

    ctx.beginPath();
    let started = false;
    for (let i = 0; i < this.trailLat.length; i++) {
      const p = project(this.trailLat[i], this.trailLon[i]);
      if (!started) {
        ctx.moveTo(p.x, p.y);
        started = true;
      } else ctx.lineTo(p.x, p.y);
    }
    ctx.strokeStyle = "rgba(126,224,255,0.7)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    const drawPaths = (paths: MapPath[], dim: boolean) => {
      for (const p of paths) {
        if (p.lat.length < 2) continue;
        ctx.beginPath();
        for (let i = 0; i < p.lat.length; i++) {
          const q = project(p.lat[i], p.lon[i]);
          if (i === 0) ctx.moveTo(q.x, q.y);
          else ctx.lineTo(q.x, q.y);
        }
        const [r, g, b] = fitRgb(
          p.f,
          paths.reduce((m, x) => Math.min(m, x.f), Infinity),
          paths.reduce((m, x) => Math.max(m, x.f), -Infinity),
          p.ok,
          p.best,
        );
        const a = dim ? 0.28 : p.best ? 0.95 : 0.7;
        ctx.strokeStyle = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${a})`;
        ctx.lineWidth = p.best ? 1.8 : 0.9;
        ctx.stroke();
      }
    };
    drawPaths(this.mapPrev, true);
    drawPaths(this.mapLive, false);

    const padG = project(
      (28.4856 * Math.PI) / 180,
      (-80.5444 * Math.PI) / 180,
    );
    ctx.fillStyle = "#ffcc33";
    ctx.beginPath();
    ctx.arc(padG.x, padG.y, 3.2, 0, Math.PI * 2);
    ctx.fill();

    const veh = project(s.lat, s.lon);
    ctx.fillStyle = "#7ee0ff";
    ctx.beginPath();
    ctx.arc(veh.x, veh.y, 3.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "rgba(255,204,102,0.5)";
    ctx.beginPath();
    ctx.moveTo(veh.x, veh.y);
    ctx.lineTo(padG.x, padG.y);
    ctx.stroke();

    const rangeKm = s.range_h / 1000;
    ctx.fillStyle = "#7f93a6";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`LZ-1  ${rangeKm.toFixed(0)} km`, 8, h - 8);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  debugCam() {
    return {
      mode: this.camMode,
      dist: this.camera.position.distanceTo(this.rocket.position),
      padDist: this.camera.position.distanceTo(this._pad),
      camLen: this.camera.position.length(),
      rokLen: this.rocket.position.length(),
      near: this.camera.near,
      far: this.camera.far,
      zoom: this.zoom,
      yaw: this.yaw,
      pitch: this.pitch,
      cam: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      rok: [this.rocket.position.x, this.rocket.position.y, this.rocket.position.z],
    };
  }
}
