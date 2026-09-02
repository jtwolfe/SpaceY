import * as THREE from "three";
import type { Snapshot } from "./types";

export type CamMode = "chase" | "pad" | "orbit";

const EARTH_R = 6_371_000;

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

export class SceneApp {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(55, 1, 20, 1.2e8);
  camMode: CamMode = "chase";

  private earth: THREE.Mesh;
  private atmo: THREE.Mesh;
  private rocket = new THREE.Group();
  private plume: THREE.Mesh;
  private plasma: THREE.Mesh;
  private debris = new THREE.Group();
  private trail: THREE.Line;
  private trailPts: THREE.Vector3[] = [];
  private pad: THREE.Group;
  private sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
  private camReady = false;
  private marker: THREE.Sprite;

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
        varying vec3 vN;
        void main() {
          vN = normalize(normalMatrix * normal);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }`,
      fragmentShader: `
        varying vec3 vN;
        void main() {
          float f = pow(1.0 - abs(vN.z), 2.2);
          gl_FragColor = vec4(0.25, 0.55, 1.0, 0.22 * f);
        }`,
    });
    this.atmo = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R * 1.035, 64, 48), atmoMat);
    this.scene.add(this.atmo);

    this.scene.add(makeStars());
    this.sun.position.set(2e7, 8e6, 1.2e7);
    this.scene.add(this.sun);
    this.scene.add(new THREE.AmbientLight(0x6b88aa, 0.35));

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

    const tgeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
    this.trail = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color: 0x7ee0ff, transparent: true, opacity: 0.65 }));
    this.scene.add(this.trail);

    const mark = document.createElement("canvas");
    mark.width = mark.height = 64;
    const mg = mark.getContext("2d")!;
    const grd = mg.createRadialGradient(32, 32, 2, 32, 32, 30);
    grd.addColorStop(0, "rgba(255,240,200,1)");
    grd.addColorStop(0.3, "rgba(126,224,255,0.85)");
    grd.addColorStop(1, "rgba(126,224,255,0)");
    mg.fillStyle = grd;
    mg.fillRect(0, 0, 64, 64);
    this.marker = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(mark), transparent: true, depthWrite: false }));
    this.scene.add(this.marker);

    this.scene.add(this.debris);
    this.resize();
    window.addEventListener("resize", () => this.resize());
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
    return g;
  }

  private buildRocket() {
    const white = new THREE.MeshStandardMaterial({ color: 0xf2f4f7, metalness: 0.35, roughness: 0.4 });
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
    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 2.4, 1.7),
        new THREE.MeshStandardMaterial({ color: 0x9aa3ad, metalness: 0.7, roughness: 0.3 }),
      );
      const a = (i * Math.PI) / 2 + Math.PI / 4;
      fin.position.set(Math.cos(a) * 2.05, 21, Math.sin(a) * 2.05);
      fin.lookAt(0, 21, 0);
      this.rocket.add(fin);
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
    const [x, y, z] = s.r_ecef;
    const pos = new THREE.Vector3(x, z, -y);
    const [pw, px, py, pz] = s.quat_ecef;
    // Body +X (nose) → Three.js +Y
    const qEcef = new THREE.Quaternion(px, py, pz, pw);
    const bodyToThree = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2);
    // Map ECEF (x,y,z) → Three (x,z,-y)
    const mapped = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(0, 0, -1),
        new THREE.Vector3(0, 1, 0),
      ),
    );
    this.rocket.position.copy(pos);
    this.rocket.quaternion.copy(mapped.clone().multiply(qEcef).multiply(bodyToThree));
    this.marker.position.copy(pos);
    this.marker.visible = this.camMode !== "chase";
    const orbitMark = Math.max(12_000, Math.min(80_000, s.range_h * 0.012));
    this.marker.scale.setScalar(this.camMode === "orbit" ? orbitMark : 400);

    const [padx, pady, padz] = s.pad_ecef;
    this.pad.position.set(padx, padz, -pady);
    const up = this.pad.position.clone().normalize();
    this.pad.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);

    this.earth.rotation.y = 0;

    this.plume.visible = s.throttle > 0.05 && s.intact;
    (this.plume.material as THREE.MeshBasicMaterial).opacity = 0.35 + 0.55 * s.throttle;
    this.plume.scale.setScalar(0.7 + s.throttle * 1.4);

    const heat = Math.min(1, s.heat / 8e6);
    (this.plasma.material as THREE.MeshBasicMaterial).opacity = s.intact ? heat * 0.55 : 0;
    this.plasma.scale.setScalar(1 + heat * 3);

    this.rocket.visible = s.intact;
    if (!s.intact && this.debris.children.length === 0) {
      this.spawnDebris(pos);
    }
    if (s.intact && this.debris.children.length) {
      this.debris.clear();
    }

    if (s.intact) {
      this.trailPts.push(pos.clone());
      if (this.trailPts.length > 800) this.trailPts.shift();
      this.trail.geometry.dispose();
      this.trail.geometry = new THREE.BufferGeometry().setFromPoints(this.trailPts);
    }

    this.updateCamera(s, pos, up);
  }

  resetTrail() {
    this.trailPts = [];
    this.debris.clear();
    this.camReady = false;
  }

  private spawnDebris(pos: THREE.Vector3) {
    for (let i = 0; i < 18; i++) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(4 + Math.random() * 8, 1 + Math.random() * 4, 1 + Math.random() * 3),
        new THREE.MeshStandardMaterial({ color: 0xbbb4a8, metalness: 0.4 }),
      );
      m.position.copy(pos);
      m.userData.v = new THREE.Vector3(
        (Math.random() - 0.5) * 80,
        (Math.random() - 0.5) * 80,
        (Math.random() - 0.5) * 80,
      );
      this.debris.add(m);
    }
  }

  tickDebris(dt: number) {
    for (const c of this.debris.children) {
      const m = c as THREE.Mesh;
      m.position.addScaledVector(m.userData.v, dt);
      m.rotation.x += dt * 2;
      m.rotation.z += dt * 1.4;
    }
  }

  private updateCamera(s: Snapshot, pos: THREE.Vector3, up: THREE.Vector3) {
    const pad = this.pad.position;
    if (this.camMode === "chase") {
      const pull = 110 + Math.min(220, s.alt / 2_000);
      const back = new THREE.Vector3(0, -1, 0).applyQuaternion(this.rocket.quaternion).multiplyScalar(pull);
      const side = new THREE.Vector3(1, 0.45, 0.35).applyQuaternion(this.rocket.quaternion).multiplyScalar(36);
      const camPos = pos.clone().add(back).add(side);
      if (!this.camReady) this.camera.position.copy(camPos);
      else this.camera.position.lerp(camPos, 0.22);
      this.camera.up.copy(up);
      this.camera.lookAt(pos);
    } else if (this.camMode === "pad") {
      const radial = up.clone().multiplyScalar(s.range_h > 200_000 ? 1_800 : 220);
      const east = new THREE.Vector3(0, 1, 0).cross(up).normalize().multiplyScalar(s.range_h > 200_000 ? 2_400 : 260);
      this.camera.position.copy(pad).add(radial).add(east);
      this.camera.up.copy(up);
      if (s.range_h < 250_000) {
        this.camera.lookAt(pos);
      } else {
        this.camera.lookAt(pad.clone().add(up.clone().multiplyScalar(80_000)));
      }
    } else {
      const padN = pad.clone().normalize();
      const vehN = pos.clone().normalize();
      let along = padN.clone().add(vehN);
      if (along.lengthSq() < 0.05) {
        along = new THREE.Vector3(0, 1, 0).cross(padN);
      }
      along.normalize();
      let side = padN.clone().cross(vehN);
      if (side.lengthSq() < 1e-6) side = new THREE.Vector3(0, 1, 0).cross(padN);
      side.normalize();
      const view = along.multiplyScalar(0.28).add(side.multiplyScalar(0.96)).normalize();
      this.camera.position.copy(view.multiplyScalar(EARTH_R + 5_200_000));
      this.camera.up.copy(padN);
      this.camera.lookAt(new THREE.Vector3(0, 0, 0));
    }
    this.camReady = true;
    this.camera.near = this.camMode === "orbit" ? 2000 : 2;
    this.camera.far = 1.2e8;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
