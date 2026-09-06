import * as THREE from "three";
import {
  EARTH_RADIUS_EQ,
  LZ1_APRON_M,
  LZ1_DIAMETER_M,
  LZ2_NORTHWEST_M,
  PAD_LAT_DEG,
  PAD_LON_DEG,
} from "./constants";
import { SLAM_SPAWN_MIN_R_M, SLAM_SPAWN_RADIUS_M } from "./scenario";

const R = EARTH_RADIUS_EQ;
const CAPE_HALF_M = 12_000;
const REGION_HALF_M = 1_100_000;
const CAP_RADIUS_M = 820_000;
const GLOBE_CLIP_M = 780_000;
const PAD_R = LZ1_DIAMETER_M * 0.5;
const APRON_R = PAD_R + LZ1_APRON_M;
const DECK_Y = 1.22;
const MARK_Y = 1.28;

const DEG = Math.PI / 180;
const LAT0 = PAD_LAT_DEG * DEG;
const LON0 = PAD_LON_DEG * DEG;
const M_PER_DEG_LAT = 111_195;
const M_PER_DEG_LON = 111_195 * Math.cos(LAT0);

export type World = {
  group: THREE.Group;
  pad: THREE.Group;
  sunDir: THREE.Vector3;
};

export function makeWorld(): World {
  const g = new THREE.Group();
  const sunDir = new THREE.Vector3(-0.74, 0.2, 0.18).normalize();

  const dayMap = loadTex("/textures/earth-day.jpg");
  const nightMap = loadTex("/textures/earth-night.jpg");
  const capeMap = new THREE.CanvasTexture(drawCape());
  capeMap.colorSpace = THREE.NoColorSpace;
  capeMap.anisotropy = 8;
  capeMap.wrapS = capeMap.wrapT = THREE.ClampToEdgeWrapping;
  const regionMap = new THREE.CanvasTexture(drawRegion());
  regionMap.colorSpace = THREE.NoColorSpace;
  regionMap.anisotropy = 8;
  regionMap.wrapS = regionMap.wrapT = THREE.ClampToEdgeWrapping;

  const { ecefEast, ecefNorth, ecefUp } = ecefBasis();
  const earthCenter = new THREE.Vector3(0, -R, 0);
  const uniforms = {
    dayMap: { value: dayMap },
    nightMap: { value: nightMap },
    capeMap: { value: capeMap },
    regionMap: { value: regionMap },
    sunDir: { value: sunDir },
    earthCenter: { value: earthCenter },
    ecefEast: { value: ecefEast },
    ecefNorth: { value: ecefNorth },
    ecefUp: { value: ecefUp },
    capeHalf: { value: CAPE_HALF_M },
    regionHalf: { value: REGION_HALF_M },
    globeClip: { value: 0 },
  };

  const globeMat = makeEarthMat(uniforms, true);
  const capMat = makeEarthMat(uniforms, false);

  const globe = new THREE.Mesh(new THREE.SphereGeometry(R - 48, 128, 96), globeMat);
  globe.position.copy(earthCenter);
  globe.frustumCulled = false;
  globe.renderOrder = 0;
  g.add(globe);

  const cap = new THREE.Mesh(makeSphericalCap(R, CAP_RADIUS_M), capMat);
  cap.frustumCulled = false;
  cap.renderOrder = 1;
  g.add(cap);

  g.add(makeInnerSky(sunDir));
  g.add(makeLimb(sunDir));
  g.add(makeStars());

  const pad = makeLandingZone();
  g.add(pad);

  g.add(makeScrub());

  return { group: g, pad, sunDir };
}

function loadTex(url: string) {
  const t = new THREE.TextureLoader().load(url);
  t.colorSpace = THREE.NoColorSpace;
  t.anisotropy = 8;
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function ecefBasis() {
  const cl = Math.cos(LAT0);
  const sl = Math.sin(LAT0);
  const co = Math.cos(LON0);
  const so = Math.sin(LON0);
  const ecefUp = new THREE.Vector3(cl * co, cl * so, sl);
  const ecefEast = new THREE.Vector3(-so, co, 0);
  const ecefNorth = new THREE.Vector3(-sl * co, -sl * so, cl);
  return { ecefEast, ecefNorth, ecefUp };
}

function makeEarthMat(
  u: {
    dayMap: { value: THREE.Texture };
    nightMap: { value: THREE.Texture };
    capeMap: { value: THREE.Texture };
    regionMap: { value: THREE.Texture };
    sunDir: { value: THREE.Vector3 };
    earthCenter: { value: THREE.Vector3 };
    ecefEast: { value: THREE.Vector3 };
    ecefNorth: { value: THREE.Vector3 };
    ecefUp: { value: THREE.Vector3 };
    capeHalf: { value: number };
    regionHalf: { value: number };
    globeClip: { value: number };
  },
  isGlobe: boolean,
) {
  const uniforms = {
    ...u,
    globeClip: { value: isGlobe ? GLOBE_CLIP_M : 0 },
  };
  return new THREE.ShaderMaterial({
    uniforms,
    toneMapped: true,
    fog: false,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      varying vec3 vN;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        vN = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D dayMap;
      uniform sampler2D nightMap;
      uniform sampler2D capeMap;
      uniform sampler2D regionMap;
      uniform vec3 sunDir;
      uniform vec3 earthCenter;
      uniform vec3 ecefEast;
      uniform vec3 ecefNorth;
      uniform vec3 ecefUp;
      uniform float capeHalf;
      uniform float regionHalf;
      uniform float globeClip;
      varying vec3 vWorld;
      varying vec3 vN;
      #include <common>
      #include <logdepthbuf_pars_fragment>
      vec3 srgb(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
      void main() {
        #include <logdepthbuf_fragment>
        float horiz = length(vWorld.xz);
        if (globeClip > 0.5 && horiz < globeClip) discard;

        vec3 n = normalize(vWorld - earthCenter);
        float eastC = dot(n, vec3(1.0, 0.0, 0.0));
        float upC = dot(n, vec3(0.0, 1.0, 0.0));
        float northC = dot(n, vec3(0.0, 0.0, -1.0));
        vec3 ecef = normalize(eastC * ecefEast + northC * ecefNorth + upC * ecefUp);
        float lat = asin(clamp(ecef.z, -1.0, 1.0));
        float lon = atan(ecef.y, ecef.x);
        vec2 uv = vec2(lon * ${0.5 / Math.PI} + 0.5, lat * ${1 / Math.PI} + 0.5);

        vec3 day = srgb(texture2D(dayMap, uv).rgb);
        vec3 night = srgb(texture2D(nightMap, uv).rgb);
        float sun = dot(n, sunDir);
        float dayW = smoothstep(-0.08, 0.16, sun);
        float wrap = max(sun, 0.0) * 0.82 + 0.08;
        vec3 albedo = mix(night * 1.35, day, dayW);
        vec3 col = albedo * (0.07 + 0.93 * wrap);

        float specMask = smoothstep(0.18, 0.4, day.b - (day.r + day.g) * 0.48);
        vec3 halfV = normalize(sunDir + normalize(cameraPosition - vWorld));
        float spec = pow(max(0.0, dot(n, halfV)), 42.0) * specMask * max(sun, 0.0);
        col += vec3(0.55, 0.62, 0.7) * spec * 0.45;

        float twilight = 1.0 - smoothstep(0.0, 0.22, abs(sun));
        col += vec3(0.55, 0.28, 0.1) * twilight * 0.12 * specMask;

        vec2 enu = vec2(vWorld.x, -vWorld.z);
        if (horiz < regionHalf * 1.42) {
          vec2 ruv = enu / (regionHalf * 2.0) + 0.5;
          if (ruv.x > 0.0 && ruv.x < 1.0 && ruv.y > 0.0 && ruv.y < 1.0) {
            vec4 region = texture2D(regionMap, ruv);
            vec3 rlin = srgb(region.rgb);
            float rw = region.a * (1.0 - smoothstep(regionHalf * 0.42, regionHalf * 0.92, horiz));
            vec3 rlit = rlin * (0.07 + 0.93 * wrap);
            col = mix(col, rlit, clamp(rw, 0.0, 1.0));
          }
        }
        if (horiz < capeHalf * 1.42) {
          vec2 cuv = enu / (capeHalf * 2.0) + 0.5;
          if (cuv.x > 0.0 && cuv.x < 1.0 && cuv.y > 0.0 && cuv.y < 1.0) {
            vec4 cape = texture2D(capeMap, cuv);
            vec3 clin = srgb(cape.rgb);
            float cw = cape.a * (1.0 - smoothstep(capeHalf * 0.48, capeHalf * 0.96, horiz));
            vec3 clit = clin * (0.1 + 0.95 * wrap);
            col = mix(col, clit, clamp(cw, 0.0, 1.0));
          }
        }

        float ndv = max(0.0, dot(n, normalize(cameraPosition - vWorld)));
        col += vec3(0.25, 0.4, 0.7) * pow(1.0 - ndv, 4.0) * 0.16;

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
}

function makeSphericalCap(radius: number, maxR: number) {
  const radii = [
    0, 30, 70, 120, 200, 320, 500, 800, 1_300, 2_200, 3_600, 6_000, 10_000, 16_000, 26_000, 42_000,
    70_000, 110_000, 180_000, 280_000, 420_000, 600_000, maxR,
  ];
  const segs = 96;
  const pos: number[] = [];
  const nrm: number[] = [];
  pos.push(0, 0, 0);
  nrm.push(0, 1, 0);
  for (let ri = 1; ri < radii.length; ri++) {
    const r = radii[ri];
    const y = Math.sqrt(Math.max(0, radius * radius - r * r)) - radius;
    for (let i = 0; i < segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      const x = Math.cos(a) * r;
      const z = -Math.sin(a) * r;
      pos.push(x, y, z);
      const nx = x;
      const ny = y + radius;
      const nz = z;
      const len = Math.hypot(nx, ny, nz) || 1;
      nrm.push(nx / len, ny / len, nz / len);
    }
  }
  const idx: number[] = [];
  for (let i = 0; i < segs; i++) {
    const a = 1 + i;
    const b = 1 + ((i + 1) % segs);
    idx.push(0, a, b);
  }
  const ringStart = (ri: number) => 1 + (ri - 1) * segs;
  for (let ri = 1; ri < radii.length - 1; ri++) {
    const s0 = ringStart(ri);
    const s1 = ringStart(ri + 1);
    for (let i = 0; i < segs; i++) {
      const n = (i + 1) % segs;
      const a = s0 + i;
      const b = s0 + n;
      const c = s1 + i;
      const d = s1 + n;
      idx.push(a, c, b);
      idx.push(b, c, d);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.Float32BufferAttribute(nrm, 3));
  geo.setIndex(idx);
  geo.computeBoundingSphere();
  return geo;
}

function makeInnerSky(sunDir: THREE.Vector3) {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      sunDir: { value: sunDir },
      earthCenter: { value: new THREE.Vector3(0, -R, 0) },
    },
    side: THREE.BackSide,
    depthWrite: false,
    fog: false,
    toneMapped: true,
    transparent: true,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 sunDir;
      uniform vec3 earthCenter;
      varying vec3 vWorld;
      #include <common>
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        vec3 view = normalize(vWorld - cameraPosition);
        float up = view.y;
        float h = clamp(up * 0.72 + 0.28, 0.0, 1.0);
        vec3 zenith = vec3(0.025, 0.04, 0.07);
        vec3 horizon = vec3(0.16, 0.2, 0.26);
        vec3 col = mix(horizon, zenith, pow(h, 0.65));
        float sunAmt = pow(max(0.0, dot(view, sunDir)), 48.0);
        float sunHaze = pow(max(0.0, dot(view, sunDir)), 6.0);
        col += vec3(1.0, 0.62, 0.28) * sunAmt * 0.85;
        col += vec3(0.7, 0.35, 0.14) * sunHaze * 0.22;
        float alt = length(cameraPosition - earthCenter) - ${R.toFixed(1)};
        float fade = 1.0 - smoothstep(18000.0, 92000.0, alt);
        float alpha = fade * mix(0.92, 0.55, h);
        if (alpha < 0.02) discard;
        gl_FragColor = vec4(col, alpha);
      }
    `,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(R + 82_000, 64, 32), mat);
  mesh.position.set(0, -R, 0);
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  return mesh;
}

function makeLimb(sunDir: THREE.Vector3) {
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      sunDir: { value: sunDir },
      earthCenter: { value: new THREE.Vector3(0, -R, 0) },
    },
    side: THREE.DoubleSide,
    depthWrite: false,
    fog: false,
    toneMapped: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */ `
      varying vec3 vWorld;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        #include <logdepthbuf_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 sunDir;
      uniform vec3 earthCenter;
      varying vec3 vWorld;
      #include <common>
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        vec3 n = normalize(vWorld - earthCenter);
        vec3 view = normalize(cameraPosition - vWorld);
        float ndv = abs(dot(n, view));
        float fres = pow(1.0 - ndv, 2.8);
        float sun = dot(n, sunDir);
        vec3 dayC = vec3(0.28, 0.5, 0.95);
        vec3 setC = vec3(1.0, 0.42, 0.12);
        vec3 col = mix(setC, dayC, smoothstep(-0.15, 0.45, sun));
        float night = smoothstep(-0.35, 0.05, sun);
        float a = fres * (0.22 + 0.7 * night);
        if (a < 0.01) discard;
        gl_FragColor = vec4(col * a, a);
      }
    `,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(R + 118_000, 80, 48), mat);
  mesh.position.set(0, -R, 0);
  mesh.frustumCulled = false;
  mesh.renderOrder = 4;
  return mesh;
}

function makeStars() {
  const n = 2800;
  const pos = new Float32Array(n * 3);
  const r = 1.35e7;
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const b = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(b) * Math.cos(a);
    pos[i * 3 + 1] = r * Math.cos(b);
    pos[i * 3 + 2] = r * Math.sin(b) * Math.sin(a);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const pts = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      color: 0xe8eef4,
      size: 1.25,
      sizeAttenuation: false,
      depthWrite: false,
      fog: false,
    }),
  );
  pts.frustumCulled = false;
  pts.renderOrder = -1;
  return pts;
}

function makeLandingZone() {
  const g = new THREE.Group();
  g.add(makeOnePad(0, 0, true));
  const nw = LZ2_NORTHWEST_M / Math.SQRT2;
  g.add(makeOnePad(-nw, nw, false));
  g.add(makeRoads());
  g.add(makeSpawnRings());
  return g;
}

function makeOnePad(east: number, north: number, primary: boolean) {
  const g = new THREE.Group();
  g.position.set(east, 0, -north);
  const gravel = new THREE.MeshStandardMaterial({
    color: 0x6a6458,
    roughness: 0.95,
    metalness: 0.02,
    emissive: 0x1a1814,
    emissiveIntensity: 0.12,
  });
  const conc = new THREE.MeshStandardMaterial({
    color: 0xd8d3c6,
    roughness: 0.78,
    metalness: 0.04,
    emissive: 0x2a2822,
    emissiveIntensity: 0.16,
  });
  const apron = new THREE.Mesh(new THREE.CylinderGeometry(APRON_R, APRON_R, 0.55, 64), gravel);
  apron.position.y = 0.28;
  g.add(apron);
  const deck = new THREE.Mesh(new THREE.CylinderGeometry(PAD_R, PAD_R, 1.15, 72), conc);
  deck.position.y = DECK_Y * 0.5;
  g.add(deck);

  const markTex = new THREE.CanvasTexture(drawPadMarks(primary));
  markTex.colorSpace = THREE.SRGBColorSpace;
  markTex.anisotropy = 8;
  const mark = new THREE.Mesh(
    new THREE.CircleGeometry(PAD_R * 0.995, 72),
    new THREE.MeshStandardMaterial({
      map: markTex,
      roughness: 0.62,
      metalness: 0.05,
      emissive: 0xffffff,
      emissiveMap: markTex,
      emissiveIntensity: 0.18,
    }),
  );
  mark.rotation.x = -Math.PI / 2;
  mark.position.y = MARK_Y;
  g.add(mark);

  const nLights = 8;
  for (let i = 0; i < nLights; i++) {
    const a = (i / nLights) * Math.PI * 2 + 0.2;
    const rr = PAD_R - 1.6;
    const lamp = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 8, 6),
      new THREE.MeshBasicMaterial({ color: 0xfff1c8 }),
    );
    lamp.position.set(Math.cos(a) * rr, MARK_Y + 0.15, Math.sin(a) * rr);
    g.add(lamp);
  }

  if (primary) {
    for (let i = 0; i < 4; i++) {
      const a = i * (Math.PI / 2) + Math.PI / 4;
      const d = APRON_R + 7;
      const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18, 0.22, 11, 8),
        new THREE.MeshStandardMaterial({ color: 0x3a3d42, metalness: 0.4, roughness: 0.45 }),
      );
      pole.position.set(Math.cos(a) * d, 5.5, Math.sin(a) * d);
      g.add(pole);
      const head = new THREE.Mesh(
        new THREE.BoxGeometry(1.1, 0.35, 0.7),
        new THREE.MeshBasicMaterial({ color: 0xffe4b0 }),
      );
      head.position.set(Math.cos(a) * d, 11.1, Math.sin(a) * d);
      g.add(head);
      const light = new THREE.PointLight(0xffe0b0, 22, 95, 2);
      light.position.set(Math.cos(a) * (d - 2), 10.4, Math.sin(a) * (d - 2));
      g.add(light);
    }
  }
  return g;
}

function makeRoads() {
  const g = new THREE.Group();
  const asphalt = new THREE.MeshStandardMaterial({
    color: 0x2a2c2e,
    roughness: 0.9,
    metalness: 0.08,
    emissive: 0x0a0c0e,
    emissiveIntensity: 0.2,
  });
  const strip = new THREE.Mesh(new THREE.BoxGeometry(18, 0.12, 420), asphalt);
  strip.position.set(-72, 0.08, 40);
  strip.rotation.y = 0.18;
  g.add(strip);
  const spur = new THREE.Mesh(new THREE.BoxGeometry(12, 0.12, 90), asphalt);
  spur.position.set(-28, 0.09, 18);
  spur.rotation.y = 1.05;
  g.add(spur);
  const lz2 = new THREE.Mesh(new THREE.BoxGeometry(10, 0.12, 220), asphalt);
  const nw = LZ2_NORTHWEST_M / Math.SQRT2;
  lz2.position.set(-nw * 0.5 - 20, 0.08, -nw * 0.5);
  lz2.rotation.y = -0.78;
  g.add(lz2);
  return g;
}

function makeSpawnRings() {
  const g = new THREE.Group();
  const spawnMat = new THREE.MeshBasicMaterial({
    color: 0xb7c4d0,
    transparent: true,
    opacity: 0.16,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const outer = new THREE.Mesh(
    new THREE.RingGeometry(SLAM_SPAWN_RADIUS_M - 1.4, SLAM_SPAWN_RADIUS_M + 1.4, 72),
    spawnMat,
  );
  outer.rotation.x = -Math.PI / 2;
  outer.position.y = MARK_Y + 0.04;
  g.add(outer);
  const inner = new THREE.Mesh(
    new THREE.RingGeometry(SLAM_SPAWN_MIN_R_M - 0.9, SLAM_SPAWN_MIN_R_M + 0.9, 48),
    new THREE.MeshBasicMaterial({
      color: 0x8a96a4,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  inner.rotation.x = -Math.PI / 2;
  inner.position.y = MARK_Y + 0.04;
  g.add(inner);
  return g;
}

function makeScrub() {
  const geo = new THREE.ConeGeometry(1.1, 1.8, 5);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x2a3a28,
    roughness: 0.95,
    metalness: 0,
    emissive: 0x0c140c,
    emissiveIntensity: 0.2,
  });
  const n = 220;
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  const dummy = new THREE.Object3D();
  let k = 0;
  for (let i = 0; i < 800 && k < n; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = 70 + Math.random() * 380;
    const x = Math.cos(a) * r;
    const z = Math.sin(a) * r;
    const distLz1 = Math.hypot(x, z);
    const nw = LZ2_NORTHWEST_M / Math.SQRT2;
    const distLz2 = Math.hypot(x + nw, z + nw);
    if (distLz1 < APRON_R + 8 || distLz2 < APRON_R + 6) continue;
    if (x > 420) continue;
    dummy.position.set(x, 0.9, z);
    dummy.rotation.y = Math.random() * 6;
    dummy.scale.setScalar(0.6 + Math.random() * 1.4);
    dummy.updateMatrix();
    mesh.setMatrixAt(k++, dummy.matrix);
  }
  mesh.count = k;
  mesh.instanceMatrix.needsUpdate = true;
  mesh.frustumCulled = false;
  return mesh;
}

function drawPadMarks(primary: boolean) {
  const s = 1024;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#d5d0c3";
  ctx.fillRect(0, 0, s, s);
  for (let i = 0; i < 16000; i++) {
    const x = Math.random() * s;
    const y = Math.random() * s;
    const v = 170 + Math.random() * 55;
    ctx.fillStyle = `rgba(${v},${v - 5},${v - 14},${0.06 + Math.random() * 0.1})`;
    ctx.fillRect(x, y, 2 + Math.random() * 3, 1);
  }
  const cx = s / 2;
  const cy = s / 2;
  ctx.beginPath();
  ctx.arc(cx, cy, s * 0.36, 0, Math.PI * 2);
  ctx.fillStyle = "#ece7db";
  ctx.fill();

  const scorch = ctx.createRadialGradient(cx, cy, 8, cx, cy, s * 0.2);
  scorch.addColorStop(0, "rgba(62,54,46,0.38)");
  scorch.addColorStop(0.5, "rgba(90,80,68,0.12)");
  scorch.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = scorch;
  ctx.beginPath();
  ctx.arc(cx, cy, s * 0.2, 0, Math.PI * 2);
  ctx.fill();

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-0.48);
  ctx.strokeStyle = "#141414";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = s * 0.038;
  const arm = s * 0.13;
  ctx.beginPath();
  ctx.moveTo(-arm, -arm * 0.18);
  ctx.lineTo(arm, arm * 0.18);
  ctx.moveTo(-arm * 0.18, arm);
  ctx.lineTo(arm * 0.18, -arm);
  ctx.stroke();
  ctx.fillStyle = "#141414";
  ctx.beginPath();
  ctx.arc(0, 0, s * 0.028, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(cx, cy, s * 0.492, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = "rgba(28,28,30,0.78)";
  ctx.font = "600 40px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(primary ? "LZ-1" : "LZ-2", cx, s * 0.84);
  return c;
}

function hash2(x: number, y: number) {
  const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return n - Math.floor(n);
}

function noise2(x: number, y: number) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a * (1 - ux) * (1 - uy) + b * ux * (1 - uy) + c * (1 - ux) * uy + d * ux * uy;
}

function fbm(x: number, y: number) {
  return (
    noise2(x, y) * 0.5 + noise2(x * 2.1, y * 2.1) * 0.25 + noise2(x * 4.3, y * 4.3) * 0.125
  );
}

function drawCape() {
  const s = 2048;
  const half = CAPE_HALF_M;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(s, s);
  const data = img.data;
  const shore = 450;
  for (let py = 0; py < s; py++) {
    for (let px = 0; px < s; px++) {
      const east = ((px + 0.5) / s) * (half * 2) - half;
      const north = half - ((py + 0.5) / s) * (half * 2);
      const n = fbm(east / 180, north / 180);
      const n2 = fbm(east / 55 + 8, north / 55);
      const edge = Math.min(
        1,
        Math.min(half - Math.abs(east), half - Math.abs(north)) / (half * 0.18),
      );
      let r = 10;
      let g = 28;
      let b = 48;
      let a = 255;
      const banana = east < -4200 && east > -5600 && Math.abs(north) < 9000;
      const indian = east < -9800 && east > -11200;
      if (east > shore + n * 40) {
        const deep = Math.min(1, (east - shore) / 4000);
        r = 7 + n * 6;
        g = 26 + n * 12 - deep * 6;
        b = 44 + n * 16 + deep * 8;
        if (east < shore + 70) {
          r = 168;
          g = 152;
          b = 118;
        } else if (east < shore + 180) {
          r = 32 + n2 * 16;
          g = 72 + n2 * 22;
          b = 92;
        }
        a = Math.floor(255 * (0.35 + 0.65 * edge));
      } else if (banana || indian) {
        r = 10 + n * 8;
        g = 40 + n * 12;
        b = 58 + n * 14;
        a = Math.floor(220 * edge);
      } else {
        const scrub = 0.28 + n * 0.45 + n2 * 0.18;
        r = 28 + scrub * 32;
        g = 42 + scrub * 42;
        b = 22 + scrub * 16;
        if (Math.abs(east + 72) < 12 && north < 220 && north > -200) {
          r = 36;
          g = 38;
          b = 40;
        }
        a = Math.floor(255 * (0.55 + 0.45 * edge));
      }
      const i = (py * s + px) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = a;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function llToEnu(lat: number, lon: number): [number, number] {
  return [(lon - PAD_LON_DEG) * M_PER_DEG_LON, (lat - PAD_LAT_DEG) * M_PER_DEG_LAT];
}

function drawRegion() {
  const s = 2048;
  const half = REGION_HALF_M;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const ctx = c.getContext("2d")!;
  ctx.clearRect(0, 0, s, s);

  const toPx = (lat: number, lon: number) => {
    const [e, n] = llToEnu(lat, lon);
    return [((e + half) / (half * 2)) * s, ((half - n) / (half * 2)) * s] as const;
  };

  const fillPoly = (pts: [number, number][], fill: string) => {
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [x, y] = toPx(p[0], p[1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  };

  fillPoly(
    [
      [35.0, -76.0],
      [33.9, -77.9],
      [32.9, -79.6],
      [32.1, -80.9],
      [31.1, -81.45],
      [30.7, -81.47],
      [30.4, -81.41],
      [29.9, -81.27],
      [29.23, -81.01],
      [28.8, -80.82],
      [28.49, -80.53],
      [28.41, -80.53],
      [28.08, -80.56],
      [27.64, -80.37],
      [27.2, -80.2],
      [26.72, -80.03],
      [26.12, -80.1],
      [25.79, -80.12],
      [25.4, -80.25],
      [25.09, -80.45],
      [24.7, -81.1],
      [24.55, -81.8],
      [24.7, -81.95],
      [25.15, -81.08],
      [25.8, -81.5],
      [26.15, -81.8],
      [26.64, -82.15],
      [27.3, -82.55],
      [27.77, -82.75],
      [28.3, -82.75],
      [28.9, -82.7],
      [29.2, -83.05],
      [29.7, -83.5],
      [29.9, -84.4],
      [30.16, -85.66],
      [30.4, -87.05],
      [30.98, -87.55],
      [31.0, -85.0],
      [31.0, -82.4],
      [32.0, -81.2],
      [33.5, -79.2],
      [35.0, -76.5],
    ],
    "#24382c",
  );

  fillPoly(
    [
      [23.5, -84.9],
      [23.15, -82.4],
      [23.1, -81.3],
      [22.5, -79.5],
      [21.6, -77.0],
      [20.3, -74.3],
      [19.9, -75.1],
      [20.7, -77.5],
      [21.8, -80.0],
      [22.4, -83.6],
      [22.9, -84.9],
    ],
    "#2a3f30",
  );

  const islands: [number, number, number][] = [
    [26.66, -78.32, 38],
    [25.06, -77.35, 22],
    [24.7, -77.9, 28],
    [24.0, -77.5, 16],
    [26.55, -78.78, 14],
  ];
  ctx.fillStyle = "#2c4032";
  for (const [lat, lon, rad] of islands) {
    const [x, y] = toPx(lat, lon);
    ctx.beginPath();
    ctx.ellipse(x, y, rad, rad * 0.55, 0.4, 0, Math.PI * 2);
    ctx.fill();
  }

  const cities: [number, number, number][] = [
    [28.54, -81.38, 7],
    [25.76, -80.19, 9],
    [27.95, -82.46, 8],
    [30.33, -81.66, 7],
    [26.12, -80.14, 6],
    [27.77, -82.64, 5],
    [32.08, -81.09, 5],
    [33.75, -84.39, 8],
  ];
  for (const [lat, lon, rad] of cities) {
    const [x, y] = toPx(lat, lon);
    const grd = ctx.createRadialGradient(x, y, 0, x, y, rad * 3);
    grd.addColorStop(0, "rgba(120,112,96,0.4)");
    grd.addColorStop(1, "rgba(120,112,96,0)");
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(x, y, rad * 3, 0, Math.PI * 2);
    ctx.fill();
  }

  const noise = ctx.getImageData(0, 0, s, s);
  const d = noise.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 8) continue;
    const j = i / 4;
    const x = j % s;
    const y = (j / s) | 0;
    const n = (fbm(x / 90, y / 90) - 0.5) * 16;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n * 0.7));
    const px = j % s;
    const py = (j / s) | 0;
    const edge = Math.min(px, py, s - 1 - px, s - 1 - py) / (s * 0.08);
    d[i + 3] = Math.floor(d[i + 3] * Math.max(0, Math.min(1, edge)));
  }
  ctx.putImageData(noise, 0, 0);
  return c;
}
