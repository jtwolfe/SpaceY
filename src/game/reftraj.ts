/** Spawn-time 3DOF corridor (2 km miss scoring) plus a receding-horizon landing goal. */

import {
  DT,
  DRY_MASS_KG,
  G0,
  MERLIN_ISP_SL_S,
  MERLIN_THRUST_SL_N,
  REF_AREA_M2,
  STAGE_LENGTH_M,
  SUCCESS_ENGINE_ALT_M,
  SUCCESS_HVEL_MPS,
  SUCCESS_PAD_OFFSET_M,
  SUCCESS_SPEED_MPS,
  THROTTLE_MIN,
  wetMass,
} from "./constants";
import { lookup } from "./atmosphere";
import { clamp, saturate, Vec3 } from "./math";

export const REF_CORRIDOR_M = 40;
export const REF_HZ = 2;

export type RefPt = {
  t: number;
  p: Vec3;
  v: Vec3;
  throttle: number;
};

export type RefTraj = {
  pts: RefPt[];
  tLight: number;
  ok: boolean;
  range: number;
  speed: number;
};

export type RefSample = {
  p: Vec3;
  v: Vec3;
  throttle: number;
  dist: number;
  dv: number;
};

const ENG = STAGE_LENGTH_M * 0.5;
const CD = 0.82;

function engineAlt(pz: number) {
  return pz - ENG;
}

function windAt(z: number, windScale: number, windDir: number) {
  const wSpeed = 14 * windScale * Math.min(1, Math.max(0, z) / 800);
  return new Vec3(Math.cos(windDir) * wSpeed, Math.sin(windDir) * wSpeed, 0);
}

function accel(
  p: Vec3,
  v: Vec3,
  mass: number,
  throttle: number,
  dir: Vec3,
  windScale: number,
  windDir: number,
) {
  const air = lookup(Math.max(0, p.z));
  const vRel = v.sub(windAt(p.z, windScale, windDir));
  const spd = vRel.len();
  const q = 0.5 * air.density * spd * spd;
  const drag = spd > 0.4 ? vRel.scale((-CD * q * REF_AREA_M2) / spd) : new Vec3();
  const thrust = throttle > 0.02 ? dir.scale((MERLIN_THRUST_SL_N * throttle) / mass) : new Vec3();
  return new Vec3(0, 0, -G0).add(drag.scale(1 / mass)).add(thrust);
}

function sLightOf(p: Vec3, v: Vec3, mass: number) {
  const vDown = Math.max(0, -v.z);
  const range = Math.hypot(p.x, p.y);
  const vh = Math.hypot(v.x, v.y);
  const Tm = MERLIN_THRUST_SL_N * 0.95;
  const a = Tm / mass;
  const aUp = Math.max(4, a * Math.cos(0.35) - G0);
  const aH = Math.max(4, a * Math.sin(0.55));
  const sVert = Math.max(0, vDown * vDown - 36) / (2 * aUp) + 8;
  const tVert = vDown / aUp;
  const tNeedH = (vh + range * 0.4) / aH;
  const extra = Math.max(0, tNeedH - tVert) * 0.5;
  return { sLight: sVert + extra * vDown + 0.45 * G0 * extra * extra, sVert };
}

function thrustDir(p: Vec3, v: Vec3, tilt: number) {
  const ax = -0.32 * p.x - 2.4 * v.x;
  const ay = -0.32 * p.y - 2.4 * v.y;
  let h = new Vec3(ax, ay, 0);
  if (h.len() < 1e-6) h = new Vec3(-p.x, -p.y, 0);
  if (h.len() < 1e-6) return new Vec3(0, 0, 1);
  h = h.normalized();
  return new Vec3(0, 0, 1).scale(Math.cos(tilt)).add(h.scale(Math.sin(tilt))).normalized();
}

export function planRef(args: {
  p: Vec3;
  v: Vec3;
  fuel: number;
  timeout: number;
  windScale: number;
  windDir: number;
}): RefTraj {
  let p = args.p.clone();
  let v = args.v.clone();
  let fuel = args.fuel;
  let t = 0;
  let throttle = 0;
  let tLight = -1;
  const pts: RefPt[] = [{ t: 0, p: p.clone(), v: v.clone(), throttle: 0 }];
  const dt = DT;
  const tEnd = Math.min(args.timeout, 90);
  let nextRec = 1 / REF_HZ;

  while (t < tEnd && engineAlt(p.z) > 4 && fuel > 40) {
    const mass = Math.max(DRY_MASS_KG, wetMass(fuel));
    const range = Math.hypot(p.x, p.y);
    const vh = Math.hypot(v.x, v.y);
    const vDown = Math.max(0, -v.z);
    const pz = engineAlt(p.z);
    const overPad = range < 22 && vh < 5.5;
    const { sLight: sL, sVert } = sLightOf(p, v, mass);

    if (tLight < 0) {
      throttle = pz <= Math.max(sVert + 50, sL) ? 0.95 : 0;
      if (throttle > 0.02) tLight = t;
    }

    let tilt = 0;
    let dir = new Vec3(0, 0, 1);
    if (throttle > 0.02) {
      tilt = clamp(range / 160, 0.2, 0.62);
      if (pz < sVert + 70) tilt = Math.min(tilt, range > 32 ? 0.4 : 0.26);
      if (pz < sVert + 30) tilt = Math.min(tilt, range > 28 ? 0.22 : 0.1);
      if (pz < 26) tilt = Math.min(tilt, range < 28 ? 0.05 : 0.14);
      const needA = Math.max(0, (vDown * vDown - 25) / (2 * Math.max(pz - 5, 4)));
      throttle = saturate(((needA + G0) * mass) / MERLIN_THRUST_SL_N);
      if (vDown > 22) throttle = Math.max(throttle, 0.88);
      if (pz < 40 && vDown > 8) throttle = 1;
      throttle = Math.max(THROTTLE_MIN, throttle);
      dir = thrustDir(p, v, tilt);
      if (overPad && pz < 80 && vDown > 0.8 && v.z < 3) throttle = Math.max(throttle, 0.9);
      if (v.z > 3 && pz > 80 && vDown < 8) throttle = 0;
    }

    const a = accel(p, v, mass, throttle, dir, args.windScale, args.windDir);
    v = v.add(a.scale(dt));
    p = p.add(v.scale(dt));
    if (throttle > 0.02) fuel -= ((throttle * MERLIN_THRUST_SL_N) / (MERLIN_ISP_SL_S * G0)) * dt;
    t += dt;

    if (t >= nextRec || engineAlt(p.z) < 16) {
      pts.push({ t, p: p.clone(), v: v.clone(), throttle });
      nextRec += 1 / REF_HZ;
    }
    if (engineAlt(p.z) <= SUCCESS_ENGINE_ALT_M && v.len() < SUCCESS_SPEED_MPS + 4) break;
  }
  if (pts[pts.length - 1].t < t - 1e-6) pts.push({ t, p: p.clone(), v: v.clone(), throttle });

  const range = Math.hypot(p.x, p.y);
  const speed = v.len();
  const vh = Math.hypot(v.x, v.y);
  const ok =
    tLight >= 0 &&
    engineAlt(p.z) <= SUCCESS_ENGINE_ALT_M + 12 &&
    range <= 55 &&
    speed <= 16 &&
    vh <= 12;

  return { pts, tLight: tLight < 0 ? tEnd : tLight, ok, range, speed };
}

export function sampleRef(ref: RefTraj, p: Vec3): RefSample {
  const pts = ref.pts;
  if (!pts.length) {
    return { p: p.clone(), v: new Vec3(), throttle: 0, dist: 0, dv: 0 };
  }
  let bestI = 0;
  let bestD = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const d = pts[i].p.sub(p).lenSq();
    if (d < bestD) {
      bestD = d;
      bestI = i;
    }
  }
  const a = pts[bestI];
  const b = pts[Math.min(pts.length - 1, bestI + 1)];
  let t = 0;
  const ab = b.p.sub(a.p);
  const ab2 = ab.lenSq();
  if (ab2 > 1) t = clamp(p.sub(a.p).dot(ab) / ab2, 0, 1);
  const rp = a.p.lerp(b.p, t);
  const rv = a.v.lerp(b.v, t);
  return {
    p: rp,
    v: rv,
    throttle: a.throttle * (1 - t) + b.throttle * t,
    dist: rp.sub(p).len(),
    dv: 0,
  };
}

export function refTrackCost(ref: RefTraj | null, p: Vec3, v: Vec3): number {
  if (!ref?.ok || ref.pts.length < 2) return 0;
  const s = sampleRef(ref, p);
  const pos = Math.max(0, s.dist - REF_CORRIDOR_M);
  const vel = Math.max(0, v.sub(s.v).len() - 12);
  return 1.6 * pos + 2.5 * vel;
}

/** Receding-horizon landing goal from current state. Not a polyline to track. */
export type GoalCmd = {
  tLight: number;
  sLight: number;
  dir: Vec3;
  dv: Vec3;
  predMiss: Vec3;
  tiltCmd: number;
  goVertical: boolean;
};

export function ballisticMissRange(p: Vec3, v: Vec3): number {
  const pz = Math.max(0.5, engineAlt(p.z));
  const tG = ballisticTgo(pz, v.z);
  return Math.hypot(p.x + v.x * tG, p.y + v.y * tG);
}

function ballisticTgo(pz: number, vz: number) {
  if (pz <= 0) return 0;
  if (vz >= 0) {
    const tUp = vz / G0;
    const zPeak = pz + vz * tUp - 0.5 * G0 * tUp * tUp;
    return tUp + Math.sqrt(Math.max(0, (2 * zPeak) / G0));
  }
  const vDown = -vz;
  return (Math.sqrt(Math.max(0, vDown * vDown + 2 * G0 * pz)) - vDown) / G0;
}

export function goalFromState(args: {
  p: Vec3;
  v: Vec3;
  fuel: number;
  lit: boolean;
  prev?: GoalCmd | null;
}): GoalCmd {
  const mass = Math.max(DRY_MASS_KG, wetMass(args.fuel));
  const range = Math.hypot(args.p.x, args.p.y);
  const vh = Math.hypot(args.v.x, args.v.y);
  const vDown = Math.max(0, -args.v.z);
  const pz = Math.max(0.5, engineAlt(args.p.z));
  const { sLight, sVert } = sLightOf(args.p, args.v, mass);
  const pad = SUCCESS_PAD_OFFSET_M;
  const goVertical = (range < pad && vh < SUCCESS_HVEL_MPS) || (range < pad * 0.55 && pz < 80);

  let tilt = 0;
  if (!goVertical) {
    tilt = clamp(range / 160, 0.08, 0.55);
    if (pz < sVert + 70) tilt = Math.min(tilt, range > 32 ? 0.38 : 0.22);
    if (pz < sVert + 30) tilt = Math.min(tilt, range > 28 ? 0.2 : 0.08);
    if (pz < 26) tilt = Math.min(tilt, range < pad ? 0.04 : 0.12);
  }
  let dir = goVertical ? new Vec3(0, 0, 1) : thrustDir(args.p, args.v, tilt);
  const upright = saturate(1 - range / Math.max(pad, 1));
  if (range < pad * 1.6) {
    dir = dir.lerp(new Vec3(0, 0, 1), upright * 0.65);
    const n = dir.len();
    dir = n < 1e-8 ? new Vec3(0, 0, 1) : dir.scale(1 / n);
    tilt *= 1 - upright * 0.8;
  }

  const tFall = (pz - sLight) / Math.max(vDown, 4);
  let tLight = args.lit ? 0 : tFall;

  const aUp = Math.max(4, (MERLIN_THRUST_SL_N * 0.95) / mass - G0);
  const tBurn = Math.max(0.4, vDown / aUp);
  const dv = new Vec3(-args.v.x - args.p.x / tBurn, -args.v.y - args.p.y / tBurn, vDown);
  const tG = ballisticTgo(pz, args.v.z);
  const predMiss = new Vec3(args.p.x + args.v.x * tG, args.p.y + args.v.y * tG, 0);

  if (!args.prev) {
    return { tLight, sLight, dir, dv, predMiss, tiltCmd: tilt, goVertical };
  }

  const mix = args.lit || Math.abs(args.prev.tLight) < 1 ? 0.12 : 0.4;
  let nextDir = args.prev.dir.lerp(dir, mix);
  const dn = nextDir.len();
  nextDir = dn < 1e-8 ? dir : nextDir.scale(1 / dn);
  const nextGo = goVertical || (args.lit && args.prev.goVertical && range < pad * 1.2);
  if (nextGo) {
    nextDir = nextDir.lerp(new Vec3(0, 0, 1), 0.35);
    const gn = nextDir.len();
    nextDir = gn < 1e-8 ? new Vec3(0, 0, 1) : nextDir.scale(1 / gn);
    tilt *= 0.7;
  }
  return {
    tLight: args.lit ? 0 : args.prev.tLight * (1 - mix) + tLight * mix,
    sLight,
    dir: nextDir,
    dv: args.prev.dv.lerp(dv, args.lit ? 0.15 : mix),
    predMiss: args.prev.predMiss.lerp(predMiss, args.lit ? 0.15 : mix),
    tiltCmd: args.prev.tiltCmd * (1 - mix) + tilt * mix,
    goVertical: nextGo,
  };
}

export function goalToRefObs(goal: GoalCmd, p: Vec3) {
  return {
    dp: new Vec3(goal.predMiss.x, goal.predMiss.y, engineAlt(p.z) - goal.sLight),
    dv: goal.dv,
    tLight: goal.tLight,
  };
}
